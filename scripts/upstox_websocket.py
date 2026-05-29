"""
upstox_websocket.py
===================
Persistent WebSocket connection to Upstox V3 Market Data Feed.

Three components:
  - LiveDataCache   : Thread-safe dictionary that stores the latest LTP and
                      completed OHLCV candles for each subscribed instrument.
  - CandleBuilder   : Aggregates raw ticks into 1-min candles, and 1-min
                      candles into 15-min candles, per instrument.
  - UpstoxWebSocketManager : Background daemon thread that maintains the
                      WebSocket connection, decodes protobuf messages,
                      drives the CandleBuilder, and populates the cache.

Design principles:
  - Zero coupling with the rest of the system.  The manager is attached to
    UpstoxSandboxBroker *after* construction and is entirely optional.
  - If the WebSocket is not running or its data is stale the broker falls
    back to REST automatically — no trade is ever missed.
  - Uses the official upstox-python-sdk MarketDataStreamerV3 so that auth,
    protobuf decoding, and ping/pong are handled by Upstox's own library.
  - Thread safety:  RLock on the cache, locks held for microseconds.

Usage:
    from scripts.upstox_websocket import UpstoxWebSocketManager

    manager = UpstoxWebSocketManager(
        access_token="YOUR_ANALYTICS_TOKEN",
        instrument_keys=["NSE_EQ|INE002A01018", ...],
        mode="ltpc",          # lightweight, supports 5000 instruments
    )
    manager.start()           # launches background daemon thread
    broker.attach_websocket(manager)
"""

import threading
import time
import logging
from collections import deque
from datetime import datetime, timedelta

import pandas as pd

logger = logging.getLogger("upstox_ws")

# ──────────────────────────────────────────────────────────────────────────────
# Market session constants (IST, no timezone objects)
# ──────────────────────────────────────────────────────────────────────────────
_MARKET_OPEN_H  = 9
_MARKET_OPEN_M  = 15
_MARKET_CLOSE_H = 15
_MARKET_CLOSE_M = 30

# How stale a cache entry can be before we fall back to REST
_DEFAULT_MAX_STALENESS_SECONDS = 10

# Rolling candle window kept in memory (minutes)
_CANDLE_WINDOW_MINUTES = 120


# ──────────────────────────────────────────────────────────────────────────────
# LiveDataCache
# ──────────────────────────────────────────────────────────────────────────────

class LiveDataCache:
    """
    Thread-safe in-memory store of real-time market data.

    Stores per-instrument:
        _ltp_store   : {instrument_key: (price: float, ts: datetime)}
        _candle_store: {instrument_key: {interval: deque of completed candle dicts}}
    """

    def __init__(self):
        self._lock        = threading.RLock()
        self._ltp_store   = {}   # key → (price, datetime)
        self._candle_store = {}  # key → {"1minute": deque, "15minute": deque}

    # ── LTP ───────────────────────────────────────────────────────────────────

    def update_ltp(self, instrument_key: str, price: float, ts: datetime = None):
        ts = ts or datetime.now()
        with self._lock:
            self._ltp_store[instrument_key] = (price, ts)

    def get_ltp(self, instrument_key: str):
        """Returns (price, timestamp) or (None, None) if not in cache."""
        with self._lock:
            return self._ltp_store.get(instrument_key, (None, None))

    def is_fresh(self, instrument_key: str,
                 max_age_seconds: float = _DEFAULT_MAX_STALENESS_SECONDS) -> bool:
        """True if cache entry exists and is younger than max_age_seconds."""
        with self._lock:
            entry = self._ltp_store.get(instrument_key)
            if entry is None:
                return False
            _, ts = entry
            return (datetime.now() - ts).total_seconds() < max_age_seconds

    # ── Candles ───────────────────────────────────────────────────────────────

    def push_candle(self, instrument_key: str, interval: str, candle: dict):
        """
        Store a completed candle.  Candle dict must have:
            timestamp, open, high, low, close, volume
        """
        with self._lock:
            if instrument_key not in self._candle_store:
                self._candle_store[instrument_key] = {}
            if interval not in self._candle_store[instrument_key]:
                self._candle_store[instrument_key][interval] = deque(
                    maxlen=_CANDLE_WINDOW_MINUTES
                )
            self._candle_store[instrument_key][interval].append(candle)

    def get_candles(self, instrument_key: str, interval: str,
                    count: int = 30) -> pd.DataFrame | None:
        """
        Returns the last `count` completed candles as a DataFrame,
        columns: timestamp, open, high, low, close, volume
        Returns None if fewer than 1 candle available.
        """
        with self._lock:
            store = self._candle_store.get(instrument_key, {})
            candles = store.get(interval)
            if not candles:
                return None
            tail = list(candles)[-count:]

        if not tail:
            return None

        df = pd.DataFrame(tail)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def has_candles(self, instrument_key: str, interval: str,
                    min_count: int = 1) -> bool:
        with self._lock:
            store = self._candle_store.get(instrument_key, {})
            q = store.get(interval)
            return bool(q and len(q) >= min_count)

    def clear(self):
        with self._lock:
            self._ltp_store.clear()
            self._candle_store.clear()


# ──────────────────────────────────────────────────────────────────────────────
# CandleBuilder
# ──────────────────────────────────────────────────────────────────────────────

class CandleBuilder:
    """
    Aggregates raw LTP ticks into 1-minute and 15-minute OHLCV candles.

    Called by the WebSocket manager on every price tick.  When a candle
    period boundary is crossed the completed candle is pushed to the cache.
    """

    def __init__(self, cache: LiveDataCache):
        self._cache     = cache
        self._lock      = threading.Lock()
        # per-instrument state: {key: {interval: current_candle_dict}}
        self._wip       = {}   # work-in-progress (open candle)

    def _bucket(self, ts: datetime, minutes: int) -> datetime:
        """Floor timestamp to the nearest `minutes`-minute boundary."""
        total_minutes = ts.hour * 60 + ts.minute
        floored       = (total_minutes // minutes) * minutes
        return ts.replace(hour=floored // 60, minute=floored % 60,
                          second=0, microsecond=0)

    def on_tick(self, instrument_key: str, price: float, volume: int = 0,
                ts: datetime = None):
        """
        Process a single price tick.  Thread-safe.
        Updates the LTP cache and pushes completed candles to the cache.
        """
        ts = ts or datetime.now()

        # Always keep LTP cache current (one call updates everything)
        self._cache.update_ltp(instrument_key, price, ts)

        with self._lock:
            if instrument_key not in self._wip:
                self._wip[instrument_key] = {}

            for minutes, interval in [(1, "1minute"), (15, "15minute")]:
                bucket = self._bucket(ts, minutes)
                wip    = self._wip[instrument_key].get(interval)

                if wip is None:
                    # Start first candle for this instrument / interval
                    self._wip[instrument_key][interval] = {
                        "timestamp": bucket,
                        "open":  price, "high":  price,
                        "low":   price, "close": price,
                        "volume": volume,
                    }
                    continue

                if bucket == wip["timestamp"]:
                    # Same candle — update OHLCV
                    wip["high"]   = max(wip["high"],  price)
                    wip["low"]    = min(wip["low"],   price)
                    wip["close"]  = price
                    wip["volume"] += volume
                else:
                    # Candle closed — push completed candle to cache
                    completed = dict(wip)
                    self._cache.push_candle(instrument_key, interval, completed)
                    logger.debug(
                        "[CANDLE] %s %s O=%.2f H=%.2f L=%.2f C=%.2f",
                        instrument_key, interval,
                        completed["open"], completed["high"],
                        completed["low"],  completed["close"],
                    )
                    # Open new candle
                    self._wip[instrument_key][interval] = {
                        "timestamp": bucket,
                        "open":  price, "high":  price,
                        "low":   price, "close": price,
                        "volume": volume,
                    }


# ──────────────────────────────────────────────────────────────────────────────
# UpstoxWebSocketManager
# ──────────────────────────────────────────────────────────────────────────────

class UpstoxWebSocketManager:
    """
    Manages a persistent WebSocket connection to Upstox V3 market data feed.

    Runs in a background daemon thread.  The main engine thread never blocks
    on this manager — it only reads from the shared LiveDataCache.

    Reconnection strategy:
        Exponential backoff: 2, 4, 8, 16, 32, 60, 60, 60 ... seconds.
        Re-subscribes to all instruments automatically after reconnect.

    State machine:
        STOPPED → CONNECTING → STREAMING → DISCONNECTED → CONNECTING → ...
    """

    STATES = {"STOPPED", "CONNECTING", "STREAMING", "DISCONNECTED", "FAILED"}

    def __init__(
        self,
        access_token:    str,
        instrument_keys: list[str],
        mode:            str = "ltpc",
        max_retries:     int = 10,
    ):
        """
        Parameters
        ----------
        access_token     Upstox analytics token (UPSTOX_ANALYTICS_ACCESS_TOKEN)
        instrument_keys  List of Upstox instrument keys to subscribe to
                         e.g. ["NSE_EQ|INE002A01018", "NSE_INDEX|Nifty 50"]
        mode             Subscription mode: "ltpc" | "full" | "option_greeks"
                         "ltpc" supports up to 5000 instruments and is lightest.
        max_retries      Max reconnection attempts before entering FAILED state.
                         Set to 0 for unlimited.
        """
        if not access_token:
            raise ValueError("access_token is required for UpstoxWebSocketManager")

        self._token           = access_token
        self._instrument_keys = list(instrument_keys)
        self._mode            = mode
        self._max_retries     = max_retries

        self.cache            = LiveDataCache()
        self._candle_builder  = CandleBuilder(self.cache)

        self._state           = "STOPPED"
        self._state_lock      = threading.Lock()
        self._retry_count     = 0
        self._streamer        = None   # upstox_client.MarketDataStreamerV3 instance
        self._thread          = None
        self._stop_event      = threading.Event()

        # Stats for monitoring
        self.stats = {
            "ticks_received":     0,
            "reconnect_count":    0,
            "last_tick_time":     None,
            "last_connect_time":  None,
            "last_error":         None,
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        """Launch the background WebSocket daemon thread."""
        if self._thread and self._thread.is_alive():
            logger.warning("[WS] Already running — ignoring start()")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="UpstoxWebSocket",
            daemon=True,  # dies automatically when main process exits
        )
        self._thread.start()
        logger.info("[WS] Background thread started (mode=%s, instruments=%d)",
                    self._mode, len(self._instrument_keys))
        print(f"[WS] WebSocket manager started — subscribing to "
              f"{len(self._instrument_keys)} instruments in '{self._mode}' mode.")

    def stop(self):
        """Gracefully stop the WebSocket thread."""
        self._stop_event.set()
        if self._streamer:
            try:
                self._streamer.disconnect()
            except Exception:
                pass
        self._set_state("STOPPED")
        logger.info("[WS] Manager stopped.")

    def subscribe(self, instrument_keys: list[str], mode: str = None):
        """
        Dynamically add instruments to the live subscription.
        Safe to call from any thread.
        """
        mode = mode or self._mode
        new_keys = [k for k in instrument_keys if k not in self._instrument_keys]
        if new_keys:
            self._instrument_keys = self._instrument_keys + new_keys

        if self._streamer and self._state == "STREAMING":
            try:
                self._streamer.subscribe(instrument_keys, mode)
                logger.info("[WS] Subscribed to %d new instruments (%s)",
                            len(instrument_keys), mode)
            except Exception as e:
                logger.warning("[WS] subscribe() failed: %s", e)

    def unsubscribe(self, instrument_keys: list[str]):
        """Dynamically remove instruments from the live subscription."""
        self._instrument_keys = [k for k in self._instrument_keys if k not in instrument_keys]

        if self._streamer and self._state == "STREAMING":
            try:
                self._streamer.unsubscribe(instrument_keys)
            except Exception as e:
                logger.warning("[WS] unsubscribe() failed: %s", e)

    def change_mode(self, instrument_keys: list[str], mode: str):
        """Upgrade or downgrade subscription mode for specific instruments."""
        if self._streamer and self._state == "STREAMING":
            try:
                self._streamer.change_mode(instrument_keys, mode)
                logger.info("[WS] Mode changed to '%s' for %d instruments",
                            mode, len(instrument_keys))
            except Exception as e:
                logger.warning("[WS] change_mode() failed: %s", e)

    @property
    def is_streaming(self) -> bool:
        return self._state == "STREAMING"

    @property
    def state(self) -> str:
        return self._state

    def get_stats(self) -> dict:
        return dict(self.stats)

    # ── Internal state machine ────────────────────────────────────────────────

    def _set_state(self, new_state: str):
        with self._state_lock:
            old = self._state
            self._state = new_state
            if old != new_state:
                logger.info("[WS] State: %s → %s", old, new_state)

    def _run_loop(self):
        """
        Main reconnection loop.  Runs in the daemon thread.
        Each iteration attempts one full connection lifetime.
        """
        backoff_schedule = [10, 20, 30, 60, 90, 90]  # Upstox rate-limits rapid reconnects with 403; give server time to clear

        while not self._stop_event.is_set():
            try:
                self._set_state("CONNECTING")
                self._connect_and_stream()      # blocks until disconnected
            except Exception as e:
                self.stats["last_error"] = str(e)
                logger.error("[WS] Connection error: %s", e)

            if self._stop_event.is_set():
                break

            self._retry_count += 1
            self.stats["reconnect_count"] += 1

            if self._max_retries > 0 and self._retry_count >= self._max_retries:
                self._set_state("FAILED")
                print(f"[WS] ⚠ Max retries ({self._max_retries}) exhausted. "
                      f"System will continue on REST API only. "
                      f"WebSocket will retry at next market open.")
                # Wait until next market open before trying again
                self._wait_for_market_open()
                self._retry_count = 0
                continue

            idx     = min(self._retry_count - 1, len(backoff_schedule) - 1)
            wait    = backoff_schedule[idx]
            self._set_state("DISCONNECTED")
            print(f"[WS] Reconnecting in {wait}s (attempt {self._retry_count})...")
            self._stop_event.wait(timeout=wait)

        self._set_state("STOPPED")

    def _connect_and_stream(self):
        """
        Establish one WebSocket session using the Upstox SDK streamer.
        Blocks until the connection is closed or an error occurs.
        """
        import upstox_client

        # Build SDK configuration with the analytics (live) token
        configuration                  = upstox_client.Configuration()
        configuration.access_token     = self._token
        # Force live endpoint (not sandbox)
        configuration.sandbox          = False
        configuration.host             = "https://api.upstox.com"

        api_client  = upstox_client.ApiClient(configuration)
        streamer    = upstox_client.MarketDataStreamerV3(api_client=api_client)
        self._streamer = streamer

        # Connection lifecycle events
        connected_event   = threading.Event()
        error_event       = threading.Event()
        error_holder      = [None]

        def on_open(*args, **kwargs):
            self._set_state("STREAMING")
            self._retry_count          = 0     # reset on successful connection
            self.stats["last_connect_time"] = datetime.now()
            connected_event.set()
            print(f"[WS] ✓ Connected to Upstox market feed "
                  f"({len(self._instrument_keys)} instruments).")

            # Subscribe to the full instrument universe
            if self._instrument_keys:
                try:
                    streamer.subscribe(self._instrument_keys, self._mode)
                    logger.info("[WS] Subscribed: %d instruments, mode=%s",
                                len(self._instrument_keys), self._mode)
                except Exception as sub_err:
                    logger.warning("[WS] Initial subscribe failed: %s", sub_err)

        def on_message(*args, **kwargs):
            """Called on every incoming protobuf frame."""
            # Extract the actual message (usually the second arg if (ws, msg) is passed, or the first if (msg) is passed)
            message = args[1] if len(args) >= 2 else (args[0] if len(args) == 1 else None)
            if message is None:
                message = kwargs.get("message")
            if message is not None:
                try:
                    self._handle_message(message)
                except Exception as msg_err:
                    logger.debug("[WS] Message parse error: %s", msg_err)

        def on_error(*args, **kwargs):
            # Extract the actual error (usually the second arg if (ws, err) is passed, or the first if (err) is passed)
            error = args[1] if len(args) >= 2 else (args[0] if len(args) == 1 else "Unknown Error")
            if error == "Unknown Error":
                error = kwargs.get("error", "Unknown Error")
            self.stats["last_error"] = str(error)
            logger.error("[WS] WebSocket error: %s", error)
            error_holder[0] = error
            error_event.set()

        def on_close(*args, **kwargs):
            logger.info("[WS] Connection closed. Args: %s, Kwargs: %s", args, kwargs)
            error_event.set()   # unblock the wait below

        # Register handlers
        streamer.on("open",    on_open)
        streamer.on("message", on_message)
        streamer.on("error",   on_error)
        streamer.on("close",   on_close)

        # Connect (non-blocking — handlers run in SDK-managed thread)
        streamer.connect()

        # Block this thread until the connection closes or we're asked to stop
        while not self._stop_event.is_set():
            if error_event.wait(timeout=5):
                break

        # Clean disconnect
        try:
            streamer.disconnect()
        except Exception:
            pass
        self._streamer = None

    def _handle_message(self, message):
        """
        Decode a protobuf message from the Upstox V3 WebSocket feed and
        update the cache + candle builder.

        The SDK's MarketDataStreamerV3 already decodes the protobuf for us
        and passes the decoded FeedResponse object (or raw bytes depending
        on SDK version).  We handle both cases.
        """
        now = datetime.now()

        # ── Case 1: SDK pre-decoded the message into a dict-like object ───────
        if hasattr(message, "feeds"):
            # FeedResponse protobuf object with .feeds dict
            self._process_feed_response(message, now)
            return

        # ── Case 2: SDK passed raw bytes — decode with protobuf ───────────────
        if isinstance(message, (bytes, bytearray)):
            try:
                from scripts import MarketDataFeed_pb2 as pb
                feed_response = pb.FeedResponse()
                feed_response.ParseFromString(message)
                self._process_feed_response(feed_response, now)
            except ImportError:
                # proto file not compiled yet — parse minimally
                logger.warning(
                    "[WS] MarketDataFeed_pb2 not found. "
                    "Run scripts/compile_proto.py to enable full tick parsing."
                )
            except Exception as decode_err:
                logger.debug("[WS] Protobuf decode error: %s", decode_err)
            return

        # ── Case 3: SDK returned a plain dict (some SDK versions) ─────────────
        if isinstance(message, dict):
            self._process_dict_message(message, now)

    def _process_feed_response(self, feed_response, now: datetime):
        """Extract LTPC data from a protobuf FeedResponse object."""
        try:
            for instrument_key, feed in feed_response.feeds.items():
                price  = None
                volume = 0

                # LTPC mode
                if feed.HasField("ltpc"):
                    price = feed.ltpc.ltp

                # Full mode — also has LTPC inside
                elif feed.HasField("fullFeed"):
                    ff = feed.fullFeed
                    if ff.HasField("marketFF"):
                        price = ff.marketFF.ltpc.ltp
                    elif ff.HasField("indexFF"):
                        price = ff.indexFF.ltpc.ltp

                if price and price > 0:
                    self._candle_builder.on_tick(instrument_key, float(price),
                                                 volume, now)
                    self.stats["ticks_received"] += 1
                    self.stats["last_tick_time"]  = now

        except Exception as e:
            logger.debug("[WS] _process_feed_response error: %s", e)

    def _process_dict_message(self, data: dict, now: datetime):
        """Fallback parser for SDK versions that return plain dicts."""
        try:
            feeds = data.get("feeds", {})
            for instrument_key, feed in feeds.items():
                ltpc  = feed.get("ltpc", {})
                price = ltpc.get("ltp") or ltpc.get("last_price")
                if price and float(price) > 0:
                    p = float(price)
                    self._candle_builder.on_tick(instrument_key, p, 0, now)
                    self.stats["ticks_received"] += 1
                    self.stats["last_tick_time"]  = now
        except Exception as e:
            logger.debug("[WS] _process_dict_message error: %s", e)

    def _wait_for_market_open(self):
        """
        Sleep until 09:15 IST on the next trading day.
        Called when max retries are exhausted.
        """
        while not self._stop_event.is_set():
            now    = datetime.now()
            target = now.replace(
                hour=_MARKET_OPEN_H, minute=_MARKET_OPEN_M,
                second=0, microsecond=0
            )
            if now >= target:
                target += timedelta(days=1)

            wait_seconds = (target - now).total_seconds()
            print(f"[WS] Waiting {wait_seconds/3600:.1f}h for market open "
                  f"before retrying WebSocket connection...")
            # Sleep in 60-second chunks so we can be interrupted by stop()
            for _ in range(int(wait_seconds // 60)):
                if self._stop_event.is_set():
                    return
                time.sleep(60)

            # Check if it's actually market hours now
            now = datetime.now()
            open_time  = now.replace(hour=_MARKET_OPEN_H,  minute=_MARKET_OPEN_M,  second=0)
            close_time = now.replace(hour=_MARKET_CLOSE_H, minute=_MARKET_CLOSE_M, second=0)
            if open_time <= now <= close_time:
                return

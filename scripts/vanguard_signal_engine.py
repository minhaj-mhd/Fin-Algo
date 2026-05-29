import os
import sys

# Add project root to path before importing local modules
sys.path.append(os.getcwd())

import time
import threading
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import json
import yfinance as yf
from google import genai
from google.genai import types
import traceback
from datetime import datetime, timedelta
from dotenv import load_dotenv

from scripts.database_manager import init_db, log_trade, get_trades_by_status
from scripts.upstox_broker import UpstoxSandboxBroker
from scripts.tv_ta import get_tv_sentiment
from scripts.terminal_utils import log

# Load API keys from .env
load_dotenv()

try:
    from scripts.tickers import TICKERS
except ImportError:
    TICKERS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]

try:
    from scripts.database_manager import (
        init_db,
        log_trade,
        get_trades_by_status,
        log_system_stats,
    )
except ImportError:

    def init_db():
        pass

    def log_trade(data):
        pass

    def get_trades_by_status(status, limit=50):
        return []

class GeminiRateTracker:
    def __init__(self, state_file="data/gemini_usage.json", max_keys=3):
        self.state_file = state_file
        self.models = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
        self.max_requests_per_day = 20
        self.max_keys = max_keys
        self.state = self._load_state()

    def _load_state(self):
        today = datetime.now().strftime("%Y-%m-%d")
        default_state = {
            "date": today,
            "usage": {m: {str(i): 0 for i in range(self.max_keys)} for m in self.models}
        }
        
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    state = json.load(f)
                if state.get("date") == today:
                    modified = False
                    for m in self.models:
                        if m not in state["usage"]:
                            state["usage"][m] = {str(i): 0 for i in range(self.max_keys)}
                            modified = True
                        else:
                            # Ensure all active key indexes are present
                            for i in range(self.max_keys):
                                k_str = str(i)
                                if k_str not in state["usage"][m]:
                                    state["usage"][m][k_str] = 0
                                    modified = True
                            # Prune extra keys that are no longer active
                            keys_to_remove = [k for k in state["usage"][m] if int(k) >= self.max_keys]
                            if keys_to_remove:
                                for k in keys_to_remove:
                                    del state["usage"][m][k]
                                modified = True
                    if modified:
                        self._save_state(state)
                    return state
            except Exception:
                pass
        
        self._save_state(default_state)
        return default_state

    def _save_state(self, state=None):
        if state is None:
            state = self.state
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=4)

    def get_next_available(self, max_keys=3):
        """Returns (model_name, key_idx), resetting all usages to 0 if all are exhausted."""
        for model in self.models:
            for i in range(max_keys):
                key_idx = str(i)
                if self.state["usage"][model].get(key_idx, 0) < self.max_requests_per_day:
                    return model, i
        
        # If all keys/models are exhausted, reset them to zero to rotate again
        print("[ROTATE-RESET] All API keys are exhausted. Resetting all usage statistics to 0 to rotate again.")
        for model in self.models:
            for i in range(max_keys):
                key_idx = str(i)
                self.state["usage"][model][key_idx] = 0
        self._save_state()
        
        if self.models and max_keys > 0:
            return self.models[0], 0
        return None, None

    def increment_usage(self, model, key_idx):
        idx_str = str(key_idx)
        if model in self.state["usage"] and idx_str in self.state["usage"][model]:
            self.state["usage"][model][idx_str] += 1
            self._save_state()

    def mark_exhausted(self, model, key_idx):
        """Used when a 429 rate limit is encountered before reaching 20."""
        idx_str = str(key_idx)
        if model in self.state["usage"] and idx_str in self.state["usage"][model]:
            self.state["usage"][model][idx_str] = self.max_requests_per_day
            self._save_state()



class VanguardEngine:
    def __init__(self, model_path, scaler_path, meta_path):
        log("\n" + "=" * 60)
        log("VANGUARD ENSEMBLE V2.3 - INDUSTRIAL HARDENED")
        log("=" * 60)

        # 1. LOAD MODELS (via Model Registry — falls back to legacy paths)
        try:
            from scripts.model_registry import ModelRegistry
            registry = ModelRegistry()
            active = registry.get_active_model()
            long_model_path  = active["long_model"]
            short_model_path = active["short_model"]
            resolved_meta    = active["meta"]
            resolved_scaler  = active["scaler"]
            self._active_model_name = active["name"]
            log(f"[REGISTRY] Active model: {active['name']}")
        except Exception as reg_err:
            # Fallback: use the paths passed to __init__ (legacy behaviour)
            log(f"[WARN] Registry unavailable ({reg_err}). Using legacy paths.")
            model_dir        = os.path.dirname(model_path)
            long_model_path  = os.path.join(model_dir, "xgb_long_model.json")
            short_model_path = os.path.join(model_dir, "xgb_short_model.json")
            resolved_meta    = meta_path
            resolved_scaler  = scaler_path
            self._active_model_name = "v1_yfinance_ranker"

        try:
            self.bst_long = xgb.Booster()
            self.bst_long.load_model(long_model_path)
            self.bst_long.set_param({'device': 'cuda'})

            self.bst_short = xgb.Booster()
            self.bst_short.load_model(short_model_path)
            self.bst_short.set_param({'device': 'cuda'})

            # Load scaler only if the registry specifies a path (V4 doesn't use one)
            if resolved_scaler and os.path.exists(resolved_scaler):
                self.scaler = pickle.load(open(resolved_scaler, "rb"))
                log(f"[INFO] Scaler loaded from {resolved_scaler}")
            else:
                self.scaler = None
                log(f"[INFO] No scaler configured for {self._active_model_name} (scale-invariant XGBoost)")
            self.feature_cols = json.load(open(resolved_meta))["features"]
            log(f"[OK] ML Models: {len(self.feature_cols)} features | LOADED ({self._active_model_name})")
        except Exception as e:
            log(f"[ERROR] Critical Load Error: {e}")
            sys.exit(1)

        # 2. INITIALIZE DB
        init_db()

        # 3. CONFIGURE GEMINI AI (ENABLED)
        self.gemini_enabled = True
        self.api_keys = []
        self.clients = []
        
        # S1 Persistent Rotation State
        self.s1_model_tiers = ["gemini-3.5-flash", "gemini-3.1-flash-lite"]
        self.s1_active_tier_idx = 0
        self.s1_active_key_idx = 0

        try:
            keys_env = os.getenv("GEMINI_API_KEYS") or os.getenv("GEMINI_API_KEY")
            if keys_env:
                self.api_keys = [k.strip() for k in keys_env.split(",") if k.strip()]

            if self.api_keys:
                self.clients = [genai.Client(api_key=k) for k in self.api_keys]
                self.client = self.clients[0]
                log(
                    f"[OK] Gemini AI Audit Layer: ACTIVE ({len(self.api_keys)} Keys Loaded)"
                )
            else:
                self.gemini_enabled = False
                log("[WARN] Gemini API Keys not found. AI Audit Layer: DISABLED.")
        except Exception as e:
            self.gemini_enabled = False
            log(f"[ERROR] Gemini Config Error: {e}")

        # Initialize tracker after knowing the number of keys
        self.gemini_tracker = GeminiRateTracker(max_keys=len(self.api_keys) if self.api_keys else 1)

        # 4. INTERNAL STATE & THREADING
        self.active_shadow_trades = []
        self.sentiment_cache = {}  # ticker -> (sentiment, reason, timestamp)
        self.atr_cache = {}        # ticker -> (sl_pct, tp_pct, timestamp)
        self.lock = threading.Lock()
        # Cache of the most recent full-universe conviction scores.
        # Used by _get_current_conviction so single-ticker re-scoring doesn't
        # collapse all Z-scored features to 0 (std=NaN on a 1-row DataFrame).
        self.latest_full_scores = None

        # Cooldown tracker: tickers closed/expired in the last 30 min are blocked from new entries
        # Format: {ticker: datetime_closed}
        self.recently_closed: dict = {}
        self.recent_vetoes: dict = {}  # Format: {ticker: datetime_vetoed}

        # Initialize Upstox Broker for execution and live data
        self.broker = UpstoxSandboxBroker()

        # Tracks the last time a conviction-flip check ran per trade_id.
        # Format: {trade_id: datetime}
        self._conviction_flip_checked: dict = {}

        # ── WebSocket: real-time market data feed ─────────────────────────────
        # Runs in a background daemon thread. The main engine reads from the
        # shared LiveDataCache instead of making REST calls for live prices.
        # Falls back to REST automatically if the WebSocket is not connected.
        self._ws_manager = None
        self._start_websocket()

        # --- Capital Management Config ---
        self.max_trade_slots = 5
        self.margin_multiplier = 5.0
        self.brokerage_per_order = 10.0  # ₹10 Buy + ₹10 Sell = ₹20 Round-trip
        self.stt_rate = 0.00025  # 0.025% on Selling side

        self.stats_file = "upstox_stats.json"
        self.initial_capital = 99517.68   # Baseline — start of 2026-05-21
        self.virtual_capital = 99517.68
        self.used_margin = 0.0
        self.realized_charges = 0.0
        self._load_virtual_stats()        # Restores capital + charges from JSON

        self.current_date = datetime.now().date()
        self.day_start_capital = self.virtual_capital

        self.min_conviction = 0.10
        self.min_raw_score = 0.12

        # ── SESSION VETO STATS (reset each trading day) ───────────────────
        self._veto_stats_date = datetime.now().date()
        self.veto_stats = {
            "s1_vetoes":  0,   # Stage 1 (fast technical) vetoed
            "s2_vetoes":  0,   # Stage 2 (news/CRO) vetoed
            "s1_passes":  0,   # Stage 1 passed → went to S2
            "s2_passes":  0,   # Stage 2 passed → trade placed
            "s1_tickers": [],  # [(ticker, side, reason)]
            "s2_tickers": [],  # [(ticker, side, rule, reason)]
        }

        # 5. FETCH STATIC METRICS (Market Cap, 52W High)
        self.ticker_metadata = {}
        self.fetch_static_metadata(TICKERS)

        # 6. RESUME OPEN TRADES FROM DATABASE
        self.load_open_trades()

        # Start Shadow Tracker Thread
        threading.Thread(target=self.shadow_tracker_loop, daemon=True).start()

    def _start_websocket(self):
        """
        Initialise the Upstox V3 WebSocket and attach it to the broker.
        Called once during __init__.  Errors are non-fatal — the system
        simply continues with REST-only data if the WS fails to start.
        """
        try:
            from scripts.upstox_websocket import UpstoxWebSocketManager

            analytics_token = os.getenv("UPSTOX_ANALYTICS_ACCESS_TOKEN")
            if not analytics_token:
                log("[WS] UPSTOX_ANALYTICS_ACCESS_TOKEN not set — WebSocket disabled.")
                return

            # Resolve instrument keys for the full ticker universe.
            # get_instrument_key() uses a local JSON cache so this is fast
            # (no REST calls for already-cached symbols).
            log(f"[WS] Resolving instrument keys for {len(TICKERS)} tickers...")
            instrument_keys = []
            for ticker in TICKERS:
                try:
                    key = self.broker.get_instrument_key(ticker)
                    if key:
                        instrument_keys.append(key)
                except Exception:
                    pass

            if not instrument_keys:
                log("[WS] No instrument keys resolved — WebSocket disabled.")
                return

            self._ws_manager = UpstoxWebSocketManager(
                access_token=analytics_token,
                instrument_keys=instrument_keys,
                mode="ltpc",      # lightweight: LTP + timestamp + close price
                max_retries=10,   # exponential backoff, retries at market open after failure
            )
            self._ws_manager.start()
            self.broker.attach_websocket(self._ws_manager)
            log(f"[WS] WebSocket started — {len(instrument_keys)} instruments subscribed.")
        except Exception as e:
            log(f"[WS] WebSocket init failed ({e}) — continuing with REST-only data.")
            self._ws_manager = None

    def _get_active_s1_client(self):
        """Returns the current active S1 model and client."""
        return self.s1_model_tiers[self.s1_active_tier_idx], self.clients[self.s1_active_key_idx]

    def fetch_static_metadata(self, tickers):
        print(f"[INIT] Fetching Market Metadata for {len(tickers)} symbols...")
        # To avoid blocking startup for too long, we fetch in chunks or use a faster method if possible
        # For now, we fetch 1-year daily data for 52W Highs and try to get info for Market Cap
        try:
            # 52W Highs/Lows from daily data
            data = yf.download(
                tickers,
                period="1y",
                interval="1d",
                progress=False,
                auto_adjust=True,
                timeout=30,
            )
            for ticker in tickers:
                try:
                    if len(tickers) > 1:
                        df = data.xs(ticker, axis=1, level=1).dropna()
                    else:
                        df = data.dropna()

                    if not df.empty:
                        self.ticker_metadata[ticker] = {
                            "high_52w": float(df["High"].max()),
                            "low_52w": float(df["Low"].min()),
                            "avg_vol_20d": float(df["Volume"].tail(20).mean()),
                            "market_cap": 0,  # Placeholder for now, yf.download doesn't provide MC
                        }
                except Exception:
                    continue

            # Note: Fetching raw Market Cap for 170+ symbols via yf.Ticker().info is very slow.
            # We will use Volume/Liquidity as a proxy or fetch it lazily if needed.
            # For this implementation, we will focus on Dollar Volume and RVOL which are more dynamic.
            log(
                f"[OK] Market Metadata: {len(self.ticker_metadata)} symbols processed."
            )
        except Exception as e:
            log(f"[WARN] Static Metadata Error: {e}")

    def load_open_trades(self):
        """Resumes tracking OPEN and PENDING_ENTRY trades from the DB after a restart.
        - Recomputes used_margin from resumed trades so the dashboard is correct immediately.
        - Extends the exit_time of any trade whose window has already expired while the
          engine was offline, giving the shadow tracker a 30-minute grace period to close
          it cleanly before a new scan cycle can place a duplicate entry.
        """
        try:
            open_trades = get_trades_by_status(["OPEN", "PENDING_ENTRY"], 100)
            if not open_trades:
                print("[INIT] No active trades (OPEN or PENDING_ENTRY) found in DB to resume.")
                self.update_upstox_stats()  # Save clean baseline state
                return

            # Deduplicate: keep only the latest OPEN trade per (ticker, side) pair
            seen = {}
            for trade in sorted(open_trades, key=lambda t: t["timestamp"]):
                key = (trade["ticker"], trade["side"])
                seen[key] = trade  # later entry overwrites earlier — keeps latest

            # Mark stale duplicates as CLOSED in DB
            latest_ids = {t["trade_id"] for t in seen.values()}
            for trade in open_trades:
                if trade["trade_id"] not in latest_ids:
                    trade["status"] = "CLOSED"
                    trade["final_profit_pct"] = 0.0
                    trade["comment"] = "Duplicate – closed on restart"
                    log_trade(trade)

            # Load deduplicated trades into memory
            recovered_margin = 0.0
            now = datetime.now()
            grace_extension = timedelta(minutes=30)   # keep open for shadow tracker to close cleanly
            with self.lock:
                for trade in seen.values():
                    t = dict(trade)

                    # --- GRACE PERIOD FOR EXPIRED TRADES ---
                    # If the trade's exit window passed while we were offline, extend it
                    # so the shadow tracker can close it properly (fetch live price, log P&L)
                    # rather than having a new scan immediately re-enter the same ticker.
                    try:
                        if t["status"] == "PENDING_ENTRY":
                            pending_since = datetime.fromisoformat(t.get("pending_since") or t["timestamp"])
                            minute = pending_since.minute
                            next_15 = ((minute // 15) + 1) * 15
                            next_boundary = pending_since.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=next_15)
                            if now > next_boundary + timedelta(minutes=5):
                                print(f"[RESTART] PENDING_ENTRY for {t['ticker']} has expired (timestamp: {pending_since.strftime('%H:%M')}). Cancelling.")
                                t["status"] = "CANCELLED"
                                t["comment"] = "Cancelled on restart - confirmation window expired."
                                log_trade(t)
                                continue
                        else:
                            exit_dt = datetime.fromisoformat(t["exit_time"])
                            if exit_dt < now:
                                new_exit = now + grace_extension
                                t["exit_time"] = new_exit.isoformat()
                                log(f"[RESTART] {t['ticker']} exit_time extended to {new_exit.strftime('%H:%M')} "
                                      f"(was {exit_dt.strftime('%H:%M')}, engine was offline)")
                    except Exception:
                        pass

                    self.active_shadow_trades.append(t)

                    # Recalculate margin for trades that may lack the margin_used field
                    m = t.get("margin_used")
                    if not m or m == 0:
                        qty   = t.get("quantity", 1)
                        price = t.get("entry_price", 0)
                        m = (qty * price) / self.margin_multiplier
                    recovered_margin += m

            self.used_margin = recovered_margin
            log(f"[INIT] Resumed {len(seen)} open trade(s) from DB. "
                  f"Used Margin restored: ₹{self.used_margin:.2f}")
        except Exception as e:
            log(f"[WARN] Could not load open trades from DB: {e}")
        finally:
            # Always persist the state (with or without trades) so the dashboard is correct
            self.update_upstox_stats()

    def get_last_completed_15min_candle(self, ticker):
        """
        Returns the last completed 15-min candle for entry confirmation.
        Uses WebSocket candle cache if available (no REST call), otherwise
        falls back to fetching 1-min data from Upstox REST and resampling.
        """
        # ── Fast path: try the 15-min candle cache from the WS builder ────────
        if self._ws_manager is not None:
            try:
                instrument_key = self.broker.get_instrument_key(ticker)
                df_15m = self._ws_manager.cache.get_candles(
                    instrument_key, "15minute", count=5
                )
                if df_15m is not None and not df_15m.empty:
                    # Candles in cache are completed (builder only pushes on boundary)
                    last_row = df_15m.iloc[-1]
                    return {
                        'timestamp': last_row['timestamp'],
                        'open':  float(last_row['open']),
                        'high':  float(last_row['high']),
                        'low':   float(last_row['low']),
                        'close': float(last_row['close']),
                    }
            except Exception:
                pass   # fall through to REST path

        # ── Slow path: REST fetch + resample (original logic) ─────────────────
        df = self.broker.get_recent_candles(ticker, interval='1minute', count=120)
        if df is None or df.empty:
            return None

        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')

        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        resampled = df.resample('15min', origin='start_day').agg({
            'open': 'first',
            'high': 'max',
            'low':  'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        if resampled.empty:
            return None

        now = datetime.now()
        completed = resampled[resampled.index + pd.Timedelta(minutes=15) <= now]
        if completed.empty:
            return None

        last_idx = completed.index[-1]
        last_row = completed.iloc[-1]
        return {
            'timestamp': last_idx,
            'open':  float(last_row['open']),
            'high':  float(last_row['high']),
            'low':   float(last_row['low']),
            'close': float(last_row['close']),
        }

    def compute_15min_atr(self, ticker):
        """
        Computes 14-period EWM ATR from 15-min candles (2 days of data).
        Returns (sl_pct, tp_pct) — both as percentages, already multiplied and capped.
        """
        ATR_MULTIPLIER = 1.5
        ATR_TP_MULTIPLIER = 3.0
        ATR_SL_MIN_PCT = 0.30
        ATR_SL_MAX_PCT = 1.50
        ATR_SL_DEFAULT = 0.50
        ATR_TP_MIN_PCT = 0.75
        ATR_TP_MAX_PCT = 2.50
        ATR_TP_DEFAULT = 1.00

        # Check cache first (15-min TTL)
        cached = self.atr_cache.get(ticker)
        if cached:
            c_sl, c_tp, c_ts = cached
            if (datetime.now() - c_ts).total_seconds() < 900:  # 15 min TTL
                return c_sl, c_tp

        try:
            # get_recent_candles() reads from the WS candle cache first;
            # only makes a REST call if the cache doesn't have enough data.
            df = self.broker.get_recent_candles(ticker, interval='1minute', count=120)
            if df is None or df.empty:
                return ATR_SL_DEFAULT, ATR_TP_DEFAULT

            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            import pandas as pd
            resampled = df.resample('15min', origin='start_day').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
            }).dropna()

            now = datetime.now()
            completed = resampled[resampled.index + pd.Timedelta(minutes=15) <= now]

            if len(completed) < 5:
                return ATR_SL_DEFAULT, ATR_TP_DEFAULT

            high = completed['high']
            low = completed['low']
            close = completed['close']
            prev_close = close.shift(1)

            tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
            atr_value = tr.ewm(span=14, adjust=False).mean().iloc[-1]

            if pd.isna(atr_value) or atr_value <= 0:
                return ATR_SL_DEFAULT, ATR_TP_DEFAULT

            last_price = float(close.iloc[-1])
            atr_pct = (atr_value / last_price) * 100

            sl_pct = max(ATR_SL_MIN_PCT, min(ATR_SL_MAX_PCT, atr_pct * ATR_MULTIPLIER))
            tp_pct = max(ATR_TP_MIN_PCT, min(ATR_TP_MAX_PCT, atr_pct * ATR_TP_MULTIPLIER))

            r_sl = round(sl_pct, 3)
            r_tp = round(tp_pct, 3)
            self.atr_cache[ticker] = (r_sl, r_tp, datetime.now())
            return r_sl, r_tp
        except Exception as e:
            print(f"[ATR] Failed for {ticker}: {e}. Using defaults.")
            return ATR_SL_DEFAULT, ATR_TP_DEFAULT

    def calculate_conviction_scores(self, tickers):
        all_latest_data = []
        valid_tickers = []

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Pulse Scan initiated ({len(tickers)} symbols)...")

        # ── FETCH NIFTY & VIX CONTEXT ──────────────────────────────────────
        nifty_features = {
            'Nifty_1H_Return': 0.0, 'Nifty_3H_Return': 0.0, 'Nifty_5H_Return': 0.0,
            'Nifty_RSI': 50.0, 'Nifty_HL_Range': 0.0, 'Nifty_20H_Std': 0.0
        }
        try:
            print("[INFO] Fetching Nifty50 hourly context...")
            nifty_raw = yf.download("^NSEI", period="15d", interval="1h", progress=False, auto_adjust=True)
            if not nifty_raw.empty:
                if isinstance(nifty_raw.columns, pd.MultiIndex):
                    nifty_raw.columns = [col[0] for col in nifty_raw.columns]
                nifty_raw['Nifty_1H_Return'] = nifty_raw['Close'].pct_change(1)
                nifty_raw['Nifty_3H_Return'] = nifty_raw['Close'].pct_change(3)
                nifty_raw['Nifty_5H_Return'] = nifty_raw['Close'].pct_change(5)
                from scripts.feature_utils import RSI
                nifty_raw['Nifty_RSI'] = RSI(nifty_raw['Close'], 14)
                nifty_raw['Nifty_HL_Range'] = (nifty_raw['High'] - nifty_raw['Low']) / nifty_raw['Close']
                nifty_raw['Nifty_20H_Std'] = nifty_raw['Nifty_1H_Return'].rolling(20).std()
                
                latest_nifty = nifty_raw.iloc[-1]
                for key in nifty_features:
                    val = latest_nifty.get(key)
                    if pd.notna(val):
                        nifty_features[key] = float(val)
        except Exception as e:
            print(f"[WARN] Nifty50 fetch failed: {e}")

        vix_features = {
            'VIX_Level': 15.0, 'VIX_Change': 0.0, 'VIX_5D_MA': 15.0,
            'VIX_High': 0, 'VIX_Extreme': 0
        }
        try:
            print("[INFO] Fetching India VIX daily context...")
            vix_raw = yf.download("^INDIAVIX", period="15d", interval="1d", progress=False, auto_adjust=True)
            if not vix_raw.empty:
                if isinstance(vix_raw.columns, pd.MultiIndex):
                    vix_raw.columns = [col[0] for col in vix_raw.columns]
                vix_raw['VIX_Level'] = vix_raw['Close']
                vix_raw['VIX_Change'] = vix_raw['Close'].pct_change(1)
                vix_raw['VIX_5D_MA'] = vix_raw['Close'].rolling(5).mean()
                vix_raw['VIX_High'] = (vix_raw['Close'] > 18).astype(int)
                vix_raw['VIX_Extreme'] = (vix_raw['Close'] > 22).astype(int)
                
                latest_vix = vix_raw.iloc[-1]
                for key in vix_features:
                    val = latest_vix.get(key)
                    if pd.notna(val):
                        vix_features[key] = float(val)
        except Exception as e:
            print(f"[WARN] India VIX fetch failed: {e}")

        # ── BATCH FETCH DAILY DATA FOR MULTI-TF ────────────────────────────
        daily_data = pd.DataFrame()
        try:
            print(f"[INFO] Fetching 60d daily history for {len(tickers)} symbols from yfinance...")
            daily_data = yf.download(
                tickers, period="60d", interval="1d",
                progress=False, auto_adjust=True, timeout=30
            )
        except Exception as e:
            print(f"[WARN] Batch daily fetch failed: {e}")

        all_dfs = {}
        try:
            # Batch fetch historical data from Upstox (one by one due to API limitations)
            print(f"[INFO] Fetching 60d hourly history for {len(tickers)} symbols from Upstox...")
            for i, ticker in enumerate(tickers):
                try:
                    hist_df = self.broker.get_historical_data(ticker, interval="60minute", days=60, fallback=False)
                    if hist_df is not None and not hist_df.empty:
                        # Rename columns to match expected format
                        hist_df = hist_df.rename(columns={
                            "open": "Open", "high": "High", "low": "Low", 
                            "close": "Close", "volume": "Volume"
                        })
                        if "timestamp" in hist_df.columns:
                            hist_df.set_index("timestamp", inplace=True)
                        all_dfs[ticker] = hist_df
                except Exception as e_indiv:
                    print(f"[WARN] Error fetching {ticker} from Upstox: {e_indiv}")
                if i % 5 == 0:
                    time.sleep(0.1)
        except Exception as e:
            print(f"[WARN] Upstox Scan Loop Error: {e}")

        # If some tickers failed to fetch from Upstox (and failed individual fallback),
        # fetch the missing tickers in batch from yfinance.
        missing_tickers = [t for t in tickers if t not in all_dfs]
        if missing_tickers:
            print(f"[INFO] Fetching {len(missing_tickers)} missing symbols in batch from yfinance...")
            try:
                batch_data = yf.download(
                    missing_tickers, period="60d", interval="1h",
                    progress=False, auto_adjust=True, timeout=45
                )
                if not batch_data.empty:
                    for ticker in missing_tickers:
                        try:
                            if len(missing_tickers) > 1:
                                if ticker in batch_data.columns.get_level_values(1):
                                    df_ticker = batch_data.xs(ticker, axis=1, level=1).dropna()
                                    if not df_ticker.empty:
                                        all_dfs[ticker] = df_ticker
                            else:
                                if not batch_data.empty:
                                    all_dfs[ticker] = batch_data.dropna()
                        except Exception as e_parse:
                            print(f"[WARN] Error parsing batch data for {ticker}: {e_parse}")
            except Exception as e_batch:
                print(f"[WARN] Batch yfinance download failed: {e_batch}")

        from scripts.feature_utils import compute_features, RSI

        for ticker in tickers:
            try:
                # ── Get Hourly DataFrame for this ticker ──
                if ticker in all_dfs:
                    df = all_dfs[ticker].ffill().dropna()
                else:
                    continue

                if len(df) < 25:
                    continue  # Need enough history for 24-period lookbacks

                # Compute base 81 technical features (V6+ use fixed math, older models use legacy math)
                is_legacy = not (hasattr(self, "_active_model_name") and any(str(self._active_model_name).startswith(p) for p in ["v6", "v7", "v8", "v9"]))
                df = compute_features(df, legacy=is_legacy)

                # ── COMPUTE V3 FEATURES (Hourly additions) ──
                df['Return_6H'] = df['Close'].pct_change(6).fillna(0)
                df['Return_1D'] = df['Close'].pct_change(7).fillna(0)
                price_dir = np.sign(df['Return'])
                vol_change = df['Volume'].pct_change()
                df['VP_Divergence'] = (price_dir * vol_change).fillna(0)
                green_bar = (df['Close'] > df['Open']).astype(float)
                high_vol = (df['Volume'] > df['Volume'].rolling(20).mean()).astype(float)
                df['Accumulation_5'] = (green_bar * high_vol).rolling(5).sum().fillna(0)
                df['Bar_Position'] = ((df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-8)).fillna(0)
                df['Green_Bar_Ratio_5'] = green_bar.rolling(5).mean().fillna(0)
                
                df['RSI_14_Raw'] = df['RSI_14'].copy()
                df['Stoch_K_Raw'] = df['Stoch_K'].copy()
                df['PercentB_Raw'] = df['PercentB'].copy()

                # ── COMPUTE V4 FEATURES (Daily Timeframe Context) ──
                daily_rsi_lag = 50.0
                daily_sma20_lag = None
                daily_atr_lag = 0.0
                daily_close_lag5 = None
                
                if not daily_data.empty:
                    try:
                        if len(tickers) > 1:
                            if ticker in daily_data.columns.get_level_values(1):
                                df_daily = daily_data.xs(ticker, axis=1, level=1).ffill().dropna()
                            else:
                                df_daily = pd.DataFrame()
                        else:
                            df_daily = daily_data.ffill().dropna()
                            
                        if len(df_daily) >= 20:
                            # ATR helper inline
                            high = df_daily['High']
                            low = df_daily['Low']
                            close = df_daily['Close']
                            prev_close = close.shift(1)
                            tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
                            atr_series = tr.ewm(span=14, adjust=False).mean()

                            latest_daily = df_daily.iloc[-2] if len(df_daily) > 1 else df_daily.iloc[-1]
                            # Use shift logic equivalent by taking the second to last day (or just previous bar)
                            daily_rsi_lag = RSI(df_daily['Close'], 14).iloc[-2] if len(df_daily) > 1 else 50.0
                            daily_sma20_lag = df_daily['Close'].rolling(20).mean().iloc[-2] if len(df_daily) > 1 else None
                            daily_atr_lag = atr_series.iloc[-2] if len(atr_series) > 1 else 0.0
                            daily_close_lag5 = df_daily['Close'].iloc[-6] if len(df_daily) >= 6 else None
                    except Exception:
                        pass
                
                # Apply daily context features
                def _safe_float(v, default):
                    if v is None: return default
                    if isinstance(v, pd.Series):
                        if v.empty: return default
                        v = v.iloc[0]
                    if pd.isna(v): return default
                    return float(v)

                daily_rsi_lag = _safe_float(daily_rsi_lag, 50.0)
                daily_sma20_lag = _safe_float(daily_sma20_lag, None)
                daily_close_lag5 = _safe_float(daily_close_lag5, None)
                daily_atr_lag = _safe_float(daily_atr_lag, 0.0)

                df['Daily_RSI'] = daily_rsi_lag
                if daily_sma20_lag is not None:
                    df['Daily_SMA20_Dist'] = (df['Close'] / daily_sma20_lag) - 1
                else:
                    df['Daily_SMA20_Dist'] = 0.0
                
                if daily_close_lag5 is not None:
                    df['Daily_Trend'] = np.sign(df['Close'] - daily_close_lag5)
                else:
                    df['Daily_Trend'] = 0.0
                    
                df['Daily_ATR_Pct'] = (daily_atr_lag / df['Close']) if daily_atr_lag else 0.0

                # Get latest row
                latest = df.iloc[-1].copy()
                latest["ticker"] = ticker
                all_latest_data.append(latest)
                valid_tickers.append(ticker)
            except Exception as e:
                print(f'[DEBUG] Error computing features for {ticker}: {e}')
                import traceback; traceback.print_exc()
                continue

        if not all_latest_data:
            print("[WARN] No valid ticker data processed.")
            return pd.DataFrame()

        # Create batch dataframe
        scores_df = pd.DataFrame(all_latest_data)

        # ── MERGE MARKET CONTEXT (NIFTY & VIX) ──
        for key, val in nifty_features.items():
            scores_df[key] = val
        for key, val in vix_features.items():
            scores_df[key] = val

        scores_df['Stock_vs_Nifty'] = scores_df['Return'] - scores_df['Nifty_1H_Return']
        scores_df['Beta_Adjusted_Mom'] = (scores_df['Return'] / (scores_df['Nifty_1H_Return'].abs() + 1e-6)).clip(-10, 10)

        # ── COMPUTE REGIME ──
        nifty_5h = nifty_features['Nifty_5H_Return']
        vix_lvl = vix_features['VIX_Level']
        vix_ma = vix_features['VIX_5D_MA']
        vix_ext = vix_features['VIX_Extreme']
        
        regime = 0
        if nifty_5h > 0.003 and vix_lvl <= vix_ma:
            regime = 1
        elif nifty_5h < -0.003 or vix_ext == 1:
            regime = -1
        scores_df['Market_Regime'] = regime

        # ── SECTOR FEATURES ──
        try:
            from scripts.sector_map import SECTOR_MAP
        except ImportError:
            SECTOR_MAP = {}
        scores_df['Sector'] = scores_df['ticker'].map(lambda t: SECTOR_MAP.get(t, "MISC"))
        
        scores_df['Sector_Mean_Return'] = scores_df.groupby('Sector')['Return'].transform('mean')
        scores_df['Stock_vs_Sector'] = scores_df['Return'] - scores_df['Sector_Mean_Return']
        scores_df['Sector_Up_Flag'] = (scores_df['Return'] > 0).astype(float)
        scores_df['Sector_Breadth'] = scores_df.groupby('Sector')['Sector_Up_Flag'].transform('mean')
        scores_df.drop(columns=['Sector_Up_Flag'], inplace=True, errors='ignore')
        
        scores_df['Sector_Count'] = scores_df.groupby('Sector')['Return'].transform('count')
        scores_df['Sector_Volatility'] = scores_df.groupby('Sector')['Return'].transform('std').fillna(0.0)
        
        # Cross-sectional logic: must handle if there's only 1 sector
        sector_perf = scores_df.groupby('Sector')['Return'].mean().reset_index()
        sector_perf['Sector_Rank'] = sector_perf['Return'].rank(ascending=False)
        scores_df = scores_df.merge(sector_perf[['Sector', 'Sector_Rank']], on='Sector', how='left')

        # ── BASELINE MARKET CONTEXT FEATURES ──
        scores_df["Market_Mean_Return"] = scores_df["Return"].mean()
        scores_df["Relative_Return"] = (
            scores_df["Return"] - scores_df["Market_Mean_Return"]
        )
        scores_df["Market_Mean_Volatility"] = scores_df["HL_Range"].mean()
        scores_df["Relative_Volatility"] = scores_df["HL_Range"] / (
            scores_df["Market_Mean_Volatility"] + 1e-8
        )

        # CAPTURE RAW DISPLAY VALUES before Z-scoring
        raw_display = scores_df[
            ["ticker", "Dollar_Volume", "RVOL", "Dist_52W_High", "Return"]
        ].copy()
        raw_display["High_52W_Actual"] = raw_display["ticker"].map(
            lambda t: (
                (
                    scores_df.loc[scores_df["ticker"] == t, "Close"].values[0]
                    - self.ticker_metadata.get(t, {}).get("high_52w", 0)
                )
                / (self.ticker_metadata.get(t, {}).get("high_52w", 1e-8))
                if self.ticker_metadata.get(t, {}).get("high_52w", 0) > 0
                else 0
            )
        )

        # ── Z-SCORING ──
        exclude_from_z = [
            "ticker", "DateTime", "Close", "Open", "High", "Low", "Volume",
            "Market_Mean_Return", "Relative_Return", "Market_Mean_Volatility", "Relative_Volatility",
            "Hour", "DayOfWeek", "Sector",
            # V3/V4 Exclusions
            "Nifty_1H_Return", "Nifty_3H_Return", "Nifty_5H_Return", "Nifty_RSI", "Nifty_HL_Range", "Nifty_20H_Std",
            "Sector_Mean_Return", "Stock_vs_Sector", "Sector_Breadth", "Sector_Count", "Sector_Volatility", "Sector_Rank",
            "VIX_Level", "VIX_Change", "VIX_5D_MA", "VIX_High", "VIX_Extreme", "Market_Regime",
            "Is_Open_Hour", "Is_Close_Hour", "Time_To_Close", "Up_Streak", "Down_Streak",
            "RSI_14_Raw", "Stoch_K_Raw", "PercentB_Raw", "Daily_RSI", "Daily_SMA20_Dist", "Daily_Trend", "Daily_ATR_Pct",
            "Bar_Position", "Green_Bar_Ratio_5", "Accumulation_5"
        ]
        
        features_to_zscore = [c for c in self.feature_cols if c not in exclude_from_z]
        for col in features_to_zscore:
            if col in scores_df.columns:
                mean = scores_df[col].mean()
                std = scores_df[col].std()
                if pd.isna(std) or std == 0:
                    scores_df[col] = 0.0
                else:
                    scores_df[col] = (scores_df[col] - mean) / std

        # ── PREDICT ──
        try:
            missing = [c for c in self.feature_cols if c not in scores_df.columns]
            if missing:
                print(f"[ERROR] Missing feature columns: {missing}")
                return pd.DataFrame()

            X = scores_df[self.feature_cols].values
            X_clean = np.nan_to_num(X)

            # Only apply scaler if it was actually fitted during training.
            # V4 (rank:pairwise on Z-scored data) was never trained with a scaler;
            # applying a stale V1 scaler would silently corrupt predictions.
            scaler_is_fitted = (
                self.scaler is not None
                and hasattr(self.scaler, 'scale_')
                and self.scaler.scale_ is not None
            )
            if scaler_is_fitted:
                X_final = self.scaler.transform(X_clean)
                print(f"[INFO] Scaler applied ({self._active_model_name})")
            else:
                X_final = X_clean  # V4: trees are scale-invariant, no scaler needed

            dmatrix = xgb.DMatrix(X_final, feature_names=self.feature_cols)
            
            # Defragment DataFrame to avoid PerformanceWarnings
            scores_df = scores_df.copy()
            
            scores_df["long_score"] = self.bst_long.predict(dmatrix)
            scores_df["short_score"] = self.bst_short.predict(dmatrix)

            scores_df["Long_Conviction"] = scores_df["long_score"] - scores_df["short_score"]
            scores_df["Short_Conviction"] = scores_df["short_score"] - scores_df["long_score"]
            scores_df["Long_Rank"] = scores_df["Long_Conviction"].rank(ascending=False)
            scores_df["Short_Rank"] = scores_df["Short_Conviction"].rank(ascending=False)

            scores_df = scores_df.reset_index(drop=True)
            raw_display_aligned = raw_display.reset_index(drop=True)
            scores_df["dv_raw"] = raw_display_aligned["Dollar_Volume"]
            scores_df["rvol_raw"] = raw_display_aligned["RVOL"]
            scores_df["dist_52h_model"] = raw_display_aligned["Dist_52W_High"]
            scores_df["dist_52h_actual"] = raw_display_aligned["High_52W_Actual"]
            scores_df["Return_Raw"] = raw_display_aligned["Return"]

            # Save for dashboard
            dashboard_cols = [
                "ticker", "Close", "long_score", "short_score", "Long_Conviction",
                "Short_Conviction", "Long_Rank", "Short_Rank", "dv_raw", "rvol_raw",
                "dist_52h_model", "dist_52h_actual",
            ]
            latest_data = (
                scores_df[dashboard_cols]
                .fillna(0)
                .sort_values("Long_Rank")
                .to_dict(orient="records")
            )
            with open("data/latest_scores.json", "w") as f:
                json.dump(latest_data, f)

        except Exception as e:
            print(f"[ERROR] Prediction Error: {e}")
            return pd.DataFrame()

        # ── Cache full-universe scores for _get_current_conviction ──
        # Only cache when we ran on the whole universe (len > 1) so that
        # the relative Z-score normalization is meaningful.
        if len(tickers) > 1:
            with self.lock:
                self.latest_full_scores = scores_df.copy()

        return scores_df

    @staticmethod
    def _compute_sr_levels(features: dict, price: float) -> dict:
        """Derives key support/resistance levels from pre-computed feature values.
        All values in INR (absolute price), ready to embed in the AI prompt.
        """
        def pct_to_price(pct_dist):
            """Convert a distance feature (fractional) back to absolute price."""
            try:
                v = float(pct_dist)
                return round(price / (1 - v), 2) if abs(v) < 0.5 else None
            except Exception:
                return None

        # Bollinger Band walls
        bb_upper = pct_to_price(features.get("Dist_BB_Upper", 0))
        bb_lower = pct_to_price(-abs(float(features.get("Dist_BB_Lower", 0))))

        # Donchian channel (20-period high/low = recent swing extremes)
        don_upper = pct_to_price(features.get("Dist_Donchian_Upper", 0))
        don_lower = pct_to_price(-abs(float(features.get("Dist_Donchian_Lower", 0))))

        # ATR (1-ATR proximity zones above/below)
        atr_pct  = float(features.get("HL_Range", 0.015))   # HL_Range ≈ ATR/Close
        atr_abs  = round(price * atr_pct, 2)
        r1_atr   = round(price + atr_abs, 2)      # 1-ATR resistance
        s1_atr   = round(price - atr_abs, 2)      # 1-ATR support

        # SMA levels (act as dynamic support/resistance)
        sma6_dist  = float(features.get("Dist_SMA_6",  0) or 0)
        sma12_dist = float(features.get("Dist_SMA_12", 0) or 0)
        sma50_dist = float(features.get("Dist_SMA_50", 0) or 0)
        
        # Use more robust formula: SMA = Price / (1 + dist)
        sma6   = round(price / (1 + sma6_dist),  2) if abs(sma6_dist) < 5 else price
        sma12  = round(price / (1 + sma12_dist), 2) if abs(sma12_dist) < 5 else price
        sma50  = round(price / (1 + sma50_dist), 2) if not np.isnan(sma50_dist) and abs(sma50_dist) < 5 else price

        # 52-week high proximity
        dist_52h = float(features.get("dist_52h_actual") or features.get("Dist_52W_High", -0.5))
        high_52w = round(price / (1 + dist_52h), 2) if abs(dist_52h) < 1.0 else None

        return {
            "price":       round(price, 2),
            "bb_upper":    bb_upper,
            "bb_lower":    bb_lower,
            "don_upper":   don_upper,   # 20-bar swing high — key resistance
            "don_lower":   don_lower,   # 20-bar swing low  — key support
            "r1_atr":      r1_atr,
            "s1_atr":      s1_atr,
            "sma6":        sma6,
            "sma12":       sma12,
            "sma50":       sma50,
            "high_52w":    high_52w,
            "atr_abs":     atr_abs,
        }

    def gemini_audit(self, ticker, side, conviction, features):
        if not self.gemini_enabled:
            return "NEUTRAL", "Audit Disabled"

        # 0. CHECK CACHE (Valid for 1 hour)
        cache_key = f"{ticker}_{side}"
        if cache_key in self.sentiment_cache:
            # Handle old cache format without prob
            cached_data = self.sentiment_cache[cache_key]
            if len(cached_data) == 4:
                sent, reason, ts, prob = cached_data
            else:
                sent, reason, ts = cached_data
                prob = "N/A"

            if time.time() - ts < 3600:
                log(f"[CACHE-HIT] Using cached sentiment for {ticker}")
                return sent, f"[CACHED] {reason}", prob

        # ── PRE-COMPUTE S/R LEVELS ──────────────────────────────────────────────
        current_price = float(features.get("Close", 0))
        sr = self._compute_sr_levels(dict(features), current_price)

        # ── FETCH RECENT PRICE HISTORY (Time + Price Context) ───────────────────
        price_history_str = "N/A"
        try:
            # Fetch 30 mins of 1-min data for 'The Magnet Effect' check.
            # get_recent_candles() uses WS cache if available (no REST call).
            hist_df = self.broker.get_recent_candles(ticker, interval='1minute', count=30)
            if isinstance(hist_df, pd.DataFrame) and not hist_df.empty:
                recent = hist_df.tail(30)
                # Format: "Time: Price, Time: Price..."
                history_points = []
                for _, row in recent.iterrows():
                    t_str = row['timestamp'].strftime('%H:%M') if 'timestamp' in row else "??:??"
                    history_points.append(f"{t_str}: ₹{row['close']:.2f}")
                price_history_str = " | ".join(history_points)
        except Exception as e:
            print(f"[WARN] Failed to fetch price history for {ticker}: {e}")

        # Human-readable S/R summary injected into both prompts
        if side == "LONG":
            sr_context = (
                f"CURRENT PRICE: ₹{sr['price']}\n"
                f"RESISTANCE LEVELS ABOVE (potential reversal zones for LONG):\n"
                f"  • Bollinger Upper Band : ₹{sr['bb_upper']}  "
                f"(dist: {round((sr['bb_upper']/sr['price']-1)*100,2) if sr['bb_upper'] else 'N/A'}%)\n"
                f"  • 20-bar Donchian High : ₹{sr['don_upper']}  "
                f"(dist: {round((sr['don_upper']/sr['price']-1)*100,2) if sr['don_upper'] else 'N/A'}%)\n"
                f"  • 1-ATR Resistance     : ₹{sr['r1_atr']}  (ATR ≈ ₹{sr['atr_abs']})\n"
                f"  • SMA-6                : ₹{sr['sma6']}\n"
                f"  • SMA-12               : ₹{sr['sma12']}\n"
                f"  • SMA-50 (50-day MA)   : ₹{sr['sma50']}\n"
                f"  • 52-Week High         : ₹{sr['high_52w']}\n"
                f"QUESTION: Is there a SIGNIFICANT RESISTANCE ZONE within 0.5% above ₹{sr['price']} "
                f"that could cap this LONG trade and cause a reversal? "
                f"Answer in 'resistance_risk': HIGH / MEDIUM / LOW"
            )
        else:  # SHORT
            sr_context = (
                f"CURRENT PRICE: ₹{sr['price']}\n"
                f"SUPPORT LEVELS BELOW (potential bounce zones for SHORT):\n"
                f"  • Bollinger Lower Band : ₹{sr['bb_lower']}  "
                f"(dist: {round((1-sr['bb_lower']/sr['price'])*100,2) if sr['bb_lower'] else 'N/A'}%)\n"
                f"  • 20-bar Donchian Low  : ₹{sr['don_lower']}  "
                f"(dist: {round((1-sr['don_lower']/sr['price'])*100,2) if sr['don_lower'] else 'N/A'}%)\n"
                f"  • 1-ATR Support        : ₹{sr['s1_atr']}  (ATR ≈ ₹{sr['atr_abs']})\n"
                f"  • SMA-6                : ₹{sr['sma6']}\n"
                f"  • SMA-12               : ₹{sr['sma12']}\n"
                f"  • SMA-50 (50-day MA)   : ₹{sr['sma50']}\n"
                f"  • 52-Week High         : ₹{sr['high_52w']}\n"
                f"QUESTION: Is there a SIGNIFICANT SUPPORT ZONE within 0.5% below ₹{sr['price']} "
                f"that could bounce this SHORT trade and cause a reversal? "
                f"Answer in 'support_risk': HIGH / MEDIUM / LOW"
            )

        # STAGE 1: FAST TECHNICAL AUDIT (Internal context only)
        # ── Helper: safe feature extractor ──────────────────────────────────
        def _f(key, default="N/A", pct=False, decimals=2):
            v = features.get(key, default)
            try:
                v = float(v)
                return f"{v * 100:.{decimals}f}%" if pct else f"{v:.{decimals}f}"
            except Exception:
                return str(default)

        # ── PRICE STRUCTURE ────────────────────────────────────────────────
        price        = sr["price"]
        sma6         = sr["sma6"]
        sma12        = sr["sma12"]
        sma50        = sr["sma50"]
        bb_upper     = sr["bb_upper"]
        bb_lower     = sr["bb_lower"]
        don_upper    = sr["don_upper"]
        don_lower    = sr["don_lower"]
        atr_abs      = sr["atr_abs"]
        high_52w     = sr["high_52w"]
        r1_atr       = sr["r1_atr"]
        s1_atr       = sr["s1_atr"]

        # ── TREND ALIGNMENT: price vs SMAs ────────────────────────────────
        def _vs(level):
            if not level: return "N/A"
            diff = round((price / level - 1) * 100, 2)
            return f"{'above' if diff > 0 else 'below'} by {abs(diff)}%"

        # ── BOLLINGER %B: position in band (0=lower, 1=upper) ─────────────
        percent_b_raw = _f("PercentB_Raw", decimals=3)

        # ── STOCHASTIC ────────────────────────────────────────────────────
        stoch_k = _f("Stoch_K_Raw", decimals=1)

        # ── VOLUME & MOMENTUM ─────────────────────────────────────────────
        rvol         = _f("rvol_raw", decimals=2)
        dv_cr        = features.get("dv_raw", 0)
        try:    dv_str = f"₹{float(dv_cr)/1e7:.1f} Cr"
        except: dv_str = "N/A"

        up_streak    = int(features.get("Up_Streak", 0) or 0)
        dn_streak    = int(features.get("Down_Streak", 0) or 0)
        green_ratio  = _f("Green_Bar_Ratio_5", pct=True, decimals=0)
        bar_pos      = _f("Bar_Position", decimals=2)          # 0=bearish close, 1=bullish close
        accumulation = _f("Accumulation_5", decimals=2)

        # ── MARKET & SECTOR CONTEXT ───────────────────────────────────────
        nifty_1h     = _f("Nifty_1H_Return", pct=True, decimals=2)
        nifty_5h     = _f("Nifty_5H_Return", pct=True, decimals=2)
        vix          = _f("VIX_Level", decimals=1)
        vix_ma       = _f("VIX_5D_MA", decimals=1)
        vix_extreme  = features.get("VIX_Extreme", 0)
        regime_raw   = int(features.get("Market_Regime", 0) or 0)
        regime_str   = {1: "BULL", -1: "BEAR", 0: "NEUTRAL"}.get(regime_raw, "NEUTRAL")
        stock_vs_nifty = _f("Stock_vs_Nifty", pct=True, decimals=2)
        vs_sector    = _f("Stock_vs_Sector", pct=True, decimals=2)
        sector_breadth = _f("Sector_Breadth", pct=True, decimals=0)

        # ── DAILY CONTEXT ─────────────────────────────────────────────────
        daily_rsi    = _f("Daily_RSI", decimals=1)
        daily_trend  = int(features.get("Daily_Trend", 0) or 0)
        daily_trend_str = {1: "UPTREND (5D)", -1: "DOWNTREND (5D)", 0: "SIDEWAYS (5D)"}.get(daily_trend, "N/A")
        daily_sma20  = _f("Daily_SMA20_Dist", pct=True, decimals=2)
        daily_atr    = _f("Daily_ATR_Pct", pct=True, decimals=2)

        # ── NEAR S/R: distance to nearest wall ───────────────────────────
        if side == "LONG":
            nearest_wall_price = min([x for x in [bb_upper, don_upper, r1_atr] if x and x > price], default=None)
            nearest_wall_label = "nearest RESISTANCE"
        else:
            nearest_wall_price = max([x for x in [bb_lower, don_lower, s1_atr] if x and x < price], default=None)
            nearest_wall_label = "nearest SUPPORT"
        if nearest_wall_price:
            wall_dist = round(abs(price / nearest_wall_price - 1) * 100, 2)
            wall_str = f"₹{nearest_wall_price} ({wall_dist}% away)"
        else:
            wall_str = "N/A"

        prompt_flash = f"""You are a professional intraday technical analyst reviewing a real-time trade signal.
All data below is LIVE from the current 1-hour bar. No external data is available — evaluate ONLY what is provided.

═══ TRADE PROPOSAL ═══════════════════════════════════════════════
TICKER  : {ticker}
SIDE    : {side}
PRICE   : ₹{price}
ML CONVICTION : {conviction:.4f}
  → Universe Rank : #{int(features.get("Long_Rank" if side == "LONG" else "Short_Rank", 999))} of ~172 stocks screened this cycle
  → Min gate      : {self.min_conviction:.2f}  (signals below this are discarded — never reach audit)
  → Scale context : 0.15=gate | 0.25=moderate | 0.35+=STRONG | 0.50+=very strong
  → IMPORTANT: This signal survived ML pre-filtering and ranks in the TOP candidates. Do NOT label it "low conviction".

═══ MOMENTUM & OSCILLATORS ═══════════════════════════════════════
RSI-14 (intraday)  : {_f("RSI_14_Raw", decimals=1)}  {'→ OVERBOUGHT' if float(_f("RSI_14_Raw", "50")) > 70 else ('→ OVERSOLD' if float(_f("RSI_14_Raw", "50")) < 30 else '→ NEUTRAL ZONE')}
RSI-14 (daily)     : {daily_rsi}  (daily trend alignment)
Stochastic %K      : {stoch_k}  {'→ OVERBOUGHT' if float(stoch_k) > 80 else ('→ OVERSOLD' if float(stoch_k) < 20 else '→ MID')}
Bollinger %B       : {percent_b_raw}  (0=at lower band, 1=at upper band, >1=breakout, <0=breakdown)
1H Return          : {_f("Return_Raw", pct=True, decimals=2)}
Up Streak          : {up_streak} consecutive green bars
Down Streak        : {dn_streak} consecutive red bars
Green Bar Ratio(5) : {green_ratio}  (last 5 bars)
Bar Close Position : {bar_pos}  (0=bearish close, 1=bullish close, 0.5=doji)
Accumulation(5)    : {accumulation}  (buy pressure, >0 = net buying)

═══ PRICE STRUCTURE & KEY LEVELS ═════════════════════════════════
vs SMA-6  (fast)   : {_vs(sma6)}   [₹{sma6}]
vs SMA-12 (medium) : {_vs(sma12)}  [₹{sma12}]
vs SMA-50 (daily)  : {_vs(sma50)}  [₹{sma50}]
BB Upper           : ₹{bb_upper}  ({round((bb_upper/price-1)*100,2) if bb_upper else 'N/A'}% above)
BB Lower           : ₹{bb_lower}  ({round((1-bb_lower/price)*100,2) if bb_lower else 'N/A'}% below)
Donchian High(20)  : ₹{don_upper}  (20-bar swing high — key resistance)
Donchian Low(20)   : ₹{don_lower}  (20-bar swing low  — key support)
1-ATR Resistance   : ₹{r1_atr}    ATR = ₹{atr_abs}
1-ATR Support      : ₹{s1_atr}
52-Week High       : ₹{high_52w}  ({_f("dist_52h_actual", pct=True, decimals=1)} from 52W high)
{nearest_wall_label.upper()} : {wall_str}

═══ VOLUME ════════════════════════════════════════════════════════
RVOL (Relative Vol): {rvol}x  {'→ HIGH ACTIVITY' if float(rvol) > 2 else ('→ MODERATE' if float(rvol) > 1 else '→ THIN')}
Dollar Volume      : {dv_str}

═══ MARKET & SECTOR CONTEXT ══════════════════════════════════════
Market Regime      : {regime_str}
Nifty 1H Return    : {nifty_1h}
Nifty 5H Return    : {nifty_5h}
Stock vs Nifty     : {stock_vs_nifty}  (positive = outperforming index)
Stock vs Sector    : {vs_sector}  (positive = outperforming sector)
Sector Breadth     : {sector_breadth}  (% of sector stocks up)
VIX Level          : {vix}  (5D MA: {vix_ma}){'  ← EXTREME FEAR' if vix_extreme else ''}

═══ DAILY TREND ALIGNMENT ════════════════════════════════════════
Daily Trend (5D)   : {daily_trend_str}
Daily RSI          : {daily_rsi}
Daily vs SMA-20    : {daily_sma20}
Daily ATR %        : {daily_atr}

═══ RECENT PRICE ACTION (last 30 min, 1-min bars) ════════════════
{price_history_str}

═══ TASK ══════════════════════════════════════════════════════════
1. Evaluate if ALL of the following align for a {side} trade:
   - Momentum direction (RSI, Stoch, streak)
   - Price structure (price vs SMAs, band position)
   - Volume confirmation (RVOL > 1.5 preferred)
   - Market regime support (is Nifty + Regime aligned with {side}?)
   - Daily trend alignment (intraday trade WITH the daily trend is safer)
   - Proximity risk: Is {nearest_wall_label} too close (< 0.3%) to allow a clean move?

2. Output STRICT JSON only — no markdown, no extra text:
{{"sentiment": "LABEL", "reason": "concise 1-sentence rationale covering the key confluences or conflicts", "probability": "XX%"}}

Labels: STRONG BULLISH | BULLISH | NEUTRAL | BEARISH | STRONG BEARISH
Probability: estimated % chance price moves in favor of {side} direction in the next 1 hour.
"""

        sent1, reason1, prob1 = "NEUTRAL", "N/A", "N/A"
        stage1_success = False

        # ── STAGE 1: TWO-TIER MODEL STRATEGY ─────────────────────────────────
        # Tier 1: gemini-3.5-flash (superior reasoning, strict JSON adherence)
        #   - Rotate through all API keys on quota/rate-limit errors.
        # Tier 2: gemini-3.1-flash-lite (fallback when all keys are exhausted on Tier 1)
        #   - Same key rotation logic applied independently.
        S1_PRIMARY  = "gemini-3.5-flash"
        S1_FALLBACK = "gemini-3.1-flash-lite"

        total_combinations = len(self.s1_model_tiers) * len(self.api_keys)
        attempts = 0
        
        while attempts < total_combinations:
            current_model = self.s1_model_tiers[self.s1_active_tier_idx]
            current_key = self.s1_active_key_idx
            
            try:
                layer1_client = self.clients[current_key]
                log(f"[S1] Attempting {current_model} (Key {current_key}) for {ticker}...")
                resp1 = layer1_client.models.generate_content(
                    model=current_model, contents=prompt_flash,
                    config=types.GenerateContentConfig(temperature=0.1)
                )
                data1 = self.parse_gemini_json(resp1.text)
                sent1 = data1.get("sentiment", "NEUTRAL").upper()
                reason1 = data1.get("reason", "N/A")
                prob1 = data1.get("probability", "N/A")
                stage1_success = True
                log(f"[S1-OK] {current_model} (Key {current_key}) succeeded for {ticker}.")
                break  # Success! We leave s1_active_tier_idx and s1_active_key_idx exactly where they are.
                
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "503" in err_str or "quota" in err_str.lower():
                    print(f"[S1-ROTATE] {current_model} Key {current_key} exhausted ({err_str[:80]}). Rotating...")
                    # Rotate Key first
                    self.s1_active_key_idx += 1
                    if self.s1_active_key_idx >= len(self.api_keys):
                        self.s1_active_key_idx = 0
                        # If we tried all keys for this model, rotate the model tier
                        self.s1_active_tier_idx += 1
                        if self.s1_active_tier_idx >= len(self.s1_model_tiers):
                            self.s1_active_tier_idx = 0 # wrap around
                        print(f"[S1-FALLBACK] All keys exhausted. Rotating to {self.s1_model_tiers[self.s1_active_tier_idx]}...")
                    attempts += 1
                    time.sleep(1)
                    continue
                else:
                    print(f"[WARN] Stage 1 non-quota error ({current_model}, Key {current_key}) for {ticker}: {e}")
                    time.sleep(2)
                    break # Break out completely on hard error
                    
        if not stage1_success:
            print(f"[S1-FAILED] Both model tiers failed for {ticker}. Skipping trade.")
            return "NEUTRAL", "Stage 1 Audit Error (All Models/Keys Exhausted)", "N/A"

        is_vetoed = False
        if side == "LONG" and sent1 in ["STRONG BEARISH", "BEARISH", "NEUTRAL"]:
            is_vetoed = True
        elif side == "SHORT" and sent1 in ["STRONG BULLISH", "BULLISH", "NEUTRAL"]:
            is_vetoed = True

        if is_vetoed:
            final_reason = f"[S1-VETO] {reason1}"
            self.sentiment_cache[cache_key] = (sent1, final_reason, time.time(), prob1)
            return sent1, final_reason, prob1

        # STAGE 2: HIERARCHICAL GOVERNANCE AUDIT (P0 > P1 > P2)
        log(f"[STAGE 2] Triggering Hierarchical Super-Veto Audit for {ticker}...")

        # ── Derive clean company name for search (strip exchange suffix) ──────
        company_name = ticker.replace(".NS", "").replace(".BO", "").replace(".", " ").strip()
        current_dt   = datetime.now().strftime("%Y-%m-%d %H:%M IST")

        # ── S1 technical summary for CRO context ─────────────────────────────
        s1_tech_summary = (
            f"RSI={_f('RSI_14_Raw', decimals=1)} | Stoch%K={stoch_k} | %B={percent_b_raw} | "
            f"RVOL={rvol}x | Regime={regime_str} | Nifty1H={nifty_1h} | "
            f"vs Nifty={stock_vs_nifty} | DailyTrend={daily_trend_str} | "
            f"UpStreak={up_streak} | DnStreak={dn_streak} | Conviction={conviction:.4f}"
        )

        prompt_search = f"""# ROLE: Institutional Risk Auditor (CRO) — Capital Protection Division
# CURRENT DATE/TIME: {current_dt}
# YOUR ONLY OBJECTIVE: Determine if fundamental/news reality CONTRADICTS the technical signal.
# You are NOT here to predict price. You are here to VETO trades where news overrides technicals.

════════════════════════════════════════════════════════════════
AUDIT PROFILE
════════════════════════════════════════════════════════════════
Company         : {company_name} (NSE: {ticker})
Proposed Trade  : {side}
Current Price   : ₹{price}
ML Conviction   : {conviction:.4f}
  → Universe Rank : #{int(features.get("Long_Rank" if side == "LONG" else "Short_Rank", 999))} of ~172 stocks screened this cycle
  → Min gate      : {self.min_conviction:.2f}  (signals below this never reach this audit)
  → Scale context : 0.15=gate | 0.25=moderate | 0.35+=STRONG | 0.50+=very strong
  → Do NOT label this "low conviction" — it is a top-ranked, pre-filtered signal.
RVOL            : {rvol}x  (>2.0 = high activity; confirms institutional participation)

S1 TECHNICAL VERDICT  : {sent1}
S1 PROBABILITY        : {prob1}  ← your output probability should be ANCHORED to this unless news materially overrides it
S1 TECHNICAL SUMMARY  : {s1_tech_summary}

KEY S/R CONTEXT
{sr_context}

RECENT PRICE ACTION (last 30 min, 1-min bars):
{price_history_str}

════════════════════════════════════════════════════════════════
STEP 1 — GROUNDED NEWS SEARCH
════════════════════════════════════════════════════════════════
Search for ALL of the following (use Google Search grounding):
  • "{company_name} stock news"
  • "{company_name} NSE results earnings"
  • "{company_name} block deal bulk deal today"

Classify findings into two buckets:

BUCKET A — Structural (P0, last 7 days):
  Examples: Earnings beat/miss, revenue guidance, brokerage upgrades/downgrades,
  regulatory action, FII/DII stake change, promoter pledge, dividend, merger/acquisition,
  industry policy shock.
  → These OVERRIDE technicals. A rating downgrade for a LONG = mandatory VETO.

BUCKET B — Tactical (P1, last 6 hours):
  Examples: Block deal, bulk deal, news headline, volume spike explanation,
  AGM, analyst meet, court order.
  → These MODIFY conviction but may not VETO unless strong directional conflict.

If NO material news is found in either bucket, explicitly state "No material catalyst found."
Do NOT hallucinate news. If you are uncertain, say so.

════════════════════════════════════════════════════════════════
STEP 2 — VETO DECISION MATRIX
════════════════════════════════════════════════════════════════
Apply the following rules IN ORDER (first triggered rule wins):

RULE 1 — HARD VETO (Fundamental Conflict):
  • {side}=LONG  + Bucket A shows: earnings miss, rating downgrade, promoter sell, regulatory ban → VETO=TRUE
  • {side}=SHORT + Bucket A shows: earnings beat, rating upgrade, buyback, strong guidance → VETO=TRUE

RULE 2 — SOFT VETO (Tactical Conflict):
  • {side}=LONG  + Bucket B shows: large block sell, negative headline → consider VETO if S1 conviction < 0.25
  • {side}=SHORT + Bucket B shows: large block buy, positive headline → consider VETO if S1 conviction < 0.25

RULE 3 — BREAKOUT OVERRIDE (Do NOT veto):
  • If RVOL > 3.0 AND Bucket A/B shows POSITIVE catalysts AND {side}=LONG:
    Resistance levels are TARGETS in a genuine breakout, not walls. Do NOT veto.

RULE 4 — MAGNET EFFECT (Price Action):
  • VETO SHORT if price has been grinding within 0.5% of a resistance level for >15 minutes
    (absorption = imminent breakout, shorting here is dangerous).
  • VETO LONG if price is sitting on a support level without a bounce for >15 minutes
    (support is being eroded = imminent breakdown).
  Detect this from the 30-min price history above.

RULE 5 — S/R PROXIMITY RISK:
  • If {nearest_wall_label} is within 0.3% of current price AND no strong catalyst supports pushing through → VETO=TRUE
    (Rationale: Not enough room for the trade to breathe; risk/reward is unfavorable.)

RULE 6 — NO NEWS = DEFER TO S1:
  • If no material news found in either bucket, set veto_decision=FALSE and
    anchor your probability to S1's probability ({prob1}).
    Do NOT veto a technically sound signal just because you found no news.

════════════════════════════════════════════════════════════════
STEP 3 — FINAL OUTPUT
════════════════════════════════════════════════════════════════
Output STRICT JSON only — absolutely no markdown, no extra text, no commentary:
{{
  "news_found": "Brief description of what you found in Bucket A and B, or 'No material catalyst found'",
  "chain_of_thought": "1-2 sentences: does the news CONFIRM, CONTRADICT, or is NEUTRAL to the {side} technical signal?",
  "structural_bias": "BULLISH|BEARISH|NEUTRAL",
  "veto_decision": "TRUE|FALSE",
  "veto_rule_triggered": "RULE 1|RULE 2|RULE 3|RULE 4|RULE 5|RULE 6|NONE",
  "final_sentiment": "STRONG BULLISH|BULLISH|NEUTRAL|BEARISH|STRONG BEARISH",
  "support_resistance_risk": "HIGH|MEDIUM|LOW",
  "probability": "XX%",
  "risk_factor": "Single sentence: the one most likely reason this trade fails"
}}

IMPORTANT: 
- If veto_decision=FALSE, final_sentiment MUST align with the {side} direction (BULLISH for LONG, BEARISH for SHORT).
- If veto_decision=TRUE, final_sentiment should reflect the override direction.
- Probability should stay close to S1's {prob1} unless you found a strong news catalyst that materially changes the outlook.
- Responding with NEUTRAL as final_sentiment is acceptable ONLY if news genuinely creates irresolvable ambiguity. Be aware this will pause the trade for 30 minutes.
"""

        stage2_retries = 2 # try current key, if fail try another key once
        
        for _ in range(stage2_retries):
            model_name, key_idx = self.gemini_tracker.get_next_available(len(self.api_keys))
            if model_name is None:
                print(f"[L2-FAILED] Stage 2 limits exhausted for {ticker}. Skipping for this cycle.")
                final_reason = f"[L2-FAILED] Layer 2 audit skipped: API limits exhausted."
                return "SYSTEM_ERROR", final_reason, "N/A"

            try:
                stage2_client = self.clients[key_idx]
                resp2 = stage2_client.models.generate_content(
                    model=model_name,
                    contents=prompt_search,
                    config=types.GenerateContentConfig(
                        tools=[{"google_search": {}}], 
                        temperature=0.1
                    )
                )
                resp2_text = self._extract_response_text(resp2)
                data2 = self.parse_gemini_json(resp2_text)

                # ── EMPTY RESPONSE GUARD ───────────────────────────────────────────
                # If parse returned {} (empty/malformed), do NOT silently default to
                # NEUTRAL — that lets bad trades through. Treat as L2 failure instead.
                if not data2 or not any(k in data2 for k in ("veto_decision", "final_sentiment", "chain_of_thought")):
                    print(f"[L2-EMPTY] Stage 2 returned empty/unparseable JSON for {ticker} (using {model_name}, Key {key_idx}). Skipping.")
                    print(f"[DEBUG] Raw Stage 2 Text ({model_name}, Key {key_idx}): {repr(resp2.text)[:500]}...")
                    final_reason = f"[L2-EMPTY ({model_name}, Key {key_idx})] Layer 2 returned no usable data."
                    return "SYSTEM_ERROR", final_reason, "N/A"

                # Success - increment tracker
                self.gemini_tracker.increment_usage(model_name, key_idx)

                news_found     = data2.get("news_found", "N/A")
                cot            = data2.get("chain_of_thought", "N/A")
                bias           = data2.get("structural_bias", "NEUTRAL")
                veto_triggered = str(data2.get("veto_decision", "FALSE")).upper() == "TRUE"
                veto_rule      = data2.get("veto_rule_triggered", "NONE")
                sent2          = data2.get("final_sentiment", "NEUTRAL").upper()
                prob2          = data2.get("probability", prob1)  # fallback to S1 prob if missing
                risk           = data2.get("risk_factor", "N/A")
                sr_risk        = data2.get("support_resistance_risk", "LOW").upper()

                # ── HIERARCHICAL VETO OVERRIDE ────────────────────────────────────
                if sr_risk == "HIGH":
                    veto_triggered = True
                    veto_rule = "RULE 5 (S/R Proximity)"
                    risk = f"[S/R RISK: HIGH] {risk}"

                # NEUTRAL only auto-vetoes if the model found an active news conflict
                # (veto_rule != NONE/RULE6). If no news, NEUTRAL from model = defer to S1.
                if sent2 == "NEUTRAL" and veto_rule not in ("NONE", "RULE 6", "N/A"):
                    veto_triggered = True
                    risk = f"[NEUTRAL — News Conflict] {risk}"

                if veto_triggered:
                    sent2 = "VETOED"
                    final_reason = (
                        f"[VETOED by {model_name} | {veto_rule}] "
                        f"News: {news_found} | {cot} | Risk: {risk} | Bias: {bias}"
                    )
                    print(f"[HIERARCHICAL VETO] {ticker} {side} VETOED ({veto_rule}) by {model_name} (Key {key_idx}): {cot}")
                else:
                    final_reason = (
                        f"[S2-PASS by {model_name} | {veto_rule}] "
                        f"News: {news_found} | {cot} | Bias: {bias} (S1: {sent1})"
                    )
                    print(f"[STAGE 2 PASS] {ticker} {side} cleared by {model_name} (Key {key_idx}) [{veto_rule}]: {cot}")

                self.sentiment_cache[cache_key] = (sent2, final_reason, time.time(), prob2)
                return sent2, final_reason, prob2

            except Exception as e_search:
                err_str = str(e_search)
                if "429" in err_str or "503" in err_str or "UNAVAILABLE" in err_str or "quota" in err_str.lower():
                    print(f"[RETRY] Stage 2 Rate Limited ({model_name}, Key {key_idx}). Retrying...")
                    self.gemini_tracker.mark_exhausted(model_name, key_idx)
                    time.sleep(1)
                    continue
                else:
                    print(f"[WARN] Stage 2 Search Error for {ticker}: {e_search}")
                    break # Break retry loop on non-rate-limit errors

        # Stage 2 failed after all retries — skip the trade
        print(f"[L2-FAILED] Stage 2 audit failed for {ticker}. Skipping for this cycle.")
        final_reason = f"[L2-FAILED] Layer 2 search audit failed (API error/timeout)."
        return "SYSTEM_ERROR", final_reason, "N/A"
    def _load_virtual_stats(self):
        """Restores full capital state from JSON. Called once on startup before open trades load."""
        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.initial_capital    = data.get("initial_capital", 99517.68)
                    self.virtual_capital    = data.get("virtual_capital", 99517.68)
                    self.realized_charges   = data.get("realized_charges", 0.0)
                    # used_margin is intentionally NOT restored here —
                    # it will be recomputed from open trades in load_open_trades()
                    print(f"[RESTORE] Capital: Rs{self.virtual_capital:.2f} | Charges: Rs{self.realized_charges:.2f}")
            except Exception as e:
                print(f"[WARN] Could not load stats file: {e}")
        # Do NOT call update_upstox_stats() here — wait until open trades are loaded

    def calculate_exit_charges(self, sell_value):
        """Full regulatory charge stack for Indian equity intraday."""
        brokerage = self.brokerage_per_order  # ₹10 flat
        stt = sell_value * self.stt_rate      # 0.025% on sell
        txn_charges = sell_value * 0.0000345  # NSE transaction (0.00345%)
        gst = (brokerage + txn_charges) * 0.18  # 18% GST
        sebi_fee = sell_value * 0.000001      # ₹10 per crore
        stamp_duty = 0.0                      # Only on buy-side, absorbed
        return brokerage + stt + txn_charges + gst + sebi_fee + stamp_duty

    def calculate_trade_quantity(self, price, stop_loss_pct=0.50):
        """
        Risk-parity sizing: every SL hit costs exactly RISK_PER_TRADE % of capital.
        Volatile stocks get fewer shares, quiet stocks get more.
        """
        RISK_PER_TRADE = 0.005  # 0.5% of capital risked per trade
        risk_amount = self.day_start_capital * RISK_PER_TRADE
        sl_distance = price * (stop_loss_pct / 100)
        
        if sl_distance <= 0:
            sl_distance = price * 0.005  # fallback
        
        ideal_qty = int(risk_amount / sl_distance)
        
        # Cap: never exceed the old fixed-slot exposure (safety net)
        max_slot_capital = (self.day_start_capital / self.max_trade_slots) * self.margin_multiplier
        max_qty = int(max_slot_capital / price)
        
        qty = max(1, min(ideal_qty, max_qty))
        return qty

    def _extract_response_text(self, resp) -> str:
        """
        Extract text from a Gemini response robustly.
        When google_search tool is used, .text may be None because the response
        is multi-part (search grounding parts + text part). We walk all parts.
        """
        # Convenience property works for non-tool responses
        try:
            if resp.text:
                return resp.text
        except Exception:
            pass

        # Walk candidates → content → parts for grounded/tool responses
        try:
            for candidate in resp.candidates:
                for part in candidate.content.parts:
                    t = getattr(part, "text", None)
                    if t:
                        return t
        except Exception:
            pass

        return ""

    def parse_gemini_json(self, text):
        """
        Robust JSON parser for Gemini responses.
        Handles: markdown fences, literal newlines inside string values,
        trailing commas, truncated responses, and mixed-format outputs.
        """
        import re
        if not text:
            return {}
        text = text.strip()

        # Strategy 1: Strip markdown code fences (```json ... ``` or ``` ... ```)
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1)
        else:
            # Strategy 2: Extract first {...} block
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                text = text[start:end + 1]

        # Strategy 3: Fix literal newlines/tabs inside JSON string values
        # Replace raw \n, \r, \t within string values with their escaped equivalents
        def fix_control_chars(m):
            return m.group(0).replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')

        # Apply fix only inside quoted string values
        text = re.sub(r'"(?:[^"\\]|\\.)*"', fix_control_chars, text, flags=re.DOTALL)

        # Strategy 4: Try standard parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strategy 5: Aggressively strip all control chars and retry
        try:
            cleaned = re.sub(r'[\x00-\x1f\x7f]', ' ', text)
            if not cleaned.strip().startswith("{"):
                cleaned = "{" + cleaned
            if not cleaned.strip().endswith("}"):
                cleaned = cleaned + "}"
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            # Strategy 6: Regex Fallback Extraction (Salvage Operation)
            extracted = {}
            keys_to_salvage = [
                "news_found", "chain_of_thought", "structural_bias", "veto_decision",
                "veto_rule_triggered", "final_sentiment", "support_resistance_risk",
                "probability", "risk_factor", "sentiment", "reason"
            ]
            for k in keys_to_salvage:
                # Match "key": "value" allowing escaped quotes inside value
                m = re.search(rf'"{k}"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.IGNORECASE | re.DOTALL)
                if m:
                    extracted[k] = m.group(1).replace('\\"', '"').replace('\\n', '\n')
            
            if extracted and any(k in extracted for k in ("veto_decision", "final_sentiment", "sentiment")):
                print(f"[JSON-RECOVERY] Salvaged {len(extracted)} keys via regex from broken JSON.")
                return extracted

            print(f"[ERROR] JSON Parse Error: {e} | Raw snippet: {text[:150]}...")
            return {}

    def _mark_recently_closed(self, ticker: str):
        """Call whenever a trade is closed so the ticker enters the 30-min cooldown."""
        self.recently_closed[ticker] = datetime.now()

    def _mark_recently_vetoed(self, ticker: str):
        """Call whenever a ticker is vetoed so it enters a 30-min cooling period."""
        self.recent_vetoes[ticker] = datetime.now()

    def _is_in_cooldown(self, ticker: str, minutes: int = 30) -> bool:
        """Returns True if this ticker was closed within the last `minutes` minutes."""
        closed_at = self.recently_closed.get(ticker)
        if closed_at is None:
            return False
        return (datetime.now() - closed_at).total_seconds() < minutes * 60

    def _is_veto_cooldown(self, ticker: str, minutes: int = 30) -> bool:
        """Returns True if this ticker was vetoed within the last `minutes` minutes.
        Checks BOTH the in-memory timestamp dict AND active_shadow_trades for VETOED status.
        This dual-check handles restarts and race conditions.
        """
        # Check 1: In-memory timestamp (survives within a single session)
        vetoed_at = self.recent_vetoes.get(ticker)
        if vetoed_at is not None:
            if (datetime.now() - vetoed_at).total_seconds() < minutes * 60:
                return True
        # Check 2: active_shadow_trades (handles case where _mark_recently_vetoed
        # was called but recent_vetoes was cleared by a restart or is missing)
        with self.lock:
            for t in self.active_shadow_trades:
                if t["ticker"] == ticker and t["status"] == "VETOED":
                    try:
                        vetoed_ts = datetime.fromisoformat(t["timestamp"])
                        if (datetime.now() - vetoed_ts).total_seconds() < minutes * 60:
                            return True
                    except Exception:
                        pass
        return False

    def start_shadow_trade(
        self,
        ticker,
        score,
        sentiment,
        entry_price,
        side="LONG",
        comment="",
        one_hour_prob="N/A",
        long_score=None,
        short_score=None,
    ):
        # GUARD 0 — COOLDOWN: block re-entry on any ticker closed in the last 30 min.
        # This is the primary defence against the restart-duplicate bug: even if a resumed
        # trade gets closed by the shadow tracker a few seconds after startup, the cooldown
        # prevents the very next scan from immediately re-entering the same position.
        if self._is_in_cooldown(ticker):
            closed_at = self.recently_closed[ticker]
            mins_ago = max(0, int((datetime.now() - closed_at).total_seconds() / 60))
            print(f"[COOLDOWN] {ticker} exited {mins_ago}m ago. Skipping to prevent duplicate entry.")
            return

        # GUARD 1 — DB-LEVEL CHECK: query the database directly for any OPEN position on
        # this ticker. The in-memory list can be stale immediately after a restart (race
        # between shadow_tracker_loop closing trades and the main scan loop checking them).
        try:
            existing_open = get_trades_by_status("OPEN", 100)
            if any(t["ticker"] == ticker and t["status"] == "OPEN" for t in existing_open):
                print(f"[DB-GUARD] Existing OPEN position found in DB for {ticker}. "
                      f"Aborting new trade to prevent duplicate entry.")
                return
        except Exception as e:
            print(f"[WARN] DB guard check failed for {ticker}: {e}")

        # 1. ENFORCE CONCURRENCY LIMIT
        with self.lock:
            open_count = len(
                [t for t in self.active_shadow_trades if t["status"] == "OPEN"]
            )

        if open_count >= self.max_trade_slots:
            return

        # Remove any existing VETOED tracking for this ticker to avoid live tracking overlaps
        with self.lock:
            self.active_shadow_trades = [
                t for t in self.active_shadow_trades 
                if not (t["ticker"] == ticker and t["status"] == "VETOED")
            ]

        # 2. CALCULATE QUANTITY & STOP LOSS
        # Dynamic ATR-based stop-loss and take-profit
        stop_loss_pct, take_profit_pct = self.compute_15min_atr(ticker)
        
        # Apply Slippage (2 bps)
        SLIPPAGE_BPS = 2
        slippage = entry_price * (SLIPPAGE_BPS / 10000)
        entry_price = entry_price + slippage if side == "LONG" else entry_price - slippage

        quantity = self.calculate_trade_quantity(entry_price, stop_loss_pct)
        trade_value = quantity * entry_price
        required_margin = trade_value / self.margin_multiplier
        
        sl_mult = 1 - (stop_loss_pct / 100) if side == "LONG" else 1 + (stop_loss_pct / 100)
        sl_price = round(entry_price * sl_mult, 2)
        print(f"[ATR-SL] {ticker} | SL: {stop_loss_pct:.2f}% | TP: {take_profit_pct:.2f}% | SL Price: {sl_price}")

        # Check early entry confirmation (Momentum Continuation with Retracement Tolerance)
        is_immediate_entry = False
        early_entry_reason = ""
        try:
            past_candle = self.get_last_completed_15min_candle(ticker)
            if past_candle is not None:
                past_open = past_candle['open']
                past_close = past_candle['close']
                past_body = past_close - past_open
                
                if side == "LONG":
                    if past_close > past_open:  # past candle is green
                        # Midpoint of past candle body
                        max_retract = past_close - (past_body * 0.5)
                        if entry_price >= max_retract:
                            is_immediate_entry = True
                            early_entry_reason = f"Past candle bullish, entry {entry_price} >= max retract {max_retract:.2f}"
                elif side == "SHORT":
                    if past_close < past_open:  # past candle is red
                        # Midpoint of past candle body
                        max_retract = past_close + (abs(past_body) * 0.5)
                        if entry_price <= max_retract:
                            is_immediate_entry = True
                            early_entry_reason = f"Past candle bearish, entry {entry_price} <= max retract {max_retract:.2f}"
        except Exception as e:
            print(f"[WARN] Failed to evaluate early entry for {ticker}: {e}")

        # 3. PLACE ORDER DEFERRED or IMMEDIATE
        if is_immediate_entry:
            try:
                upstox_order = self.broker.place_order(ticker, side, quantity=quantity, price=entry_price, stop_loss=sl_price)
                order_id = "SANDBOX-ERROR"
                if hasattr(upstox_order, "data") and upstox_order.data:
                    order_id = getattr(upstox_order.data, "order_id", "SANDBOX-SUCCESS")
                elif isinstance(upstox_order, dict) and "data" in upstox_order:
                    order_id = upstox_order["data"].get("order_id", "SANDBOX-SUCCESS")
                upstox_order_id = order_id
                status = "OPEN"
                print(f"[IMMEDIATE ENTRY] Confirmed {ticker} at {entry_price} (Order: {order_id}) - Reason: {early_entry_reason}")
            except Exception as e:
                print(f"[ERROR] Immediate order placement failed for {ticker}: {e}. Falling back to deferred entry.")
                upstox_order_id = "PENDING-CANDLE-CONFIRMATION"
                status = "PENDING_ENTRY"
                is_immediate_entry = False
        else:
            upstox_order_id = "PENDING-CANDLE-CONFIRMATION"
            status = "PENDING_ENTRY"

        # 4. FETCH TRADINGVIEW SENTIMENT (Analytics only)
        tv_sentiment = get_tv_sentiment(ticker)

        trade = {
            "trade_id": f"T-{int(time.time())}-{ticker}",
            "timestamp": datetime.now().isoformat(),
            "ticker": ticker,
            "side": side,
            "tech_score": float(score),
            "nlp_sentiment": float(sentiment),
            "long_score": float(long_score) if long_score is not None else None,
            "short_score": float(short_score) if short_score is not None else None,
            "entry_price": entry_price,
            "peak_price": entry_price,
            "peak_profit_pct": 0.0,
            "final_profit_pct": 0.0,
            "exit_time": (datetime.now() + timedelta(hours=1)).isoformat(),
            "status": status,
            "comment": f"{comment} | UpstoxID: {upstox_order_id}" + (f" | {early_entry_reason}" if is_immediate_entry else ""),
            "one_hour_prob": one_hour_prob,
            "quantity": quantity,
            "margin_used": required_margin,
            "buy_brokerage": self.brokerage_per_order,
            "tv_sentiment": tv_sentiment,
            "pending_since": datetime.now().isoformat() if status == "PENDING_ENTRY" else None,
            # ── Loss-Extension State ─────────────────────────────────────
            # extension_count  : number of 15-min extensions granted (max 2)
            # extended_exit_time: ISO string, set when an extension is active
            # extension_pending : True while waiting out a 15-min extension window
            "extension_count": 0,
            "extended_exit_time": None,
            "extension_pending": False,
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
            "trailing_active": False,
            "breakeven_locked": False,
        }
        with self.lock:
            self.active_shadow_trades.append(trade)
            self.used_margin += required_margin
            self.realized_charges += self.brokerage_per_order

        log_trade(trade)
        self.update_upstox_stats()
        self.update_markdown_ledger(trade)
        print(
            f"[SIGNAL] {side} {ticker} | Conviction: {score:.4f} | Audit: {sentiment:.2f}"
        )

    def start_vetoed_tracking(
        self,
        ticker,
        score,
        sentiment,
        entry_price,
        side="LONG",
        comment="",
        one_hour_prob="N/A",
        long_score=None,
        short_score=None,
    ):
        """Logs a vetoed signal to the DB and tracks it for 1 hour to see potential performance."""
        # GUARD: Don't duplicate if already being tracked (Open or Vetoed for the same side)
        with self.lock:
            if any(
                t["ticker"] == ticker
                and t["side"] == side
                and t["status"] in ["OPEN", "VETOED"]
                for t in self.active_shadow_trades
            ):
                print(f"[VETO-SKIP] {side} {ticker} already tracked (OPEN/VETOED). Skipping duplicate.")
                return

        # FETCH TRADINGVIEW SENTIMENT (Analytics only)
        tv_sentiment = get_tv_sentiment(ticker)
        
        # Mark as recently vetoed to prevent immediate repeat audits
        self._mark_recently_vetoed(ticker)

        trade = {
            "trade_id": f"V-{int(time.time())}-{ticker}",
            "timestamp": datetime.now().isoformat(),
            "ticker": ticker,
            "side": side,
            "tech_score": float(score),
            "nlp_sentiment": float(sentiment),
            "long_score": float(long_score) if long_score is not None else None,
            "short_score": float(short_score) if short_score is not None else None,
            "entry_price": entry_price,
            "peak_price": entry_price,
            "peak_profit_pct": 0.0,
            "final_profit_pct": 0.0,
            "exit_time": (datetime.now() + timedelta(hours=1)).isoformat(),
            "status": "VETOED",
            "comment": comment,
            "one_hour_prob": one_hour_prob,
            "tv_sentiment": tv_sentiment,
            "quantity": 0,
            "stop_loss_pct": 0.0,
            "take_profit_pct": 0.0,
            "margin_used": 0.0,
            "buy_brokerage": 0.0,
            "trailing_active": False,
            "breakeven_locked": False,
            "extension_count": 0,
            "extension_pending": False,
            "extended_exit_time": None
        }
        with self.lock:
            self.active_shadow_trades.append(trade)
        log_trade(trade)
        print(f"[VETO-TRACK] {side} {ticker} added to performance tracking.")

    # ══════════════════════════════════════════════════════════════════════
    # LOSS-EXTENSION SUBSYSTEM
    # When a stop-loss fires on a negative position we run up to 2 iterative
    # checks before actually closing the trade:
    #   1. Re-score XGB — if conviction flipped, close immediately.
    #   2. Ask a fast Gemini triage agent — if it says CLOSE, close immediately.
    #   3. If EXTEND: wait 15 minutes, then repeat (max 2 extensions).
    #   4. After 2 extensions: force-close regardless.
    # ══════════════════════════════════════════════════════════════════════

    def _get_current_conviction(self, ticker: str, side: str):
        """
        Re-scores a single ticker live via XGB to check whether the model
        conviction still aligns with the original trade direction.

        BUG-FIX (Cross-Sectional Z-Score Collapse):
          Passing a single ticker to calculate_conviction_scores() produces a
          1-row DataFrame whose std() is always NaN, forcing every Z-scored
          feature to 0.0 and making XGBoost predictions meaningless.
          We now look up the ticker's score from the cached full-universe scan
          (self.latest_full_scores) instead.  If the cache is absent we fall
          back to a fresh full-universe scan so Z-scores stay valid.

        Returns:
            (long_conviction, short_conviction, aligned: bool)
            aligned = True  -> conviction still supports original trade
            aligned = False -> conviction has flipped against the trade
        """
        try:
            scores_df = None

            # 1. Try the cached full-universe scan first (preferred path).
            with self.lock:
                cached = self.latest_full_scores

            if cached is not None and not cached.empty:
                match = cached[cached["ticker"] == ticker]
                if not match.empty:
                    scores_df = match
                    log(f"[INFO] _get_current_conviction: using cached full-universe scores for {ticker}.")

            # 2. Cache miss or ticker not found – run a fresh full-universe scan
            #    so cross-sectional Z-scores are correctly normalised.
            if scores_df is None or scores_df.empty:
                log(f"[INFO] _get_current_conviction: cache miss for {ticker}. Running full-universe scan.")
                full_df = self.calculate_conviction_scores(TICKERS)
                if full_df is None or full_df.empty:
                    log(f"[WARN] _get_current_conviction: full scan returned no data for {ticker}. Defaulting aligned=False.")
                    return 0.0, 0.0, False
                match = full_df[full_df["ticker"] == ticker]
                if match.empty:
                    log(f"[WARN] _get_current_conviction: {ticker} not found in full scan. Defaulting aligned=False.")
                    return 0.0, 0.0, False
                scores_df = match

            row = scores_df.iloc[0]
            long_conv  = float(row.get("Long_Conviction", 0.0))
            short_conv = float(row.get("Short_Conviction", 0.0))

            if side == "LONG":
                # Buffer zone: only flip if conviction drops below -0.10
                aligned = long_conv > -0.10
            else:
                # Buffer zone: only flip if conviction drops below -0.10
                aligned = short_conv > -0.10

            log(f"[INFO] XGB re-score {ticker}: Long={long_conv:.4f} Short={short_conv:.4f} | Aligned={aligned} (Buffer: -0.10)")
            return long_conv, short_conv, aligned

        except Exception as e:
            log(f"[WARN] _get_current_conviction error for {ticker}: {e}")
            return 0.0, 0.0, False


    def shadow_tracker_loop(self):
        while True:
            try:
                with self.lock:
                    current_trades = list(self.active_shadow_trades)

                if not current_trades:
                    # No open trades — check every 5s so we react fast when
                    # a trade is placed. WebSocket keeps prices fresh for free.
                    time.sleep(5)
                    continue

                for trade in current_trades:
                    try:
                        price = self.broker.get_live_price(trade["ticker"])
                        if price is None:
                            continue

                        now = datetime.now()

                        if trade["status"] == "PENDING_ENTRY":
                            # Expiry check: cancel if confirmation window has expired (more than 5m past next 15-min boundary)
                            try:
                                pending_since = datetime.fromisoformat(trade.get("pending_since") or trade["timestamp"])
                                minute = pending_since.minute
                                next_15 = ((minute // 15) + 1) * 15
                                next_boundary = pending_since.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=next_15)
                                if now > next_boundary + timedelta(minutes=5):
                                    print(f"[PENDING -> CANCELLED] {trade['ticker']} expired (signal time: {pending_since.strftime('%H:%M')}, now: {now.strftime('%H:%M')}).")
                                    trade["status"] = "CANCELLED"
                                    trade["comment"] = "Cancelled - Confirmation window expired."
                                    with self.lock:
                                        self.realized_charges -= self.brokerage_per_order
                                        self.used_margin -= trade.get("margin_used", 0.0)
                                    log_trade(trade)
                                    self.update_upstox_stats()
                                    continue
                            except Exception as e_exp:
                                print(f"[WARN] Expiry check failed for {trade['ticker']}: {e_exp}")

                            # Time cutoff check: immediately abort/cancel any pending entry trades after 3:00 PM (15:00 IST)
                            if now.strftime("%H:%M") >= "15:00":
                                print(f"[PENDING -> CANCELLED] {trade['ticker']} confirmation aborted (Time cutoff: after 3:00 PM).")
                                trade["status"] = "CANCELLED"
                                trade["comment"] = "Cancelled - Aborted after 3:00 PM time cutoff."
                                with self.lock:
                                    self.realized_charges -= self.brokerage_per_order
                                    self.used_margin -= trade.get("margin_used", 0.0)
                                log_trade(trade)
                                self.update_upstox_stats()
                                continue

                            candle = self.get_last_completed_15min_candle(trade["ticker"])
                            raw_since = trade.get("pending_since")
                            if raw_since:
                                pending_since = datetime.fromisoformat(raw_since)
                            else:
                                pending_since = datetime.fromisoformat(trade["timestamp"])
                            if candle is not None:
                                candle_close_time = candle["timestamp"] + timedelta(minutes=15)
                                if candle_close_time > pending_since:
                                    # Candle closed after signal was generated. Check direction!
                                    is_confirmed = False
                                    if trade["side"] == "LONG" and candle["close"] > candle["open"]:
                                        is_confirmed = True
                                    elif trade["side"] == "SHORT" and candle["close"] < candle["open"]:
                                        is_confirmed = True

                                    if is_confirmed:
                                        # XGBOOST LIVE RE-VERIFICATION CHECK
                                        long_conv, short_conv, aligned = self._get_current_conviction(trade["ticker"], trade["side"])
                                        conv_score = long_conv if trade["side"] == "LONG" else short_conv
                                        orig_score = trade.get("tech_score", 0.0)
                                        
                                        # Enforce strict confirmation gates
                                        is_conv_ok = True
                                        reason_low_conv = ""
                                        if orig_score >= self.min_conviction:
                                            if conv_score < 0.10:
                                                is_conv_ok = False
                                                reason_low_conv = f"dropped from {orig_score:.4f} to {conv_score:.4f} (Required >= 0.10)"
                                        else:
                                            if conv_score < (orig_score - 0.05) or conv_score < 0.05:
                                                is_conv_ok = False
                                                reason_low_conv = f"faded from {orig_score:.4f} to {conv_score:.4f} (Required >= {max(0.05, orig_score - 0.05):.2f})"
                                                
                                        if not aligned or not is_conv_ok:
                                            block_reason = "XGBoost conviction flipped" if not aligned else f"conviction too low at entry ({reason_low_conv})"
                                            print(f"[PENDING -> CANCELLED] {trade['ticker']} confirmation blocked. {block_reason}.")
                                            trade["status"] = "CANCELLED"
                                            trade["comment"] = f"Cancelled - {block_reason}."
                                            # BUG-FIX (Charge Leakage): refund the buy brokerage that was
                                            # pre-charged when the PENDING_ENTRY signal was created.
                                            with self.lock:
                                                self.realized_charges -= self.brokerage_per_order
                                                self.used_margin -= trade.get("margin_used", 0.0)
                                            log_trade(trade)
                                            self.update_upstox_stats()
                                            continue

                                        # Execute entry!
                                        entry_price = price
                                        stop_loss_pct, take_profit_pct = self.compute_15min_atr(trade["ticker"])

                                        SLIPPAGE_BPS = 2
                                        slippage = entry_price * (SLIPPAGE_BPS / 10000)
                                        entry_price = entry_price + slippage if trade["side"] == "LONG" else entry_price - slippage

                                        qty = self.calculate_trade_quantity(entry_price, stop_loss_pct)
                                        sl_mult = 1 - (stop_loss_pct / 100) if trade["side"] == "LONG" else 1 + (stop_loss_pct / 100)
                                        sl_price = round(entry_price * sl_mult, 2)
                                        print(f"[ATR-SL] {trade['ticker']} | SL: {stop_loss_pct:.2f}% | TP: {take_profit_pct:.2f}%")
                                        
                                        # Place order
                                        upstox_order = self.broker.place_order(trade["ticker"], trade["side"], quantity=qty, price=entry_price, stop_loss=sl_price)
                                        order_id = "SANDBOX-ERROR"
                                        if hasattr(upstox_order, "data") and upstox_order.data:
                                            order_id = getattr(upstox_order.data, "order_id", "SANDBOX-SUCCESS")
                                        elif isinstance(upstox_order, dict) and "data" in upstox_order:
                                            order_id = upstox_order["data"].get("order_id", "SANDBOX-SUCCESS")

                                        print(f"[PENDING -> OPEN] Confirmed {trade['ticker']} at {entry_price} (Order: {order_id})")
                                        
                                        trade["status"] = "OPEN"
                                        trade["stop_loss_pct"] = stop_loss_pct
                                        trade["take_profit_pct"] = take_profit_pct
                                        trade["entry_price"] = entry_price
                                        trade["peak_price"] = entry_price
                                        trade["quantity"] = qty
                                        trade["margin_used"] = (qty * entry_price) / self.margin_multiplier
                                        trade["timestamp"] = now.isoformat()
                                        trade["exit_time"] = (now + timedelta(hours=1)).isoformat()
                                        trade["comment"] = trade.get("comment", "").replace("PENDING-CANDLE-CONFIRMATION", order_id)
                                        
                                        log_trade(trade)
                                        self.update_upstox_stats()
                                    else:
                                        # Candle closed against us or neutral. Wait for next or timeout.
                                        pass
                            
                            # Timeout check: 45 minutes
                            if now - pending_since > timedelta(minutes=45) and trade["status"] == "PENDING_ENTRY":
                                print(f"[PENDING -> CANCELLED] {trade['ticker']} confirmation timed out.")
                                trade["status"] = "CANCELLED"
                                trade["comment"] = "Cancelled - 15-min candle confirmation timed out."
                                # BUG-FIX (Charge Leakage): refund the buy brokerage that was
                                # pre-charged when the PENDING_ENTRY signal was created.
                                with self.lock:
                                    self.realized_charges -= self.brokerage_per_order
                                    self.used_margin -= trade.get("margin_used", 0.0)
                                log_trade(trade)
                                self.update_upstox_stats()
                            
                            # Live sync for pending entry trades before continuing
                            if trade["status"] == "PENDING_ENTRY":
                                pnl = ((price - trade["entry_price"]) / trade["entry_price"] * 100) if trade["side"] == "LONG" else ((trade["entry_price"] - price) / trade["entry_price"] * 100)
                                trade["exit_price"] = price
                                trade["final_profit_pct"] = round(pnl, 4)
                                
                                # Update peak price and peak pnl
                                if trade["side"] == "LONG":
                                    if price > trade.get("peak_price", 0.0):
                                        trade["peak_price"] = price
                                else:
                                    if price < trade.get("peak_price", 99999999.0):
                                        trade["peak_price"] = price
                                trade["peak_profit_pct"] = max(trade.get("peak_profit_pct", 0.0), pnl)
                                
                                log_trade(trade)

                            continue

                        if trade["side"] == "LONG":
                            pnl = (
                                (price - trade["entry_price"])
                                / trade["entry_price"]
                                * 100
                            )
                            if price > trade["peak_price"]:
                                trade["peak_price"] = price
                        else:
                            pnl = (
                                (trade["entry_price"] - price)
                                / trade["entry_price"]
                                * 100
                            )
                            if price < trade["peak_price"]:
                                trade["peak_price"] = price

                        trade["peak_profit_pct"] = max(trade["peak_profit_pct"], pnl)

                        now = datetime.now()

                        # ── VETOED TRADE FAST PATH ──────────────────────────────────
                        # VETOED trades must NOT go through OPEN-trade machinery
                        # (SL/TP/conviction-flip/breakeven/trailing-stop). They have
                        # no quantity or stop_loss_pct, so those checks produce wrong
                        # results. Instead: update live price + P&L every tick, and
                        # expire only when the 1-hour window closes or market shuts.
                        if trade["status"] == "VETOED":
                            trade["exit_price"] = price
                            trade["final_profit_pct"] = round(pnl, 4)
                            veto_expiry = datetime.fromisoformat(trade["exit_time"])
                            veto_expired = now >= veto_expiry or now.strftime("%H:%M") >= "15:15"
                            if veto_expired:
                                trade["status"] = "VETOED_EXPIRED"
                                log_trade(trade)
                                self.update_markdown_ledger(trade)
                                print(
                                    f"[VETOED_EXPIRED] {trade['ticker']} {trade['side']} "
                                    f"| 1h Close: ₹{price:.2f} | P&L if taken: {pnl:.2f}%"
                                )
                            else:
                                log_trade(trade)
                            continue  # skip all OPEN-trade logic below
                        # ── END VETOED FAST PATH ────────────────────────────────────

                        is_stop_loss_hit = pnl <= -trade.get("stop_loss_pct", 0.50)

                        # ── 15-MIN CONVICTION FLIP CHECK ───────────────────────────
                        # Every 15 minutes, re-score via XGBoost.  If the model's
                        # conviction has flipped against the trade direction, close
                        # immediately — before SL is even hit.
                        # Cost: ~0ms (reads from latest_full_scores cache).
                        is_conviction_flip = False
                        trade_id = trade.get("trade_id", trade.get("ticker"))
                        last_flip_check = self._conviction_flip_checked.get(trade_id)
                        flip_check_due = (
                            last_flip_check is None
                            or (now - last_flip_check).total_seconds() >= 900  # 15 min
                        )

                        if flip_check_due and trade["status"] == "OPEN":
                            self._conviction_flip_checked[trade_id] = now
                            long_conv, short_conv, aligned = self._get_current_conviction(
                                trade["ticker"], trade["side"]
                            )
                            conv_score = long_conv if trade["side"] == "LONG" else short_conv

                            if not aligned:
                                is_conviction_flip = True
                                flip_note = (
                                    f" | Conviction Flip @ 15-min check "
                                    f"(Long={long_conv:.3f} Short={short_conv:.3f})"
                                )
                                print(
                                    f"[CONVICTION-FLIP] {trade['ticker']} {trade['side']} "
                                    f"— XGBoost flipped. Closing trade. "
                                    f"Long={long_conv:.3f} Short={short_conv:.3f} P&L={pnl:.2f}%"
                                )
                            else:
                                print(
                                    f"[CONVICTION-OK] {trade['ticker']} {trade['side']} "
                                    f"— XGBoost still aligned at 15-min check "
                                    f"(score={conv_score:.3f}). Holding."
                                )
                        # ── END 15-MIN CONVICTION FLIP CHECK ───────────────────────

                        # ── TRAILING STOP & BREAKEVEN MANAGEMENT ──────────────────
                        sl_pct = trade.get("stop_loss_pct", 0.50)

                        if not trade.get("breakeven_locked") and pnl >= sl_pct:
                            trade["breakeven_locked"] = True
                            print(f"[BREAKEVEN] {trade['ticker']} locked at entry. P&L={pnl:.2f}%")

                        if not trade.get("trailing_active") and pnl >= (sl_pct * 2.0):
                            trade["trailing_active"] = True
                            print(f"[TRAILING] {trade['ticker']} trailing stop activated. P&L={pnl:.2f}%")

                        is_trailing_exit = False
                        is_breakeven_exit = False
                        if trade.get("trailing_active"):
                            trailing_stop_level = trade["peak_profit_pct"] - sl_pct
                            if pnl <= trailing_stop_level:
                                is_trailing_exit = True
                        elif trade.get("breakeven_locked"):
                            if pnl <= 0.0:
                                is_breakeven_exit = True

                        # ── TAKE-PROFIT CHECK ──────────────────────────────────────
                        is_take_profit_hit = pnl >= trade.get("take_profit_pct", 1.00)
                        if (is_take_profit_hit or is_trailing_exit) and trade["status"] == "OPEN":
                            if is_trailing_exit:
                                tp_note = f" | Trailing Stop @ {pnl:.2f}% (peak {trade['peak_profit_pct']:.2f}%)"
                            else:
                                tp_pct_used = trade.get("take_profit_pct", 1.00)
                                tp_note = f" | TP Hit @ {tp_pct_used:.2f}%"

                            trade["status"] = "TAKE_PROFIT"
                            trade["comment"] = trade.get("comment", "") + tp_note
                            trade["exit_price"] = price
                            trade["final_profit_pct"] = round(pnl, 4)

                            # Finalize P&L with charges
                            sell_value = trade["quantity"] * price
                            total_exit_costs = self.calculate_exit_charges(sell_value)
                            if trade["side"] == "LONG":
                                gross_pnl_amt = (price - trade["entry_price"]) * trade["quantity"]
                            else:
                                gross_pnl_amt = (trade["entry_price"] - price) * trade["quantity"]
                            net_pnl_amt = gross_pnl_amt - total_exit_costs - (trade.get("buy_brokerage") or 0.0)
                            trade["final_profit_pct"] = (net_pnl_amt / (trade["entry_price"] * trade["quantity"])) * 100
                            trade["net_pnl_amt"] = net_pnl_amt

                            with self.lock:
                                self.virtual_capital += net_pnl_amt
                                self.used_margin -= (trade.get("margin_used") or 0.0)
                                self.realized_charges += total_exit_costs

                            self._mark_recently_closed(trade["ticker"])
                            self.update_upstox_stats()
                            log_trade(trade)
                            self.update_markdown_ledger(trade)
                            print(f"[{trade['status']}] {trade['ticker']} | Net P&L: {trade['final_profit_pct']:.2f}% (₹{net_pnl_amt:.2f})")
                            continue  # skip all other exit checks — TP is final
                        # ── END TAKE-PROFIT CHECK ──────────────────────────────────

                        # ── TIME EXPIRY EXTENSION CHECK ────────────────────────────
                        raw_time_expiry = now >= datetime.fromisoformat(trade["exit_time"])
                        if (
                            raw_time_expiry
                            and not trade.get("extension_pending", False)
                            and pnl < 0
                            and trade.get("extension_count", 0) < 2
                            and now.strftime("%H:%M") < "15:15"
                            and trade["status"] == "OPEN"
                            and not is_conviction_flip   # don't extend if XGBoost already flipped
                        ):
                            ext_count = trade.get("extension_count", 0)
                            print(f"[EXPIRY-CHECK] {trade['ticker']} is at time expiry with loss ({pnl:.2f}%). Re-evaluating...")
                            long_conv, short_conv, aligned = self._get_current_conviction(trade["ticker"], trade["side"])
                            conv_score = long_conv if trade["side"] == "LONG" else short_conv
                            
                            if aligned and conv_score > self.min_conviction:
                                print(f"[EXPIRY-CHECK] {trade['ticker']} Conviction {conv_score:.4f} is high. Asking Gemini in background...")
                                if getattr(self, 'gemini_enabled', False) and hasattr(self, 'clients') and self.clients:
                                    trade["extension_pending"] = True
                                    trade["extension_started"] = now.isoformat()
                                    
                                    def _gemini_check_extension(target_trade, t_price, t_pnl, current_ext_count):
                                        prompt = f"We are holding a {target_trade['side']} position in {target_trade['ticker']} on the Indian Stock Market with Entry Price: {target_trade['entry_price']:.2f}, Current Price: {t_price:.2f}, PnL: {t_pnl:.2f}%. The trade is about to close due to time expiry. If extended for 1 hour, can we mitigate this loss or turn it into profit? Analyze current news and market sentiment. Answer with a clear 'EXTEND' if we should wait, or 'CLOSE' if we should cut losses now."
                                        
                                        from google.genai import types
                                        total_combinations = len(self.s1_model_tiers) * len(self.api_keys)
                                        attempts = 0
                                        success = False
                                        ans = "CLOSE"
                                        
                                        while attempts < total_combinations:
                                            with self.lock:
                                                current_key = self.s1_active_key_idx
                                                current_model = self.s1_model_tiers[self.s1_active_tier_idx]
                                            
                                            try:
                                                ext_client = self.clients[current_key]
                                                print(f"[EXPIRY-CHECK] Attempting extension evaluation with {current_model} (Key {current_key}) for {target_trade['ticker']}...")
                                                response = ext_client.models.generate_content(
                                                    model=current_model,
                                                    contents=prompt,
                                                    config=types.GenerateContentConfig(
                                                        tools=[{"google_search": {}}],
                                                    )
                                                )
                                                ans = self._extract_response_text(response).upper()
                                                success = True
                                                break
                                            except Exception as e:
                                                err_str = str(e)
                                                if "429" in err_str or "503" in err_str or "quota" in err_str.lower():
                                                    print(f"\n[EXTENSION-ROTATE] {current_model} Key {current_key} exhausted ({err_str[:80]}). Rotating...")
                                                    with self.lock:
                                                        self.s1_active_key_idx += 1
                                                        if self.s1_active_key_idx >= len(self.api_keys):
                                                            self.s1_active_key_idx = 0
                                                            self.s1_active_tier_idx += 1
                                                            if self.s1_active_tier_idx >= len(self.s1_model_tiers):
                                                                self.s1_active_tier_idx = 0
                                                            print(f"[EXTENSION-FALLBACK] Rotating to model {self.s1_model_tiers[self.s1_active_tier_idx]}...")
                                                    attempts += 1
                                                    time.sleep(1)
                                                    continue
                                                else:
                                                    print(f"\n[EXTENSION-ERROR] Gemini non-quota error ({current_model}, Key {current_key}) for {target_trade['ticker']}: {e}")
                                                    break
                                        
                                        with self.lock:
                                            if success:
                                                if "EXTEND" in ans:
                                                    target_trade["extended_exit_time"] = (datetime.fromisoformat(target_trade["exit_time"]) + timedelta(minutes=15)).isoformat()
                                                    target_trade["exit_time"] = target_trade["extended_exit_time"]
                                                    target_trade["extension_count"] = current_ext_count + 1
                                                    target_trade["comment"] = target_trade.get("comment", "") + f" | Ext {current_ext_count + 1} (Gemini)"
                                                    log_trade(target_trade)
                                                    print(f"\n[EXTENSION] {target_trade['ticker']} extended by 15 mins. (Count: {current_ext_count+1}/2)")
                                                else:
                                                    print(f"\n[EXTENSION-REJECTED] Gemini suggested CLOSE for {target_trade['ticker']}.")
                                            else:
                                                print(f"\n[EXTENSION-ERROR] Gemini call failed after trying all options.")
                                            target_trade["extension_pending"] = False

                                    threading.Thread(target=_gemini_check_extension, args=(trade, price, pnl, ext_count), daemon=True).start()
                                    continue  # Skip exit check this loop iteration
                                else:
                                    print("[EXTENSION-ERROR] Gemini is not enabled.")
                            else:
                                print(f"[EXTENSION-REJECTED] Conviction insufficient (Aligned: {aligned}, Score: {conv_score:.4f})")
                        # ── END TIME EXPIRY EXTENSION CHECK ────────────────────────

                        pending_time = trade.get("extension_started")
                        if trade.get("extension_pending") and pending_time:
                            if now - datetime.fromisoformat(pending_time) > timedelta(minutes=5):
                                print(f"[EXTENSION-TIMEOUT] Gemini extension check timed out for {trade['ticker']}.")
                                trade["extension_pending"] = False

                        is_time_expiry = raw_time_expiry and not trade.get("extension_pending", False)

                        if (
                            is_conviction_flip
                            or is_time_expiry
                            or now.strftime("%H:%M") >= "15:15"
                            or is_stop_loss_hit
                            or is_breakeven_exit
                        ):
                            net_pnl_amt = 0.0
                            # Determine close reason for OPEN trades
                            if is_conviction_flip:
                                trade["status"] = "CLOSED"
                                trade["comment"] = trade.get("comment", "") + flip_note
                            elif is_breakeven_exit:
                                trade["status"] = "CLOSED"
                                be_note = f" | Breakeven Exit (peak {trade.get('peak_profit_pct', 0):.2f}%)"
                                trade["comment"] = trade.get("comment", "") + be_note
                            elif is_stop_loss_hit:
                                ext_used = trade.get("extension_count", 0)
                                sl_pct_used = trade.get("stop_loss_pct", 0.50)
                                sl_note = f" | SL Hit @ {sl_pct_used:.2f}%" + (f" (after {ext_used} ext)" if ext_used else "")
                                trade["status"] = "STOP_LOSS"
                                trade["comment"] = trade.get("comment", "") + sl_note
                            else:
                                trade["status"] = "CLOSED"

                            # Finalize P&L with charges
                            sell_value = trade["quantity"] * price
                            total_exit_costs = self.calculate_exit_charges(sell_value)

                            # Gross P&L Amount
                            if trade["side"] == "LONG":
                                gross_pnl_amt = (
                                    price - trade["entry_price"]
                                ) * trade["quantity"]
                            else:
                                gross_pnl_amt = (
                                    trade["entry_price"] - price
                                ) * trade["quantity"]

                            net_pnl_amt = (
                                gross_pnl_amt
                                - total_exit_costs
                                - (trade.get("buy_brokerage") or 0.0)
                            )

                            # Convert Net P&L back to percentage relative to entry price (for UI consistency)
                            trade["final_profit_pct"] = (
                                net_pnl_amt
                                / (trade["entry_price"] * trade["quantity"])
                            ) * 100
                            trade["net_pnl_amt"] = net_pnl_amt

                            # Update Global Stats
                            with self.lock:
                                self.virtual_capital += net_pnl_amt
                                self.used_margin -= (trade.get("margin_used") or 0.0)
                                self.realized_charges += total_exit_costs

                            # Mark cooldown so the same ticker can't be re-entered for 30 min
                            self._mark_recently_closed(trade["ticker"])

                            self.update_upstox_stats()

                            trade["exit_price"] = price
                            trade["comment"] = (
                                (
                                    trade.get("comment", "")
                                    + f" | Net: ₹{net_pnl_amt:.2f}"
                                )
                                if trade["status"] == "CLOSED"
                                else trade.get("comment", "")
                            )
                            log_trade(trade)
                            if trade["status"] in ["CLOSED", "STOP_LOSS", "TAKE_PROFIT", "VETOED_EXPIRED"]:
                                self.update_markdown_ledger(trade)
                            print(
                                f"[{trade['status']}] {trade['ticker']} | Net P&L: {trade['final_profit_pct']:.2f}% (₹{net_pnl_amt:.2f})"
                            )
                        else:
                            # --- LIVE SYNC: push current price & P&L to DB for real-time dashboard ---
                            trade["final_profit_pct"] = round(
                                pnl, 4
                            )  # live current P&L
                            trade["exit_price"] = price  # live current price
                            log_trade(trade)
                    except Exception as e:
                        print(
                            f"[ERROR] Shadow tracker error for {trade['ticker']}: {e}"
                        )
                        continue

                with self.lock:
                    self.active_shadow_trades = [
                        t
                        for t in self.active_shadow_trades
                        if t["status"] in ["OPEN", "VETOED", "PENDING_ENTRY"]
                    ]
                
                # Update upstox_stats.json snapshot with live price, P&L, margin, and time-left updates
                self.update_upstox_stats()
            except Exception as e:
                print(f"[CRITICAL] Shadow tracker loop error: {e}")
                pass
            # Reduced from 60s → 5s. With WebSocket, get_live_price() reads
            # from the in-memory cache (no REST calls), so polling every 5s
            # costs nothing and gives ~12x faster SL/TP detection.
            time.sleep(5)

    def update_markdown_ledger(self, trade):
        ledger_path = "data/VANGUARD_DEMO_LEDGER.md"
        os.makedirs("data", exist_ok=True)
        if not os.path.exists(ledger_path):
            with open(ledger_path, "w", encoding="utf-8") as f:
                f.write(
                    "# VANGUARD ELITE COMMAND CENTER LEDGER\n\n| Timestamp | Ticker | Side | Qty | Entry | Exit | Net ₹ | Final % | Status | Comment |\n| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
                )
        with open(ledger_path, "a", encoding="utf-8") as f:
            qty = trade.get("quantity", 1)
            net_amt = trade.get("net_pnl_amt", 0)
            f.write(
                f"| {trade['timestamp'][:16]} | {trade['ticker']} | {trade['side']} | {qty} | {trade['entry_price']:.2f} | {trade.get('exit_price', 0):.2f} | ₹{net_amt:.2f} | {trade['final_profit_pct']:.2f}% | {trade['status']} | {trade.get('comment', '')} |\n"
            )

    def _recompute_used_margin(self):
        """Always derive used_margin from the current live trade list (not from increments)."""
        with self.lock:
            total = sum(
                t.get("margin_used") or
                ((t.get("quantity", 1) * t.get("entry_price", 0)) / self.margin_multiplier)
                for t in self.active_shadow_trades
                if t["status"] == "OPEN"
            )
        self.used_margin = total
        return total

    def update_upstox_stats(self):
        """Saves a full virtual portfolio snapshot to JSON and SQLite for persistence.
        All values are recomputed from ground truth (live trade list + capital) so
        there is no drift between restarts.
        """
        # Always recompute used_margin from live state
        self._recompute_used_margin()

        # Compute unrealized P&L on open trades (sum of live running pnl amounts)
        unrealized_pnl_inr = 0.0
        open_positions = []
        pending_positions = []
        now = datetime.now()
        with self.lock:
            for t in self.active_shadow_trades:
                if t["status"] == "PENDING_ENTRY":
                    pending_positions.append({
                        "ticker":              t["ticker"],
                        "side":                t["side"],
                        "quantity":            t.get("quantity", 0),
                        "entry_price":         round(t.get("entry_price", 0), 2),
                        "entry_time":          t["timestamp"],
                        "one_hour_prob":       t.get("one_hour_prob", "N/A"),
                        "comment":             t.get("comment", ""),
                        "status":              "PENDING_ENTRY",
                        "tech_score":          float(t.get("tech_score") or 0.0) if t.get("tech_score") is not None else None,
                        "long_score":          float(t.get("long_score") or 0.0) if t.get("long_score") is not None else None,
                        "short_score":         float(t.get("short_score") or 0.0) if t.get("short_score") is not None else None,
                    })
                    continue

                if t["status"] != "OPEN":
                    continue
                qty         = t.get("quantity", 1)
                entry       = t.get("entry_price", 0)
                current     = t.get("exit_price") or entry   # shadow tracker writes live price here
                side        = t.get("side", "LONG")
                margin_used = t.get("margin_used") or ((qty * entry) / self.margin_multiplier)

                if side == "LONG":
                    gross_pnl_inr = (current - entry) * qty
                else:
                    gross_pnl_inr = (entry - current) * qty

                pnl_pct = (gross_pnl_inr / (entry * qty) * 100) if entry > 0 else 0.0

                # Time left in minutes
                # During an active extension, count down to the EXTENSION deadline,
                # not the original exit_time (which would already show 0).
                try:
                    ext_pending  = t.get("extension_pending", False)
                    ext_exit_str = t.get("extended_exit_time")
                    if ext_pending and ext_exit_str:
                        effective_exit_dt = datetime.fromisoformat(ext_exit_str)
                    else:
                        effective_exit_dt = datetime.fromisoformat(t["exit_time"])
                    time_left = max(0, int((effective_exit_dt - now).total_seconds() / 60))
                except Exception:
                    time_left = 0

                unrealized_pnl_inr += gross_pnl_inr
                ext_count = t.get("extension_count", 0)
                open_positions.append({
                    "ticker":              t["ticker"],
                    "side":                side,
                    "quantity":            qty,
                    "entry_price":         round(entry, 2),
                    "current_price":       round(current, 2),
                    "trade_value":         round(qty * entry, 2),
                    "margin_used":         round(margin_used, 2),
                    "unrealized_pnl_inr":  round(gross_pnl_inr, 2),
                    "unrealized_pnl_pct":  round(pnl_pct, 4),
                    "peak_profit_pct":     round(t.get("peak_profit_pct", 0), 4),
                    "entry_time":          t["timestamp"],
                    "exit_time":           t["exit_time"],
                    "time_left_min":       time_left,
                    "one_hour_prob":       t.get("one_hour_prob", "N/A"),
                    "comment":             t.get("comment", ""),
                    # Extension fields — consumed by the dashboard for status badges
                    "extension_count":     ext_count,
                    "extension_pending":   t.get("extension_pending", False),
                    "extended_exit_time":  t.get("extended_exit_time"),
                    "tech_score":          float(t.get("tech_score") or 0.0) if t.get("tech_score") is not None else None,
                    "long_score":          float(t.get("long_score") or 0.0) if t.get("long_score") is not None else None,
                    "short_score":         float(t.get("short_score") or 0.0) if t.get("short_score") is not None else None,
                })

        # Today's realized P&L in INR (from closed trades in DB)
        today_pnl_inr = 0.0
        try:
            from scripts.database_manager import get_today_realized_pnl
            today_pnl_inr = get_today_realized_pnl()
        except Exception:
            pass

        total_pnl_inr = self.virtual_capital - self.initial_capital
        total_pnl_pct = (total_pnl_inr / self.initial_capital * 100) if self.initial_capital > 0 else 0.0

        stats = {
            "initial_capital":        self.initial_capital,
            "virtual_capital":        round(self.virtual_capital, 2),
            "used_margin":            round(self.used_margin, 2),
            "available_margin":       round(self.virtual_capital - self.used_margin, 2),
            "realized_charges":       round(self.realized_charges, 2),
            "open_positions_count":   len(open_positions),
            "unrealized_pnl_inr":     round(unrealized_pnl_inr, 2),
            "day_realized_pnl_inr":   round(today_pnl_inr, 2),
            "total_pnl_inr":          round(total_pnl_inr, 2),
            "total_pnl_pct":          round(total_pnl_pct, 4),
            "positions":              open_positions,
            "pending_positions":      pending_positions,
            "timestamp":              now.isoformat(),
        }
        with open("upstox_stats.json", "w") as f:
            json.dump(stats, f, indent=4)

        # Also log to SQLite for historical tracking
        log_system_stats(stats)

    def sync_upstox_portfolio(self):
        """Re-derives all portfolio metrics from the live trade list.
        Called periodically; also triggers update_upstox_stats() which persists everything.
        """
        try:
            # used_margin is recomputed inside update_upstox_stats via _recompute_used_margin()
            self.update_upstox_stats()
        except Exception as e:
            print(f"[WARN] Portfolio Sync Error: {e}")

    def run(self):
        while True:
            try:
                # 1. Daily Capital Reset & Intraday Slot Refresh
                now_date = datetime.now().date()
                with self.lock:
                    open_count = len(
                        [t for t in self.active_shadow_trades if t["status"] == "OPEN"]
                    )

                # Reset baseline if it's a new day OR if we have zero active trades (to scale with profits)
                if now_date > self.current_date or open_count == 0:
                    if now_date > self.current_date:
                        self.current_date = now_date
                        log(
                            f"[INFO] Day Reset: Initial Capital for today: Rs{self.virtual_capital:.2f}"
                        )

                    if self.day_start_capital != self.virtual_capital:
                        self.day_start_capital = self.virtual_capital
                        log(
                            f"[INFO] Slot Refresh: Scaling position size to current capital: Rs{self.day_start_capital:.2f}"
                        )

                now_str = datetime.now().strftime("%H:%M")
                if not ("09:00" <= now_str < "15:30"):
                    print(f"[{now_str}] Outside Market Hours. Waiting...")
                    time.sleep(600)
                    continue

                log("\n" + "-" * 40)
                log(f"VANGUARD SCAN CYCLE: {datetime.now().strftime('%H:%M:%S')}")
                with self.lock:
                    open_count = len(
                        [t for t in self.active_shadow_trades if t["status"] == "OPEN"]
                    )
                log(f"Active Trades: {open_count}/{self.max_trade_slots}")
                log(
                    f"Virtual Capital: Rs{self.virtual_capital:.2f} | Charges: Rs{self.realized_charges:.2f}"
                )
                log("-" * 40)

                # 2. Fetch live scores for all tickers
                scores_df = self.calculate_conviction_scores(TICKERS)
                if scores_df.empty:
                    log("[WARN] No scores generated. Checking connection...")
                    time.sleep(60)
                    continue

                # 3. Execution Window Check
                is_trading_window = "10:15" <= now_str < "15:05"

                # 4. Process Signals
                for side in ["LONG", "SHORT"]:
                    # 4a. Filter out tickers in cooldown (Closed or Vetoed) before selecting top signals
                    eligible_df = scores_df[~scores_df['ticker'].apply(
                        lambda x: self._is_in_cooldown(x) or self._is_veto_cooldown(x)
                    )]
                    
                    conv_col = "Long_Conviction" if side == "LONG" else "Short_Conviction"
                    raw_col = "long_score" if side == "LONG" else "short_score"
                    rank_col = "Long_Rank" if side == "LONG" else "Short_Rank"

                    # Print top 3 candidates for visibility
                    if not eligible_df.empty:
                        top3 = eligible_df.sort_values(conv_col, ascending=False).head(3)
                        log(f"[SCAN] Top 3 {side} Candidates:")
                        for _, row in top3.iterrows():
                            log(f"  • {row['ticker']:<15} | Conviction: {row[conv_col]:.4f} (Raw: {row[raw_col]:.4f})")

                    # Get top 2 Hybrid (Net) Candidates that meet min_conviction
                    top_net = eligible_df[eligible_df[conv_col] >= self.min_conviction].sort_values(rank_col, ascending=True).head(2)
                    
                    # Get top 2 Pure Directional Candidates that meet min_raw_score
                    eligible_raw_df = eligible_df[~eligible_df['ticker'].isin(top_net['ticker'])]
                    top_raw = eligible_raw_df[eligible_raw_df[raw_col] >= self.min_raw_score].sort_values(raw_col, ascending=False).head(2)
                    
                    top_signals = pd.concat([top_net, top_raw])

                    for _, sig in top_signals.iterrows():
                        conviction = sig[conv_col]
                        raw_score = sig[raw_col]

                        if True:
                            if not is_trading_window:
                                if int(sig.get(rank_col, 99)) == 1:
                                    log(
                                        f"[SCAN ONLY] {side} {sig['ticker']} | Conviction: {conviction:.4f} | Raw: {raw_score:.4f} (Outside Trade Window)"
                                    )
                                continue

                            # CONCURRENCY CHECK
                            with self.lock:
                                if (
                                    len(
                                        [
                                            t
                                            for t in self.active_shadow_trades
                                            if t["status"] in ["OPEN", "PENDING_ENTRY"]
                                        ]
                                    )
                                    >= self.max_trade_slots
                                ):
                                    log(
                                        f"[SKIP] Concurrency Limit Reached (Max {self.max_trade_slots})."
                                    )
                                    break

                                already_open = any(
                                    t["ticker"] == sig["ticker"]
                                    and t["status"] in ["OPEN", "VETOED", "PENDING_ENTRY"]
                                    for t in self.active_shadow_trades
                                )

                            if already_open:
                                log(f"[SKIP] {side} {sig['ticker']} already OPEN/VETOED in shadow tracker.")
                                continue  # Check next rank if already open

                            # 4b. Veto cooldown check (belt-and-suspenders before calling Gemini)
                            if self._is_veto_cooldown(sig["ticker"]):
                                mins_ago = int(
                                    (datetime.now() - self.recent_vetoes[sig["ticker"]]).total_seconds() / 60
                                ) if sig["ticker"] in self.recent_vetoes else "?"
                                log(f"[VETO-COOLDOWN] {side} {sig['ticker']} vetoed {mins_ago}m ago. Skipping (30m cooldown).")
                                continue

                            # 5. Gemini AI Audit
                            log(
                                f"[AUDIT] {side} {sig['ticker']} | Conviction: {conviction:.4f} | Raw Score: {raw_score:.4f} | Verifying..."
                            )
                            sentiment, reason, one_hour_prob = self.gemini_audit(
                                sig["ticker"], side, conviction, sig
                            )

                            if sentiment == "SYSTEM_ERROR":
                                log(f"[SKIP] {sig['ticker']} skipped due to AI API error. Will retry next scan.")
                                continue

                            # ── Reset veto_stats if trading day changed ────────────────────
                            if datetime.now().date() != self._veto_stats_date:
                                self._veto_stats_date = datetime.now().date()
                                self.veto_stats = {
                                    "s1_vetoes": 0, "s2_vetoes": 0,
                                    "s1_passes": 0, "s2_passes": 0,
                                    "s1_tickers": [], "s2_tickers": []
                                }

                            # ── Determine which stage issued the veto/pass ─────────────────
                            # gemini_audit() prefixes reason with [S1-VETO], [S2-PASS], etc.
                            is_s1 = reason.startswith("[S1-")
                            veto_stage = "S1" if is_s1 else "S2"

                            # ── VETO LOGIC ──────────────────────────────────────────────────
                            is_vetoed = False
                            sentiment_upper = sentiment.upper()
                            if "VETO" in sentiment_upper:
                                is_vetoed = True
                            elif side == "LONG" and sentiment_upper in [
                                "STRONG BEARISH", "BEARISH", "NEUTRAL",
                            ]:
                                is_vetoed = True
                            elif side == "SHORT" and sentiment_upper in [
                                "STRONG BULLISH", "BULLISH", "NEUTRAL",
                            ]:
                                is_vetoed = True

                            entry_price = self.broker.get_live_price(sig["ticker"])
                            if not entry_price or entry_price <= 0:
                                entry_price = float(sig["Close"])

                            if not is_vetoed:
                                sent_map = {
                                    "STRONG BULLISH": 1.0, "BULLISH": 0.75,
                                    "NEUTRAL": 0.5, "BEARISH": 0.25, "STRONG BEARISH": 0.0,
                                }
                                sent_score = sent_map.get(sentiment, 0.5)

                                # Track S2 pass (S1 already passed to reach here)
                                self.veto_stats["s1_passes"] += 1
                                self.veto_stats["s2_passes"] += 1
                                log(
                                    f"[✓ PASS] {side} {sig['ticker']} confirmed by AI | Sentiment: {sentiment} | {reason}"
                                )
                                self.start_shadow_trade(
                                    sig["ticker"], conviction, sent_score, entry_price,
                                    side, f"[{sentiment}] {reason}", one_hour_prob,
                                    long_score=sig["long_score"],
                                    short_score=sig["short_score"]
                                )
                                break  # Trade placed, move to next side
                            else:
                                # ── Increment per-stage counters ─────────────────────────
                                if is_s1:
                                    self.veto_stats["s1_vetoes"] += 1
                                    self.veto_stats["s1_tickers"].append(
                                        (sig["ticker"], side, reason)
                                    )
                                    print(
                                        f"[✗ S1-VETO] Rank {int(sig[rank_col])} {side} {sig['ticker']} "
                                        f"| Conviction: {conviction:.4f} | {reason}"
                                    )
                                else:
                                    self.veto_stats["s1_passes"] += 1  # S1 passed
                                    self.veto_stats["s2_vetoes"] += 1
                                    # Extract rule from reason e.g. [VETOED by model | RULE 1 ...]
                                    import re as _re
                                    rule_match = _re.search(r'RULE \d+[^]]*', reason)
                                    rule_tag = rule_match.group(0) if rule_match else "S2"
                                    self.veto_stats["s2_tickers"].append(
                                        (sig["ticker"], side, rule_tag, reason)
                                    )
                                    print(
                                        f"[✗ S2-VETO] Rank {int(sig[rank_col])} {side} {sig['ticker']} "
                                        f"| Conviction: {conviction:.4f} | {rule_tag} | {reason}"
                                    )

                                self.start_vetoed_tracking(
                                    sig["ticker"], conviction, 0.0, entry_price,
                                    side, f"[{veto_stage}-VETO] {reason}", one_hour_prob,
                                    long_score=sig["long_score"],
                                    short_score=sig["short_score"]
                                )
                        else:
                            if int(sig[rank_col]) == 1:
                                print(
                                    f"[SKIP] No high-conviction {side} found (Best: {conviction:.4f})"
                                )
                            break

                # ── SESSION SUMMARY (End of Scan Cycle) ───────────────────────────
                print("\n" + "═"*70)
                print(f" SESSION VETO SUMMARY ({datetime.now().strftime('%H:%M')})")
                print("═"*70)
                print(f" S1 (Technical) : VETOED={self.veto_stats['s1_vetoes']} | PASSED={self.veto_stats['s1_passes']}")
                print(f" S2 (Governance): VETOED={self.veto_stats['s2_vetoes']} | PASSED={self.veto_stats['s2_passes']}")
                
                if self.veto_stats["s1_tickers"]:
                    print("\n S1 VETOES:")
                    for t, s, r in self.veto_stats["s1_tickers"]:
                        print(f"  • {s} {t} : {r[:80]}...")
                
                if self.veto_stats["s2_tickers"]:
                    print("\n S2 VETOES:")
                    for t, s, rule, r in self.veto_stats["s2_tickers"]:
                        print(f"  • {s} {t} [{rule}] : {r[:80]}...")
                print("═"*70 + "\n")

                # ── ALIGNED 15-MIN CANDLE SCHEDULER ─────────────────────────
                # Fire exactly on :00, :15, :30, :45 — right after each 15-min
                # candle closes.  No drift: we sleep the precise remainder of
                # the current 15-min window, not a fixed 300 s.
                _now       = datetime.now()
                _secs_past = (_now.minute % 15) * 60 + _now.second + _now.microsecond / 1e6
                _wait      = (15 * 60) - _secs_past
                if _wait <= 2:          # already on the boundary — skip to next window
                    _wait += 15 * 60
                _next_fire = _now + timedelta(seconds=_wait)
                print(f"[SCHEDULER] Next 15-min scan at {_next_fire.strftime('%H:%M:%S')} "
                      f"(sleeping {int(_wait)}s)")
                time.sleep(_wait)
                # ── END ALIGNED SCHEDULER ────────────────────────────────────

            except Exception as e:
                print(f"[CRITICAL ERROR] Main Engine Loop: {e}")
                traceback.print_exc()
                time.sleep(60)


if __name__ == "__main__":
    VanguardEngine(
        "models/xgb_long_model.json", "models/scaler.pkl", "models/model_metadata.json"
    ).run()

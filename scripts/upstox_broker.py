import upstox_client
from upstox_client.rest import ApiException
import os
import json
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

INSTRUMENT_CACHE_PATH = os.path.join(os.path.dirname(__file__), "instrument_cache.json")
try:
    with open(INSTRUMENT_CACHE_PATH, "r") as f:
        INSTRUMENT_MAP = json.load(f)
except Exception as e:
    print(f"[WARN] Failed to load instrument cache in UpstoxBroker: {e}")
    INSTRUMENT_MAP = {}

load_dotenv()

class UpstoxSandboxBroker:
    def __init__(self):
        self.sandbox_token = os.getenv("UPSTOX_SANDBOX_ACCESS_TOKEN")
        self.analytics_token = os.getenv("UPSTOX_ANALYTICS_ACCESS_TOKEN")
        
        # 1. Order Configuration (Sandbox)
        self.order_config = upstox_client.Configuration(sandbox=True)
        self.order_config.access_token = self.sandbox_token
        # Monkeypatch for sandbox restrictions
        self.order_config.sandbox_urls.add("/v2/order/retrieve-all")
        self.order_config.sandbox_urls.add("/v3/user/get-funds-and-margin")
        self.order_config.sandbox_urls.add("/v2/portfolio/short-term-positions")
        self.order_api_client = upstox_client.ApiClient(self.order_config)
        
        # 2. Data Configuration (Analytics / Live)
        # NOTE: The Upstox SDK has a singleton-like behavior for Configuration.
        # We must manually override fields to ensure the Data config is LIVE.
        self.data_config = upstox_client.Configuration(sandbox=False)
        self.data_config.sandbox = False
        self.data_config.host = "https://api.upstox.com"
        self.data_config.access_token = self.analytics_token
        self.data_api_client = upstox_client.ApiClient(self.data_config)

        # WebSocket manager (optional — attached after construction)
        # When attached, get_live_price() reads from the in-memory cache first.
        self.ws_manager = None

    def attach_websocket(self, ws_manager):
        """
        Attach a running UpstoxWebSocketManager so that get_live_price()
        and get_recent_candles() read from the local cache instead of REST.

        Call this AFTER ws_manager.start() has been called.
        """
        self.ws_manager = ws_manager
        print("[BROKER] WebSocket manager attached — live prices will be "
              "served from cache (REST fallback active).")
        
    def test_connection(self):
        """Verify connection by fetching order book (profile is not available in sandbox)."""
        try:
            api_instance = upstox_client.OrderApi(self.order_api_client)
            return api_instance.get_order_book('2.0')
        except ApiException as e:
            return {"error": f"Exception when calling OrderApi->get_order_book: {e}"}

    def _load_cache(self):
        cache_path = os.path.join(os.path.dirname(__file__), 'instrument_cache.json')
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_cache(self, cache):
        cache_path = os.path.join(os.path.dirname(__file__), 'instrument_cache.json')
        with open(cache_path, 'w') as f:
            json.dump(cache, f, indent=4)

    def get_instrument_key(self, ticker):
        """Maps a standard ticker (e.g. RELIANCE.NS) to Upstox instrument key with caching."""
        symbol = ticker.replace('.NS', '')
        
        cache = self._load_cache()
        if symbol in cache:
            return cache[symbol]
        
        try:
            print(f"[INFO] Searching Upstox Instrument Key for {symbol}...")
            api_instance = upstox_client.InstrumentsApi(self.data_api_client)
            api_response = api_instance.search_instrument(symbol)
            
            if api_response.status == 'success' and api_response.data:
                # Find exact match for trading_symbol
                for inst in api_response.data:
                    t_symbol = inst.get('trading_symbol')
                    if t_symbol == symbol:
                        key = inst.get('instrument_key')
                        cache[symbol] = key
                        self._save_cache(cache)
                        return key
                
                # Fallback to first result if no exact trading_symbol match
                key = api_response.data[0].get('instrument_key')
                cache[symbol] = key
                self._save_cache(cache)
                return key
                
            raise Exception("No instrument found")
        except Exception as e:
            print(f"[WARN] Instrument Search failed for {symbol}: {e}")
            return f"NSE_EQ|{symbol}" # Final guess

    def get_live_price(self, ticker):
        """
        Fetches current LTP.  Priority order:
          1. WebSocket cache (microseconds, zero network I/O)
          2. Upstox REST API  (fallback when WS not attached or cache stale)
          3. yfinance          (last resort)
        """
        # ── 1. WebSocket cache (fast path) ────────────────────────────────────
        if self.ws_manager is not None:
            try:
                instrument_key = self.get_instrument_key(ticker)
                if self.ws_manager.cache.is_fresh(instrument_key, max_age_seconds=10):
                    price, _ = self.ws_manager.cache.get_ltp(instrument_key)
                    if price is not None and price > 0:
                        return float(price)   # [WS-HIT] — returned in ~1µs
            except Exception:
                pass   # fall through to REST

        # ── 2. Upstox REST API (slow path / fallback) ─────────────────────────
        try:
            instrument_key = self.get_instrument_key(ticker)
            api_instance = upstox_client.MarketQuoteApi(self.data_api_client)
            api_response = api_instance.get_full_market_quote(instrument_key, '2.0')

            if api_response.status == 'success' and api_response.data:
                # 1. Direct lookup by requested instrument key
                if instrument_key in api_response.data:
                    return api_response.data[instrument_key].last_price

                # 2. Inconsistent symbol/ISIN key search fallback
                symbol = ticker.replace('.NS', '')
                for key, val in api_response.data.items():
                    if symbol in key or instrument_key in key or key in instrument_key:
                        return val.last_price

                # 3. Single entry fallback
                if len(api_response.data) == 1:
                    return list(api_response.data.values())[0].last_price
            raise Exception("Quote data missing in response")
        except Exception as e:
            print(f"[WARN] Upstox Live Price failed for {ticker}: {e}. Falling back to yfinance.")

        # ── 3. yfinance last resort ───────────────────────────────────────────
        try:
            stock = yf.Ticker(ticker)
            return stock.history(period='1d')['Close'].iloc[-1]
        except Exception:
            return None

    def get_recent_candles(self, ticker, interval='1minute', count=30):
        """
        Returns the most recent `count` completed candles for `ticker`.

        Priority:
          1. WebSocket candle builder cache (zero REST calls)
          2. Upstox REST API historical endpoint (fallback)

        Parameters
        ----------
        ticker   : Standard ticker e.g. "RELIANCE.NS"
        interval : "1minute" | "15minute"
        count    : How many completed candles to return

        Returns
        -------
        pd.DataFrame with columns: timestamp, open, high, low, close, volume
        or None if no data is available.
        """
        # ── 1. WebSocket candle cache (fast path) ─────────────────────────────
        if self.ws_manager is not None:
            try:
                instrument_key = self.get_instrument_key(ticker)
                # Require at least 80% of requested candles to trust the cache
                min_required = max(1, int(count * 0.8))
                if self.ws_manager.cache.has_candles(
                        instrument_key, interval, min_count=min_required):
                    df = self.ws_manager.cache.get_candles(
                        instrument_key, interval, count=count)
                    if df is not None and not df.empty:
                        return df   # [WS-CANDLE-HIT]
            except Exception:
                pass   # fall through to REST

        # ── 2. REST fallback ──────────────────────────────────────────────────
        days = 2 if interval == '1minute' else 1
        return self.get_historical_data(ticker, interval=interval, days=days)

    def get_historical_data(self, ticker, interval='day', days=30, fallback=True):
        """Fetches historical OHLC from Upstox with yfinance fallback."""
        try:
            instrument_key = self.get_instrument_key(ticker)
            api_instance = upstox_client.HistoryApi(self.data_api_client)
            
            # Use at least 10 days for daily interval to prevent empty weekend responses
            is_daily = interval in ['day', '1d']
            query_days = max(days, 10) if is_daily else days
            
            to_date = datetime.now().strftime('%Y-%m-%d')
            from_date = (datetime.now() - timedelta(days=query_days)).strftime('%Y-%m-%d')
            
            # Handle 60minute by merging Intraday + Historical, then resampling
            is_60m = interval == '60minute'
            upstox_interval = '30minute' if is_60m else ('day' if interval == '1d' else '1minute' if interval == '1m' else interval)
            
            # 1. Fetch Historical (up to yesterday)
            hist_response = api_instance.get_historical_candle_data1(
                instrument_key, upstox_interval, to_date, from_date, '2.0'
            )
            
            # 2. Fetch Intraday (today's live candles)
            intra_response = None
            if upstox_interval in ['1minute', '30minute']:
                try:
                    intra_response = api_instance.get_intra_day_candle_data(instrument_key, upstox_interval, '2.0')
                except:
                    pass

            all_candles = []
            if hist_response.status == 'success' and hist_response.data and hist_response.data.candles:
                all_candles.extend(hist_response.data.candles)
            if intra_response and intra_response.status == 'success' and intra_response.data and intra_response.data.candles:
                all_candles.extend(intra_response.data.candles)

            # 3. Fetch today's live daily candle from quote API
            if upstox_interval == 'day':
                try:
                    quote_api = upstox_client.MarketQuoteApi(self.data_api_client)
                    quote_resp = quote_api.get_full_market_quote(instrument_key, '2.0')
                    if quote_resp.status == 'success' and quote_resp.data:
                        quote_data = None
                        if instrument_key in quote_resp.data:
                            quote_data = quote_resp.data[instrument_key]
                        else:
                            symbol = ticker.replace('.NS', '')
                            for key, val in quote_resp.data.items():
                                if symbol in key or instrument_key in key or key in instrument_key:
                                    quote_data = val
                                    break
                        
                        if quote_data and hasattr(quote_data, 'ohlc') and quote_data.ohlc:
                            ohlc = quote_data.ohlc
                            ts_now = datetime.now()
                            volume = getattr(quote_data, 'volume', 0) or 0
                            oi = getattr(quote_data, 'oi', 0) or 0
                            if volume > 0 or quote_data.last_price > 0:
                                today_candle = [
                                    ts_now.strftime('%Y-%m-%dT00:00:00+05:30'),
                                    float(ohlc.open),
                                    float(ohlc.high),
                                    float(ohlc.low),
                                    float(quote_data.last_price),
                                    int(volume),
                                    int(oi)
                                ]
                                all_candles.append(today_candle)
                except Exception as e_quote:
                    print(f"[WARN] Failed to fetch live daily candle from quote: {e_quote}")

            if all_candles:
                # Upstox returns list of lists: [timestamp, open, high, low, close, volume, oi]
                df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                # keep='last' preserves today's live quote over historical duplicates
                df = df.drop_duplicates(subset=['timestamp'], keep='last').sort_values('timestamp')
                
                if is_60m:
                    df = df.set_index('timestamp')
                    df = df.resample('1h', origin='start_day').agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum',
                        'oi': 'last'
                    }).dropna().reset_index()
                    
                return df
            raise Exception("Historical data missing in response")
        except Exception as e:
            if not fallback:
                print(f"[WARN] Upstox Historical failed for {ticker}: {e} (yfinance fallback disabled).")
                return pd.DataFrame()
            print(f"[WARN] Upstox Historical failed for {ticker}: {e}. Falling back to yfinance.")
            # Map intervals for yfinance
            interval_map = {
                'day': '1d',
                '1d': '1d',
                '60minute': '1h',
                '30minute': '30m',
                '15minute': '15m',
                '5minute': '5m',
                '1minute': '1m',
                '1m': '1m'
            }
            yf_interval = interval_map.get(interval, interval)
            return yf.download(ticker, period=f"{days}d", interval=yf_interval, progress=False, auto_adjust=True)

    def get_user_margins(self):
        """Fetch available margin from Sandbox."""
        try:
            api_instance = upstox_client.UserApi(self.order_api_client)
            return api_instance.get_user_fund_margin_v3()
        except ApiException as e:
            return {"error": f"Margin Fetch Error: {e}"}

    def get_positions(self):
        """Fetch current open positions in the Sandbox."""
        try:
            api_instance = upstox_client.PortfolioApi(self.order_api_client)
            return api_instance.get_positions('2.0')
        except ApiException as e:
            return {"error": f"Positions Fetch Error: {e}"}

    def get_order_book(self):
        """Fetch today's order book from the Sandbox."""
        try:
            api_instance = upstox_client.OrderApi(self.order_api_client)
            return api_instance.get_order_book('2.0')
        except ApiException as e:
            return {"error": f"Order Book Fetch Error: {e}"}

    def place_order(self, ticker, side, quantity, price, stop_loss=None):
        """Places a limit order in the sandbox environment."""
        try:
            api_instance = upstox_client.OrderApi(self.order_api_client)
            tag = "VANGUARD_AUDIT"
            if stop_loss:
                tag = f"VANGUARD_SL_{stop_loss:.2f}"

            instrument_token = self.get_instrument_key(ticker)

            order_data = {
                "quantity": quantity,
                "product": "I", # Intraday
                "validity": "DAY",
                "price": price,
                "tag": tag,
                "instrument_token": instrument_token,
                "order_type": "LIMIT",
                "transaction_type": "BUY" if side == 'LONG' else "SELL",
                "disclosed_quantity": 0,
                "trigger_price": 0.0,
                "is_amo": False
            }
            # Note: The actual method name might vary slightly by SDK version, 
            # but this is the standard structure for V3
            return api_instance.place_order(order_data, '2.0')
        except ApiException as e:
            return {"error": f"Order placement failed: {e}"}

if __name__ == "__main__":
    broker = UpstoxSandboxBroker()
    if not broker.sandbox_token:
        print("[ERROR] UPSTOX_SANDBOX_ACCESS_TOKEN not found in .env")
    else:
        print("Testing Upstox Sandbox Connection (Fetching Order Book)...")
        orders = broker.test_connection()
        print(f"Order Book Response: {orders}")

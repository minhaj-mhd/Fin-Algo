import os
import sys
import time
import json
import threading
import traceback
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import xgboost as xgb

from scripts.vanguard import config
from scripts.vanguard.model_inference import ModelManager
from scripts.vanguard.signal_generation import SignalGenerator
from scripts.vanguard.trade_state import TradeStateManager
from scripts.vanguard.risk_manager import RiskManager
from scripts.vanguard.broker_adapter import BrokerAdapter
from scripts.vanguard.ai_veto import AIVetoManager
from scripts.vanguard.persistence import log_trade, save_latest_scores

from scripts.database_manager import init_db, get_trades_by_status
from scripts.tv_ta import get_tv_sentiment
from scripts.strategy_filters import StrategyFilters
from scripts.terminal_utils import log
from scripts.tickers import TICKERS

class VanguardOrchestrator:
    def __init__(self, model_path=None, scaler_path=None, meta_path=None):
        log("\n" + "=" * 60)
        log("VANGUARD SYSTEM - FULLY ORCHESTRATED V2.3 MIGRATED")
        log("=" * 60)

        # 1. Initialize Components
        self.strategy_filters = StrategyFilters()
        self.broker = BrokerAdapter()
        
        self.model_manager = ModelManager()
        self.model_manager.load_active_models(model_path, scaler_path, meta_path)
        self.model_manager.load_daily_gatekeepers()
        self.model_manager.load_multi_tf_models()

        self.signal_generator = SignalGenerator(self.strategy_filters)
        self.risk_manager = RiskManager()
        self.ai_veto_manager = AIVetoManager(self.risk_manager.min_conviction)

        # 2. Internal state variables
        self.active_shadow_trades = []
        self.sentiment_cache = {}
        self.atr_cache = {}
        self.lock = threading.Lock()
        
        self.latest_full_scores = None
        self.recently_closed = {}
        self.recent_vetoes = {}
        self._conviction_flip_checked = {}

        self._veto_stats_date = datetime.now().date()
        self.veto_stats = {
            "s1_vetoes":  0,
            "s2_vetoes":  0,
            "s1_passes":  0,
            "s2_passes":  0,
            "s1_tickers": [],
            "s2_tickers": [],
        }

        self.long_eligible_tickers = set(TICKERS)
        self.short_eligible_tickers = set(TICKERS)

        self.current_date = datetime.now().date()

        # 3. Setup WebSocket
        self._ws_manager = None
        self._start_websocket()

        # 4. Resume states
        init_db()
        self.ticker_metadata = {}
        self.fetch_static_metadata(TICKERS)
        self.load_open_trades()
        self.update_daily_macro_filters()

        # 5. Start Background Threads
        threading.Thread(target=self.shadow_tracker_loop, daemon=True).start()

    def _start_websocket(self):
        if not config.WEBSOCKET_ENABLED:
            log("[WS] WebSocket disabled by config.")
            return

        try:
            from scripts.upstox_websocket import UpstoxWebSocketManager

            analytics_token = os.getenv("UPSTOX_ANALYTICS_ACCESS_TOKEN")
            if not analytics_token:
                log("[WS] UPSTOX_ANALYTICS_ACCESS_TOKEN not set — WebSocket disabled.")
                return

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
                mode="ltpc",
                max_retries=10,
            )
            self._ws_manager.start()
            self.broker.attach_websocket(self._ws_manager)
            log(f"[WS] WebSocket started — {len(instrument_keys)} instruments subscribed.")
        except Exception as e:
            log(f"[WS] WebSocket init failed ({e}) — continuing with REST-only data.")
            self._ws_manager = None

    def fetch_static_metadata(self, tickers):
        print(f"[INIT] Fetching Market Metadata for {len(tickers)} symbols...")
        try:
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
                            "market_cap": 0,
                        }
                except Exception:
                    continue
            log(f"[OK] Market Metadata: {len(self.ticker_metadata)} symbols processed.")
        except Exception as e:
            log(f"[WARN] Static Metadata Error: {e}")

    def load_open_trades(self):
        try:
            open_trades = get_trades_by_status(["OPEN", "PENDING_ENTRY"], 100)
            if not open_trades:
                print("[INIT] No active trades (OPEN or PENDING_ENTRY) found in DB to resume.")
                self.risk_manager.update_upstox_stats(self.active_shadow_trades)
                return

            seen = {}
            for trade in sorted(open_trades, key=lambda t: t["timestamp"]):
                key = (trade["ticker"], trade["side"])
                seen[key] = trade

            latest_ids = {t["trade_id"] for t in seen.values()}
            for trade in open_trades:
                if trade["trade_id"] not in latest_ids:
                    trade["status"] = "CLOSED"
                    trade["final_profit_pct"] = 0.0
                    trade["comment"] = "Duplicate – closed on restart"
                    log_trade(trade)

            recovered_margin = 0.0
            now = datetime.now()
            grace_extension = timedelta(minutes=30)
            with self.lock:
                for trade in seen.values():
                    t = dict(trade)
                    try:
                        if t["status"] == "PENDING_ENTRY":
                            pending_since = datetime.fromisoformat(t.get("pending_since") or t["timestamp"])
                            minute = pending_since.minute
                            next_15 = ((minute // 15) + 1) * 15
                            next_boundary = pending_since.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=next_15)
                            if now > next_boundary + timedelta(minutes=5):
                                print(f"[RESTART] PENDING_ENTRY for {t['ticker']} has expired. Cancelling.")
                                t["status"] = "CANCELLED"
                                t["comment"] = "Cancelled on restart - confirmation window expired."
                                log_trade(t)
                                continue
                        else:
                            exit_dt = datetime.fromisoformat(t["exit_time"])
                            if exit_dt < now:
                                new_exit = now + grace_extension
                                t["exit_time"] = new_exit.isoformat()
                                log(f"[RESTART] {t['ticker']} exit_time extended to {new_exit.strftime('%H:%M')} (engine was offline)")
                    except Exception:
                        pass

                    self.active_shadow_trades.append(t)

                    m = t.get("margin_used")
                    if not m or m == 0:
                        qty = t.get("quantity", 1)
                        price = t.get("entry_price", 0)
                        m = (qty * price) / config.MARGIN_MULTIPLIER
                    recovered_margin += m

            self.risk_manager.used_margin = recovered_margin
            log(f"[INIT] Resumed {len(seen)} open trade(s) from DB. Used Margin restored: Rs{self.risk_manager.used_margin:.2f}")
        except Exception as e:
            log(f"[WARN] Could not load open trades from DB: {e}")
        finally:
            self.risk_manager.update_upstox_stats(self.active_shadow_trades)

    def update_daily_macro_filters(self):
        log("\n" + "=" * 60)
        log("RUNNING DAILY MACRO TREND SCAN (GATEKEEPER SELECTION)")
        log("=" * 60)
        try:
            os.makedirs("data", exist_ok=True)
            with open("data/daily_gatekeepers.json", "w") as f:
                json.dump({
                    "status": "RUNNING DAILY MACRO TREND SCAN (GATEKEEPER SELECTION)",
                    "timestamp": datetime.now().isoformat(),
                    "long_eligible": [],
                    "short_eligible": [],
                    "long_eligible_count": 0,
                    "short_eligible_count": 0
                }, f)
        except Exception as e:
            log(f"[WARN] Failed to write running status to daily_gatekeepers: {e}")

        try:
            tickers = list(TICKERS)
            log(f"[DAILY-SCAN] Fetching 1y daily bars for {len(tickers)} symbols via yfinance...")
            
            df_batch = yf.download(
                tickers, period="1y", interval="1d",
                progress=False, auto_adjust=True, timeout=45
            )
            
            if df_batch.empty:
                raise ValueError("Downloaded empty batch from yfinance.")
                
            from scripts.feature_utils import compute_features_daily_xgb
            
            all_ticker_dfs = {}
            for ticker in tickers:
                try:
                    if len(tickers) > 1:
                        if ticker in df_batch.columns.get_level_values(1):
                            ticker_df = df_batch.xs(ticker, level=1, axis=1).copy()
                        else:
                            continue
                    else:
                        ticker_df = df_batch.copy()
                        
                    ticker_df = ticker_df.dropna(subset=['Close'])
                    if len(ticker_df) < 50:
                        continue
                        
                    ticker_df = ticker_df.rename(columns={
                        'Open': 'Open', 'High': 'High', 'Low': 'Low',
                        'Close': 'Close', 'Volume': 'Volume'
                    })
                    ticker_df['Ticker'] = ticker
                    
                    df_feat = compute_features_daily_xgb(ticker_df)
                    
                    high_52w = df_feat['High'].rolling(250, min_periods=50).max()
                    low_52w  = df_feat['Low'].rolling(250, min_periods=50).min()
                    df_feat['Dist_52W_High'] = (df_feat['Close'] - high_52w) / (high_52w + 1e-8)
                    df_feat['Dist_52W_Low']  = (df_feat['Close'] - low_52w)  / (low_52w  + 1e-8)
                    
                    df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
                    all_ticker_dfs[ticker] = df_feat
                except Exception:
                    pass

            if not all_ticker_dfs:
                raise ValueError("Could not compute daily features for any ticker.")

            all_dates = []
            for t, df_t in all_ticker_dfs.items():
                all_dates.extend(df_t.index.tolist())
            unique_dates = sorted(list(set(all_dates)))
            
            recent_dates = unique_dates[-15:]
            log(f"[DAILY-SCAN] Z-scoring over last {len(recent_dates)} dates: {recent_dates[0].strftime('%Y-%m-%d')} to {recent_dates[-1].strftime('%Y-%m-%d')}")
            
            aligned_records = []
            for ticker, df_t in all_ticker_dfs.items():
                for dt in recent_dates:
                    if dt in df_t.index:
                        row = df_t.loc[dt].copy()
                        row['DateTime'] = dt
                        row['Ticker'] = ticker
                        aligned_records.append(row)
                        
            df_aligned = pd.DataFrame(aligned_records)
            df_aligned['Query_ID'] = df_aligned.groupby(df_aligned['DateTime'].dt.date).ngroup()
            
            df_aligned['Market_Mean_Return']     = df_aligned.groupby('Query_ID')['Return'].transform('mean')
            df_aligned['Relative_Return']        = df_aligned['Return'] - df_aligned['Market_Mean_Return']
            df_aligned['Market_Mean_Volatility'] = df_aligned.groupby('Query_ID')['HL_Range'].transform('mean')
            df_aligned['Relative_Volatility']    = df_aligned['HL_Range'] / (df_aligned['Market_Mean_Volatility'] + 1e-8)
            
            exclude_cols = {
                'DateTime', 'Query_ID', 'Ticker', 'Next_Day_Return',
                'Open', 'High', 'Low', 'Close', 'Volume',
                'Market_Mean_Return', 'Relative_Return',
                'Market_Mean_Volatility', 'Relative_Volatility',
                'Hour', 'DayOfWeek', 'Is_Open_Hour', 'Is_Close_Hour', 'Time_To_Close'
            }
            feature_cols = [c for c in self.model_manager.daily_feature_cols if c not in exclude_cols]
            
            for col in feature_cols:
                if col in df_aligned.columns:
                    grp_mean = df_aligned.groupby('Query_ID')[col].transform('mean')
                    grp_std  = df_aligned.groupby('Query_ID')[col].transform('std')
                    df_aligned[col] = (df_aligned[col] - grp_mean) / (grp_std + 1e-8)
                    
            df_aligned[self.model_manager.daily_feature_cols] = df_aligned[self.model_manager.daily_feature_cols].fillna(0)
            df_aligned = df_aligned.sort_values(['Ticker', 'DateTime']).reset_index(drop=True)
            
            xgb_long_scores = {}
            xgb_short_scores = {}
            
            for ticker, g in df_aligned.groupby('Ticker'):
                if len(g) < 10:
                    continue
                    
                latest_row = g.iloc[-1]
                X_xgb = latest_row[self.model_manager.daily_feature_cols].values.astype(np.float32).reshape(1, -1)
                
                dmatrix_te = xgb.DMatrix(X_xgb)
                xgb_l = float(self.model_manager.daily_xgb_long.predict(dmatrix_te)[0])
                xgb_s = float(self.model_manager.daily_xgb_short.predict(dmatrix_te)[0])
                xgb_long_scores[ticker] = xgb_l
                xgb_short_scores[ticker] = xgb_s
                
            tickers_evaluated = list(xgb_long_scores.keys())
            if not tickers_evaluated:
                raise ValueError("No tickers were successfully evaluated.")
                
            df_ranks = pd.DataFrame({
                'Ticker': tickers_evaluated,
                'xgb_l': [xgb_long_scores[t] for t in tickers_evaluated],
                'xgb_s': [xgb_short_scores[t] for t in tickers_evaluated]
            })
            
            df_ranks['xgb_l_rank'] = df_ranks['xgb_l'].rank(ascending=False, method='min')
            df_ranks['xgb_s_rank'] = df_ranks['xgb_s'].rank(ascending=False, method='min')
            
            k_eligible = max(2, int(len(df_ranks) * 0.40))
            
            long_eligible = df_ranks.sort_values('xgb_l_rank').head(k_eligible)['Ticker'].tolist()
            short_eligible = df_ranks.sort_values('xgb_s_rank').head(k_eligible)['Ticker'].tolist()
            
            with self.lock:
                self.long_eligible_tickers = set(long_eligible)
                self.short_eligible_tickers = set(short_eligible)
                
            gatekeeper_data = {
                "status": "COMPLETED",
                "timestamp": datetime.now().isoformat(),
                "long_eligible": [t.replace('.NS', '') for t in long_eligible],
                "short_eligible": [t.replace('.NS', '') for t in short_eligible],
                "long_eligible_count": len(long_eligible),
                "short_eligible_count": len(short_eligible)
            }
            os.makedirs("data", exist_ok=True)
            with open("data/daily_gatekeepers.json", "w") as f:
                json.dump(gatekeeper_data, f)
                
            log(f"[DAILY-SCAN] Macro Trend Gatekeepers updated (XGBoost Pure):")
            log(f"  • LONG Eligible (Top40%): {len(self.long_eligible_tickers)} tickers")
            log(f"  • SHORT Eligible (Top40%): {len(self.short_eligible_tickers)} tickers")
            log("=" * 60)
        except Exception as scan_err:
            log(f"[ERROR] Daily Macro Scan failed: {scan_err}")
            try:
                with open("data/daily_gatekeepers.json", "w") as f:
                    json.dump({
                        "status": f"FAILED: {scan_err}",
                        "timestamp": datetime.now().isoformat(),
                        "long_eligible": [],
                        "short_eligible": [],
                        "long_eligible_count": 0,
                        "short_eligible_count": 0
                    }, f)
            except Exception:
                pass
            with self.lock:
                self.long_eligible_tickers = set(TICKERS)
                self.short_eligible_tickers = set(TICKERS)

    def get_last_completed_15min_candle(self, ticker):
        if self._ws_manager is not None:
            try:
                instrument_key = self.broker.get_instrument_key(ticker)
                df_15m = self._ws_manager.cache.get_candles(instrument_key, "15minute", count=5)
                if df_15m is not None and not df_15m.empty:
                    now = pd.Timestamp.now()
                    ts_series = pd.to_datetime(df_15m['timestamp'])
                    if ts_series.dt.tz is not None:
                        ts_series = ts_series.dt.tz_localize(None)
                    completed = df_15m[ts_series + pd.Timedelta(minutes=15) <= now]
                    if not completed.empty:
                        last_row = completed.iloc[-1]
                        return {
                            'timestamp': pd.Timestamp(last_row['timestamp']).to_pydatetime(),
                            'open':  float(last_row['open']),
                            'high':  float(last_row['high']),
                            'low':   float(last_row['low']),
                            'close': float(last_row['close']),
                        }
            except Exception:
                pass

        df = self.broker.get_recent_candles(ticker, interval='1minute', count=120)
        if df is None or df.empty:
            return None

        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        resampled = df.resample('15min', origin='start_day').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
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
        ATR_SL_DEFAULT = 0.50
        ATR_TP_DEFAULT = 1.00

        cached = self.atr_cache.get(ticker)
        if cached:
            c_sl, c_tp, c_ts = cached
            if (datetime.now() - c_ts).total_seconds() < 900:
                return c_sl, c_tp

        try:
            df = self.broker.get_recent_candles(ticker, interval='1minute', count=120)
            if df is None or df.empty:
                return ATR_SL_DEFAULT, ATR_TP_DEFAULT

            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

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
            atr_series = tr.ewm(span=14, adjust=False).mean()
            atr = atr_series.iloc[-1]

            cur_price = float(close.iloc[-1])
            atr_pct = (atr / cur_price) * 100

            # For a 1-hour hold (4 x 15min bars), the expected price move is ~sqrt(4) * ATR = 2.0 * ATR.
            # Using 3.0 * ATR for TP is highly improbable to hit within 1 hour.
            # Adjusted to realistic levels: SL = 1.0x ATR, TP = 1.8x ATR
            sl_pct = max(0.25, min(1.20, atr_pct * 1.0))
            tp_pct = max(0.50, min(2.00, atr_pct * 1.8))

            self.atr_cache[ticker] = (sl_pct, tp_pct, datetime.now())
            return sl_pct, tp_pct
        except Exception:
            return ATR_SL_DEFAULT, ATR_TP_DEFAULT

    def calculate_conviction_scores(self, tickers):
        all_latest_data = []
        valid_tickers = []

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Pulse Scan initiated ({len(tickers)} symbols)...")

        nifty_features = {
            'Nifty_1H_Return': 0.0, 'Nifty_3H_Return': 0.0, 'Nifty_5H_Return': 0.0,
            'Nifty_RSI': 50.0, 'Nifty_HL_Range': 0.0, 'Nifty_20H_Std': 0.0
        }
        try:
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

        daily_data = pd.DataFrame()
        try:
            daily_data = yf.download(
                tickers, period="60d", interval="1d",
                progress=False, auto_adjust=True, timeout=30
            )
        except Exception as e:
            print(f"[WARN] Batch daily fetch failed: {e}")

        all_dfs = {}
        def _fetch_hist_data(ticker):
            try:
                from scripts.upstox_broker import UpstoxSandboxBroker
                temp_broker = UpstoxSandboxBroker()
                hist_df = temp_broker.get_historical_data(ticker, interval="60minute", days=60, fallback=False)
                if hist_df is not None and not hist_df.empty:
                    hist_df = hist_df.rename(columns={
                        "open": "Open", "high": "High", "low": "Low", 
                        "close": "Close", "volume": "Volume"
                    })
                    if "timestamp" in hist_df.columns:
                        hist_df.set_index("timestamp", inplace=True)
                    return ticker, hist_df
            except Exception:
                pass
            return ticker, None

        import concurrent.futures
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
                future_to_ticker = {executor.submit(_fetch_hist_data, ticker): ticker for ticker in tickers}
                for future in concurrent.futures.as_completed(future_to_ticker):
                    ticker, df = future.result()
                    if df is not None:
                        all_dfs[ticker] = df
        except Exception as e:
            print(f"[WARN] Upstox Scan Loop Error: {e}")

        missing_tickers = [t for t in tickers if t not in all_dfs]
        if missing_tickers:
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
                        except Exception:
                            pass
            except Exception as e_batch:
                print(f"[WARN] Batch yfinance download failed: {e_batch}")

        from scripts.feature_utils import compute_features, RSI

        def _compute_features_for_ticker(ticker, df_input, daily_data_input, is_legacy_flag):
            try:
                df_curr = df_input.ffill().dropna()
                if len(df_curr) < 25:
                    return None

                df_curr = compute_features(df_curr, legacy=is_legacy_flag)

                df_curr['Return_6H'] = df_curr['Close'].pct_change(6).fillna(0)
                df_curr['Return_1D'] = df_curr['Close'].pct_change(7).fillna(0)
                price_dir = np.sign(df_curr['Return'])
                vol_change = df_curr['Volume'].pct_change()
                df_curr['VP_Divergence'] = (price_dir * vol_change).fillna(0)
                green_bar = (df_curr['Close'] > df_curr['Open']).astype(float)
                high_vol = (df_curr['Volume'] > df_curr['Volume'].rolling(20).mean()).astype(float)
                df_curr['Accumulation_5'] = (green_bar * high_vol).rolling(5).sum().fillna(0)
                df_curr['Bar_Position'] = ((df_curr['Close'] - df_curr['Low']) / (df_curr['High'] - df_curr['Low'] + 1e-8)).fillna(0)
                df_curr['Green_Bar_Ratio_5'] = green_bar.rolling(5).mean().fillna(0)
                
                df_curr['RSI_14_Raw'] = df_curr['RSI_14'].copy()
                df_curr['Stoch_K_Raw'] = df_curr['Stoch_K'].copy()
                df_curr['PercentB_Raw'] = df_curr['PercentB'].copy()

                daily_rsi_lag = 50.0
                daily_sma20_lag = None
                daily_atr_lag = 0.0
                daily_close_lag5 = None
                
                if not daily_data_input.empty:
                    try:
                        if isinstance(daily_data_input.columns, pd.MultiIndex):
                            if ticker in daily_data_input.columns.get_level_values(1):
                                df_daily = daily_data_input.xs(ticker, axis=1, level=1).ffill().dropna()
                            else:
                                df_daily = pd.DataFrame()
                        else:
                            df_daily = daily_data_input.ffill().dropna()
                            
                        if len(df_daily) >= 20:
                            high = df_daily['High']
                            low = df_daily['Low']
                            close = df_daily['Close']
                            prev_close = close.shift(1)
                            tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
                            atr_series = tr.ewm(span=14, adjust=False).mean()

                            latest_daily = df_daily.iloc[-2] if len(df_daily) > 1 else df_daily.iloc[-1]
                            daily_rsi_lag = RSI(df_daily['Close'], 14).iloc[-2] if len(df_daily) > 1 else 50.0
                            daily_sma20_lag = df_daily['Close'].rolling(20).mean().iloc[-2] if len(df_daily) > 1 else None
                            daily_atr_lag = atr_series.iloc[-2] if len(atr_series) > 1 else 0.0
                            daily_close_lag5 = df_daily['Close'].iloc[-6] if len(df_daily) >= 6 else None
                    except Exception:
                        pass
                
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

                df_curr['Daily_RSI'] = daily_rsi_lag
                if daily_sma20_lag is not None:
                    df_curr['Daily_SMA20_Dist'] = (df_curr['Close'] / daily_sma20_lag) - 1
                else:
                    df_curr['Daily_SMA20_Dist'] = 0.0
                
                if daily_close_lag5 is not None:
                    df_curr['Daily_Trend'] = np.sign(df_curr['Close'] - daily_close_lag5)
                else:
                    df_curr['Daily_Trend'] = 0.0
                    
                df_curr['Daily_ATR_Pct'] = (daily_atr_lag / df_curr['Close']) if daily_atr_lag else 0.0

                latest = df_curr.iloc[-1].copy()
                latest["ticker"] = ticker
                return latest
            except Exception:
                return None

        is_legacy = not any(str(self.model_manager.active_model_name).startswith(p) for p in ["v6", "v7", "v8", "v9"])

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_ticker = {
                executor.submit(_compute_features_for_ticker, ticker, all_dfs[ticker], daily_data, is_legacy): ticker
                for ticker in tickers if ticker in all_dfs
            }
            for future in concurrent.futures.as_completed(future_to_ticker):
                res = future.result()
                if res is not None:
                    all_latest_data.append(res)
                    valid_tickers.append(res["ticker"])

        if not all_latest_data:
            print("[WARN] No valid ticker data processed.")
            return pd.DataFrame()

        scores_df = pd.DataFrame(all_latest_data)

        for key, val in nifty_features.items():
            scores_df[key] = val
        for key, val in vix_features.items():
            scores_df[key] = val

        scores_df['Stock_vs_Nifty'] = scores_df['Return'] - scores_df['Nifty_1H_Return']
        scores_df['Beta_Adjusted_Mom'] = (scores_df['Return'] / (scores_df['Nifty_1H_Return'].abs() + 1e-6)).clip(-10, 10)

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
        
        sector_perf = scores_df.groupby('Sector')['Return'].mean().reset_index()
        sector_perf['Sector_Rank'] = sector_perf['Return'].rank(ascending=False)
        scores_df = scores_df.merge(sector_perf[['Sector', 'Sector_Rank']], on='Sector', how='left')

        scores_df["Market_Mean_Return"] = scores_df["Return"].mean()
        scores_df["Relative_Return"] = scores_df["Return"] - scores_df["Market_Mean_Return"]
        scores_df["Market_Mean_Volatility"] = scores_df["HL_Range"].mean()
        scores_df["Relative_Volatility"] = scores_df["HL_Range"] / (scores_df["Market_Mean_Volatility"] + 1e-8)

        raw_display = scores_df[["ticker", "Dollar_Volume", "RVOL", "Dist_52W_High", "Return"]].copy()
        raw_display["High_52W_Actual"] = raw_display["ticker"].map(
            lambda t: (
                (scores_df.loc[scores_df["ticker"] == t, "Close"].values[0] - self.ticker_metadata.get(t, {}).get("high_52w", 0))
                / (self.ticker_metadata.get(t, {}).get("high_52w", 1e-8))
                if self.ticker_metadata.get(t, {}).get("high_52w", 0) > 0 else 0
            )
        )

        scores_df = self.model_manager.score_universe(scores_df, self.ticker_metadata)
        if scores_df.empty:
            return pd.DataFrame()

        raw_display_aligned = raw_display.reset_index(drop=True)
        scores_df = scores_df.assign(
            dv_raw=raw_display_aligned["Dollar_Volume"],
            rvol_raw=raw_display_aligned["RVOL"],
            dist_52h_model=raw_display_aligned["Dist_52W_High"],
            dist_52h_actual=raw_display_aligned["High_52W_Actual"],
            Return_Raw=raw_display_aligned["Return"]
        )

        save_latest_scores(scores_df, self.long_eligible_tickers, self.short_eligible_tickers)

        if len(tickers) > 1:
            with self.lock:
                self.latest_full_scores = scores_df.copy()

        return scores_df

    def start_shadow_trade(self, ticker, conviction, entry_price, side, reason, one_hour_prob,
                           nlp_sentiment=None, tv_sentiment=None,
                           long_score=None, short_score=None, strategy_id=None,
                           score_15m=None, score_30m=None, score_1d=None, is_ensemble=False):
        now = datetime.now()
        trade = {
            "trade_id":            f"TRADE-{ticker}-{side}-{now.strftime('%y%m%d%H%M%S')}",
            "ticker":              ticker,
            "side":                side,
            "quantity":            0,
            "entry_price":         entry_price,
            "exit_price":          entry_price,
            "stop_loss_pct":       0.50,
            "take_profit_pct":     1.00,
            "peak_profit_pct":     0.0,
            "peak_price":          entry_price,
            "timestamp":           now.isoformat(),
            "exit_time":           (now + timedelta(hours=1)).isoformat(),
            "status":              "PENDING_ENTRY",
            "comment":             f"PENDING-CANDLE-CONFIRMATION | {reason}",
            "margin_used":         0.0,
            "buy_brokerage":       config.BROKERAGE_PER_ORDER,
            "final_profit_pct":    0.0,
            "pending_since":       now.isoformat(),
            
            "strategy_id":         strategy_id,
            "one_hour_prob":       one_hour_prob,
            "tech_score":          float(conviction) if conviction is not None else None,
            "nlp_sentiment":       float(nlp_sentiment) if nlp_sentiment is not None else None,
            "tv_sentiment":        str(tv_sentiment) if tv_sentiment is not None else None,
            "long_score":          float(long_score) if long_score is not None else None,
            "short_score":         float(short_score) if short_score is not None else None,
            "score_15m":           float(score_15m) if score_15m is not None else None,
            "score_30m":           float(score_30m) if score_30m is not None else None,
            "score_1d":            float(score_1d) if score_1d is not None else None,
            "is_ensemble":         is_ensemble,
            "net_pnl_amt":         0.0,
        }

        with self.lock:
            self.active_shadow_trades.append(trade)
            self.risk_manager.realized_charges += config.BROKERAGE_PER_ORDER

        self.risk_manager.update_upstox_stats(self.active_shadow_trades)
        log_trade(trade)

    def start_vetoed_tracking(self, ticker, conviction, entry_price, side, reason, one_hour_prob,
                              nlp_sentiment=None, tv_sentiment=None,
                              long_score=None, short_score=None, strategy_id=None,
                              score_15m=None, score_30m=None, score_1d=None, is_ensemble=False):
        now = datetime.now()
        trade = {
            "trade_id":            f"VETO-{ticker}-{side}-{now.strftime('%y%m%d%H%M%S')}",
            "ticker":              ticker,
            "side":                side,
            "quantity":            0,
            "entry_price":         entry_price,
            "exit_price":          entry_price,
            "stop_loss_pct":       0.0,
            "take_profit_pct":     0.0,
            "peak_profit_pct":     0.0,
            "peak_price":          entry_price,
            "timestamp":           now.isoformat(),
            "exit_time":           (now + timedelta(hours=1)).isoformat(),
            "status":              "VETOED",
            "comment":             reason,
            "margin_used":         0.0,
            "buy_brokerage":       0.0,
            "final_profit_pct":    0.0,
            
            "strategy_id":         strategy_id,
            "one_hour_prob":       one_hour_prob,
            "tech_score":          float(conviction) if conviction is not None else None,
            "nlp_sentiment":       float(nlp_sentiment) if nlp_sentiment is not None else None,
            "tv_sentiment":        str(tv_sentiment) if tv_sentiment is not None else None,
            "long_score":          float(long_score) if long_score is not None else None,
            "short_score":         float(short_score) if short_score is not None else None,
            "score_15m":           float(score_15m) if score_15m is not None else None,
            "score_30m":           float(score_30m) if score_30m is not None else None,
            "score_1d":            float(score_1d) if score_1d is not None else None,
            "is_ensemble":         is_ensemble,
            "net_pnl_amt":         0.0,
        }

        self._mark_recently_vetoed(ticker)
        with self.lock:
            self.active_shadow_trades.append(trade)
        log_trade(trade)

    def _mark_recently_closed(self, ticker):
        self.recently_closed[ticker] = datetime.now()

    def _mark_recently_vetoed(self, ticker):
        self.recent_vetoes[ticker] = datetime.now()

    def _is_in_cooldown(self, ticker, minutes=30):
        closed_at = self.recently_closed.get(ticker)
        if closed_at is None:
            return False
        return (datetime.now() - closed_at).total_seconds() < minutes * 60

    def _is_veto_cooldown(self, ticker, minutes=30):
        vetoed_at = self.recent_vetoes.get(ticker)
        if vetoed_at is not None:
            if (datetime.now() - vetoed_at).total_seconds() < minutes * 60:
                return True
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

    def _get_current_conviction(self, ticker, side):
        with self.lock:
            df = self.latest_full_scores
        if df is None or df.empty:
            return 0.0, 0.0, False

        row = df[df["ticker"] == ticker]
        if row.empty:
            return 0.0, 0.0, False

        long_conv = float(row.iloc[0]["Long_Conviction"])
        short_conv = float(row.iloc[0]["Short_Conviction"])
        
        is_aligned = False
        if side == "LONG" and long_conv > short_conv:
            is_aligned = True
        elif side == "SHORT" and short_conv > long_conv:
            is_aligned = True

        return long_conv, short_conv, is_aligned

    def shadow_tracker_loop(self):
        while True:
            try:
                with self.lock:
                    current_trades = list(self.active_shadow_trades)

                if not current_trades:
                    time.sleep(5)
                    continue

                for trade in current_trades:
                    try:
                        price = self.broker.get_live_price(trade["ticker"])
                        if price is None:
                            continue

                        now = datetime.now()
                        pnl = ((price - trade["entry_price"]) / trade["entry_price"] * 100) if trade["side"] == "LONG" else ((trade["entry_price"] - price) / trade["entry_price"] * 100)

                        if trade["status"] == "PENDING_ENTRY":
                            # 1. Check Expiry
                            if TradeStateManager.check_pending_entry_expiry(trade, now):
                                print(f"[PENDING -> CANCELLED] {trade['ticker']} expired.")
                                trade["status"] = "CANCELLED"
                                trade["comment"] = "Cancelled - Confirmation window expired."
                                with self.lock:
                                    self.risk_manager.realized_charges -= config.BROKERAGE_PER_ORDER
                                    self.risk_manager.used_margin -= trade.get("margin_used", 0.0)
                                log_trade(trade)
                                self.risk_manager.update_upstox_stats(self.active_shadow_trades)
                                continue

                            # 2. Time Cutoff
                            if now.strftime("%H:%M") >= "15:00":
                                print(f"[PENDING -> CANCELLED] {trade['ticker']} confirmation aborted (Time cutoff: after 3:00 PM).")
                                trade["status"] = "CANCELLED"
                                trade["comment"] = "Cancelled - Aborted after 3:00 PM time cutoff."
                                with self.lock:
                                    self.risk_manager.realized_charges -= config.BROKERAGE_PER_ORDER
                                    self.risk_manager.used_margin -= trade.get("margin_used", 0.0)
                                log_trade(trade)
                                self.risk_manager.update_upstox_stats(self.active_shadow_trades)
                                continue

                            # 3. Candle Confirmation check
                            candle = self.get_last_completed_15min_candle(trade["ticker"])
                            raw_since = trade.get("pending_since") or trade["timestamp"]
                            pending_since = datetime.fromisoformat(raw_since)
                            
                            if candle is not None:
                                candle_close_time = candle["timestamp"] + timedelta(minutes=15)
                                if candle_close_time > pending_since:
                                    is_confirmed = TradeStateManager.check_candle_confirmation(trade, candle)
                                    if is_confirmed:
                                        # Strict conviction gate
                                        long_conv, short_conv, aligned = self._get_current_conviction(trade["ticker"], trade["side"])
                                        conv_score = long_conv if trade["side"] == "LONG" else short_conv
                                        orig_score = trade.get("tech_score", 0.0)
                                        
                                        is_conv_ok, reason_low = TradeStateManager.check_conviction_gate(conv_score, orig_score, self.risk_manager.min_conviction)
                                        if not aligned or not is_conv_ok:
                                            block_reason = "XGBoost conviction flipped" if not aligned else f"conviction too low at entry ({reason_low})"
                                            print(f"[PENDING -> CANCELLED] {trade['ticker']} confirmation blocked. {block_reason}.")
                                            trade["status"] = "CANCELLED"
                                            trade["comment"] = f"Cancelled - {block_reason}."
                                            with self.lock:
                                                self.risk_manager.realized_charges -= config.BROKERAGE_PER_ORDER
                                                self.risk_manager.used_margin -= trade.get("margin_used", 0.0)
                                            log_trade(trade)
                                            self.risk_manager.update_upstox_stats(self.active_shadow_trades)
                                            continue

                                        # Execute entry order
                                        entry_price = price
                                        stop_loss_pct, take_profit_pct = self.compute_15min_atr(trade["ticker"])
                                        qty = self.risk_manager.calculate_trade_quantity(entry_price, stop_loss_pct)
                                        
                                        sl_mult = 1 - (stop_loss_pct / 100) if trade["side"] == "LONG" else 1 + (stop_loss_pct / 100)
                                        sl_price = round(entry_price * sl_mult, 2)
                                        
                                        order_res = self.broker.place_order(trade["ticker"], trade["side"], quantity=qty, price=entry_price, stop_loss=sl_price)
                                        order_id = order_res.get("order_id", "SANDBOX-SUCCESS")

                                        print(f"[PENDING -> OPEN] Confirmed {trade['ticker']} at {entry_price} (Order: {order_id})")
                                        trade["status"] = "OPEN"
                                        trade["stop_loss_pct"] = stop_loss_pct
                                        trade["take_profit_pct"] = take_profit_pct
                                        trade["entry_price"] = entry_price
                                        trade["peak_price"] = entry_price
                                        trade["quantity"] = qty
                                        trade["margin_used"] = (qty * entry_price) / config.MARGIN_MULTIPLIER
                                        trade["timestamp"] = now.isoformat()
                                        trade["exit_time"] = (now + timedelta(hours=1)).isoformat()
                                        trade["comment"] = trade.get("comment", "").replace("PENDING-CANDLE-CONFIRMATION", order_id)
                                        
                                        log_trade(trade)
                                        self.risk_manager.update_upstox_stats(self.active_shadow_trades)

                            # Timeout check
                            if now - pending_since > timedelta(minutes=45) and trade["status"] == "PENDING_ENTRY":
                                print(f"[PENDING -> CANCELLED] {trade['ticker']} confirmation timed out.")
                                trade["status"] = "CANCELLED"
                                trade["comment"] = "Cancelled - 15-min candle confirmation timed out."
                                with self.lock:
                                    self.risk_manager.realized_charges -= config.BROKERAGE_PER_ORDER
                                    self.risk_manager.used_margin -= trade.get("margin_used", 0.0)
                                log_trade(trade)
                                self.risk_manager.update_upstox_stats(self.active_shadow_trades)

                            if trade["status"] == "PENDING_ENTRY":
                                trade["exit_price"] = price
                                trade["final_profit_pct"] = round(pnl, 4)
                                if trade["side"] == "LONG":
                                    trade["peak_price"] = max(trade.get("peak_price", 0.0), price)
                                else:
                                    trade["peak_price"] = min(trade.get("peak_price", 99999999.0), price)
                                trade["peak_profit_pct"] = max(trade.get("peak_profit_pct", 0.0), pnl)
                                log_trade(trade)

                            continue

                        # Peak Profit tracking
                        if trade["side"] == "LONG":
                            trade["peak_price"] = max(trade.get("peak_price", 0.0), price)
                        else:
                            trade["peak_price"] = min(trade.get("peak_price", 99999999.0), price)
                        trade["peak_profit_pct"] = max(trade["peak_profit_pct"], pnl)

                        # Vetoed trade check EOD or 1H expiry
                        if trade["status"] == "VETOED":
                            trade["exit_price"] = price
                            trade["final_profit_pct"] = round(pnl, 4)
                            veto_expiry = datetime.fromisoformat(trade["exit_time"])
                            if now >= veto_expiry or now.strftime("%H:%M") >= "15:15":
                                trade["status"] = "VETOED_EXPIRED"
                                log_trade(trade)
                                print(f"[VETOED_EXPIRED] {trade['ticker']} {trade['side']} | 1h Close: Rs{price:.2f} | P&L if taken: {pnl:.2f}%")
                            else:
                                log_trade(trade)
                            continue

                        # ── 15-MIN CONVICTION FLIP CHECK ───────────────────────────
                        is_conviction_flip = False
                        trade_id = trade.get("trade_id", trade.get("ticker"))
                        last_flip_check = self._conviction_flip_checked.get(trade_id)
                        
                        if (last_flip_check is None or (now - last_flip_check).total_seconds() >= 900) and trade["status"] == "OPEN":
                            self._conviction_flip_checked[trade_id] = now
                            long_conv, short_conv, aligned = self._get_current_conviction(trade["ticker"], trade["side"])
                            conv_score = long_conv if trade["side"] == "LONG" else short_conv

                            if not aligned:
                                is_conviction_flip = True
                                flip_note = f" | Conviction Flip @ 15-min check (Long={long_conv:.3f} Short={short_conv:.3f})"
                                print(f"[CONVICTION-FLIP] {trade['ticker']} {trade['side']} — XGBoost flipped. Closing.")
                            else:
                                print(f"[CONVICTION-OK] {trade['ticker']} {trade['side']} — XGBoost still aligned.")

                        # --- Time Expiry Extension Check ---
                        raw_time_expiry = now >= datetime.fromisoformat(trade["exit_time"])
                        if (
                            raw_time_expiry
                            and not trade.get("extension_pending", False)
                            and pnl < 0
                            and trade.get("extension_count", 0) < 2
                            and now.strftime("%H:%M") < "15:15"
                            and trade["status"] == "OPEN"
                            and not is_conviction_flip
                        ):
                            ext_count = trade.get("extension_count", 0)
                            print(f"[EXPIRY-CHECK] {trade['ticker']} time expiry with loss ({pnl:.2f}%).")
                            long_conv, short_conv, aligned = self._get_current_conviction(trade["ticker"], trade["side"])
                            conv_score = long_conv if trade["side"] == "LONG" else short_conv
                            
                            if aligned and conv_score > self.risk_manager.min_conviction:
                                if self.ai_veto_manager.gemini_enabled:
                                    trade["extension_pending"] = True
                                    trade["extension_started"] = now.isoformat()
                                    
                                    def _gemini_check_extension(target_trade, t_price, t_pnl, current_ext):
                                        prompt = f"We are holding a {target_trade['side']} position in {target_trade['ticker']} with entry {target_trade['entry_price']:.2f}, current {t_price:.2f}, pnl {t_pnl:.2f}%. Check recent news and analyze if extending by 15 mins is likely to recover our loss. Answer with 'EXTEND' or 'CLOSE'."
                                        try:
                                            client = self.ai_veto_manager.clients[0]
                                            response = client.models.generate_content(
                                                model=self.ai_veto_manager.s1_model_tiers[0],
                                                contents=prompt,
                                                config=types.GenerateContentConfig(tools=[{"google_search": {}}])
                                            )
                                            ans = self.ai_veto_manager._extract_response_text(response).upper()
                                            
                                            with self.lock:
                                                if "EXTEND" in ans:
                                                    target_trade["extended_exit_time"] = (datetime.fromisoformat(target_trade["exit_time"]) + timedelta(minutes=15)).isoformat()
                                                    target_trade["exit_time"] = target_trade["extended_exit_time"]
                                                    target_trade["extension_count"] = current_ext + 1
                                                    target_trade["comment"] = target_trade.get("comment", "") + f" | Ext {current_ext + 1}"
                                                    log_trade(target_trade)
                                                    print(f"[EXTENSION] {target_trade['ticker']} extended 15 mins.")
                                                else:
                                                    print(f"[EXTENSION-REJECTED] suggested CLOSE.")
                                        except Exception as err:
                                            print(f"[EXTENSION-ERROR] Gemini failed: {err}")
                                        finally:
                                            target_trade["extension_pending"] = False
                                            
                                    threading.Thread(target=_gemini_check_extension, args=(trade, price, pnl, ext_count), daemon=True).start()
                                    continue

                        # Evaluated Exits (SL, BE, Trailing Stop, Time Expiry, EOD)
                        should_exit, exit_status, exit_note = TradeStateManager.evaluate_open_trade_exit(trade, price, pnl, now)

                        if should_exit or is_conviction_flip:
                            if is_conviction_flip:
                                trade["status"] = "CLOSED"
                                trade["comment"] = trade.get("comment", "") + flip_note
                            else:
                                trade["status"] = exit_status
                                trade["comment"] = trade.get("comment", "") + (exit_note or "")

                            trade["exit_price"] = price
                            
                            sell_value = trade["quantity"] * price
                            total_exit_costs = self.risk_manager.calculate_exit_charges(sell_value)
                            
                            if trade["side"] == "LONG":
                                gross_pnl = (price - trade["entry_price"]) * trade["quantity"]
                            else:
                                gross_pnl = (trade["entry_price"] - price) * trade["quantity"]
                                
                            net_pnl = gross_pnl - total_exit_costs - (trade.get("buy_brokerage") or 0.0)
                            trade["final_profit_pct"] = (net_pnl / (trade["entry_price"] * trade["quantity"])) * 100
                            trade["net_pnl_amt"] = net_pnl

                            with self.lock:
                                self.risk_manager.virtual_capital += net_pnl
                                self.risk_manager.used_margin -= trade.get("margin_used", 0.0)
                                self.risk_manager.realized_charges += total_exit_costs

                            self._mark_recently_closed(trade["ticker"])
                            self.risk_manager.update_upstox_stats(self.active_shadow_trades)
                            log_trade(trade)
                            
                            strat_disp = f"S{trade.get('strategy_id')}" if trade.get('strategy_id') is not None else "AI"
                            print(f"[{trade['status']}] {trade['ticker']} ({strat_disp}) | Net P&L: {trade['final_profit_pct']:.2f}% (Rs{net_pnl:.2f})")
                            continue

                        # Live updates
                        if trade["status"] == "OPEN":
                            trade["exit_price"] = price
                            trade["final_profit_pct"] = round(pnl, 4)
                            log_trade(trade)

                    except Exception as e_indiv:
                        print(f"[WARN] Error tracking trade {trade.get('ticker')}: {e_indiv}")

                # Clean concluded trades from memory periodically
                with self.lock:
                    self.active_shadow_trades = [
                        t for t in self.active_shadow_trades 
                        if t["status"] in ["OPEN", "PENDING_ENTRY", "VETOED"]
                    ]

            except Exception as e:
                print(f"[CRITICAL ERROR] Shadow Tracker Loop: {e}")
            time.sleep(1)

    def run(self):
        while True:
            try:
                # 1. Daily Capital Reset
                now_date = datetime.now().date()
                with self.lock:
                    open_count = len([t for t in self.active_shadow_trades if t["status"] == "OPEN"])

                if now_date > self.current_date or open_count == 0:
                    if now_date > self.current_date:
                        self.current_date = now_date
                        log(f"[INFO] Day Reset: Rs{self.risk_manager.virtual_capital:.2f}")
                        self.update_daily_macro_filters()

                    if self.risk_manager.day_start_capital != self.risk_manager.virtual_capital:
                        self.risk_manager.day_start_capital = self.risk_manager.virtual_capital
                        log(f"[INFO] Scaling slots to Rs{self.risk_manager.day_start_capital:.2f}")

                now_str = datetime.now().strftime("%H:%M")
                if not ("09:00" <= now_str < "15:30"):
                    print(f"[{now_str}] Outside Market Hours. Waiting...")
                    time.sleep(600)
                    continue

                log("\n" + "-" * 40)
                log(f"VANGUARD SCAN CYCLE: {datetime.now().strftime('%H:%M:%S')}")
                log(f"Active Trades: {open_count}/{config.MAX_TRADE_SLOTS}")
                log(f"Virtual Capital: Rs{self.risk_manager.virtual_capital:.2f} | Charges: Rs{self.risk_manager.realized_charges:.2f}")
                log("-" * 40)

                # 2. Fetch live scores
                scores_df = self.calculate_conviction_scores(TICKERS)
                if scores_df.empty:
                    log("[WARN] No scores generated.")
                    time.sleep(60)
                    continue

                is_trading_window = "10:15" <= now_str < "15:05"

                # 3. Generate candidate signals
                top_signals = self.signal_generator.generate_candidate_signals(
                    scores_df,
                    self.long_eligible_tickers,
                    self.short_eligible_tickers,
                    self._is_in_cooldown,
                    self._is_veto_cooldown,
                    self.risk_manager.min_conviction,
                    self.risk_manager.min_raw_score
                )

                if not top_signals.empty:
                    for side_loop in ["LONG", "SHORT"]:
                        side_signals = top_signals[top_signals['side'] == side_loop]
                        if side_signals.empty: continue
                        for _, sig in side_signals.iterrows():
                            side = sig['side']
                            ticker = sig['ticker']
                            
                            if side == "LONG" and ticker not in self.long_eligible_tickers: continue
                            if side == "SHORT" and ticker not in self.short_eligible_tickers: continue
                            
                            conviction = sig['conviction']
                            raw_score = sig['raw_score']
                            strategy_id = sig['strategy_id']
                            if pd.isna(strategy_id) or strategy_id is None:
                                strategy_id = None
                            else:
                                strategy_id = int(strategy_id)
                                
                            strat_display = f"S{strategy_id}" if strategy_id is not None else "AI"
                            rank_col = "Long_Rank" if side == "LONG" else "Short_Rank"
                            is_eligible_time = is_trading_window or (strategy_id is not None and "09:15" <= now_str < "15:05")

                            if not is_eligible_time:
                                log(f"[SCAN ONLY] {side} {ticker} ({strat_display}) | Conviction: {conviction:.4f} (Outside Window)")
                                continue

                            # Concurrency Slots
                            with self.lock:
                                active_slots = len([t for t in self.active_shadow_trades if t["status"] in ["OPEN", "PENDING_ENTRY"]])
                                if active_slots >= config.MAX_TRADE_SLOTS:
                                    log(f"[SKIP] Concurrency Limit Reached (Max {config.MAX_TRADE_SLOTS}).")
                                    break

                                already_open = any(
                                    t["ticker"] == ticker
                                    and t["status"] in ["OPEN", "VETOED", "PENDING_ENTRY"]
                                    for t in self.active_shadow_trades
                                )

                            if already_open:
                                log(f"[SKIP] {side} {ticker} already active in shadow tracker.")
                                continue

                            # Veto cooldown
                            if self._is_veto_cooldown(ticker):
                                log(f"[VETO-COOLDOWN] {side} {ticker} vetoed recently. Skipping.")
                                continue

                            # 4. Gemini AI Veto Audit
                            log(f"[AUDIT] {side} {ticker} ({strat_display}) | Conviction: {conviction:.4f} | Verifying...")
                            full_feature_row = scores_df[scores_df['ticker'] == ticker].iloc[0].copy()
                            full_feature_row['strategy_id'] = sig.get('strategy_id')
                            full_feature_row['signal_source'] = sig.get('source')
                            full_feature_row['is_ensemble'] = sig.get('is_ensemble', False)
                            
                            sentiment, reason, one_hour_prob = self.ai_veto_manager.gemini_audit(
                                ticker, side, conviction, full_feature_row, self.broker.get_recent_candles
                            )

                            if sentiment == "SYSTEM_ERROR":
                                log(f"[SKIP] {ticker} skipped due to AI API error.")
                                continue

                            # Session stats refresh
                            if datetime.now().date() != self._veto_stats_date:
                                self._veto_stats_date = datetime.now().date()
                                self.veto_stats = {
                                    "s1_vetoes": 0, "s2_vetoes": 0,
                                    "s1_passes": 0, "s2_passes": 0,
                                    "s1_tickers": [], "s2_tickers": []
                                }

                            is_s1 = reason.startswith("[S1-")
                            veto_stage = "S1" if is_s1 else "S2"

                            is_vetoed = "VETO" in sentiment.upper()

                            entry_price = self.broker.get_live_price(ticker)
                            if not entry_price or entry_price <= 0:
                                entry_price = float(full_feature_row["Close"])

                            # Fetch TradingView Sentiment (Analytics only)
                            tv_sentiment = get_tv_sentiment(ticker)

                            sent_map = {
                                "STRONG BULLISH": 1.0, "BULLISH": 0.75,
                                "NEUTRAL": 0.5, "BEARISH": 0.25, "STRONG BEARISH": 0.0,
                                "VETOED": 0.0,
                            }
                            nlp_score = sent_map.get(sentiment, 0.5)

                            if not is_vetoed:
                                self.veto_stats["s1_passes"] += 1
                                self.veto_stats["s2_passes"] += 1
                                log(f"[✓ PASS] {side} {ticker} confirmed by AI | Sentiment: {sentiment} | {reason}")
                                
                                self.start_shadow_trade(
                                    ticker, conviction, entry_price,
                                    side, f"[{sentiment}] {reason}", one_hour_prob,
                                    nlp_sentiment=nlp_score,
                                    tv_sentiment=tv_sentiment,
                                    long_score=full_feature_row.get("long_score"),
                                    short_score=full_feature_row.get("short_score"),
                                    strategy_id=strategy_id,
                                    score_15m=full_feature_row.get("score_15m"),
                                    score_30m=full_feature_row.get("score_30m"),
                                    score_1d=full_feature_row.get("score_1d"),
                                    is_ensemble=full_feature_row.get("is_ensemble", False)
                                )
                                break
                            else:
                                if is_s1:
                                    self.veto_stats["s1_vetoes"] += 1
                                    self.veto_stats["s1_tickers"].append((sig["ticker"], side, reason, strategy_id))
                                    print(f"[✗ S1-VETO] Rank {int(sig[rank_col])} {side} {sig['ticker']} | Conv: {conviction:.4f} | {reason}")
                                else:
                                    self.veto_stats["s1_passes"] += 1
                                    self.veto_stats["s2_vetoes"] += 1
                                    import re as _re
                                    rule_match = _re.search(r'RULE \d+[^]]*', reason)
                                    rule_tag = rule_match.group(0) if rule_match else "S2"
                                    self.veto_stats["s2_tickers"].append((sig["ticker"], side, rule_tag, reason, strategy_id))
                                    print(f"[✗ S2-VETO] Rank {int(sig[rank_col])} {side} {sig['ticker']} | Conv: {conviction:.4f} | {rule_tag} | {reason}")

                                self.start_vetoed_tracking(
                                    sig["ticker"], conviction, entry_price,
                                    side, f"[{veto_stage}-VETO] {reason}", one_hour_prob,
                                    nlp_sentiment=nlp_score,
                                    tv_sentiment=tv_sentiment,
                                    long_score=sig.get("long_score"),
                                    short_score=sig.get("short_score"),
                                    strategy_id=strategy_id,
                                    score_15m=sig.get("score_15m"),
                                    score_30m=sig.get("score_30m"),
                                    score_1d=sig.get("score_1d"),
                                    is_ensemble=sig.get("is_ensemble", False)
                                )

                # End scan cycle summary
                print("\n" + "═"*70)
                print(f" SESSION VETO SUMMARY ({datetime.now().strftime('%H:%M')})")
                print("═"*70)
                print(f" S1 (Technical) : VETOED={self.veto_stats['s1_vetoes']} | PASSED={self.veto_stats['s1_passes']}")
                print(f" S2 (Governance): VETOED={self.veto_stats['s2_vetoes']} | PASSED={self.veto_stats['s2_passes']}")
                print("═"*70 + "\n")

                # Boundary Alignment Sleep
                _now       = datetime.now()
                _secs_past = (_now.minute % 15) * 60 + _now.second + _now.microsecond / 1e6
                _wait      = (15 * 60) - _secs_past
                if _wait <= 2:
                    _wait += 15 * 60
                _next_fire = _now + timedelta(seconds=_wait)
                print(f"[SCHEDULER] Next 15-min scan at {_next_fire.strftime('%H:%M:%S')} (sleeping {int(_wait)}s)")
                time.sleep(_wait)

            except Exception as e:
                print(f"[CRITICAL ERROR] Main Orchestrator Loop: {e}")
                traceback.print_exc()
                time.sleep(60)

"""
backtest_multi_tf_v2.py — Fixed Multi-Timeframe Confluence Backtester

Key fixes from v1:
1. Return computation: Uses actual price-based P&L (entry_open → exit_close)
   instead of mixing next_return with open/close prices.
2. Dynamic exit: Uses conviction-strength thresholds (not raw sign flips).
3. Proper multi-bar tracking with actual price series.
4. Added Strategy 3: Trailing stop-loss / take-profit.

Strategy logic:
1. Hourly Signal (1H model): Picks Top-3 Long & Short at each hour.
2. Trend Confirmation (30M model): 1H picks must be in 30M Top-5 same direction.
3. Entry Timing (15M model): Must be in 15M Top-5 same direction + IBS pullback.
4. Exit strategies:
   - S1: Single-Bar (15-min) hold — exit at current bar's close.
   - S2: Dynamic conviction exit — exits when conviction weakens significantly.
   - S3: Trailing stop / take-profit — mechanical risk management.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm
from datetime import datetime

sys.path.append(os.getcwd())

# ========================================
# CONFIG
# ========================================
TEST_MONTH = "2026-05"
TRANSACTION_COST_PCT = 0.03  # 0.03% round-trip (each side)

# Top-K thresholds for confluence
HOURLY_TOPK = 3       # Pick top-3 from hourly model
CONV_30M_TOPK = 5     # Must be in top-5 of 30M model
CONV_15M_TOPK = 5     # Must be in top-5 of 15M model

# IBS Pullback thresholds
IBS_LONG_THRESHOLD = 0.40
IBS_SHORT_THRESHOLD = 0.60

# Strategy 2: Dynamic Exit Parameters
S2_MAX_BARS = 4              # Max hold = 4 bars = 60 minutes
S2_RANK_DROP_THRESHOLD = 15  # Exit if rank drops below top-15 (was top-10, too aggressive)
S2_CONV_WEAKNESS_RATIO = 0.3 # Exit if conviction drops below 30% of entry conviction

# Strategy 3: Trailing Stop / Take-Profit
S3_STOP_LOSS_PCT = -0.005    # -0.5% stop loss
S3_TAKE_PROFIT_PCT = 0.008   # +0.8% take profit
S3_MAX_BARS = 6              # Max 90 minutes

print("=" * 70)
print("MULTI-TIMEFRAME CONFLUENCE BACKTESTER v2 (FIXED)")
print(f"Simulating 1H + 30M + 15M Confluence on Holdout Month: {TEST_MONTH}")
print("=" * 70)

# ========================================
# Helper: Align Timeframes
# ========================================
def align_timeframes(t_str):
    """
    Given a 15-minute bar timestamp (e.g. '2026-05-18 10:30:00+05:30'),
    return the latest closed 1H bar timestamp and latest closed 30M bar timestamp.
    
    Actual data timestamps:
      1H:  09:30, 10:30, 11:30, 12:30, 13:30, 14:30  (bar open times)
      30M: 09:15, 09:45, 10:15, 10:45, ...            (bar open times)
      15M: 09:15, 09:30, 09:45, 10:00, 10:15, ...     (bar open times)
    
    A bar at timestamp S with interval I covers [S, S+I).
    It's "closed" (data finalized) at S+I.
    So 1H bar "09:30" closes at 10:30 → available when T >= 10:30.
    And 30M bar "09:15" closes at 09:45 → available when T >= 09:45.
    """
    date_part = t_str[:10]
    time_part = t_str[11:16]
    h, m = map(int, time_part.split(':'))
    minutes = h * 60 + m
    
    # 1H bars: start at 09:30, 10:30, 11:30, 12:30, 13:30, 14:30
    # They close 60 min later: 10:30, 11:30, 12:30, 13:30, 14:30, 15:30
    hourly_bars = [
        (9, 30),   # closes at 10:30
        (10, 30),  # closes at 11:30
        (11, 30),  # closes at 12:30
        (12, 30),  # closes at 13:30
        (13, 30),  # closes at 14:30
        (14, 30),  # closes at 15:30
    ]
    
    t_1h = None
    for bar_h, bar_m in hourly_bars:
        bar_start_min = bar_h * 60 + bar_m
        bar_close_min = bar_start_min + 60
        if minutes >= bar_close_min:
            t_1h = f"{date_part} {bar_h:02d}:{bar_m:02d}:00+05:30"
    
    # 30M bars: start at 09:15, 09:45, 10:15, 10:45, ..., 15:15
    # They close 30 min later
    t_30m = None
    bar_start = 9 * 60 + 15  # 09:15
    while bar_start < 15 * 60 + 30:
        bar_close = bar_start + 30
        if minutes >= bar_close:
            bh, bm = divmod(bar_start, 60)
            t_30m = f"{date_part} {bh:02d}:{bm:02d}:00+05:30"
        bar_start += 30
    
    return t_1h, t_30m


# ========================================
# Helper: Predict Scores for a Timeframe Model
# ========================================
def predict_timeframe_scores(df_tf, model_key, meta_path, long_model_path, short_model_path, scaler_path):
    print(f"\nScoring {model_key} dataset...")
    
    with open(meta_path) as f:
        meta = json.load(f)
    feature_cols = meta["features"]
    
    # Check column existence
    missing_cols = [c for c in feature_cols if c not in df_tf.columns]
    if missing_cols:
        print(f"  [WARN] Missing features in dataset: {missing_cols}. Filling with 0.")
        for col in missing_cols:
            df_tf[col] = 0.0
            
    # Load models
    bst_long = xgb.Booster()
    bst_long.load_model(long_model_path)
    bst_short = xgb.Booster()
    bst_short.load_model(short_model_path)
    
    # Load scaler if it exists and apply
    scaler = None
    if scaler_path and os.path.exists(scaler_path):
        with open(scaler_path, "rb") as sf:
            scaler = pickle.load(sf)
            
    X = df_tf[feature_cols].values
    X_clean = np.nan_to_num(X)
    
    scaler_is_fitted = (
        scaler is not None
        and hasattr(scaler, 'scale_')
        and scaler.scale_ is not None
    )
    if scaler_is_fitted:
        X_final = scaler.transform(X_clean)
        print("  Scaler applied")
    else:
        X_final = X_clean
        
    dmat = xgb.DMatrix(X_final, feature_names=feature_cols)
    df_tf['long_score'] = bst_long.predict(dmat)
    df_tf['short_score'] = bst_short.predict(dmat)
    
    # Compute Convictions and Ranks
    df_tf['long_conv'] = df_tf['long_score'] - df_tf['short_score']
    df_tf['short_conv'] = df_tf['short_score'] - df_tf['long_score']
    df_tf['long_rank'] = df_tf.groupby('Query_ID')['long_conv'].rank(ascending=False)
    df_tf['short_rank'] = df_tf.groupby('Query_ID')['short_conv'].rank(ascending=False)
    
    print(f"  Scoring complete: {df_tf.shape[0]:,} rows.")
    return df_tf


# ========================================
# 1. LOAD DATASETS
# ========================================
print("\nLoading datasets...")
df_1h_all = pd.read_csv("data/ranking_data_upstox_3y.csv")
df_30m_all = pd.read_csv("data/ranking_data_upstox_30min_1y.csv")
df_15m_all = pd.read_csv("data/ranking_data_upstox_15min_1y.csv")

# Filter to the holdout month
df_1h = df_1h_all[df_1h_all['DateTime'].str.startswith(TEST_MONTH)].copy()
df_30m = df_30m_all[df_30m_all['DateTime'].str.startswith(TEST_MONTH)].copy()
df_15m = df_15m_all[df_15m_all['DateTime'].str.startswith(TEST_MONTH)].copy()

print(f"  1H Data:  {df_1h.shape[0]:,} rows")
print(f"  30M Data: {df_30m.shape[0]:,} rows")
print(f"  15M Data: {df_15m.shape[0]:,} rows")

# Intersection of tickers
t_1h = set(df_1h['Ticker'].unique())
t_30m = set(df_30m['Ticker'].unique())
t_15m = set(df_15m['Ticker'].unique())
common_tickers = sorted(list(t_1h.intersection(t_30m).intersection(t_15m)))
print(f"  Tickers universe intersection: {len(common_tickers)} symbols")

df_1h = df_1h[df_1h['Ticker'].isin(common_tickers)].copy()
df_30m = df_30m[df_30m['Ticker'].isin(common_tickers)].copy()
df_15m = df_15m[df_15m['Ticker'].isin(common_tickers)].copy()

# ========================================
# 2. RUN PREDICTIONS
# ========================================
# Hourly Model (v8_upstox_3y)
df_1h = predict_timeframe_scores(
    df_1h, "1H",
    "models/v8_upstox_3y/metadata.json",
    "models/v8_upstox_3y/xgb_long_model.json",
    "models/v8_upstox_3y/xgb_short_model.json",
    "models/scaler.pkl"
)

# 30-Min Model (v1_30min)
df_30m = predict_timeframe_scores(
    df_30m, "30M",
    "models/v1_30min/metadata.json",
    "models/v1_30min/xgb_long_model.json",
    "models/v1_30min/xgb_short_model.json",
    "models/v1_30min/scaler.pkl"
)

# 15-Min Model (v1_15min)
df_15m = predict_timeframe_scores(
    df_15m, "15M",
    "models/v1_15min/metadata.json",
    "models/v1_15min/xgb_long_model.json",
    "models/v1_15min/xgb_short_model.json",
    "models/v1_15min/scaler.pkl"
)

# ========================================
# 3. BUILD LOOKUP DICTS
# ========================================
print("\nIndexing predictions for O(1) simulation lookups...")

dict_1h = {}
for qid, q_df in df_1h.groupby('DateTime'):
    dict_1h[qid] = {}
    for _, row in q_df.iterrows():
        dict_1h[qid][row['Ticker']] = {
            'long_rank': int(row['long_rank']),
            'short_rank': int(row['short_rank']),
            'long_conv': float(row['long_conv']),
            'short_conv': float(row['short_conv'])
        }

dict_30m = {}
for qid, q_df in df_30m.groupby('DateTime'):
    dict_30m[qid] = {}
    for _, row in q_df.iterrows():
        dict_30m[qid][row['Ticker']] = {
            'long_rank': int(row['long_rank']),
            'short_rank': int(row['short_rank']),
            'long_conv': float(row['long_conv']),
            'short_conv': float(row['short_conv'])
        }

# For 15M: include OHLC for price-based P&L
dict_15m = {}
for qid, q_df in df_15m.groupby('DateTime'):
    dict_15m[qid] = {}
    for _, row in q_df.iterrows():
        h, l, c, o = row['High'], row['Low'], row['Close'], row['Open']
        ibs_raw = (c - l) / (h - l + 1e-10)
        
        dict_15m[qid][row['Ticker']] = {
            'long_rank': int(row['long_rank']),
            'short_rank': int(row['short_rank']),
            'long_conv': float(row['long_conv']),
            'short_conv': float(row['short_conv']),
            'open': float(o),
            'high': float(h),
            'low': float(l),
            'close': float(c),
            'ibs_raw': float(ibs_raw),
        }

# ========================================
# 4. BUILD INTRADAY PRICE SERIES PER TICKER
# ========================================
# For multi-bar strategies, we need forward prices after entry.
# Build a ticker -> sorted list of (datetime, open, high, low, close)
print("\nBuilding intraday price series per ticker...")

price_series = {}  # ticker -> [(datetime, open, high, low, close), ...]
df_15m_sorted = df_15m.sort_values('DateTime')
for _, row in tqdm(df_15m_sorted.iterrows(), total=len(df_15m_sorted), desc="Building price series"):
    ticker = row['Ticker']
    if ticker not in price_series:
        price_series[ticker] = []
    price_series[ticker].append((
        row['DateTime'],
        float(row['Open']),
        float(row['High']),
        float(row['Low']),
        float(row['Close'])
    ))

# Build lookup: ticker -> datetime -> index in price_series
price_idx = {}  # ticker -> {datetime: idx}
for ticker, series in price_series.items():
    price_idx[ticker] = {dt: i for i, (dt, o, h, l, c) in enumerate(series)}

print(f"  Built price series for {len(price_series)} tickers")


def get_forward_prices(ticker, entry_dt, n_bars):
    """Get the next n_bars prices after entry_dt for a ticker."""
    if ticker not in price_idx or entry_dt not in price_idx[ticker]:
        return []
    start_idx = price_idx[ticker][entry_dt]
    series = price_series[ticker]
    result = []
    for i in range(start_idx, min(start_idx + n_bars + 1, len(series))):
        dt, o, h, l, c = series[i]
        result.append({'datetime': dt, 'open': o, 'high': h, 'low': l, 'close': c})
    return result


# ========================================
# 5. RUN BACKTEST SIMULATION
# ========================================
print("\nRunning backtest simulation...")

trades_s1 = []  # Strategy 1: Single Bar (15m) Hold
trades_s2 = []  # Strategy 2: Dynamic Exit (fixed)
trades_s3 = []  # Strategy 3: Trailing Stop / Take-Profit

# Tracking active trades for S2 and S3
# Each: {symbol: {side, entry_price, entry_time, entry_conv, date, entry_bar_idx}}

df_15m['Date'] = df_15m['DateTime'].str[:10]
unique_dates = sorted(df_15m['Date'].unique())

# Dedup: track which (symbol, bar) combinations already have active trades
s2_active = {}  # symbol -> trade_info
s3_active = {}  # symbol -> trade_info

total_s1_entries = 0
total_s2_entries = 0
total_s3_entries = 0

for date in tqdm(unique_dates, desc="Simulating Days"):
    df_day = df_15m[df_15m['Date'] == date]
    timestamps_15m = sorted(df_day['DateTime'].unique())
    
    # Reset daily active trades
    s2_active = {}
    s3_active = {}
    
    for bar_idx, T in enumerate(timestamps_15m):
        t_1h, t_30m = align_timeframes(T)
        
        # ─── STRATEGY 2: CHECK ACTIVE TRADE EXITS ───
        to_close_s2 = []
        for symbol, tr in s2_active.items():
            if T not in dict_15m or symbol not in dict_15m[T]:
                # Data gap — force exit at last known price
                to_close_s2.append((symbol, 'DATA_GAP', tr['last_close']))
                continue
            
            curr = dict_15m[T][symbol]
            tr['bars_held'] += 1
            tr['last_close'] = curr['close']
            
            # Check exit conditions
            exit_reason = None
            
            # 1. Conviction weakness: conviction has dropped significantly from entry
            if tr['side'] == 'LONG':
                current_conv = curr['long_conv']
                if tr['entry_conv'] > 0 and current_conv < tr['entry_conv'] * S2_CONV_WEAKNESS_RATIO:
                    exit_reason = 'CONV_WEAK'
                if curr['long_rank'] > S2_RANK_DROP_THRESHOLD:
                    exit_reason = 'RANK_DROP'
            else:
                current_conv = curr['short_conv']
                if tr['entry_conv'] > 0 and current_conv < tr['entry_conv'] * S2_CONV_WEAKNESS_RATIO:
                    exit_reason = 'CONV_WEAK'
                if curr['short_rank'] > S2_RANK_DROP_THRESHOLD:
                    exit_reason = 'RANK_DROP'
            
            # 2. Time expiry
            if tr['bars_held'] >= S2_MAX_BARS:
                exit_reason = exit_reason or 'TIME_EXPIRY'
            
            # 3. Market close
            if T.endswith("15:15:00+05:30"):
                exit_reason = exit_reason or 'MARKET_CLOSE'
            
            if exit_reason:
                to_close_s2.append((symbol, exit_reason, curr['close']))
        
        for symbol, reason, exit_price in to_close_s2:
            tr = s2_active[symbol]
            if tr['side'] == 'LONG':
                gross_ret = (exit_price / tr['entry_price']) - 1
            else:
                gross_ret = 1 - (exit_price / tr['entry_price'])
            net_ret = gross_ret - TRANSACTION_COST_PCT / 100
            
            trades_s2.append({
                'date': tr['date'], 'entry_time': tr['entry_time'], 'exit_time': T,
                'ticker': symbol, 'side': tr['side'],
                'entry_price': tr['entry_price'], 'exit_price': exit_price,
                'bars_held': tr['bars_held'], 'exit_reason': reason,
                'gross_return': gross_ret, 'net_return': net_ret,
                'is_win': net_ret > 0
            })
            del s2_active[symbol]
        
        # ─── STRATEGY 3: CHECK ACTIVE TRADE EXITS ───
        to_close_s3 = []
        for symbol, tr in s3_active.items():
            if T not in dict_15m or symbol not in dict_15m[T]:
                to_close_s3.append((symbol, 'DATA_GAP', tr['last_close']))
                continue
            
            curr = dict_15m[T][symbol]
            tr['bars_held'] += 1
            
            # Track highest/lowest for trailing stop
            if tr['side'] == 'LONG':
                current_pnl = (curr['close'] / tr['entry_price']) - 1
                bar_high_pnl = (curr['high'] / tr['entry_price']) - 1
                tr['peak_pnl'] = max(tr['peak_pnl'], bar_high_pnl)
            else:
                current_pnl = 1 - (curr['close'] / tr['entry_price'])
                bar_low_pnl = 1 - (curr['low'] / tr['entry_price'])
                # For shorts, 'peak' is best P&L (highest unrealized profit)
                tr['peak_pnl'] = max(tr['peak_pnl'], bar_low_pnl)
            
            tr['last_close'] = curr['close']
            
            exit_reason = None
            
            # Stop loss
            if current_pnl <= S3_STOP_LOSS_PCT:
                exit_reason = 'STOP_LOSS'
            
            # Take profit
            if current_pnl >= S3_TAKE_PROFIT_PCT:
                exit_reason = 'TAKE_PROFIT'
            
            # Time expiry
            if tr['bars_held'] >= S3_MAX_BARS:
                exit_reason = exit_reason or 'TIME_EXPIRY'
            
            # Market close
            if T.endswith("15:15:00+05:30"):
                exit_reason = exit_reason or 'MARKET_CLOSE'
            
            if exit_reason:
                to_close_s3.append((symbol, exit_reason, curr['close']))
        
        for symbol, reason, exit_price in to_close_s3:
            tr = s3_active[symbol]
            if tr['side'] == 'LONG':
                gross_ret = (exit_price / tr['entry_price']) - 1
            else:
                gross_ret = 1 - (exit_price / tr['entry_price'])
            net_ret = gross_ret - TRANSACTION_COST_PCT / 100
            
            trades_s3.append({
                'date': tr['date'], 'entry_time': tr['entry_time'], 'exit_time': T,
                'ticker': symbol, 'side': tr['side'],
                'entry_price': tr['entry_price'], 'exit_price': exit_price,
                'bars_held': tr['bars_held'], 'exit_reason': reason,
                'peak_pnl': tr['peak_pnl'],
                'gross_return': gross_ret, 'net_return': net_ret,
                'is_win': net_ret > 0
            })
            del s3_active[symbol]
        
        # ─── EVALUATE NEW ENTRIES ───
        # Need 1H and 30M signals
        if not t_1h or t_1h not in dict_1h:
            continue
        if not t_30m or t_30m not in dict_30m:
            continue
        if T not in dict_15m:
            continue
        
        picks_1h = dict_1h[t_1h]
        picks_30m = dict_30m[t_30m]
        picks_15m = dict_15m[T]
        
        # Sort tickers by conviction for Top-K selection
        sorted_longs = sorted(
            [t for t in picks_1h], 
            key=lambda x: picks_1h[x]['long_conv'], reverse=True
        )[:HOURLY_TOPK]
        
        sorted_shorts = sorted(
            [t for t in picks_1h], 
            key=lambda x: picks_1h[x]['short_conv'], reverse=True
        )[:HOURLY_TOPK]
        
        # ─── A. LONGS CONFLUENCE ───
        for symbol in sorted_longs:
            # 30M Trend Confirmation
            if symbol not in picks_30m or picks_30m[symbol]['long_rank'] > CONV_30M_TOPK:
                continue
            # 15M Entry Confluence
            if symbol not in picks_15m or picks_15m[symbol]['long_rank'] > CONV_15M_TOPK:
                continue
            # IBS Pullback check
            if picks_15m[symbol]['ibs_raw'] > IBS_LONG_THRESHOLD:
                continue
                
            # ═══ ENTRY SIGNAL GENERATED! ═══
            bar = picks_15m[symbol]
            entry_price = bar['close']  # Enter at current bar's close
            
            # Strategy 1: Single-bar hold — P&L = next bar's close vs this bar's close
            forward = get_forward_prices(symbol, T, 1)
            if len(forward) >= 2:
                next_bar = forward[1]
                s1_exit = next_bar['close']
                g_ret = (s1_exit / entry_price) - 1
                n_ret = g_ret - TRANSACTION_COST_PCT / 100
                trades_s1.append({
                    'date': date, 'entry_time': T, 'exit_time': next_bar['datetime'],
                    'ticker': symbol, 'side': 'LONG',
                    'entry_price': entry_price, 'exit_price': s1_exit,
                    'gross_return': g_ret, 'net_return': n_ret, 'is_win': n_ret > 0
                })
                total_s1_entries += 1
            
            # Strategy 2: Dynamic exit entry
            if symbol not in s2_active:
                s2_active[symbol] = {
                    'date': date, 'entry_time': T, 'ticker': symbol, 'side': 'LONG',
                    'entry_price': entry_price, 'entry_conv': bar['long_conv'],
                    'bars_held': 0, 'last_close': bar['close']
                }
                total_s2_entries += 1
            
            # Strategy 3: Trailing stop entry
            if symbol not in s3_active:
                s3_active[symbol] = {
                    'date': date, 'entry_time': T, 'ticker': symbol, 'side': 'LONG',
                    'entry_price': entry_price, 'bars_held': 0,
                    'last_close': bar['close'], 'peak_pnl': 0.0
                }
                total_s3_entries += 1
                
        # ─── B. SHORTS CONFLUENCE ───
        for symbol in sorted_shorts:
            # 30M Trend Confirmation
            if symbol not in picks_30m or picks_30m[symbol]['short_rank'] > CONV_30M_TOPK:
                continue
            # 15M Entry Confluence
            if symbol not in picks_15m or picks_15m[symbol]['short_rank'] > CONV_15M_TOPK:
                continue
            # IBS Pullback check (overbought/rally)
            if picks_15m[symbol]['ibs_raw'] < IBS_SHORT_THRESHOLD:
                continue
                
            # ═══ ENTRY SIGNAL GENERATED! ═══
            bar = picks_15m[symbol]
            entry_price = bar['close']  # Enter at current bar's close
            
            # Strategy 1: Single-bar hold
            forward = get_forward_prices(symbol, T, 1)
            if len(forward) >= 2:
                next_bar = forward[1]
                s1_exit = next_bar['close']
                g_ret = 1 - (s1_exit / entry_price)  # Short P&L
                n_ret = g_ret - TRANSACTION_COST_PCT / 100
                trades_s1.append({
                    'date': date, 'entry_time': T, 'exit_time': next_bar['datetime'],
                    'ticker': symbol, 'side': 'SHORT',
                    'entry_price': entry_price, 'exit_price': s1_exit,
                    'gross_return': g_ret, 'net_return': n_ret, 'is_win': n_ret > 0
                })
                total_s1_entries += 1
            
            # Strategy 2: Dynamic exit entry
            if symbol not in s2_active:
                s2_active[symbol] = {
                    'date': date, 'entry_time': T, 'ticker': symbol, 'side': 'SHORT',
                    'entry_price': entry_price, 'entry_conv': bar['short_conv'],
                    'bars_held': 0, 'last_close': bar['close']
                }
                total_s2_entries += 1
            
            # Strategy 3: Trailing stop entry
            if symbol not in s3_active:
                s3_active[symbol] = {
                    'date': date, 'entry_time': T, 'ticker': symbol, 'side': 'SHORT',
                    'entry_price': entry_price, 'bars_held': 0,
                    'last_close': bar['close'], 'peak_pnl': 0.0
                }
                total_s3_entries += 1
    
    # ─── FORCE CLOSE ALL REMAINING EOD ───
    last_T = timestamps_15m[-1] if timestamps_15m else None
    
    for symbol, tr in s2_active.items():
        exit_price = tr['last_close']
        if tr['side'] == 'LONG':
            gross_ret = (exit_price / tr['entry_price']) - 1
        else:
            gross_ret = 1 - (exit_price / tr['entry_price'])
        net_ret = gross_ret - TRANSACTION_COST_PCT / 100
        trades_s2.append({
            'date': tr['date'], 'entry_time': tr['entry_time'], 'exit_time': last_T,
            'ticker': symbol, 'side': tr['side'],
            'entry_price': tr['entry_price'], 'exit_price': exit_price,
            'bars_held': tr['bars_held'], 'exit_reason': 'FORCE_CLOSE_EOD',
            'gross_return': gross_ret, 'net_return': net_ret, 'is_win': net_ret > 0
        })
    
    for symbol, tr in s3_active.items():
        exit_price = tr['last_close']
        if tr['side'] == 'LONG':
            gross_ret = (exit_price / tr['entry_price']) - 1
        else:
            gross_ret = 1 - (exit_price / tr['entry_price'])
        net_ret = gross_ret - TRANSACTION_COST_PCT / 100
        trades_s3.append({
            'date': tr['date'], 'entry_time': tr['entry_time'], 'exit_time': last_T,
            'ticker': symbol, 'side': tr['side'],
            'entry_price': tr['entry_price'], 'exit_price': exit_price,
            'bars_held': tr['bars_held'], 'exit_reason': 'FORCE_CLOSE_EOD',
            'peak_pnl': tr.get('peak_pnl', 0),
            'gross_return': gross_ret, 'net_return': net_ret, 'is_win': net_ret > 0
        })

print(f"\n  Total S1 entries: {total_s1_entries}")
print(f"  Total S2 entries: {total_s2_entries}")
print(f"  Total S3 entries: {total_s3_entries}")


# ========================================
# 6. METRICS CALCULATION AND REPORTING
# ========================================
def evaluate_strategy_trades(trades, name):
    if not trades:
        print(f"\n  Strategy {name}: NO TRADES GENERATED")
        return None
        
    df_t = pd.DataFrame(trades)
    
    # Calculate daily statistics
    daily_stats = []
    for date, group in df_t.groupby('date'):
        wins = group['is_win'].sum()
        total = len(group)
        pnl = group['net_return'].sum()
        daily_stats.append({
            'date': date, 'trades': total, 'wins': wins, 'losses': total - wins,
            'win_rate': wins / total if total > 0 else 0, 'pnl': pnl, 'is_green': pnl > 0
        })
    df_d = pd.DataFrame(daily_stats)
    
    total_trades = len(df_t)
    total_wins = int(df_t['is_win'].sum())
    total_losses = total_trades - total_wins
    overall_wr = total_wins / total_trades
    
    avg_trades_day = df_d['trades'].mean()
    avg_wins_day = df_d['wins'].mean()
    avg_losses_day = df_d['losses'].mean()
    
    total_net_pnl = df_t['net_return'].sum()
    avg_net_trade = df_t['net_return'].mean()
    
    green_days = int(df_d['is_green'].sum())
    total_days = len(df_d)
    green_day_rate = green_days / total_days
    
    winners = df_t[df_t['is_win']]
    losers = df_t[~df_t['is_win']]
    avg_win_size = winners['net_return'].mean() if len(winners) > 0 else 0
    avg_loss_size = losers['net_return'].mean() if len(losers) > 0 else 0
    
    profit_factor = abs(winners['net_return'].sum() / losers['net_return'].sum()) if len(losers) > 0 and losers['net_return'].sum() != 0 else float('inf')
    
    # Maximum Drawdown (cumulative sum of trades)
    cum_returns = df_t['net_return'].cumsum()
    max_dd = (cum_returns - cum_returns.cummax()).min()
    
    # Avg bars held (if available)
    has_bars_held = 'bars_held' in df_t.columns
    avg_bars = df_t['bars_held'].mean() if has_bars_held else 1
    
    # Long vs Short breakdown
    longs = df_t[df_t['side'] == 'LONG']
    shorts = df_t[df_t['side'] == 'SHORT']
    long_wr = longs['is_win'].mean() if len(longs) > 0 else 0
    short_wr = shorts['is_win'].mean() if len(shorts) > 0 else 0
    
    print(f"\n{'=' * 75}")
    print(f"  STRATEGY RESULTS: {name}")
    print(f"{'=' * 75}")
    print(f"  Total Trades: {total_trades} over {total_days} active days")
    print(f"  Win / Loss  : {total_wins} Wins | {total_losses} Losses")
    print(f"  Overall WR  : {overall_wr:.1%}")
    print(f"  Long WR     : {long_wr:.1%} ({len(longs)} trades)")
    print(f"  Short WR    : {short_wr:.1%} ({len(shorts)} trades)")
    if has_bars_held:
        print(f"  Avg Hold     : {avg_bars:.1f} bars ({avg_bars * 15:.0f} min)")
    print(f"")
    print(f"  --- Daily Performance ---")
    print(f"  Avg Trades/Day  : {avg_trades_day:.1f}")
    print(f"  Avg Wins/Day    : {avg_wins_day:.1f}")
    print(f"  Avg Losses/Day  : {avg_losses_day:.1f}")
    print(f"  Green Day Rate  : {green_days}/{total_days} ({green_day_rate:.1%})")
    print(f"")
    print(f"  --- P&L and Magnitudes (Net of {TRANSACTION_COST_PCT:.2f}% Costs) ---")
    print(f"  Total Net Return     : {total_net_pnl * 100:+.2f}%")
    print(f"  Avg Net Return/Trade : {avg_net_trade * 100:+.4f}%")
    print(f"  Avg Win Size         : {avg_win_size * 100:+.4f}%")
    print(f"  Avg Loss Size        : {avg_loss_size * 100:+.4f}%")
    print(f"  Profit Factor        : {profit_factor:.2f}")
    print(f"  Max Drawdown         : {max_dd * 100:.2f}%")
    
    if 'exit_reason' in df_t.columns:
        reasons = df_t['exit_reason'].value_counts()
        print(f"\n  --- Exit Reasons ---")
        for reason, count in reasons.items():
            subset = df_t[df_t['exit_reason'] == reason]
            wr = subset['is_win'].mean()
            avg_pnl = subset['net_return'].mean()
            print(f"    {reason:<20} : {count:3d} ({count/total_trades:5.1%})  WR={wr:.1%}  AvgPnL={avg_pnl*100:+.4f}%")
    
    # Daily P&L table
    print(f"\n  --- Daily P&L Table ---")
    print(f"  {'Date':<12} {'Trades':>6} {'Wins':>5} {'WR':>6} {'PnL':>10}")
    print(f"  {'-'*12} {'-'*6} {'-'*5} {'-'*6} {'-'*10}")
    for _, d in df_d.iterrows():
        marker = "+" if d['is_green'] else "-"
        print(f"  {d['date']:<12} {int(d['trades']):>6} {int(d['wins']):>5} {d['win_rate']:>5.0%} {d['pnl']*100:>+9.3f}% {marker}")
            
    print(f"{'=' * 75}\n")
    
    return {
        'strategy': name,
        'total_trades': int(total_trades),
        'wins': int(total_wins),
        'losses': int(total_losses),
        'win_rate': float(overall_wr),
        'long_win_rate': float(long_wr),
        'short_win_rate': float(short_wr),
        'long_trades': int(len(longs)),
        'short_trades': int(len(shorts)),
        'total_return': float(total_net_pnl),
        'avg_return': float(avg_net_trade),
        'avg_trades_per_day': float(avg_trades_day),
        'green_day_rate': float(green_day_rate),
        'profit_factor': float(profit_factor),
        'max_drawdown': float(max_dd),
        'avg_bars_held': float(avg_bars)
    }

eval_s1 = evaluate_strategy_trades(trades_s1, "S1: Confluence + Single-Bar Hold")
eval_s2 = evaluate_strategy_trades(trades_s2, "S2: Confluence + Dynamic Conviction Exit")
eval_s3 = evaluate_strategy_trades(trades_s3, "S3: Confluence + Trailing Stop/TP")

# Save results
out_data = {
    'holdout_month': TEST_MONTH,
    'tickers_universe_size': len(common_tickers),
    'config': {
        'hourly_topk': HOURLY_TOPK,
        'conv_30m_topk': CONV_30M_TOPK,
        'conv_15m_topk': CONV_15M_TOPK,
        'ibs_long_threshold': IBS_LONG_THRESHOLD,
        'ibs_short_threshold': IBS_SHORT_THRESHOLD,
        's2_max_bars': S2_MAX_BARS,
        's2_rank_drop_threshold': S2_RANK_DROP_THRESHOLD,
        's2_conv_weakness_ratio': S2_CONV_WEAKNESS_RATIO,
        's3_stop_loss_pct': S3_STOP_LOSS_PCT,
        's3_take_profit_pct': S3_TAKE_PROFIT_PCT,
        's3_max_bars': S3_MAX_BARS,
        'transaction_cost_pct': TRANSACTION_COST_PCT,
    },
    'strategies': {
        'strategy_1_single_bar': eval_s1,
        'strategy_2_dynamic': eval_s2,
        'strategy_3_trailing': eval_s3
    },
    'backtested_at': datetime.now().isoformat()
}

out_path = "data/strategy_backtest_multi_tf_v2_results.json"
with open(out_path, "w") as f:
    json.dump(out_data, f, indent=2)
print(f"\n[SUCCESS] Multi-Timeframe Confluence backtest v2 complete.")
print(f"Results saved to {out_path}")

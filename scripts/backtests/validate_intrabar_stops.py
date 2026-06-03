"""
validate_intrabar_stops.py — High-resolution intrabar stop-loss validation.

Simulates Strategy 22 and Strategy 13 at 15-minute resolution, then traces 
active trades on 5-minute candles to determine realistic stop-outs.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm
from datetime import datetime, timedelta

sys.path.append(os.getcwd())

TEST_MONTH = "2026-05"
TRANSACTION_COST_PCT = 0.06

# ========================================
# Helper: Load and Score
# ========================================
def load_and_filter_csv(path, month_prefixes):
    print(f"Loading and filtering {path}...")
    chunks = []
    for chunk in pd.read_csv(path, chunksize=100000):
        mask = chunk['DateTime'].str.startswith(tuple(month_prefixes))
        filtered = chunk[mask]
        if not filtered.empty:
            chunks.append(filtered)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

def predict_timeframe_scores(df_tf, model_key, meta_path, long_model_path, short_model_path, scaler_path=None):
    print(f"Scoring {model_key} dataset...")
    df_tf = df_tf.copy()
    with open(meta_path) as f:
        meta = json.load(f)
    feature_cols = meta["features"]
    
    missing_cols = [c for c in feature_cols if c not in df_tf.columns]
    for col in missing_cols:
        df_tf[col] = 0.0
            
    bst_long = xgb.Booster(); bst_long.load_model(long_model_path); bst_long.set_param({'device': 'cpu'})
    bst_short = xgb.Booster(); bst_short.load_model(short_model_path); bst_short.set_param({'device': 'cpu'})
    
    scaler = None
    if scaler_path and os.path.exists(scaler_path):
        with open(scaler_path, "rb") as sf:
            scaler = pickle.load(sf)
            
    X = np.nan_to_num(df_tf[feature_cols].values)
    if scaler and hasattr(scaler, 'scale_') and scaler.scale_ is not None:
        X = scaler.transform(X)
        
    dmat = xgb.DMatrix(X, feature_names=feature_cols)
    df_tf['long_score'] = bst_long.predict(dmat)
    df_tf['short_score'] = bst_short.predict(dmat)
    df_tf['long_conv'] = df_tf['long_score'] - df_tf['short_score']
    df_tf['short_conv'] = df_tf['short_score'] - df_tf['long_score']
    df_tf['long_rank'] = df_tf.groupby('DateTime')['long_conv'].rank(ascending=False)
    df_tf['short_rank'] = df_tf.groupby('DateTime')['short_conv'].rank(ascending=False)
    
    df_tf = df_tf.sort_values(['Ticker', 'DateTime']).reset_index(drop=True)
    df_tf['long_conv_lag1'] = df_tf.groupby('Ticker')['long_conv'].shift(1)
    df_tf['short_conv_lag1'] = df_tf.groupby('Ticker')['short_conv'].shift(1)
    return df_tf

def find_yesterday_daily_date(trading_date, sorted_daily_dates):
    for d in reversed(sorted_daily_dates):
        if d < trading_date:
            return d
    return None

def align_timeframes(t_str):
    date_part = t_str[:10]
    time_part = t_str[11:16]
    h, m = map(int, time_part.split(':'))
    minutes = h * 60 + m
    
    hourly_bars = [(9, 30), (10, 30), (11, 30), (12, 30), (13, 30), (14, 30)]
    t_1h = None
    for bar_h, bar_m in hourly_bars:
        bar_close_min = bar_h * 60 + bar_m + 60
        if minutes >= bar_close_min:
            t_1h = f"{date_part} {bar_h:02d}:{bar_m:02d}:00+05:30"
            
    t_30m = None
    bar_start = 9 * 60 + 15
    while bar_start < 15 * 60 + 30:
        bar_close = bar_start + 30
        if minutes >= bar_close:
            bh, bm = divmod(bar_start, 60)
            t_30m = f"{date_part} {bh:02d}:{bm:02d}:00+05:30"
        bar_start += 30
    return t_1h, t_30m

# ========================================
# Helper: Evaluate Statistics
# ========================================
def evaluate_stats(trades):
    if not trades:
        return {'total_trades': 0, 'win_rate': 0.0, 'total_return': 0.0, 'profit_factor': 0.0, 'max_dd': 0.0}
    df = pd.DataFrame(trades)
    total_trades = len(df)
    wins = int(df['is_win'].sum())
    win_rate = wins / total_trades if total_trades > 0 else 0.0
    total_return = df['net_return'].sum()
    
    winners = df[df['is_win']]
    losers = df[~df['is_win']]
    profit_factor = abs(winners['net_return'].sum() / losers['net_return'].sum()) if len(losers) > 0 and losers['net_return'].sum() != 0 else float('inf')
    
    cum_returns = df['net_return'].cumsum()
    max_dd = (cum_returns - cum_returns.cummax()).min() if len(cum_returns) > 0 else 0.0
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'total_return': total_return,
        'profit_factor': profit_factor,
        'max_dd': max_dd
    }

def main():
    print("=" * 75)
    print("INTRABAR STOP-LOSS PATH VALIDATION WITH 5-MINUTE CANDLES")
    print("=" * 75)

    # 1. LOAD CACHED 5M DATA
    print("\nLoading 5-minute candles cache...")
    cache_dir = "data/raw_upstox_cache_5min"
    if not os.path.exists(cache_dir) or not os.listdir(cache_dir):
        print(f"[FATAL] Cache dir {cache_dir} is empty. Please run collect_upstox_5min_may.py first.")
        sys.exit(1)
        
    dict_5m_candles = {}
    for filename in os.listdir(cache_dir):
        if filename.endswith(".csv"):
            ticker = filename.replace(".csv", "") + ".NS"
            path = os.path.join(cache_dir, filename)
            df_5m = pd.read_csv(path)
            df_5m['timestamp'] = pd.to_datetime(df_5m['timestamp'])
            
            # Map by timestamp for O(1) lookups
            ticker_map = {}
            for _, row in df_5m.iterrows():
                # Store high, low, close, open
                ticker_map[row['timestamp']] = {
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close'])
                }
            dict_5m_candles[ticker] = ticker_map

    # 2. RUN BASICS FROM 25X BACKTESTER
    df_daily_raw = load_and_filter_csv("data/ranking_data_upstox_daily_5y.csv", ["2026-04", "2026-05"])
    df_1h_raw = load_and_filter_csv("data/ranking_data_upstox_3y.csv", [TEST_MONTH])
    df_30m_raw = load_and_filter_csv("data/ranking_data_upstox_30min_1y.csv", [TEST_MONTH])
    df_15m_raw = load_and_filter_csv("data/ranking_data_upstox_15min_1y.csv", [TEST_MONTH])

    df_daily = predict_timeframe_scores(df_daily_raw, "Daily", "models/daily_xgb/metadata.json", "models/daily_xgb/xgb_long_model.json", "models/daily_xgb/xgb_short_model.json", "models/daily_xgb/scaler.pkl")
    df_1h = predict_timeframe_scores(df_1h_raw, "1H", "models/v8_upstox_3y/metadata.json", "models/v8_upstox_3y/xgb_long_model.json", "models/v8_upstox_3y/xgb_short_model.json", "models/scaler.pkl")
    df_30m = predict_timeframe_scores(df_30m_raw, "30M", "models/v1_30min/metadata.json", "models/v1_30min/xgb_long_model.json", "models/v1_30min/xgb_short_model.json", "models/v1_30min/scaler.pkl")
    df_15m = predict_timeframe_scores(df_15m_raw, "15M", "models/v1_15min/metadata.json", "models/v1_15min/xgb_long_model.json", "models/v1_15min/xgb_short_model.json", "models/v1_15min/scaler.pkl")

    common_tickers = sorted(list(set(df_daily['Ticker']) & set(df_1h['Ticker']) & set(df_30m['Ticker']) & set(df_15m['Ticker'])))
    print(f"\nCommon Tickers: {len(common_tickers)}")

    df_daily = df_daily[df_daily['Ticker'].isin(common_tickers)].copy()
    df_1h = df_1h[df_1h['Ticker'].isin(common_tickers)].copy()
    df_30m = df_30m[df_30m['Ticker'].isin(common_tickers)].copy()
    df_15m = df_15m[df_15m['Ticker'].isin(common_tickers)].copy()

    sorted_daily_dates = sorted(df_daily['DateTime'].str[:10].unique())

    # Build 15M Indicators
    df_15m = df_15m.sort_values(['Ticker', 'DateTime']).reset_index(drop=True)
    df_15m['Date'] = df_15m['DateTime'].str[:10]
    df_15m['long_conv_daily_max'] = df_15m.groupby(['Date', 'Ticker'])['long_conv'].cummax()
    df_15m['short_conv_daily_max'] = df_15m.groupby(['Date', 'Ticker'])['short_conv'].cummax()

    # Pre-build mappings
    dict_daily = {}
    for date_str, group in df_daily.groupby(df_daily['DateTime'].str[:10]):
        dict_daily[date_str] = {}
        for _, row in group.iterrows():
            dict_daily[date_str][row['Ticker']] = {
                'long_rank': int(row['long_rank']),
                'short_rank': int(row['short_rank']),
                'long_conv': float(row['long_conv']),
                'short_conv': float(row['short_conv']),
                'count': len(group)
            }

    dict_1h = {}
    for dt, group in df_1h.groupby('DateTime'):
        dict_1h[dt] = {}
        for _, row in group.iterrows():
            dict_1h[dt][row['Ticker']] = {
                'long_rank': int(row['long_rank']),
                'short_rank': int(row['short_rank']),
                'long_conv': float(row['long_conv']),
                'short_conv': float(row['short_conv']),
            }

    dict_30m = {}
    for dt, group in df_30m.groupby('DateTime'):
        dict_30m[dt] = {}
        for _, row in group.iterrows():
            dict_30m[dt][row['Ticker']] = {
                'long_rank': int(row['long_rank']),
                'short_rank': int(row['short_rank']),
                'long_conv': float(row['long_conv']),
                'short_conv': float(row['short_conv']),
            }

    # Align multi-timeframe convictions
    aligned_1h_long_rank, aligned_1h_short_rank = [], []
    aligned_30m_long_rank, aligned_30m_short_rank = [], []
    aligned_30m_long_conv, aligned_30m_short_conv = [], []
    aligned_30m_long_conv_lag1, aligned_30m_short_conv_lag1 = [], []
    aligned_1h_long_conv, aligned_1h_short_conv = [], []
    aligned_1h_long_conv_lag1, aligned_1h_short_conv_lag1 = [], []
    
    unique_dates = df_15m['Date'].unique()
    date_to_yesterday = {d: find_yesterday_daily_date(d, sorted_daily_dates) for d in unique_dates}

    for t, dt, d in zip(df_15m['Ticker'], df_15m['DateTime'], df_15m['Date']):
        t_1h, t_30m = align_timeframes(dt)
        
        h_val = dict_1h.get(t_1h, {}).get(t) if t_1h else None
        aligned_1h_long_rank.append(h_val['long_rank'] if h_val else 9999)
        aligned_1h_short_rank.append(h_val['short_rank'] if h_val else 9999)
        aligned_1h_long_conv.append(h_val['long_conv'] if h_val else 0.0)
        aligned_1h_short_conv.append(h_val['short_conv'] if h_val else 0.0)
        
        m_val = dict_30m.get(t_30m, {}).get(t) if t_30m else None
        aligned_30m_long_rank.append(m_val['long_rank'] if m_val else 9999)
        aligned_30m_short_rank.append(m_val['short_rank'] if m_val else 9999)
        aligned_30m_long_conv.append(m_val['long_conv'] if m_val else 0.0)
        aligned_30m_short_conv.append(m_val['short_conv'] if m_val else 0.0)

    df_15m['h1_long_rank'] = aligned_1h_long_rank
    df_15m['h1_short_rank'] = aligned_1h_short_rank
    df_15m['h1_long_conv'] = aligned_1h_long_conv
    df_15m['h1_short_conv'] = aligned_1h_short_conv
    
    df_15m['m30_long_rank'] = aligned_30m_long_rank
    df_15m['m30_short_rank'] = aligned_30m_short_rank
    df_15m['m30_long_conv'] = aligned_30m_long_conv
    df_15m['m30_short_conv'] = aligned_30m_short_conv
    
    df_15m['h1_long_conv_lag1'] = df_15m.groupby('Ticker')['h1_long_conv'].shift(1).fillna(0.0)
    df_15m['h1_short_conv_lag1'] = df_15m.groupby('Ticker')['h1_short_conv'].shift(1).fillna(0.0)
    df_15m['m30_long_conv_lag1'] = df_15m.groupby('Ticker')['m30_long_conv'].shift(1).fillna(0.0)
    df_15m['m30_short_conv_lag1'] = df_15m.groupby('Ticker')['m30_short_conv'].shift(1).fillna(0.0)
    df_15m['long_conv_lag1'] = df_15m.groupby('Ticker')['long_conv'].shift(1).fillna(0.0)
    df_15m['short_conv_lag1'] = df_15m.groupby('Ticker')['short_conv'].shift(1).fillna(0.0)

    ticker_bar_map = {t: {} for t in common_tickers}
    for idx, row in df_15m.iterrows():
        t = row['Ticker']
        ticker_bar_map[t][row['DateTime']] = {
            'ticker': t, 'datetime': row['DateTime'], 'date': row['Date'],
            'open': float(row['Open']), 'high': float(row['High']), 'low': float(row['Low']), 'close': float(row['Close']),
            'long_conv': float(row['long_conv']), 'short_conv': float(row['short_conv']),
            'long_rank': int(row['long_rank']), 'short_rank': int(row['short_rank']),
            'long_conv_daily_max': float(row['long_conv_daily_max']), 'short_conv_daily_max': float(row['short_conv_daily_max']),
            'h1_long_rank': int(row['h1_long_rank']), 'h1_short_rank': int(row['h1_short_rank']),
            'm30_long_rank': int(row['m30_long_rank']), 'm30_short_rank': int(row['m30_short_rank']),
            'h1_long_conv': float(row['h1_long_conv']), 'h1_short_conv': float(row['h1_short_conv']),
            'm30_long_conv': float(row['m30_long_conv']), 'm30_short_conv': float(row['m30_short_conv']),
            'h1_long_conv_lag1': float(row['h1_long_conv_lag1']), 'h1_short_conv_lag1': float(row['h1_short_conv_lag1']),
            'm30_long_conv_lag1': float(row['m30_long_conv_lag1']), 'm30_short_conv_lag1': float(row['m30_short_conv_lag1']),
            'long_conv_lag1': float(row['long_conv_lag1']), 'short_conv_lag1': float(row['short_conv_lag1'])
        }

    unique_15m_times = sorted(df_15m['DateTime'].unique())
    dict_15m = {}
    for dt, group in df_15m.groupby('DateTime'):
        dict_15m[dt] = []
        for _, row in group.iterrows():
            t = row['Ticker']
            if t in ticker_bar_map and dt in ticker_bar_map[t]:
                dict_15m[dt].append(ticker_bar_map[t][dt])

    # 3. HIGH-RESOLUTION SIMULATOR
    def run_strategy_audit(strategy_id, name, trailing_stop=0.006, max_hold=None):
        print(f"\nAuditing Strategy {strategy_id}: {name}...")
        
        orig_trades = []
        audited_trades = []
        
        active_trades = [] # List of active trades
        daily_trade_count = {}
        
        for bar_idx, T in enumerate(unique_15m_times):
            date_str = T[:10]
            time_str = T[11:16]
            h, m = map(int, time_str.split(':'))
            t_min = h * 60 + m
            
            if date_str not in daily_trade_count:
                daily_trade_count[date_str] = 0
                
            is_last_bar = (bar_idx == len(unique_15m_times) - 1) or (unique_15m_times[bar_idx + 1][:10] != date_str) or (time_str == "15:15")
            
            # --- exits section ---
            remaining_trades = []
            for t in active_trades:
                t['bars_held'] += 1
                ticker = t['ticker']
                bar_15m = ticker_bar_map[ticker].get(T)
                
                if not bar_15m:
                    # Data gap exit
                    for tr_list, reason in [(orig_trades, 'DATA_GAP'), (audited_trades, 'DATA_GAP')]:
                        tr_list.append({
                            'ticker': ticker, 'side': t['side'], 'entry_price': t['entry_price'],
                            'exit_price': t['entry_price'], 'gross_return': 0.0, 'net_return': -TRANSACTION_COST_PCT / 100,
                            'is_win': False, 'exit_reason': reason
                        })
                    continue
                
                # A. Evaluate Original Backtest Close-of-Bar Logic
                orig_exit = False
                orig_reason = None
                orig_exit_price = bar_15m['close']
                
                # Check trailing stop for original
                if t['side'] == 'LONG':
                    cur_ret_15m = (bar_15m['close'] / t['entry_price']) - 1.0
                    high_ret_15m = (bar_15m['high'] / t['entry_price']) - 1.0
                    t['orig_peak_pnl'] = max(t['orig_peak_pnl'], high_ret_15m)
                    orig_stopped = t['orig_peak_pnl'] - cur_ret_15m >= trailing_stop
                else:
                    cur_ret_15m = 1.0 - (bar_15m['close'] / t['entry_price'])
                    low_ret_15m = 1.0 - (bar_15m['low'] / t['entry_price'])
                    t['orig_peak_pnl'] = max(t['orig_peak_pnl'], low_ret_15m)
                    orig_stopped = t['orig_peak_pnl'] - cur_ret_15m >= trailing_stop
                
                if orig_stopped:
                    orig_exit = True
                    orig_reason = 'TRAILING_STOP'
                    orig_exit_price = t['entry_price'] * (1.0 + t['orig_peak_pnl'] - trailing_stop) if t['side'] == 'LONG' else t['entry_price'] * (1.0 - (t['orig_peak_pnl'] - trailing_stop))
                elif max_hold and t['bars_held'] >= max_hold:
                    orig_exit = True
                    orig_reason = 'TIME_EXPIRY'
                elif is_last_bar:
                    orig_exit = True
                    orig_reason = 'FORCE_CLOSE_EOD'
                    
                # B. Evaluate High-Resolution Audited Intrabar Logic
                aud_exit = False
                aud_reason = None
                aud_exit_price = bar_15m['close']
                
                # Generate list of 5-minute bars covering this 15-minute window
                # If we entered at T_entry close, the first 5M bar starts at T_entry + 15M.
                # The current 15M bar T (which closes at T + 15M) is covered by 5M bars starting at T, T+5, T+10.
                t_dt = pd.to_datetime(T)
                sub_5m_times = [t_dt, t_dt + timedelta(minutes=5), t_dt + timedelta(minutes=10)]
                
                for t5 in sub_5m_times:
                    if aud_exit:
                        break
                    
                    bar_5m = dict_5m_candles.get(ticker, {}).get(t5)
                    if not bar_5m:
                        continue # Skip missing 5m candles
                        
                    if t['side'] == 'LONG':
                        low_pnl_5m = (bar_5m['low'] / t['entry_price']) - 1.0
                        high_pnl_5m = (bar_5m['high'] / t['entry_price']) - 1.0
                        
                        t['aud_peak_pnl'] = max(t['aud_peak_pnl'], high_pnl_5m)
                        
                        # Check trailing stop
                        if t['aud_peak_pnl'] - low_pnl_5m >= trailing_stop:
                            aud_exit = True
                            aud_reason = 'TRAILING_STOP_INTRABAR'
                            aud_exit_price = t['entry_price'] * (1.0 + t['aud_peak_pnl'] - trailing_stop)
                    else:
                        low_pnl_5m = 1.0 - (bar_5m['low'] / t['entry_price']) # low is high pnl for short
                        high_pnl_5m = 1.0 - (bar_5m['high'] / t['entry_price']) # high is loss for short
                        
                        t['aud_peak_pnl'] = max(t['aud_peak_pnl'], low_pnl_5m) # low price is max profit for short
                        
                        # Check trailing stop (loss for short when price reaches high price)
                        # high_pnl_5m = 1.0 - (high_price / entry)
                        loss_from_peak = t['aud_peak_pnl'] - (1.0 - (bar_5m['high'] / t['entry_price']))
                        if loss_from_peak >= trailing_stop:
                            aud_exit = True
                            aud_reason = 'TRAILING_STOP_INTRABAR'
                            aud_exit_price = t['entry_price'] * (1.0 - (t['aud_peak_pnl'] - trailing_stop))
                
                # Check remaining non-stop exits at 15-minute close boundary
                if not aud_exit:
                    if max_hold and t['bars_held'] >= max_hold:
                        aud_exit = True
                        aud_reason = 'TIME_EXPIRY'
                    elif is_last_bar:
                        aud_exit = True
                        aud_reason = 'FORCE_CLOSE_EOD'
                
                # Log closed trades
                if orig_exit:
                    g_ret = (orig_exit_price / t['entry_price']) - 1.0 if t['side'] == 'LONG' else 1.0 - (orig_exit_price / t['entry_price'])
                    n_ret = g_ret - TRANSACTION_COST_PCT / 100
                    orig_trades.append({
                        'ticker': ticker, 'side': t['side'], 'entry_price': t['entry_price'],
                        'exit_price': orig_exit_price, 'gross_return': g_ret, 'net_return': n_ret,
                        'is_win': n_ret > 0, 'exit_reason': orig_reason
                    })
                
                if aud_exit:
                    g_ret = (aud_exit_price / t['entry_price']) - 1.0 if t['side'] == 'LONG' else 1.0 - (aud_exit_price / t['entry_price'])
                    n_ret = g_ret - TRANSACTION_COST_PCT / 100
                    audited_trades.append({
                        'ticker': ticker, 'side': t['side'], 'entry_price': t['entry_price'],
                        'exit_price': aud_exit_price, 'gross_return': g_ret, 'net_return': n_ret,
                        'is_win': n_ret > 0, 'exit_reason': aud_reason
                    })
                
                # Keep active in simulation if not exited on BOTH models (to maintain state)
                # Wait, if one exited, we stop tracking it in the active list.
                # If audited exits first, we can no longer trace it.
                # So we remove it from active_trades when the AUDITED exit triggers.
                # For original, if it exited but audited didn't, we still have it active.
                # To be mathematically precise, we should track active trades for both separately.
                # Let's check: yes, we should track a separate active trade dict for original and audited!
                # That is much cleaner and mathematically correct.
                
            # Wait, let's restructure the active trade lists to avoid state pollution!
            
        # Let's do a clean separate loop for Original and Audited trades
        
        # --- ORIGINAL SIMULATION ---
        active_orig = []
        daily_count = {}
        for bar_idx, T in enumerate(unique_15m_times):
            date_str = T[:10]
            time_str = T[11:16]
            is_last_bar = (bar_idx == len(unique_15m_times) - 1) or (unique_15m_times[bar_idx + 1][:10] != date_str) or (time_str == "15:15")
            
            if date_str not in daily_count: daily_count[date_str] = 0
            
            # Process exits
            still_active = []
            for t in active_orig:
                t['bars_held'] += 1
                bar_15m = ticker_bar_map[t['ticker']].get(T)
                if not bar_15m:
                    orig_trades.append({
                        'net_return': -TRANSACTION_COST_PCT / 100, 'is_win': False, 'exit_reason': 'DATA_GAP'
                    })
                    continue
                
                orig_exit = False
                orig_reason = None
                orig_exit_price = bar_15m['close']
                
                if t['side'] == 'LONG':
                    cur_ret = (bar_15m['close'] / t['entry_price']) - 1.0
                    high_ret = (bar_15m['high'] / t['entry_price']) - 1.0
                    t['peak_pnl'] = max(t['peak_pnl'], high_ret)
                    stopped = t['peak_pnl'] - cur_ret >= trailing_stop
                else:
                    cur_ret = 1.0 - (bar_15m['close'] / t['entry_price'])
                    low_ret = 1.0 - (bar_15m['low'] / t['entry_price'])
                    t['peak_pnl'] = max(t['peak_pnl'], low_ret)
                    stopped = t['peak_pnl'] - cur_ret >= trailing_stop
                    
                if stopped:
                    orig_exit = True
                    orig_reason = 'TRAILING_STOP'
                    orig_exit_price = t['entry_price'] * (1.0 + t['peak_pnl'] - trailing_stop) if t['side'] == 'LONG' else t['entry_price'] * (1.0 - (t['peak_pnl'] - trailing_stop))
                elif max_hold and t['bars_held'] >= max_hold:
                    orig_exit = True
                    orig_reason = 'TIME_EXPIRY'
                elif is_last_bar:
                    orig_exit = True
                    orig_reason = 'FORCE_CLOSE_EOD'
                    
                if orig_exit:
                    g_ret = (orig_exit_price / t['entry_price']) - 1.0 if t['side'] == 'LONG' else 1.0 - (orig_exit_price / t['entry_price'])
                    n_ret = g_ret - TRANSACTION_COST_PCT / 100
                    orig_trades.append({
                        'net_return': n_ret, 'is_win': n_ret > 0, 'exit_reason': orig_reason
                    })
                else:
                    still_active.append(t)
            active_orig = still_active
            
            # Place entries
            if T in dict_15m:
                active_tickers = {t['ticker'] for t in active_orig}
                # Strategy-specific gates
                if strategy_id == 22:
                    top_longs = sorted(dict_15m[T], key=lambda x: x['long_rank'])
                    top_shorts = sorted(dict_15m[T], key=lambda x: x['short_rank'])
                    
                    # Long entries
                    for x in top_longs:
                        if x['long_rank'] <= 2 and x['long_conv'] >= x['long_conv_daily_max']:
                            ticker = x['ticker']
                            if ticker not in active_tickers and daily_count[date_str] < 4:
                                active_orig.append({
                                    'ticker': ticker, 'side': 'LONG', 'entry_price': x['close'],
                                    'bars_held': 0, 'peak_pnl': 0.0
                                })
                                active_tickers.add(ticker)
                                daily_count[date_str] += 1
                    # Short entries
                    for x in top_shorts:
                        if x['short_rank'] <= 2 and x['short_conv'] >= x['short_conv_daily_max']:
                            ticker = x['ticker']
                            if ticker not in active_tickers and daily_count[date_str] < 4:
                                active_orig.append({
                                    'ticker': ticker, 'side': 'SHORT', 'entry_price': x['close'],
                                    'bars_held': 0, 'peak_pnl': 0.0
                                })
                                active_tickers.add(ticker)
                                daily_count[date_str] += 1
                                
                elif strategy_id == 13:
                    t_1h, t_30m = align_timeframes(T)
                    if t_1h in dict_1h and t_30m in dict_30m:
                        top_longs = sorted(dict_15m[T], key=lambda x: x['long_rank'])
                        top_shorts = sorted(dict_15m[T], key=lambda x: x['short_rank'])
                        
                        for x in top_longs:
                            if x['long_rank'] <= 5:
                                ticker = x['ticker']
                                h1_pred = dict_1h[t_1h].get(ticker)
                                if h1_pred and h1_pred['long_rank'] <= 8:
                                    m30_pred = dict_30m[t_30m].get(ticker)
                                    if m30_pred and m30_pred['long_rank'] <= 5:
                                        if ticker not in active_tickers and daily_count[date_str] < 6:
                                            active_orig.append({
                                                'ticker': ticker, 'side': 'LONG', 'entry_price': x['close'],
                                                'bars_held': 0, 'peak_pnl': 0.0
                                            })
                                            active_tickers.add(ticker)
                                            daily_count[date_str] += 1
                                            
                        for x in top_shorts:
                            if x['short_rank'] <= 5:
                                ticker = x['ticker']
                                h1_pred = dict_1h[t_1h].get(ticker)
                                if h1_pred and h1_pred['short_rank'] <= 8:
                                    m30_pred = dict_30m[t_30m].get(ticker)
                                    if m30_pred and m30_pred['short_rank'] <= 5:
                                        if ticker not in active_tickers and daily_count[date_str] < 6:
                                            active_orig.append({
                                                'ticker': ticker, 'side': 'SHORT', 'entry_price': x['close'],
                                                'bars_held': 0, 'peak_pnl': 0.0
                                            })
                                            active_tickers.add(ticker)
                                            daily_count[date_str] += 1
                                            
        # --- AUDITED SIMULATION (WITH 5M TICK-BY-TICK PATH TRACING) ---
        active_aud = []
        daily_count = {}
        for bar_idx, T in enumerate(unique_15m_times):
            date_str = T[:10]
            time_str = T[11:16]
            is_last_bar = (bar_idx == len(unique_15m_times) - 1) or (unique_15m_times[bar_idx + 1][:10] != date_str) or (time_str == "15:15")
            
            if date_str not in daily_count: daily_count[date_str] = 0
            
            # Process exits (5-minute tick-by-tick within this 15-minute bar)
            still_active = []
            for t in active_aud:
                t['bars_held'] += 1
                ticker = t['ticker']
                bar_15m = ticker_bar_map[ticker].get(T)
                if not bar_15m:
                    audited_trades.append({
                        'net_return': -TRANSACTION_COST_PCT / 100, 'is_win': False, 'exit_reason': 'DATA_GAP'
                    })
                    continue
                
                # Check intrabar 5-minute candles
                aud_exit = False
                aud_reason = None
                aud_exit_price = bar_15m['close']
                
                t_dt = pd.to_datetime(T)
                sub_5m_times = [t_dt, t_dt + timedelta(minutes=5), t_dt + timedelta(minutes=10)]
                
                for t5 in sub_5m_times:
                    if aud_exit:
                        break
                    
                    bar_5m = dict_5m_candles.get(ticker, {}).get(t5)
                    if not bar_5m:
                        continue
                        
                    if t['side'] == 'LONG':
                        low_pnl_5m = (bar_5m['low'] / t['entry_price']) - 1.0
                        high_pnl_5m = (bar_5m['high'] / t['entry_price']) - 1.0
                        
                        t['peak_pnl'] = max(t['peak_pnl'], high_pnl_5m)
                        
                        # Check trailing stop
                        if t['peak_pnl'] - low_pnl_5m >= trailing_stop:
                            aud_exit = True
                            aud_reason = 'TRAILING_STOP_INTRABAR'
                            aud_exit_price = t['entry_price'] * (1.0 + t['peak_pnl'] - trailing_stop)
                    else:
                        low_pnl_5m = 1.0 - (bar_5m['low'] / t['entry_price']) # low price = max profit for short
                        high_pnl_5m = 1.0 - (bar_5m['high'] / t['entry_price']) # high price = loss for short
                        
                        t['peak_pnl'] = max(t['peak_pnl'], low_pnl_5m)
                        
                        loss_from_peak = t['peak_pnl'] - high_pnl_5m
                        if loss_from_peak >= trailing_stop:
                            aud_exit = True
                            aud_reason = 'TRAILING_STOP_INTRABAR'
                            aud_exit_price = t['entry_price'] * (1.0 - (t['peak_pnl'] - trailing_stop))
                            
                # If not stopped out intrabar, check time/EOD exits at bar-close
                if not aud_exit:
                    if max_hold and t['bars_held'] >= max_hold:
                        aud_exit = True
                        aud_reason = 'TIME_EXPIRY'
                    elif is_last_bar:
                        aud_exit = True
                        aud_reason = 'FORCE_CLOSE_EOD'
                        
                if aud_exit:
                    g_ret = (aud_exit_price / t['entry_price']) - 1.0 if t['side'] == 'LONG' else 1.0 - (aud_exit_price / t['entry_price'])
                    n_ret = g_ret - TRANSACTION_COST_PCT / 100
                    audited_trades.append({
                        'net_return': n_ret, 'is_win': n_ret > 0, 'exit_reason': aud_reason
                    })
                else:
                    still_active.append(t)
            active_aud = still_active
            
            # Place entries
            if T in dict_15m:
                active_tickers = {t['ticker'] for t in active_aud}
                
                # Check if it is the last bar to prevent overnight carry-over in audited!
                # Wait! We want to trace the EXACT same entries as original to isolate the stop-loss impact.
                # So we enter exactly when the original would enter (no overnight protection yet), 
                # so that the ONLY variable changing is the stop-loss check resolution.
                
                if strategy_id == 22:
                    top_longs = sorted(dict_15m[T], key=lambda x: x['long_rank'])
                    top_shorts = sorted(dict_15m[T], key=lambda x: x['short_rank'])
                    
                    for x in top_longs:
                        if x['long_rank'] <= 2 and x['long_conv'] >= x['long_conv_daily_max']:
                            ticker = x['ticker']
                            if ticker not in active_tickers and daily_count[date_str] < 4:
                                active_aud.append({
                                    'ticker': ticker, 'side': 'LONG', 'entry_price': x['close'],
                                    'bars_held': 0, 'peak_pnl': 0.0
                                })
                                active_tickers.add(ticker)
                                daily_count[date_str] += 1
                    for x in top_shorts:
                        if x['short_rank'] <= 2 and x['short_conv'] >= x['short_conv_daily_max']:
                            ticker = x['ticker']
                            if ticker not in active_tickers and daily_count[date_str] < 4:
                                active_aud.append({
                                    'ticker': ticker, 'side': 'SHORT', 'entry_price': x['close'],
                                    'bars_held': 0, 'peak_pnl': 0.0
                                })
                                active_tickers.add(ticker)
                                daily_count[date_str] += 1
                                
                elif strategy_id == 13:
                    t_1h, t_30m = align_timeframes(T)
                    if t_1h in dict_1h and t_30m in dict_30m:
                        top_longs = sorted(dict_15m[T], key=lambda x: x['long_rank'])
                        top_shorts = sorted(dict_15m[T], key=lambda x: x['short_rank'])
                        
                        for x in top_longs:
                            if x['long_rank'] <= 5:
                                ticker = x['ticker']
                                h1_pred = dict_1h[t_1h].get(ticker)
                                if h1_pred and h1_pred['long_rank'] <= 8:
                                    m30_pred = dict_30m[t_30m].get(ticker)
                                    if m30_pred and m30_pred['long_rank'] <= 5:
                                        if ticker not in active_tickers and daily_count[date_str] < 6:
                                            active_aud.append({
                                                'ticker': ticker, 'side': 'LONG', 'entry_price': x['close'],
                                                'bars_held': 0, 'peak_pnl': 0.0
                                            })
                                            active_tickers.add(ticker)
                                            daily_count[date_str] += 1
                                            
                        for x in top_shorts:
                            if x['short_rank'] <= 5:
                                ticker = x['ticker']
                                h1_pred = dict_1h[t_1h].get(ticker)
                                if h1_pred and h1_pred['short_rank'] <= 8:
                                    m30_pred = dict_30m[t_30m].get(ticker)
                                    if m30_pred and m30_pred['short_rank'] <= 5:
                                        if ticker not in active_tickers and daily_count[date_str] < 6:
                                            active_aud.append({
                                                'ticker': ticker, 'side': 'SHORT', 'entry_price': x['close'],
                                                'bars_held': 0, 'peak_pnl': 0.0
                                            })
                                            active_tickers.add(ticker)
                                            daily_count[date_str] += 1

        # 4. PRINT REPORT
        orig_res = evaluate_stats(orig_trades)
        aud_res = evaluate_stats(audited_trades)
        
        print("\n" + "-" * 75)
        print(f"AUDIT REPORT FOR STRATEGY {strategy_id}: {name}")
        print("-" * 75)
        print(f"  Metric             | Original (15M Close) | Audited (5M Intrabar)")
        print(f"  -------------------|----------------------|----------------------")
        print(f"  Total Trades       | {orig_res['total_trades']:<20} | {aud_res['total_trades']:<20}")
        print(f"  Win Rate           | {orig_res['win_rate']:<20.1%} | {aud_res['win_rate']:<20.1%}")
        print(f"  Sum of Returns %   | {orig_res['total_return']*100:<20.2f}% | {aud_res['total_return']*100:<20.2f}%")
        print(f"  Profit Factor      | {orig_res['profit_factor']:<20.2f} | {aud_res['profit_factor']:<20.2f}")
        print(f"  Max Drawdown %     | {orig_res['max_dd']*100:<20.2f}% | {aud_res['max_dd']*100:<20.2f}%")
        print("-" * 75)
        
    run_strategy_audit(22, "Momentum Spike & Hold", trailing_stop=0.006)
    run_strategy_audit(13, "Midday Momentum Extension", trailing_stop=0.003, max_hold=8)

if __name__ == "__main__":
    main()

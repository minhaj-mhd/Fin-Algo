"""
scripts/build_dynamic_exit_dataset.py
Generates a dataset for the Dynamic Exit Engine (Hold-or-Sell Supervisor).
Simulates Strategy 22 trades. At every 15m bar while active, records state features 
and calculates the `forward_peak_return` (the maximum possible return if held from 
the current bar until the best point before EOD).
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

def load_and_filter_csv(path, month_prefixes):
    chunks = []
    for chunk in pd.read_csv(path, chunksize=100000):
        mask = chunk['DateTime'].str.startswith(tuple(month_prefixes))
        filtered = chunk[mask]
        if not filtered.empty:
            chunks.append(filtered)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

def predict_timeframe_scores(df_tf, model_key, meta_path, long_model_path, short_model_path, scaler_path=None):
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
    
    # Store raw features too
    for col in feature_cols:
        if col not in df_tf.columns:
            df_tf[col] = X[:, feature_cols.index(col)]
            
    df_tf = df_tf.sort_values(['Ticker', 'DateTime']).reset_index(drop=True)
    return df_tf, feature_cols

def main():
    print("=" * 75)
    print("BUILDING DYNAMIC EXIT DATASET (Forward Peak Return)")
    print("=" * 75)

    print("\nLoading 5-minute candles cache...")
    cache_dir = "data/raw_upstox_cache_5min"
    if not os.path.exists(cache_dir) or not os.listdir(cache_dir):
        print(f"[FATAL] Cache dir {cache_dir} is empty.")
        sys.exit(1)
        
    dict_5m_candles = {}
    for filename in os.listdir(cache_dir):
        if filename.endswith(".csv"):
            ticker = filename.replace(".csv", "") + ".NS"
            path = os.path.join(cache_dir, filename)
            df_5m = pd.read_csv(path)
            df_5m['timestamp'] = pd.to_datetime(df_5m['timestamp'])
            ticker_map = {}
            for _, row in df_5m.iterrows():
                ticker_map[row['timestamp']] = {
                    'open': float(row['open']), 'high': float(row['high']), 'low': float(row['low']), 'close': float(row['close'])
                }
            dict_5m_candles[ticker] = ticker_map

    df_daily_raw = load_and_filter_csv("data/ranking_data_upstox_daily_5y.csv", ["2026-04", "2026-05"])
    df_15m_raw = load_and_filter_csv("data/ranking_data_upstox_15min_1y.csv", [TEST_MONTH])

    df_daily, _ = predict_timeframe_scores(df_daily_raw, "Daily", "models/daily_xgb/metadata.json", "models/daily_xgb/xgb_long_model.json", "models/daily_xgb/xgb_short_model.json", "models/daily_xgb/scaler.pkl")
    df_15m, raw_features = predict_timeframe_scores(df_15m_raw, "15M", "models/v1_15min/metadata.json", "models/v1_15min/xgb_long_model.json", "models/v1_15min/xgb_short_model.json", "models/v1_15min/scaler.pkl")

    common_tickers = sorted(list(set(df_daily['Ticker']) & set(df_15m['Ticker'])))
    df_daily = df_daily[df_daily['Ticker'].isin(common_tickers)].copy()
    df_15m = df_15m[df_15m['Ticker'].isin(common_tickers)].copy()

    df_15m['Date'] = df_15m['DateTime'].str[:10]
    df_15m['long_conv_daily_max'] = df_15m.groupby(['Date', 'Ticker'])['long_conv'].cummax()
    df_15m['short_conv_daily_max'] = df_15m.groupby(['Date', 'Ticker'])['short_conv'].cummax()

    ticker_bar_map = {t: {} for t in common_tickers}
    for idx, row in df_15m.iterrows():
        t = row['Ticker']
        feat_dict = {f: float(row[f]) for f in raw_features if pd.notnull(row[f])}
        ticker_bar_map[t][row['DateTime']] = {
            'ticker': t, 'datetime': row['DateTime'], 'date': row['Date'],
            'open': float(row['Open']), 'high': float(row['High']), 'low': float(row['Low']), 'close': float(row['Close']),
            'long_conv': float(row['long_conv']), 'short_conv': float(row['short_conv']),
            'long_rank': int(row['long_rank']), 'short_rank': int(row['short_rank']),
            'long_conv_daily_max': float(row['long_conv_daily_max']), 'short_conv_daily_max': float(row['short_conv_daily_max']),
            'features': feat_dict
        }

    unique_15m_times = sorted(df_15m['DateTime'].unique())
    dict_15m = {}
    for dt, group in df_15m.groupby('DateTime'):
        dict_15m[dt] = []
        for _, row in group.iterrows():
            t = row['Ticker']
            if t in ticker_bar_map and dt in ticker_bar_map[t]:
                dict_15m[dt].append(ticker_bar_map[t][dt])

    active_trades = []
    daily_count = {}
    dataset_records = []
    
    print("\nSimulating trades to extract intra-trade states...")
    for bar_idx, T in enumerate(tqdm(unique_15m_times)):
        date_str = T[:10]
        time_str = T[11:16]
        is_last_bar = (bar_idx == len(unique_15m_times) - 1) or (unique_15m_times[bar_idx + 1][:10] != date_str) or (time_str == "15:15")
        
        if date_str not in daily_count: daily_count[date_str] = 0
            
        still_active = []
        for t in active_trades:
            ticker = t['ticker']
            bar_15m = ticker_bar_map[ticker].get(T)
            
            if not bar_15m: 
                # Missing data, force close
                continue
                
            current_close = bar_15m['close']
            current_conv = bar_15m['long_conv'] if t['side'] == 'LONG' else bar_15m['short_conv']
            current_rank = bar_15m['long_rank'] if t['side'] == 'LONG' else bar_15m['short_rank']
            
            unrealized_pnl = (current_close / t['entry_price']) - 1.0 if t['side'] == 'LONG' else 1.0 - (current_close / t['entry_price'])
            conv_delta = current_conv - t['entry_conv']
            rank_delta = current_rank - t['entry_rank']
            
            # Record state at this exact bar
            state_record = {
                'ticker': ticker,
                'entry_time': t['entry_time'],
                'current_time': T,
                'side': 1 if t['side'] == 'LONG' else 0,
                'bars_held': t['bars_held'],
                'unrealized_pnl': unrealized_pnl,
                'current_conv': current_conv,
                'current_rank': current_rank,
                'conv_delta': conv_delta,
                'rank_delta': rank_delta
            }
            # Append to trade's trajectory
            t['trajectory'].append({'bar_idx': bar_idx, 'time': T, 'close': current_close, 'record': state_record})
            
            if is_last_bar:
                # Trade is finished. Now calculate forward_peak_return for EVERY recorded state in the trajectory
                # forward_peak_return = max future high (for long) from that point until EOD.
                # Actually, using 15m high/low for simplicity of peak calculation
                for i, state_dict in enumerate(t['trajectory']):
                    state_time = state_dict['time']
                    state_close = state_dict['close']
                    rec = state_dict['record']
                    
                    # Look ahead from this bar to the end of the trajectory
                    max_future_pnl = 0.0
                    for future_state in t['trajectory'][i:]:
                        f_bar_15m = ticker_bar_map[ticker].get(future_state['time'])
                        if f_bar_15m:
                            if t['side'] == 'LONG':
                                f_ret = (f_bar_15m['high'] / state_close) - 1.0
                            else:
                                f_ret = 1.0 - (f_bar_15m['low'] / state_close)
                            if f_ret > max_future_pnl:
                                max_future_pnl = f_ret
                                
                    rec['forward_peak_return'] = max_future_pnl
                    dataset_records.append(rec)
            else:
                t['bars_held'] += 1
                still_active.append(t)
                
        active_trades = still_active
        
        # Entries
        if T in dict_15m:
            active_tickers = {t['ticker'] for t in active_trades}
            top_longs = sorted(dict_15m[T], key=lambda x: x['long_rank'])
            top_shorts = sorted(dict_15m[T], key=lambda x: x['short_rank'])
            
            for x in top_longs:
                if x['long_rank'] <= 2 and x['long_conv'] >= x['long_conv_daily_max']:
                    ticker = x['ticker']
                    if ticker not in active_tickers and daily_count[date_str] < 4:
                        active_trades.append({
                            'ticker': ticker, 'side': 'LONG', 'entry_price': x['close'], 'entry_time': T,
                            'entry_conv': x['long_conv'], 'entry_rank': x['long_rank'],
                            'bars_held': 0, 'trajectory': []
                        })
                        active_tickers.add(ticker)
                        daily_count[date_str] += 1
            for x in top_shorts:
                if x['short_rank'] <= 2 and x['short_conv'] >= x['short_conv_daily_max']:
                    ticker = x['ticker']
                    if ticker not in active_tickers and daily_count[date_str] < 4:
                        active_trades.append({
                            'ticker': ticker, 'side': 'SHORT', 'entry_price': x['close'], 'entry_time': T,
                            'entry_conv': x['short_conv'], 'entry_rank': x['short_rank'],
                            'bars_held': 0, 'trajectory': []
                        })
                        active_tickers.add(ticker)
                        daily_count[date_str] += 1

    df_dataset = pd.DataFrame(dataset_records)
    print(f"\nExtracted {len(df_dataset)} intra-trade state records.")
    
    out_path = "data/dynamic_exit_dataset.csv"
    df_dataset.to_csv(out_path, index=False)
    print(f"Dataset saved to {out_path}")

if __name__ == "__main__":
    main()

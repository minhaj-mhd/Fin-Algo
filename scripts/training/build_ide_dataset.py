"""
scripts/build_ide_dataset.py
Generates a dataset for the Intelligent Decision Engine (IDE).
Simulates Strategy 22. For each triggered trade, extracts t=0 and t=15m/30m features,
and traces 5m bars to label it as 1 (hits 1.0% profit before 0.5% loss) or 0 (hits 0.5% loss or EOD close < entry).
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
PROFIT_TARGET = 0.010 # 1.0%
STOP_LOSS = 0.005     # 0.5%

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
    
    # Store raw features too
    for col in feature_cols:
        if col not in df_tf.columns:
            df_tf[col] = X[:, feature_cols.index(col)]
            
    df_tf = df_tf.sort_values(['Ticker', 'DateTime']).reset_index(drop=True)
    return df_tf, feature_cols

def main():
    print("=" * 75)
    print("BUILDING IDE DATASET (Target: 1.0%, Stop: 0.5%)")
    print("=" * 75)

    # 1. LOAD CACHED 5M DATA
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
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close'])
                }
            dict_5m_candles[ticker] = ticker_map

    # 2. LOAD & PREPARE DATAFRAMES
    df_daily_raw = load_and_filter_csv("data/ranking_data_upstox_daily_5y.csv", ["2026-04", "2026-05"])
    df_15m_raw = load_and_filter_csv("data/ranking_data_upstox_15min_1y.csv", [TEST_MONTH])

    df_daily, _ = predict_timeframe_scores(df_daily_raw, "Daily", "models/daily_xgb/metadata.json", "models/daily_xgb/xgb_long_model.json", "models/daily_xgb/xgb_short_model.json", "models/daily_xgb/scaler.pkl")
    df_15m, raw_features = predict_timeframe_scores(df_15m_raw, "15M", "models/v1_15min/metadata.json", "models/v1_15min/xgb_long_model.json", "models/v1_15min/xgb_short_model.json", "models/v1_15min/scaler.pkl")

    common_tickers = sorted(list(set(df_daily['Ticker']) & set(df_15m['Ticker'])))
    print(f"\nCommon Tickers: {len(common_tickers)}")

    df_daily = df_daily[df_daily['Ticker'].isin(common_tickers)].copy()
    df_15m = df_15m[df_15m['Ticker'].isin(common_tickers)].copy()

    df_15m['Date'] = df_15m['DateTime'].str[:10]
    df_15m['long_conv_daily_max'] = df_15m.groupby(['Date', 'Ticker'])['long_conv'].cummax()
    df_15m['short_conv_daily_max'] = df_15m.groupby(['Date', 'Ticker'])['short_conv'].cummax()

    ticker_bar_map = {t: {} for t in common_tickers}
    for idx, row in df_15m.iterrows():
        t = row['Ticker']
        feature_dict = {f: float(row[f]) for f in raw_features if pd.notnull(row[f])}
        ticker_bar_map[t][row['DateTime']] = {
            'ticker': t, 'datetime': row['DateTime'], 'date': row['Date'],
            'open': float(row['Open']), 'high': float(row['High']), 'low': float(row['Low']), 'close': float(row['Close']),
            'long_conv': float(row['long_conv']), 'short_conv': float(row['short_conv']),
            'long_rank': int(row['long_rank']), 'short_rank': int(row['short_rank']),
            'long_conv_daily_max': float(row['long_conv_daily_max']), 'short_conv_daily_max': float(row['short_conv_daily_max']),
            'features': feature_dict
        }

    unique_15m_times = sorted(df_15m['DateTime'].unique())
    dict_15m = {}
    for dt, group in df_15m.groupby('DateTime'):
        dict_15m[dt] = []
        for _, row in group.iterrows():
            t = row['Ticker']
            if t in ticker_bar_map and dt in ticker_bar_map[t]:
                dict_15m[dt].append(ticker_bar_map[t][dt])

    # ============================================================
    # SIMULATION FUNCTION FOR DATA EXTRACTION
    # ============================================================
    active_trades = []
    daily_count = {}
    dataset_records = []
    
    print("\nSimulating Strategy 22 to extract features and labels...")
    for bar_idx, T in enumerate(tqdm(unique_15m_times)):
        date_str = T[:10]
        time_str = T[11:16]
        is_last_bar = (bar_idx == len(unique_15m_times) - 1) or (unique_15m_times[bar_idx + 1][:10] != date_str) or (time_str == "15:15")
        
        if date_str not in daily_count:
            daily_count[date_str] = 0
            
        # --- Handle active trades (tracing to outcome) ---
        still_active = []
        for t in active_trades:
            ticker = t['ticker']
            
            # Check 5m bars since last check
            t_dt = pd.to_datetime(T)
            sub_5m_times = [t_dt, t_dt + timedelta(minutes=5), t_dt + timedelta(minutes=10)]
            
            exit_triggered = False
            outcome = 0
            
            # Calculate PnL on 5m path
            for t5 in sub_5m_times:
                if exit_triggered: break
                bar_5m = dict_5m_candles.get(ticker, {}).get(t5)
                if not bar_5m: continue
                
                # Update MFE/MAE
                if t['side'] == 'LONG':
                    high_ret = (bar_5m['high'] / t['entry_price']) - 1.0
                    low_ret = (bar_5m['low'] / t['entry_price']) - 1.0
                    if high_ret > t['mfe']: t['mfe'] = high_ret
                    if low_ret < t['mae']: t['mae'] = low_ret
                else:
                    high_ret = 1.0 - (bar_5m['low'] / t['entry_price'])
                    low_ret = 1.0 - (bar_5m['high'] / t['entry_price'])
                    if high_ret > t['mfe']: t['mfe'] = high_ret
                    if low_ret < t['mae']: t['mae'] = low_ret
                    
                # Check hits
                if t['side'] == 'LONG':
                    if bar_5m['high'] >= t['entry_price'] * (1.0 + PROFIT_TARGET):
                        exit_triggered = True
                        outcome = 1
                    elif bar_5m['low'] <= t['entry_price'] * (1.0 - STOP_LOSS):
                        exit_triggered = True
                        outcome = 0
                else:
                    if bar_5m['low'] <= t['entry_price'] * (1.0 - PROFIT_TARGET):
                        exit_triggered = True
                        outcome = 1
                    elif bar_5m['high'] >= t['entry_price'] * (1.0 + STOP_LOSS):
                        exit_triggered = True
                        outcome = 0
            
            if not exit_triggered and is_last_bar:
                # EOD Force Close
                exit_triggered = True
                bar_15m = ticker_bar_map[ticker].get(T)
                if bar_15m:
                    final_ret = (bar_15m['close'] / t['entry_price']) - 1.0 if t['side'] == 'LONG' else 1.0 - (bar_15m['close'] / t['entry_price'])
                    outcome = 1 if final_ret > 0 else 0
                else:
                    outcome = 0
                    
            if not exit_triggered:
                # Capture t+1 (15m) and t+2 (30m) snapshot if available
                bar_15m = ticker_bar_map[ticker].get(T)
                if bar_15m:
                    if t['bars_held'] == 0:
                        t['t1_conv'] = bar_15m['long_conv'] if t['side'] == 'LONG' else bar_15m['short_conv']
                        t['t1_rank'] = bar_15m['long_rank'] if t['side'] == 'LONG' else bar_15m['short_rank']
                        t['t1_mfe'] = t['mfe']
                        t['t1_mae'] = t['mae']
                    elif t['bars_held'] == 1:
                        t['t2_conv'] = bar_15m['long_conv'] if t['side'] == 'LONG' else bar_15m['short_conv']
                        t['t2_rank'] = bar_15m['long_rank'] if t['side'] == 'LONG' else bar_15m['short_rank']
                        t['t2_mfe'] = t['mfe']
                        t['t2_mae'] = t['mae']
                        
                t['bars_held'] += 1
                still_active.append(t)
            else:
                # Trade finished, record it
                record = {
                    'ticker': t['ticker'],
                    'entry_time': t['entry_time'],
                    'side': 1 if t['side'] == 'LONG' else 0,
                    't0_conv': t['t0_conv'],
                    't0_rank': t['t0_rank'],
                    't1_conv_delta': t.get('t1_conv', t['t0_conv']) - t['t0_conv'],
                    't1_rank_delta': t.get('t1_rank', t['t0_rank']) - t['t0_rank'],
                    't1_mfe': t.get('t1_mfe', 0.0),
                    't1_mae': t.get('t1_mae', 0.0),
                    't2_conv_delta': t.get('t2_conv', t.get('t1_conv', t['t0_conv'])) - t['t0_conv'],
                    't2_rank_delta': t.get('t2_rank', t.get('t1_rank', t['t0_rank'])) - t['t0_rank'],
                    't2_mfe': t.get('t2_mfe', 0.0),
                    't2_mae': t.get('t2_mae', 0.0),
                    'label': outcome
                }
                # Add raw t0 features
                for k, v in t['t0_features'].items():
                    record[f'feat_{k}'] = v
                dataset_records.append(record)
                
        active_trades = still_active
        
        # --- Entries section ---
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
                            'bars_held': 0, 'mfe': 0.0, 'mae': 0.0,
                            't0_conv': x['long_conv'], 't0_rank': x['long_rank'], 't0_features': x['features']
                        })
                        active_tickers.add(ticker)
                        daily_count[date_str] += 1
            for x in top_shorts:
                if x['short_rank'] <= 2 and x['short_conv'] >= x['short_conv_daily_max']:
                    ticker = x['ticker']
                    if ticker not in active_tickers and daily_count[date_str] < 4:
                        active_trades.append({
                            'ticker': ticker, 'side': 'SHORT', 'entry_price': x['close'], 'entry_time': T,
                            'bars_held': 0, 'mfe': 0.0, 'mae': 0.0,
                            't0_conv': x['short_conv'], 't0_rank': x['short_rank'], 't0_features': x['features']
                        })
                        active_tickers.add(ticker)
                        daily_count[date_str] += 1
                        
    # Save dataset
    df_dataset = pd.DataFrame(dataset_records)
    print(f"\nExtracted {len(df_dataset)} records.")
    print(df_dataset['label'].value_counts())
    
    out_path = "data/ide_dataset.csv"
    df_dataset.to_csv(out_path, index=False)
    print(f"Dataset saved to {out_path}")

if __name__ == "__main__":
    main()

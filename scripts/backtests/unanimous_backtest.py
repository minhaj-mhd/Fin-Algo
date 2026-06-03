import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm
from datetime import datetime
import warnings
from pandas.errors import PerformanceWarning
warnings.filterwarnings('ignore', category=PerformanceWarning)

sys.path.append(os.getcwd())

# ========================================
# Helper: Memory-Optimized Loading
# ========================================
def load_and_filter_csv(path, date_prefixes):
    print(f"Loading and filtering {path} for dates {date_prefixes}...")
    chunks = []
    for chunk in pd.read_csv(path, chunksize=100000):
        mask = chunk['DateTime'].str.startswith(tuple(date_prefixes))
        filtered = chunk[mask]
        if not filtered.empty:
            chunks.append(filtered)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

# ========================================
# Helper: Predict Scores
# ========================================
def predict_timeframe_scores(df_tf, model_key, meta_path, long_model_path, short_model_path, scaler_path=None):
    print(f"Scoring {model_key} dataset...")
    df_tf = df_tf.copy()
    
    with open(meta_path) as f:
        meta = json.load(f)
    feature_cols = meta["features"]
    
    missing_cols = [c for c in feature_cols if c not in df_tf.columns]
    if missing_cols:
        for col in missing_cols:
            df_tf[col] = 0.0
            
    bst_long = xgb.Booster()
    bst_long.load_model(long_model_path)
    bst_long.set_param({'device': 'cpu'})
    
    bst_short = xgb.Booster()
    bst_short.load_model(short_model_path)
    bst_short.set_param({'device': 'cpu'})
    
    scaler = None
    if scaler_path and os.path.exists(scaler_path):
        with open(scaler_path, "rb") as sf:
            scaler = pickle.load(sf)
            
    X = df_tf[feature_cols].values
    X_clean = np.nan_to_num(X)
    
    if scaler is not None and hasattr(scaler, 'scale_') and scaler.scale_ is not None:
        X_final = scaler.transform(X_clean)
    else:
        X_final = X_clean
        
    dmat = xgb.DMatrix(X_final, feature_names=feature_cols)
    df_tf['long_score'] = bst_long.predict(dmat)
    df_tf['short_score'] = bst_short.predict(dmat)
    
    df_tf['long_conv'] = df_tf['long_score'] - df_tf['short_score']
    df_tf['short_conv'] = df_tf['short_score'] - df_tf['long_score']
    
    df_tf = df_tf.sort_values(['Ticker', 'DateTime']).reset_index(drop=True)
    return df_tf

# ========================================
# Helper: Align Timeframes
# ========================================
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

def find_yesterday_daily_date(trading_date, sorted_daily_dates):
    for d in reversed(sorted_daily_dates):
        if d < trading_date:
            return d
    return None

def run_backtest():
    # Use the Monday date available in the dataset (2026-05-25)
    # To have the previous trading day's daily score, we also load Friday "2026-05-22"
    test_dates = ["2026-05-22", "2026-05-25"]
    
    print("Loading data...")
    df_daily = load_and_filter_csv("data/ranking_data_upstox_daily_5y.csv", test_dates)
    df_1h = load_and_filter_csv("data/ranking_data_upstox_3y.csv", test_dates)
    df_30m = load_and_filter_csv("data/ranking_data_upstox_30min_1y.csv", test_dates)
    df_15m = load_and_filter_csv("data/ranking_data_upstox_15min_1y.csv", test_dates)
    
    if df_15m.empty:
        print("No data found for the test dates.")
        return
        
    # Get the actual target date (the last date in the 15m dataset)
    target_date = sorted(df_15m['DateTime'].str[:10].unique())[-1]
    print(f"Target Date for Backtest: {target_date}")

    df_daily = predict_timeframe_scores(df_daily, "Daily", "models/daily_xgb_v2/metadata.json", "models/daily_xgb_v2/xgb_long_model.json", "models/daily_xgb_v2/xgb_short_model.json", "models/daily_xgb_v2/scaler.pkl")
    df_1h = predict_timeframe_scores(df_1h, "1H", "models/v8_upstox_3y/metadata.json", "models/v8_upstox_3y/xgb_long_model.json", "models/v8_upstox_3y/xgb_short_model.json", "models/scaler.pkl")
    df_30m = predict_timeframe_scores(df_30m, "30M", "models/v1_30min/metadata.json", "models/v1_30min/xgb_long_model.json", "models/v1_30min/xgb_short_model.json", "models/v1_30min/scaler.pkl")
    df_15m = predict_timeframe_scores(df_15m, "15M", "models/v1_15min/metadata.json", "models/v1_15min/xgb_long_model.json", "models/v1_15min/xgb_short_model.json", "models/v1_15min/scaler.pkl")

    df_15m['Date'] = df_15m['DateTime'].str[:10]
    sorted_daily_dates = sorted(df_daily['DateTime'].str[:10].unique())
    
    # Dicts for fast alignment
    dict_daily = {d: g.set_index('Ticker')[['long_score', 'short_score']].to_dict('index') for d, g in df_daily.groupby(df_daily['DateTime'].str[:10])}
    dict_1h = {dt: g.set_index('Ticker')[['long_score', 'short_score']].to_dict('index') for dt, g in df_1h.groupby('DateTime')}
    dict_30m = {dt: g.set_index('Ticker')[['long_score', 'short_score']].to_dict('index') for dt, g in df_30m.groupby('DateTime')}

    print("Aligning convictions...")
    a_daily_long, a_daily_short = [], []
    a_1h_long, a_1h_short = [], []
    a_30m_long, a_30m_short = [], []
    
    for _, row in df_15m.iterrows():
        t = row['Ticker']
        dt = row['DateTime']
        d = row['Date']
        
        y_date = find_yesterday_daily_date(d, sorted_daily_dates)
        d_val = dict_daily.get(y_date, {}).get(t) if y_date else None
        a_daily_long.append(d_val['long_score'] if d_val else 0.0)
        a_daily_short.append(d_val['short_score'] if d_val else 0.0)
        
        t_1h, t_30m = align_timeframes(dt)
        
        h_val = dict_1h.get(t_1h, {}).get(t) if t_1h else None
        a_1h_long.append(h_val['long_score'] if h_val else 0.0)
        a_1h_short.append(h_val['short_score'] if h_val else 0.0)
        
        m_val = dict_30m.get(t_30m, {}).get(t) if t_30m else None
        a_30m_long.append(m_val['long_score'] if m_val else 0.0)
        a_30m_short.append(m_val['short_score'] if m_val else 0.0)

    df_15m['daily_long_conv'] = a_daily_long
    df_15m['daily_short_conv'] = a_daily_short
    df_15m['h1_long_conv'] = a_1h_long
    df_15m['h1_short_conv'] = a_1h_short
    df_15m['m30_long_conv'] = a_30m_long
    df_15m['m30_short_conv'] = a_30m_short

    # Add future prices for returns calculation
    print("Calculating forward returns...")
    df_15m = df_15m.sort_values(['Ticker', 'DateTime']).reset_index(drop=True)
    df_15m['close_plus_1'] = df_15m.groupby('Ticker')['Close'].shift(-1)
    df_15m['close_plus_2'] = df_15m.groupby('Ticker')['Close'].shift(-2)
    
    df_15m['ret_15m_long'] = (df_15m['close_plus_1'] - df_15m['Close']) / df_15m['Close']
    df_15m['ret_15m_short'] = (df_15m['Close'] - df_15m['close_plus_1']) / df_15m['Close']
    
    df_15m['ret_30m_long'] = (df_15m['close_plus_2'] - df_15m['Close']) / df_15m['Close']
    df_15m['ret_30m_short'] = (df_15m['Close'] - df_15m['close_plus_2']) / df_15m['Close']

    # Filter for target date
    df_test = df_15m[df_15m['Date'] == target_date].copy()
    
    df_test['long_rank'] = df_test.groupby('DateTime')['long_score'].rank(ascending=False)
    df_test['short_rank'] = df_test.groupby('DateTime')['short_score'].rank(ascending=False)
    
    # Conditions using individual scores and ranks
    # LONG: 15m_long_rank <= 5, all scores > 0
    cond_long = (
        (df_test['long_rank'] <= 5) &
        (df_test['long_score'] > 0) &
        (df_test['m30_long_conv'] > 0) &
        (df_test['h1_long_conv'] > 0) &
        (df_test['daily_long_conv'] > 0)
    )
    
    # SHORT: 15m_short_rank <= 5, all scores > 0
    cond_short = (
        (df_test['short_rank'] <= 5) &
        (df_test['short_score'] > 0) &
        (df_test['m30_short_conv'] > 0) &
        (df_test['h1_short_conv'] > 0) &
        (df_test['daily_short_conv'] > 0)
    )
    
    long_trades = df_test[cond_long].dropna(subset=['close_plus_2'])
    short_trades = df_test[cond_short].dropna(subset=['close_plus_2'])
    
    def calc_stats(trades_df, is_long=True):
        if trades_df.empty:
            return {"trades": 0, "winrate_15m": 0.0, "ret_15m": 0.0, "winrate_30m": 0.0, "ret_30m": 0.0}
            
        r15_col = 'ret_15m_long' if is_long else 'ret_15m_short'
        r30_col = 'ret_30m_long' if is_long else 'ret_30m_short'
        
        # Deduct transaction cost (e.g. 0.06% round trip)
        tc = 0.0006
        r15 = trades_df[r15_col] - tc
        r30 = trades_df[r30_col] - tc
        
        return {
            "trades": len(trades_df),
            "winrate_15m": (r15 > 0).mean() * 100,
            "ret_15m": r15.sum() * 100, # total return in %
            "mean_ret_15m": r15.mean() * 100,
            "winrate_30m": (r30 > 0).mean() * 100,
            "ret_30m": r30.sum() * 100,
            "mean_ret_30m": r30.mean() * 100
        }

    long_stats = calc_stats(long_trades, True)
    short_stats = calc_stats(short_trades, False)
    
    print("\n" + "="*60)
    print(f"CASCADING CONVICTION BACKTEST RESULTS ({target_date})")
    print("="*60)
    
    print("\n--- LONG TRADES ---")
    print(f"Total Trades: {long_stats['trades']}")
    if long_stats['trades'] > 0:
        print(f"15 Min Hold -> WinRate: {long_stats['winrate_15m']:.2f}% | Total Net Return: {long_stats['ret_15m']:.2f}% | Avg Return: {long_stats['mean_ret_15m']:.3f}%")
        print(f"30 Min Hold -> WinRate: {long_stats['winrate_30m']:.2f}% | Total Net Return: {long_stats['ret_30m']:.2f}% | Avg Return: {long_stats['mean_ret_30m']:.3f}%")
        
    print("\n--- SHORT TRADES ---")
    print(f"Total Trades: {short_stats['trades']}")
    if short_stats['trades'] > 0:
        print(f"15 Min Hold -> WinRate: {short_stats['winrate_15m']:.2f}% | Total Net Return: {short_stats['ret_15m']:.2f}% | Avg Return: {short_stats['mean_ret_15m']:.3f}%")
        print(f"30 Min Hold -> WinRate: {short_stats['winrate_30m']:.2f}% | Total Net Return: {short_stats['ret_30m']:.2f}% | Avg Return: {short_stats['mean_ret_30m']:.3f}%")
        
    print("\n--- AGGREGATE ---")
    total_trades = long_stats['trades'] + short_stats['trades']
    if total_trades > 0:
        total_15m = long_stats['ret_15m'] + short_stats['ret_15m']
        total_30m = long_stats['ret_30m'] + short_stats['ret_30m']
        print(f"Total Trades: {total_trades}")
        print(f"Total Net Return (15m): {total_15m:.2f}%")
        print(f"Total Net Return (30m): {total_30m:.2f}%")

if __name__ == "__main__":
    run_backtest()

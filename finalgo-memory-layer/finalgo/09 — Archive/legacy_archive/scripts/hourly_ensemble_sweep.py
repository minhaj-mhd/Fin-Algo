"""
hourly_ensemble_sweep.py — Optimize ensembles strictly on hourly candle boundaries

At each hourly candle boundary (10:30, 11:30, 12:30, 13:30, 14:30, 15:30):
  - Align 1H, 30M, and 15M scores
  - Compute blended conviction: w1*1H + w2*30M + w3*15M
  - Select Top-K Longs and Shorts
  - Evaluate on a 1-hour holding period (next hourly candle close)
"""

import os, sys, json, pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm
from datetime import datetime
from itertools import product

sys.path.append(os.getcwd())

TEST_MONTH = "2026-05"
TRANSACTION_COST_PCT = 0.03

print("=" * 80)
print("HOURLY BOUNDARY ENSEMBLE SWEEP")
print(f"Finding optimal hourly-aligned ensemble configs on {TEST_MONTH}")
print("=" * 80)

def predict_timeframe_scores(df_tf, model_key, meta_path, long_model_path, short_model_path, scaler_path):
    with open(meta_path) as f:
        meta = json.load(f)
    feature_cols = meta["features"]
    missing_cols = [c for c in feature_cols if c not in df_tf.columns]
    for col in missing_cols:
        df_tf[col] = 0.0
    bst_long = xgb.Booster(); bst_long.load_model(long_model_path)
    bst_short = xgb.Booster(); bst_short.load_model(short_model_path)
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
    df_tf['long_rank'] = df_tf.groupby('Query_ID')['long_conv'].rank(ascending=False)
    df_tf['short_rank'] = df_tf.groupby('Query_ID')['short_conv'].rank(ascending=False)
    print(f"  {model_key}: {df_tf.shape[0]:,} rows scored")
    return df_tf

print("\nLoading datasets...")
df_1h_all = pd.read_csv("data/ranking_data_upstox_3y.csv")
df_30m_all = pd.read_csv("data/ranking_data_upstox_30min_1y.csv")
df_15m_all = pd.read_csv("data/ranking_data_upstox_15min_1y.csv")

df_1h = df_1h_all[df_1h_all['DateTime'].str.startswith(TEST_MONTH)].copy()
df_30m = df_30m_all[df_30m_all['DateTime'].str.startswith(TEST_MONTH)].copy()
df_15m = df_15m_all[df_15m_all['DateTime'].str.startswith(TEST_MONTH)].copy()

t_common = sorted(set(df_1h['Ticker'].unique()) & set(df_30m['Ticker'].unique()) & set(df_15m['Ticker'].unique()))
df_1h = df_1h[df_1h['Ticker'].isin(t_common)].copy()
df_30m = df_30m[df_30m['Ticker'].isin(t_common)].copy()
df_15m = df_15m[df_15m['Ticker'].isin(t_common)].copy()
print(f"  Common tickers: {len(t_common)}")

print("\nScoring models...")
df_1h = predict_timeframe_scores(df_1h, "1H", "models/v8_upstox_3y/metadata.json",
    "models/v8_upstox_3y/xgb_long_model.json", "models/v8_upstox_3y/xgb_short_model.json", "models/scaler.pkl")
df_30m = predict_timeframe_scores(df_30m, "30M", "models/v1_30min/metadata.json",
    "models/v1_30min/xgb_long_model.json", "models/v1_30min/xgb_short_model.json", "models/v1_30min/scaler.pkl")
df_15m = predict_timeframe_scores(df_15m, "15M", "models/v1_15min/metadata.json",
    "models/v1_15min/xgb_long_model.json", "models/v1_15min/xgb_short_model.json", "models/v1_15min/scaler.pkl")

# Build dictionaries
print("\nBuilding lookup indices...")
def build_dict(df, include_prices=False):
    d = {}
    for qid, q_df in df.groupby('DateTime'):
        d[qid] = {}
        for _, row in q_df.iterrows():
            entry = {
                'long_rank': int(row['long_rank']),
                'short_rank': int(row['short_rank']),
                'long_conv': float(row['long_conv']),
                'short_conv': float(row['short_conv']),
            }
            if include_prices:
                h, l, c, o = row['High'], row['Low'], row['Close'], row['Open']
                entry['ibs_raw'] = float((c - l) / (h - l + 1e-10))
                entry['close'] = float(c)
                entry['next_return'] = float(row['Next_Hour_Return']) if 'Next_Hour_Return' in row.index else 0.0
            d[qid][row['Ticker']] = entry
    return d

dict_1h = build_dict(df_1h, include_prices=True)
dict_30m = build_dict(df_30m)
dict_15m = build_dict(df_15m)

def align_timeframes_at_hour(t_1h_str):
    """
    Given an hourly candle boundary close time (e.g. 10:30, 11:30... open time 09:30, 10:30...),
    we want to align the exact same closed bar in 30M and 15M.
    Hourly open boundaries: 09:30:00, 10:30:00, 11:30:00, 12:30:00, 13:30:00, 14:30:00
    These bars close 1 hour later: 10:30:00, 11:30:00, 12:30:00, 13:30:00, 14:30:00, 15:30:00
    At 10:30:00 (when 09:30 1H candle closes):
      - Latest closed 30M candle is 10:00:00 (which closed at 10:30:00)
      - Latest closed 15M candle is 10:15:00 (which closed at 10:30:00)
    """
    date_part = t_1h_str[:10]
    h_str, m_str = t_1h_str[11:13], t_1h_str[14:16]
    h, m = int(h_str), int(m_str)
    
    # The 1H open timestamp is t_1h_str.
    # The close time is 1 hour later. At that close time, the latest closed 30M and 15M bars are:
    # 30M bar starting 15 min after 1H bar start (closes at +45 min, i.e. 15 min before 1H close).
    # 15M bar starting 45 min after 1H bar start (closes at +60 min, i.e. exactly at 1H close).
    t_30m_start = h * 60 + m + 15
    h_30, m_30 = divmod(t_30m_start, 60)
    t_30m = f"{date_part} {h_30:02d}:{m_30:02d}:00+05:30"
    
    t_15m_start = h * 60 + m + 45
    h_15, m_15 = divmod(t_15m_start, 60)
    t_15m = f"{date_part} {h_15:02d}:{m_15:02d}:00+05:30"
    
    return t_30m, t_15m

# Test various weight combinations
weight_options = [
    (1.0, 0.0, 0.0), # 1H only (control)
    (0.6, 0.3, 0.1),
    (0.5, 0.3, 0.2),
    (0.4, 0.4, 0.2),
    (0.3, 0.4, 0.3),
    (0.4, 0.3, 0.3),
    (0.2, 0.4, 0.4),
    (0.1, 0.3, 0.6),
]

configs = []
for w1, w2, w3 in weight_options:
    for topk in [3, 5, 10]:
        configs.append({
            'name': f'Hourly_Blend_w_{w1}_{w2}_{w3}_top{topk}',
            'w_1h': w1, 'w_30m': w2, 'w_15m': w3,
            'topk': topk
        })

df_1h['Date'] = df_1h['DateTime'].str[:10]
unique_dates = sorted(df_1h['Date'].unique())
timestamps_by_date_1h = {}
for date in unique_dates:
    timestamps_by_date_1h[date] = sorted(df_1h[df_1h['Date'] == date]['DateTime'].unique())

num_trading_days = len(unique_dates)
results = []

for cfg in tqdm(configs, desc="Hourly Sweep"):
    trades = []
    
    for date in unique_dates:
        for T in timestamps_by_date_1h[date]:
            t_30m, t_15m = align_timeframes_at_hour(T)
            
            if T not in dict_1h:
                continue
            if t_30m not in dict_30m:
                continue
            if t_15m not in dict_15m:
                continue
                
            picks_1h = dict_1h[T]
            picks_30m = dict_30m[t_30m]
            picks_15m = dict_15m[t_15m]
            
            common_t = [t for t in picks_1h if t in picks_30m and t in picks_15m]
            if not common_t:
                continue
                
            blend_longs = {}
            blend_shorts = {}
            for sym in common_t:
                blend_longs[sym] = (
                    cfg['w_1h'] * picks_1h[sym]['long_conv'] +
                    cfg['w_30m'] * picks_30m[sym]['long_conv'] +
                    cfg['w_15m'] * picks_15m[sym]['long_conv']
                )
                blend_shorts[sym] = (
                    cfg['w_1h'] * picks_1h[sym]['short_conv'] +
                    cfg['w_30m'] * picks_30m[sym]['short_conv'] +
                    cfg['w_15m'] * picks_15m[sym]['short_conv']
                )
            
            top_longs = sorted(blend_longs.keys(), key=lambda x: blend_longs[x], reverse=True)[:cfg['topk']]
            top_shorts = sorted(blend_shorts.keys(), key=lambda x: blend_shorts[x], reverse=True)[:cfg['topk']]
            
            # Evaluate using 1H Next_Hour_Return from the hourly model dict (close-to-close)
            for sym in top_longs:
                ret = picks_1h[sym]['next_return']
                net = ret - TRANSACTION_COST_PCT / 100
                trades.append({'date': date, 'side': 'LONG', 'ticker': sym, 'net_return': net, 'is_win': net > 0})
                
            for sym in top_shorts:
                ret = -picks_1h[sym]['next_return']
                net = ret - TRANSACTION_COST_PCT / 100
                trades.append({'date': date, 'side': 'SHORT', 'ticker': sym, 'net_return': net, 'is_win': net > 0})

    if not trades:
        results.append({
            'name': cfg['name'],
            'w_1h': cfg.get('w_1h', 0),
            'w_30m': cfg.get('w_30m', 0),
            'w_15m': cfg.get('w_15m', 0),
            'topk': cfg.get('topk', 0),
            'total_trades': 0,
            'trades_per_day': 0.0,
            'win_rate': 0.0,
            'long_wr': 0.0,
            'short_wr': 0.0,
            'total_return': 0.0,
            'avg_return': 0.0,
            'profit_factor': 0.0,
            'green_days': "0/0"
        })
        continue
        
    df_t = pd.DataFrame(trades)
    total = len(df_t)
    wins = int(df_t['is_win'].sum())
    wr = wins / total
    total_ret = df_t['net_return'].sum()
    avg_ret = df_t['net_return'].mean()
    
    winners = df_t[df_t['is_win']]
    losers = df_t[~df_t['is_win']]
    pf = abs(winners['net_return'].sum() / losers['net_return'].sum()) if len(losers) > 0 and losers['net_return'].sum() != 0 else 99.0
    
    longs = df_t[df_t['side'] == 'LONG']
    shorts = df_t[df_t['side'] == 'SHORT']
    long_wr = longs['is_win'].mean() if len(longs) > 0 else 0
    short_wr = shorts['is_win'].mean() if len(shorts) > 0 else 0
    
    daily = df_t.groupby('date').agg(
        trades=('net_return', 'count'),
        pnl=('net_return', 'sum')
    )
    green_days = (daily['pnl'] > 0).sum()
    
    results.append({
        'name': cfg['name'],
        'w_1h': cfg['w_1h'],
        'w_30m': cfg['w_30m'],
        'w_15m': cfg['w_15m'],
        'topk': cfg['topk'],
        'total_trades': total,
        'trades_per_day': total / num_trading_days,
        'win_rate': wr,
        'long_wr': long_wr,
        'short_wr': short_wr,
        'total_return': total_ret,
        'avg_return': avg_ret,
        'profit_factor': pf,
        'green_days': f"{green_days}/{len(daily)}"
    })

df_r = pd.DataFrame(results)
df_r.to_csv("data/hourly_ensemble_results.csv", index=False)

print("\n" + "=" * 120)
print("HOURLY ENSEMBLE SWEEP RESULTS (sorted by Profit Factor)")
print("=" * 120)
df_r_sorted = df_r.sort_values('profit_factor', ascending=False)
print(f"{'Config Name':<50} {'Sig/Day':>8} {'WR':>6} {'LongWR':>7} {'ShortWR':>8} {'TotRet':>9} {'PF':>6} {'Green':>7}")
print(f"{'-'*50} {'-'*8} {'-'*6} {'-'*7} {'-'*8} {'-'*9} {'-'*6} {'-'*7}")
for _, row in df_r_sorted.iterrows():
    print(f"{row['name']:<50} {row['trades_per_day']:>7.1f} "
          f"{row['win_rate']:>5.1%} {row['long_wr']:>6.1%} {row['short_wr']:>7.1%} "
          f"{row['total_return']*100:>+8.2f}% "
          f"{row['profit_factor']:>5.2f} {row['green_days']:>7}")

print(f"\n[DONE] Hourly ensemble sweep results saved to data/hourly_ensemble_results.csv")

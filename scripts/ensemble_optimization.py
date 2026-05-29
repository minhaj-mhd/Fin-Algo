"""
ensemble_optimization.py — Find the ultimate ensemble configuration of the 3 models

Tests two main ensembling paradigms:
1. Linear Conviction Blending:
   Combined Conv = w1 * 1H_Conv + w2 * 30M_Conv + w3 * 15M_Conv
2. Soft Confluence / Majority Voting:
   Signal if 1H confirms and (30M or 15M confirms) with wider thresholds.

Evaluates on:
- 15-minute hold (1 bar)
- 1-hour hold (4 bars)
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
print("ENSEMBLE HYPERPARAMETER OPTIMIZATION ENGINE")
print(f"Finding the optimal blend of 1H, 30M, and 15M models for {TEST_MONTH}")
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
                entry['next_return'] = float(row['Next_15Min_Return']) if 'Next_15Min_Return' in row.index else 0.0
            d[qid][row['Ticker']] = entry
    return d

dict_1h = build_dict(df_1h)
dict_30m = build_dict(df_30m)
dict_15m = build_dict(df_15m, include_prices=True)

# Build price series for 1-hour hold from 15M dataset (4 bars)
price_series = {}
df_15m_sorted = df_15m.sort_values('DateTime')
for _, row in df_15m_sorted.iterrows():
    ticker = row['Ticker']
    if ticker not in price_series:
        price_series[ticker] = []
    price_series[ticker].append((row['DateTime'], float(row['Close'])))

price_idx = {}
for ticker, series in price_series.items():
    price_idx[ticker] = {dt: i for i, (dt, c) in enumerate(series)}

def align_timeframes(t_str):
    date_part = t_str[:10]
    h, m = int(t_str[11:13]), int(t_str[14:16])
    minutes = h * 60 + m
    
    t_1h = None
    for bar_h, bar_m in [(9,30),(10,30),(11,30),(12,30),(13,30),(14,30)]:
        if minutes >= bar_h*60 + bar_m + 60:
            t_1h = f"{date_part} {bar_h:02d}:{bar_m:02d}:00+05:30"
    
    t_30m = None
    bar_start = 9 * 60 + 15
    while bar_start < 15 * 60 + 30:
        if minutes >= bar_start + 30:
            bh, bm = divmod(bar_start, 60)
            t_30m = f"{date_part} {bh:02d}:{bm:02d}:00+05:30"
        bar_start += 30
    
    return t_1h, t_30m

# Create configurations to test
configs = []

# --- PARADIGM 1: LINEAR CONVICTION BLENDING ---
# Test various weight combinations (w1, w2, w3) summing to 1.0
weight_options = [
    (0.6, 0.3, 0.1),
    (0.5, 0.3, 0.2),
    (0.4, 0.4, 0.2),
    (0.3, 0.4, 0.3),
    (0.4, 0.3, 0.3),
    (0.2, 0.4, 0.4),
    (0.1, 0.3, 0.6),
]

for w1, w2, w3 in weight_options:
    for hold_bars in [1, 4]:
        for topk in [3, 5]:
            configs.append({
                'name': f'Blend_w_{w1}_{w2}_{w3}_top{topk}_{"15M" if hold_bars == 1 else "1H"}_Hold',
                'paradigm': 'Linear_Blend',
                'w_1h': w1, 'w_30m': w2, 'w_15m': w3,
                'topk': topk,
                'hold_bars': hold_bars,
                'use_soft': False
            })

# --- PARADIGM 2: SOFT CONFLUENCE (MAJORITY VOTING) ---
# Long if 1H ranks <= K1 AND (30M ranks <= K2 OR 15M ranks <= K3)
soft_combos = [
    (5, 15, 15),
    (5, 20, 20),
    (10, 20, 20),
    (10, 30, 30),
]

for k1, k2, k3 in soft_combos:
    for hold_bars in [1, 4]:
        configs.append({
            'name': f'Soft_Conf_{k1}_{k2}_{k3}_{"15M" if hold_bars == 1 else "1H"}_Hold',
            'paradigm': 'Soft_Confluence',
            'k1': k1, 'k2': k2, 'k3': k3,
            'hold_bars': hold_bars,
            'use_soft': True
        })

print(f"\nTotal configurations to test: {len(configs)}")

df_15m['Date'] = df_15m['DateTime'].str[:10]
unique_dates = sorted(df_15m['Date'].unique())
timestamps_by_date = {}
for date in unique_dates:
    timestamps_by_date[date] = sorted(df_15m[df_15m['Date'] == date]['DateTime'].unique())

num_trading_days = len(unique_dates)
results = []

for cfg in tqdm(configs, desc="Optimizing Ensembles"):
    trades = []
    
    for date in unique_dates:
        for T in timestamps_by_date[date]:
            t_1h, t_30m = align_timeframes(T)
            if not t_1h or t_1h not in dict_1h:
                continue
            if T not in dict_15m:
                continue
            
            picks_1h = dict_1h[t_1h]
            picks_15m = dict_15m[T]
            picks_30m = dict_30m.get(t_30m, {}) if t_30m else {}
            
            # Get intersection tickers
            common_t = [t for t in picks_1h if t in picks_15m and t in picks_30m]
            if not common_t:
                continue
            
            if cfg['use_soft']:
                # Paradigm 2: Soft Confluence
                # Long
                long_signals = []
                for sym in common_t:
                    r1h = picks_1h[sym]['long_rank']
                    r30m = picks_30m[sym]['long_rank']
                    r15m = picks_15m[sym]['long_rank']
                    if r1h <= cfg['k1'] and (r30m <= cfg['k2'] or r15m <= cfg['k3']):
                        long_signals.append(sym)
                
                for sym in long_signals:
                    if sym in price_idx and T in price_idx[sym]:
                        idx = price_idx[sym][T]
                        series = price_series[sym]
                        exit_idx = min(idx + cfg['hold_bars'], len(series) - 1)
                        if exit_idx > idx:
                            ret = (series[exit_idx][1] / series[idx][1]) - 1
                            net = ret - TRANSACTION_COST_PCT / 100
                            trades.append({'date': date, 'side': 'LONG', 'ticker': sym, 'net_return': net, 'is_win': net > 0})
                
                # Short
                short_signals = []
                for sym in common_t:
                    r1h = picks_1h[sym]['short_rank']
                    r30m = picks_30m[sym]['short_rank']
                    r15m = picks_15m[sym]['short_rank']
                    if r1h <= cfg['k1'] and (r30m <= cfg['k2'] or r15m <= cfg['k3']):
                        short_signals.append(sym)
                
                for sym in short_signals:
                    if sym in price_idx and T in price_idx[sym]:
                        idx = price_idx[sym][T]
                        series = price_series[sym]
                        exit_idx = min(idx + cfg['hold_bars'], len(series) - 1)
                        if exit_idx > idx:
                            ret = 1 - (series[exit_idx][1] / series[idx][1])
                            net = ret - TRANSACTION_COST_PCT / 100
                            trades.append({'date': date, 'side': 'SHORT', 'ticker': sym, 'net_return': net, 'is_win': net > 0})
            
            else:
                # Paradigm 1: Linear Conviction Blending
                blend_longs = {}
                blend_shorts = {}
                
                for sym in common_t:
                    long_blend = (
                        cfg['w_1h'] * picks_1h[sym]['long_conv'] +
                        cfg['w_30m'] * picks_30m[sym]['long_conv'] +
                        cfg['w_15m'] * picks_15m[sym]['long_conv']
                    )
                    short_blend = (
                        cfg['w_1h'] * picks_1h[sym]['short_conv'] +
                        cfg['w_30m'] * picks_30m[sym]['short_conv'] +
                        cfg['w_15m'] * picks_15m[sym]['short_conv']
                    )
                    blend_longs[sym] = long_blend
                    blend_shorts[sym] = short_blend
                
                # Select Top-K
                top_longs = sorted(blend_longs.keys(), key=lambda x: blend_longs[x], reverse=True)[:cfg['topk']]
                top_shorts = sorted(blend_shorts.keys(), key=lambda x: blend_shorts[x], reverse=True)[:cfg['topk']]
                
                for sym in top_longs:
                    if sym in price_idx and T in price_idx[sym]:
                        idx = price_idx[sym][T]
                        series = price_series[sym]
                        exit_idx = min(idx + cfg['hold_bars'], len(series) - 1)
                        if exit_idx > idx:
                            ret = (series[exit_idx][1] / series[idx][1]) - 1
                            net = ret - TRANSACTION_COST_PCT / 100
                            trades.append({'date': date, 'side': 'LONG', 'ticker': sym, 'net_return': net, 'is_win': net > 0})
                
                for sym in top_shorts:
                    if sym in price_idx and T in price_idx[sym]:
                        idx = price_idx[sym][T]
                        series = price_series[sym]
                        exit_idx = min(idx + cfg['hold_bars'], len(series) - 1)
                        if exit_idx > idx:
                            ret = 1 - (series[exit_idx][1] / series[idx][1])
                            net = ret - TRANSACTION_COST_PCT / 100
                            trades.append({'date': date, 'side': 'SHORT', 'ticker': sym, 'net_return': net, 'is_win': net > 0})

    if not trades:
        results.append({**cfg, 'total_trades': 0, 'trades_per_day': 0,
                        'win_rate': 0, 'total_return': 0, 'avg_return': 0,
                        'profit_factor': 0, 'long_wr': 0, 'short_wr': 0})
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
        'paradigm': cfg['paradigm'],
        'hold_bars': cfg['hold_bars'],
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
df_r.to_csv("data/ensemble_optimization_results.csv", index=False)

print("\n" + "=" * 120)
print("TOP 20 OPTIMAL ENSEMBLE CONFIGURATIONS (sorted by Profit Factor)")
print("=" * 120)
df_r_sorted = df_r.sort_values('profit_factor', ascending=False)
print(f"{'Config Name':<55} {'Sig/Day':>8} {'WR':>6} {'LongWR':>7} {'ShortWR':>8} {'TotRet':>9} {'PF':>6} {'Green':>7}")
print(f"{'-'*55} {'-'*8} {'-'*6} {'-'*7} {'-'*8} {'-'*9} {'-'*6} {'-'*7}")
for _, row in df_r_sorted.head(20).iterrows():
    print(f"{row['name']:<55} {row['trades_per_day']:>7.1f} "
          f"{row['win_rate']:>5.1%} {row['long_wr']:>6.1%} {row['short_wr']:>7.1%} "
          f"{row['total_return']*100:>+8.2f}% "
          f"{row['profit_factor']:>5.2f} {row['green_days']:>7}")

print("\n\n" + "=" * 120)
print("SWEET SPOT ENSEMBLES: Configs generating 30-80 signals/day (sorted by PF)")
print("=" * 120)
sweet = df_r[(df_r['trades_per_day'] >= 30) & (df_r['trades_per_day'] <= 80)].sort_values('profit_factor', ascending=False)
for _, row in sweet.head(15).iterrows():
    print(f"{row['name']:<55} {row['trades_per_day']:>7.1f} "
          f"{row['win_rate']:>5.1%} {row['long_wr']:>6.1%} {row['short_wr']:>7.1%} "
          f"{row['total_return']*100:>+8.2f}% "
          f"{row['profit_factor']:>5.2f} {row['green_days']:>7}")

print(f"\n[DONE] Ensemble optimization results saved to data/ensemble_optimization_results.csv")

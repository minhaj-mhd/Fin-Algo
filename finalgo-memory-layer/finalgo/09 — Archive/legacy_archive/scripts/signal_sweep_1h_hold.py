"""
signal_sweep_1h_hold.py — Find the optimal signal generation configuration with a 1-hour holding period

Tests:
  A) 1H model only (various Top-K, 1-hour hold)
  B) 1H + 30M confluence (various Top-K, 1-hour hold)
  C) 1H + 15M confluence (various Top-K, 1-hour hold)
  D) 1H + 30M + 15M confluence (various Top-K, 1-hour hold)
  E) Best combos with IBS filter (1-hour hold)
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

print("=" * 70)
print("1-HOUR HOLD SIGNAL SWEEP")
print(f"Finding optimal config for ~50 signals/day on {TEST_MONTH} with 1-Hour Hold")
print("=" * 70)

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
            d[qid][row['Ticker']] = entry
    return d

dict_1h = build_dict(df_1h, include_prices=True)
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

# 1H price series for pure 1H model
df_1h_prices = {}
for qid, q_df in df_1h.groupby('DateTime'):
    df_1h_prices[qid] = {}
    for _, row in q_df.iterrows():
        df_1h_prices[qid][row['Ticker']] = {
            'next_return': float(row['Next_Hour_Return']) if 'Next_Hour_Return' in row.index else 0.0,
            'close': float(row['Close']),
        }

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

# Configurations to test
configs = []

# Group A: 1H only (1-hour hold)
for topk in [3, 5, 10, 15, 20, 30, 50]:
    configs.append({
        'name': f'1H_only_top{topk}',
        'group': 'A_1H_Only',
        'h_topk': topk,
        'use_30m': False, 'use_15m': False,
        'm30_topk': 999, 'm15_topk': 999,
        'use_ibs': False, 'ibs_long': 1.0, 'ibs_short': 0.0,
        'eval_tf': '1h',
    })

# Group B: 1H + 30M confluence (1-hour hold, evaluated on 15m resolution with 4-bar hold)
for h_topk, m30_topk in product([5, 10, 15], [10, 20, 30]):
    configs.append({
        'name': f'1H_top{h_topk}+30M_top{m30_topk}',
        'group': 'B_1H_30M_1H_Hold',
        'h_topk': h_topk,
        'use_30m': True, 'use_15m': False,
        'm30_topk': m30_topk, 'm15_topk': 999,
        'use_ibs': False, 'ibs_long': 1.0, 'ibs_short': 0.0,
        'eval_tf': '1h_hold_15m_res',
    })

# Group C: 1H + 15M confluence (1-hour hold, evaluated on 15m resolution with 4-bar hold)
for h_topk, m15_topk in product([5, 10, 15], [10, 20, 30]):
    configs.append({
        'name': f'1H_top{h_topk}+15M_top{m15_topk}',
        'group': 'C_1H_15M_1H_Hold',
        'h_topk': h_topk,
        'use_30m': False, 'use_15m': True,
        'm30_topk': 999, 'm15_topk': m15_topk,
        'use_ibs': False, 'ibs_long': 1.0, 'ibs_short': 0.0,
        'eval_tf': '1h_hold_15m_res',
    })

# Group D: Full confluence (1H+30M+15M) (1-hour hold, 15m res)
for h_topk, m30_topk, m15_topk in product([5, 10, 15], [10, 20, 30], [10, 20, 30]):
    configs.append({
        'name': f'1H_top{h_topk}+30M_top{m30_topk}+15M_top{m15_topk}',
        'group': 'D_Full_1H_Hold',
        'h_topk': h_topk,
        'use_30m': True, 'use_15m': True,
        'm30_topk': m30_topk, 'm15_topk': m15_topk,
        'use_ibs': False, 'ibs_long': 1.0, 'ibs_short': 0.0,
        'eval_tf': '1h_hold_15m_res',
    })

df_15m['Date'] = df_15m['DateTime'].str[:10]
unique_dates = sorted(df_15m['Date'].unique())
timestamps_by_date = {}
for date in unique_dates:
    timestamps_by_date[date] = sorted(df_15m[df_15m['Date'] == date]['DateTime'].unique())

df_1h['Date'] = df_1h['DateTime'].str[:10]
unique_dates_1h = sorted(df_1h['Date'].unique())
timestamps_by_date_1h = {}
for date in unique_dates_1h:
    timestamps_by_date_1h[date] = sorted(df_1h[df_1h['Date'] == date]['DateTime'].unique())

num_trading_days = len(unique_dates)

results = []

for cfg in tqdm(configs, desc="Sweeping configs"):
    trades = []
    
    if cfg['eval_tf'] == '1h':
        # Hourly bars evaluation (1-hour hold)
        for date in unique_dates_1h:
            for T in timestamps_by_date_1h[date]:
                if T not in dict_1h:
                    continue
                picks = dict_1h[T]
                
                # Longs
                sorted_longs = sorted(picks.keys(), key=lambda x: picks[x]['long_conv'], reverse=True)[:cfg['h_topk']]
                for sym in sorted_longs:
                    if sym in df_1h_prices.get(T, {}):
                        ret = df_1h_prices[T][sym]['next_return']
                        net = ret - TRANSACTION_COST_PCT / 100
                        trades.append({'date': date, 'side': 'LONG', 'ticker': sym, 'net_return': net, 'is_win': net > 0})
                
                # Shorts
                sorted_shorts = sorted(picks.keys(), key=lambda x: picks[x]['short_conv'], reverse=True)[:cfg['h_topk']]
                for sym in sorted_shorts:
                    if sym in df_1h_prices.get(T, {}):
                        ret = -df_1h_prices[T][sym]['next_return']
                        net = ret - TRANSACTION_COST_PCT / 100
                        trades.append({'date': date, 'side': 'SHORT', 'ticker': sym, 'net_return': net, 'is_win': net > 0})
    else:
        # 15M resolution with 1-hour (4-bar) hold
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
                
                # Longs
                sorted_longs = sorted(picks_1h.keys(), key=lambda x: picks_1h[x]['long_conv'], reverse=True)[:cfg['h_topk']]
                for sym in sorted_longs:
                    if cfg['use_30m'] and (sym not in picks_30m or picks_30m[sym]['long_rank'] > cfg['m30_topk']):
                        continue
                    if cfg['use_15m'] and (sym not in picks_15m or picks_15m[sym]['long_rank'] > cfg['m15_topk']):
                        continue
                    
                    # Entry! Calculate 4-bar (1h) return
                    if sym in price_idx and T in price_idx[sym]:
                        idx = price_idx[sym][T]
                        series = price_series[sym]
                        exit_idx = min(idx + 4, len(series) - 1)
                        if exit_idx > idx:
                            ret = (series[exit_idx][1] / series[idx][1]) - 1
                            net = ret - TRANSACTION_COST_PCT / 100
                            trades.append({'date': date, 'side': 'LONG', 'ticker': sym, 'net_return': net, 'is_win': net > 0})
                
                # Shorts
                sorted_shorts = sorted(picks_1h.keys(), key=lambda x: picks_1h[x]['short_conv'], reverse=True)[:cfg['h_topk']]
                for sym in sorted_shorts:
                    if cfg['use_30m'] and (sym not in picks_30m or picks_30m[sym]['short_rank'] > cfg['m30_topk']):
                        continue
                    if cfg['use_15m'] and (sym not in picks_15m or picks_15m[sym]['short_rank'] > cfg['m15_topk']):
                        continue
                    
                    # Entry! Calculate 4-bar (1h) return
                    if sym in price_idx and T in price_idx[sym]:
                        idx = price_idx[sym][T]
                        series = price_series[sym]
                        exit_idx = min(idx + 4, len(series) - 1)
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
        'group': cfg['group'],
        'h_topk': cfg['h_topk'],
        'use_30m': cfg['use_30m'],
        'use_15m': cfg['use_15m'],
        'm30_topk': cfg['m30_topk'],
        'm15_topk': cfg['m15_topk'],
        'use_ibs': cfg['use_ibs'],
        'eval_tf': cfg['eval_tf'],
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
df_r.to_csv("data/signal_sweep_1h_hold_results.csv", index=False)

print("\n" + "=" * 120)
print("SWEET SPOT ANALYSIS: Configs generating 30-80 signals/day (1-Hour Hold)")
print("=" * 120)
sweet = df_r[(df_r['trades_per_day'] >= 30) & (df_r['trades_per_day'] <= 80)]
sweet = sweet.sort_values('profit_factor', ascending=False)

print(f"\n  {'Config':<45} {'Sig/Day':>8} {'WR':>6} {'LongWR':>7} {'ShortWR':>8} {'TotRet':>9} {'PF':>6} {'Green':>7}")
print(f"  {'-'*45} {'-'*8} {'-'*6} {'-'*7} {'-'*8} {'-'*9} {'-'*6} {'-'*7}")
for _, row in sweet.head(20).iterrows():
    print(f"  {row['name']:<45} {row['trades_per_day']:>7.1f} "
          f"{row['win_rate']:>5.1%} {row['long_wr']:>6.1%} {row['short_wr']:>7.1%} "
          f"{row['total_return']*100:>+8.2f}% "
          f"{row['profit_factor']:>5.2f} {row['green_days']:>7}")

print("\n\nOVERALL BEST PERFORMERS (sorted by Profit Factor)")
print("=" * 120)
for _, row in df_r.sort_values('profit_factor', ascending=False).head(20).iterrows():
    print(f"  {row['name']:<45} Sig/D={row['trades_per_day']:>6.1f} WR={row['win_rate']:>5.1%} PF={row['profit_factor']:>5.2f} TotRet={row['total_return']*100:>+8.2f}%")

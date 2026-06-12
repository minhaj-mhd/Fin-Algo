"""
hourly_close_entry_sweep.py — Optimize hourly close-entry ensembles under strict 0.06% costs

Simulates entering exactly at the hourly candle close price (zero execution lag)
and exiting at the close price after 1H, 2H, 3H, 4H, and EOD.
Applies a strict 0.06% transaction cost.
"""

import os, sys, json, pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm
from datetime import datetime

sys.path.append(os.getcwd())

TEST_MONTH = "2026-05"
TRANSACTION_COST_PCT = 0.06 # Strict 0.06% round-trip cost

print("=" * 80)
print("HOURLY CLOSE-ENTRY OPTIMIZATION SWEEP")
print(f"Applying strict {TRANSACTION_COST_PCT}% transaction costs on {TEST_MONTH}")
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

print("\nScoring models...")
df_1h = predict_timeframe_scores(df_1h, "1H", "models/v8_upstox_3y/metadata.json",
    "models/v8_upstox_3y/xgb_long_model.json", "models/v8_upstox_3y/xgb_short_model.json", "models/scaler.pkl")
df_30m = predict_timeframe_scores(df_30m, "30M", "models/v1_30min/metadata.json",
    "models/v1_30min/xgb_long_model.json", "models/v1_30min/xgb_short_model.json", "models/v1_30min/scaler.pkl")
df_15m = predict_timeframe_scores(df_15m, "15M", "models/v1_15min/metadata.json",
    "models/v1_15min/xgb_long_model.json", "models/v1_15min/xgb_short_model.json", "models/v1_15min/scaler.pkl")

# Build dictionaries
print("\nBuilding lookup indices...")
dict_1h = {}
for qid, q_df in df_1h.groupby('DateTime'):
    dict_1h[qid] = {}
    for _, row in q_df.iterrows():
        dict_1h[qid][row['Ticker']] = {
            'long_rank': int(row['long_rank']),
            'short_rank': int(row['short_rank']),
            'long_conv': float(row['long_conv']),
            'short_conv': float(row['short_conv']),
        }

dict_30m = {}
for qid, q_df in df_30m.groupby('DateTime'):
    dict_30m[qid] = {}
    for _, row in q_df.iterrows():
        dict_30m[qid][row['Ticker']] = {
            'long_rank': int(row['long_rank']),
            'short_rank': int(row['short_rank']),
            'long_conv': float(row['long_conv']),
            'short_conv': float(row['short_conv']),
        }

dict_15m = {}
for qid, q_df in df_15m.groupby('DateTime'):
    dict_15m[qid] = {}
    for _, row in q_df.iterrows():
        dict_15m[qid][row['Ticker']] = {
            'long_rank': int(row['long_rank']),
            'short_rank': int(row['short_rank']),
            'long_conv': float(row['long_conv']),
            'short_conv': float(row['short_conv']),
        }

# Build price series from 15M dataset
price_series = {}
df_15m_sorted = df_15m.sort_values('DateTime')
for _, row in df_15m_sorted.iterrows():
    ticker = row['Ticker']
    if ticker not in price_series:
        price_series[ticker] = []
    price_series[ticker].append((
        row['DateTime'],
        float(row['Close'])
    ))

price_idx = {}
for ticker, series in price_series.items():
    price_idx[ticker] = {dt: i for i, (dt, c) in enumerate(series)}

def align_timeframes_at_hour(t_1h_str):
    date_part = t_1h_str[:10]
    h_str, m_str = t_1h_str[11:13], t_1h_str[14:16]
    h, m = int(h_str), int(m_str)
    
    t_30m_start = h * 60 + m + 15
    h_30, m_30 = divmod(t_30m_start, 60)
    t_30m = f"{date_part} {h_30:02d}:{m_30:02d}:00+05:30"
    
    t_15m_start = h * 60 + m + 45
    h_15, m_15 = divmod(t_15m_start, 60)
    t_15m = f"{date_part} {h_15:02d}:{m_15:02d}:00+05:30"
    
    return t_30m, t_15m

# Test parameters
holding_options = [4, 8, 12, 16, 24] # 1H, 2H, 3H, 4H, 6H hold (in terms of 15M bars)
topk_options = [3, 5, 10]
model_configs = [
    ('1H_Only', 1.0, 0.0, 0.0),
    ('Blended_Ensemble', 0.3, 0.4, 0.3),
    ('Blended_Ensemble_High_Return', 0.1, 0.3, 0.6)
]

configs = []
for model_name, w1, w2, w3 in model_configs:
    for hold_bars in holding_options:
        for topk in topk_options:
            configs.append({
                'name': f'{model_name}_w_{w1}_{w2}_{w3}_top{topk}_hold_{hold_bars*15}M',
                'model_name': model_name,
                'w_1h': w1, 'w_30m': w2, 'w_15m': w3,
                'topk': topk,
                'hold_bars': hold_bars
            })

df_1h['Date'] = df_1h['DateTime'].str[:10]
unique_dates = sorted(df_1h['Date'].unique())
timestamps_by_date_1h = {}
for date in unique_dates:
    timestamps_by_date_1h[date] = sorted(df_1h[df_1h['Date'] == date]['DateTime'].unique())

num_trading_days = len(unique_dates)
results = []

for cfg in tqdm(configs, desc="Hourly Close Entry Sweep"):
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
            
            # Entry is exactly at the close of t_15m (which corresponds to the 1-hour candle close)
            # Exit is at the close price of the bar after hold_bars
            for sym in top_longs:
                if sym in price_idx and t_15m in price_idx[sym]:
                    entry_idx = price_idx[sym][t_15m]
                    series = price_series[sym]
                    exit_idx = min(entry_idx + cfg['hold_bars'], len(series) - 1)
                    if exit_idx > entry_idx:
                        ret = (series[exit_idx][1] / series[entry_idx][1]) - 1 # close vs close
                        net = ret - TRANSACTION_COST_PCT / 100
                        trades.append({'date': date, 'side': 'LONG', 'ticker': sym, 'gross_return': ret, 'net_return': net, 'is_win': net > 0})
                        
            for sym in top_shorts:
                if sym in price_idx and t_15m in price_idx[sym]:
                    entry_idx = price_idx[sym][t_15m]
                    series = price_series[sym]
                    exit_idx = min(entry_idx + cfg['hold_bars'], len(series) - 1)
                    if exit_idx > entry_idx:
                        ret = 1 - (series[exit_idx][1] / series[entry_idx][1])
                        net = ret - TRANSACTION_COST_PCT / 100
                        trades.append({'date': date, 'side': 'SHORT', 'ticker': sym, 'gross_return': ret, 'net_return': net, 'is_win': net > 0})

    if not trades:
        results.append({**cfg, 'total_trades': 0, 'trades_per_day': 0,
                        'win_rate': 0, 'total_return': 0, 'avg_gross_return': 0, 'avg_return': 0,
                        'profit_factor': 0, 'long_wr': 0, 'short_wr': 0, 'green_days': '0/0'})
        continue
        
    df_t = pd.DataFrame(trades)
    total = len(df_t)
    wins = int(df_t['is_win'].sum())
    wr = wins / total
    total_ret = df_t['net_return'].sum()
    avg_gross = df_t['gross_return'].mean()
    avg_net = df_t['net_return'].mean()
    
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
        'model_name': cfg['model_name'],
        'hold_bars': cfg['hold_bars'],
        'topk': cfg['topk'],
        'total_trades': total,
        'trades_per_day': total / num_trading_days,
        'win_rate': wr,
        'long_wr': long_wr,
        'short_wr': short_wr,
        'total_return': total_ret,
        'avg_gross_return': avg_gross,
        'avg_return': avg_net,
        'profit_factor': pf,
        'green_days': f"{green_days}/{len(daily)}"
    })

df_r = pd.DataFrame(results)
df_r.to_csv("data/hourly_close_entry_sweep_results.csv", index=False)

print("\n" + "=" * 120)
print(f"HOURLY CLOSE-ENTRY SWEEP RESULTS (sorted by Total Net Return, 0.06% strict transaction cost)")
print("=" * 120)
df_r_sorted = df_r.sort_values('total_return', ascending=False)
print(f"{'Config Name':<55} {'Sig/Day':>8} {'WR':>6} {'AvgGross':>10} {'AvgNet':>10} {'TotNetRet':>10} {'PF':>6} {'Green':>7}")
print(f"{'-'*55} {'-'*8} {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*6} {'-'*7}")
for _, row in df_r_sorted.head(25).iterrows():
    print(f"{row['name']:<55} {row['trades_per_day']:>7.1f} "
          f"{row['win_rate']:>5.1%} {row['avg_gross_return']*100:>9.3f}% {row['avg_return']*100:>9.3f}% "
          f"{row['total_return']*100:>+9.2f}% "
          f"{row['profit_factor']:>5.2f} {row['green_days']:>7}")

print(f"\n[DONE] Hourly close-entry sweep results saved to data/hourly_close_entry_sweep_results.csv")

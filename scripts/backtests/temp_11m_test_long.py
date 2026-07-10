import pandas as pd
import json
import xgboost as xgb
import numpy as np
from datetime import time, date

# 1. Load Data
df = pd.read_parquet('data/research/v20_rolling_1h/panel.parquet')

# Filter test set (Aug 2025 to June 2026)
df['DateTime'] = pd.to_datetime(df['DateTime'])
df = df[df['DateTime'].dt.date >= date(2025, 8, 1)]

# Filter 15-minute intervals between 10:15 and 14:15
time_mask = (df['DateTime'].dt.time >= time(10, 15)) & (df['DateTime'].dt.time <= time(14, 15))
df = df[time_mask]

# 2. Load Models & Metadata
v20_feats = json.load(open('models/research/v20_rolling_1h/metadata.json'))['features']
bs = xgb.Booster()
bs.load_model('models/research/v20_rolling_1h/xgb_short_model.json')
bl = xgb.Booster()
bl.load_model('models/research/v20_rolling_1h/xgb_long_model.json')

# 3. Predict & Score
df = df.dropna(subset=v20_feats + ['Next_Hour_Return'])
X = xgb.DMatrix(np.nan_to_num(df[v20_feats].values.astype(np.float32)), feature_names=v20_feats)
df['ss'] = bs.predict(X)
df['ls'] = bl.predict(X)

ss_mean = df.groupby('DateTime')['ss'].transform('mean')
ls_mean = df.groupby('DateTime')['ls'].transform('mean')
# Long Conviction is reversed
df['long_conviction'] = (df['ls'] - ls_mean) - (df['ss'] - ss_mean)

COST = 6
NOTIONAL = 99517.68

# Sweep threshold for longs. (Typically long scores can be slightly different)
thresholds = [0.065, 0.070, 0.075, 0.080, 0.082, 0.084, 0.086, 0.090, 0.095, 0.100]
print(f'Thresh | Trades | Win% | Net BPS | Total Rs')
print('-'*50)

for th in thresholds:
    trades = []
    for ts, g in df.groupby('DateTime'):
        cands = g[g['ls'] > th].sort_values('long_conviction', ascending=False)
        if len(cands):
            p = cands.iloc[0]
            # long trade: PnL is positive of return
            trades.append((ts, p['Ticker'], p['Next_Hour_Return']*10000, p['long_conviction']))
    
    td = pd.DataFrame(trades, columns=['ts','tk','pnl','score'])
    if len(td) == 0:
        continue
    td['net6'] = td.pnl - COST
    td['bookRs'] = td.net6 / 10000 * NOTIONAL
    win = (td.net6 > 0).mean()
    net = td.net6.mean()
    total_rs = td.bookRs.sum()
    print(f'{th:.3f}  | {len(td):6d} | {win:.1%} | {net:>7.2f} | {total_rs:>8.0f}')

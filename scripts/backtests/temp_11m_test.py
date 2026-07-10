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
df['short_conviction'] = (df['ss'] - ss_mean) - (df['ls'] - ls_mean)

# 4. Extract Trades (Top 1 per DateTime)
COST = 6
NOTIONAL = 99517.68
trades = []

for ts, g in df.groupby('DateTime'):
    cands = g[g['ss'] > 0.082].sort_values('short_conviction', ascending=False)
    if len(cands):
        p = cands.iloc[0]
        trades.append((ts, p['Ticker'], -p['Next_Hour_Return']*10000, p['short_conviction']))

td = pd.DataFrame(trades, columns=['ts','tk','pnl','score'])
td['net6'] = td.pnl - COST
td['bookRs'] = td.net6 / 10000 * NOTIONAL
td['date'] = td['ts'].dt.date

# 5. Summary
print('Total Trades:', len(td))
print('Win Rate:', (td.net6 > 0).mean())
print('Avg Net BPS:', td.net6.mean())
print('Total Rs Booked:', td.bookRs.sum())
print('\nMonthly Breakdown:')
td['month'] = td['ts'].dt.to_period('M')
for m in sorted(td['month'].unique()):
    m_tr = td[td['month'] == m]
    print(f"{m}: n={len(m_tr)} | Net={m_tr.net6.mean():+.2f} | Win={(m_tr.net6>0).mean():.1%} | Rs={m_tr.bookRs.sum():+,.0f}")

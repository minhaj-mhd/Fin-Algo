import pandas as pd
import json
import xgboost as xgb
import numpy as np
from datetime import time, date

# 1. Load Nifty 50 15m
nifty = pd.read_csv('data/raw_index_cache/nifty50_15m.csv')
nifty['ts'] = pd.to_datetime(nifty['ts'])
nifty = nifty.sort_values('ts').reset_index(drop=True)
nifty['nifty_ret_2h'] = nifty['close'] / nifty['close'].shift(8) - 1
nifty_map = dict(zip(nifty['ts'], nifty['nifty_ret_2h']))

# 2. Load Data
df = pd.read_parquet('data/research/v20_rolling_1h/panel.parquet')
df['DateTime'] = pd.to_datetime(df['DateTime'])
df = df[df['DateTime'].dt.date >= date(2025, 8, 1)]

time_mask = (df['DateTime'].dt.time >= time(10, 15)) & (df['DateTime'].dt.time <= time(14, 15))
df = df[time_mask]

# Apply Nifty return
df['nifty_ret_2h'] = df['DateTime'].map(nifty_map)
df = df.dropna(subset=['nifty_ret_2h'])

# 3. Load Models & Metadata
v20_feats = json.load(open('models/research/v20_rolling_1h/metadata.json'))['features']
bs = xgb.Booster()
bs.load_model('models/research/v20_rolling_1h/xgb_short_model.json')
bl = xgb.Booster()
bl.load_model('models/research/v20_rolling_1h/xgb_long_model.json')

# 4. Predict & Score
df = df.dropna(subset=v20_feats + ['Next_Hour_Return'])
X = xgb.DMatrix(np.nan_to_num(df[v20_feats].values.astype(np.float32)), feature_names=v20_feats)
df['ss'] = bs.predict(X)
df['ls'] = bl.predict(X)

ss_mean = df.groupby('DateTime')['ss'].transform('mean')
ls_mean = df.groupby('DateTime')['ls'].transform('mean')
df['long_conviction'] = (df['ls'] - ls_mean) - (df['ss'] - ss_mean)

COST = 6
NOTIONAL = 99517.68

def run_backtest(df_subset):
    trades = []
    for ts, g in df_subset.groupby('DateTime'):
        cands = g.sort_values('long_conviction', ascending=False)
        if len(cands):
            p = cands.iloc[0]
            trades.append((ts, p['Ticker'], p['Next_Hour_Return']*10000, p['long_conviction']))
    td = pd.DataFrame(trades, columns=['ts','tk','pnl','score'])
    if len(td) == 0: return 0, 0, 0, 0
    td['net6'] = td.pnl - COST
    td['bookRs'] = td.net6 / 10000 * NOTIONAL
    return len(td), (td.net6 > 0).mean(), td.net6.mean(), td.bookRs.sum()

print(f"Condition                 | Trades | Win%  | Net BPS | Total Rs")
print("-" * 65)

conditions = [
    ("Nifty > +0.50%", df[df['nifty_ret_2h'] > 0.0050]),
    ("Nifty > +0.25%", df[df['nifty_ret_2h'] > 0.0025]),
    ("Nifty >  0.00%", df[df['nifty_ret_2h'] > 0.0000]),
    ("Nifty <  0.00%", df[df['nifty_ret_2h'] < 0.0000]),
    ("Nifty < -0.25%", df[df['nifty_ret_2h'] < -0.0025]),
    ("Nifty < -0.50%", df[df['nifty_ret_2h'] < -0.0050]),
]

for name, subset in conditions:
    n, w, b, r = run_backtest(subset)
    if n > 0:
        print(f"{name:<25} | {n:6d} | {w:.1%} | {b:>7.2f} | {r:>8.0f}")


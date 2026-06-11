"""
Tier-wise raw vs net (10 bps) win rates, avg bps, and total returns.
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
warnings.filterwarnings('ignore')
sys.path.append(os.getcwd())

MODEL_DIR  = 'models/v2_15min_3y'
META_PATH  = f'{MODEL_DIR}/metadata.json'
LONG_PATH  = f'{MODEL_DIR}/xgb_long_model.json'
SHORT_PATH = f'{MODEL_DIR}/xgb_short_model.json'
DATA_FILE  = 'data/ranking_data_upstox_15min_3y.csv'
OOS_MONTHS = 3
COST       = 10 / 10000

print("Loading...")
with open(META_PATH) as f: meta = json.load(f)
feature_cols = meta['features']
bst_long  = xgb.Booster(); bst_long.load_model(LONG_PATH)
bst_short = xgb.Booster(); bst_short.load_model(SHORT_PATH)

all_months = set()
for ch in pd.read_csv(DATA_FILE, usecols=['DateTime'], chunksize=500_000):
    all_months.update(ch['DateTime'].str[:7].unique())
oos_m = sorted(all_months)[-OOS_MONTHS:]

chunks = []
for ch in pd.read_csv(DATA_FILE, chunksize=200_000):
    sub = ch[ch['DateTime'].str[:7].isin(oos_m)]
    if len(sub): chunks.append(sub)
df = pd.concat(chunks, ignore_index=True)

X = df[feature_cols].values.astype(float)
for ci in range(X.shape[1]):
    col = X[:, ci]; bad = np.isnan(col) | np.isinf(col)
    if bad.any():
        X[bad, ci] = float(np.nanmean(col[~bad])) if (~bad).any() else 0.0

df['long_score']  = bst_long.predict(xgb.DMatrix(X))
df['short_score'] = bst_short.predict(xgb.DMatrix(X))
df['dt']          = pd.to_datetime(df['DateTime'])
df['hour']        = df['dt'].dt.hour
df['ret']         = df['Next_15Min_Return']

configs = [
    ("Tier 1  L>0.0829 @ 15h (p99)",
     (df['long_score'] > 0.0829) & (df['hour'] == 15), 'long'),
    ("Tier 2  L>0.0629 @ 15h (p95)",
     (df['long_score'] > 0.0629) & (df['hour'] == 15), 'long'),
    ("Tier 3  L>0.0514 @ 15h (p90)",
     (df['long_score'] > 0.0514) & (df['hour'] == 15), 'long'),
    ("Sniper  S>0.0514 L<-0.1112 @ 10h",
     (df['short_score'] > 0.0514) & (df['long_score'] < -0.1112) & (df['hour'] == 10), 'short'),
]

SEP = "=" * 90
print(f"\n{SEP}")
print(f"  {'Config':<36} {'N':>5}  {'Raw WR':>7} {'Net WR':>7}  {'Raw bps':>8} {'Net bps':>8}  {'Raw Ret%':>9} {'Net Ret%':>9}")
print(SEP)

for label, mask, direction in configs:
    sub = df[mask].copy()
    if len(sub) == 0:
        print(f"  {label}  — no trades")
        continue

    r = sub['ret'].values
    gross = r if direction == 'long' else -r
    net   = gross - COST

    raw_wr    = (gross > 0).mean() * 100
    net_wr    = (net   > 0).mean() * 100
    raw_bps   = gross.mean() * 10000
    net_bps   = net.mean()   * 10000
    raw_total = ((1 + gross).prod() - 1) * 100
    net_total = ((1 + net).prod()   - 1) * 100
    n         = len(sub)

    print(f"  {label:<36} {n:>5,}  {raw_wr:>6.1f}% {net_wr:>6.1f}%  "
          f"{raw_bps:>+7.2f}  {net_bps:>+7.2f}  {raw_total:>+8.1f}%  {net_total:>+8.1f}%")

print(SEP)
print(f"\n  Raw  = no fees applied")
print(f"  Net  = after 10 bps deduction per trade")
print(f"  Ret% = compounded over 3-month OOS period")

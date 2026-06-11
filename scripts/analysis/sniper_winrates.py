"""
Raw vs net-of-10bps win rates for all sniper tiers.
Raw WR  = trade went in the right direction (ret > 0 for long, ret < 0 for short)
Net WR  = trade profitable after 10 bps friction (ret > 0.001 for long, ret < -0.001 for short)
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
COST       = 10 / 10000   # 10 bps

print("Loading models...")
with open(META_PATH) as f: meta = json.load(f)
feature_cols = meta['features']
bst_long  = xgb.Booster(); bst_long.load_model(LONG_PATH)
bst_short = xgb.Booster(); bst_short.load_model(SHORT_PATH)

print("Streaming OOS data...")
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
print(f"  {len(df):,} rows ready\n")

SEP = "-" * 70

def stats(sub, direction):
    r = sub['ret'].values
    gross = r if direction == 'long' else -r
    net   = gross - COST

    raw_wr   = (gross > 0).mean() * 100
    net_wr   = (net   > 0).mean() * 100
    avg_gross = gross.mean() * 10000
    avg_net   = net.mean()   * 10000
    n         = len(sub)
    return n, raw_wr, net_wr, avg_gross, avg_net

# ── configs to evaluate ───────────────────────────────────────────────────────
configs = [
    # label, mask, direction
    ("Tier 1  | Direct Long  L>0.0829 @ 15h",
     (df['long_score'] > 0.0829) & (df['hour'] == 15),
     'long'),
    ("Tier 2  | Direct Long  L>0.0629 @ 15h",
     (df['long_score'] > 0.0629) & (df['hour'] == 15),
     'long'),
    ("Tier 3  | Direct Long  L>0.0514 @ 15h",
     (df['long_score'] > 0.0514) & (df['hour'] == 15),
     'long'),
    ("Sniper  | Dual-Lock Short  S>0.0514 L<-0.1112 @ 10h",
     (df['short_score'] > 0.0514) & (df['long_score'] < -0.1112) & (df['hour'] == 10),
     'short'),
    # broader cuts for comparison
    ("Broad   | Direct Long  L>0.0829 ALL hours",
     (df['long_score'] > 0.0829),
     'long'),
    ("Broad   | Direct Long  L>0.0629 ALL hours",
     (df['long_score'] > 0.0629),
     'long'),
]

print(f"{'Config':<52} {'N':>6}  {'Raw WR':>8}  {'Net WR':>8}  {'Raw bps':>9}  {'Net bps':>9}")
print(SEP)
for label, mask, direction in configs:
    sub = df[mask]
    if len(sub) == 0:
        print(f"  {label}  — no trades")
        continue
    n, raw_wr, net_wr, avg_gross, avg_net = stats(sub, direction)
    print(f"{label:<52} {n:>6,}  {raw_wr:>7.1f}%  {net_wr:>7.1f}%  "
          f"{avg_gross:>+8.2f}  {avg_net:>+8.2f}")

print(SEP)
print(f"\n  Raw WR  = trade moved in right direction (no fees)")
print(f"  Net WR  = trade profitable after 10 bps deduction")
print(f"  Raw bps = gross avg return per trade")
print(f"  Net bps = avg return after 10 bps")

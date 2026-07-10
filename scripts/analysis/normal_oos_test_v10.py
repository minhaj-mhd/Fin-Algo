import os, sys, json, warnings
import numpy as np, pandas as pd, xgboost as xgb
from scipy.stats import rankdata
warnings.filterwarnings('ignore'); sys.path.append(os.getcwd())

DIR, DATA, RET = 'models/v10_native_1h', 'data/ranking_data_upstox_1h_v3_3y.csv', 'Next_Hour_Return'
COST = 10/1e4

# Define the static split
# Train: up to 2025-07
# Val: 2025-08
# Test: 2025-09 to 2026-06
TRAIN_END = '2025-07'
VAL_MONTH = '2025-08'
TEST_START = '2025-09'

def load(p, ret):
    df = pd.concat([c for c in pd.read_csv(p, chunksize=200_000)], ignore_index=True)
    df['dt'] = pd.to_datetime(df['DateTime'])
    df['ym'] = df['DateTime'].str[:7]
    df['hour'] = df['dt'].dt.strftime('%H:%M')
    return df.dropna(subset=[ret]).reset_index(drop=True)

def Xmat(df, fe):
    X = df[fe].values.astype(float)
    for ci in range(X.shape[1]):
        c = X[:, ci]; b = np.isnan(c) | np.isinf(c)
        if b.any(): X[b, ci] = float(np.nanmean(c[~b])) if (~b).any() else 0.0
    return X

def iranks(y, q, inv=False):
    out = np.zeros_like(y, dtype=int)
    for qid in np.unique(q):
        m = q == qid; v = -y[m] if inv else y[m]; out[m] = rankdata(v, method='ordinal')-1
    return out

def fitdm(X, y, q, inv):
    d = xgb.DMatrix(X, label=iranks(y, q, inv))
    d.set_group(pd.Series(q).groupby(q).size().values)
    return d

def stat(net):
    n = len(net)
    if n == 0: return None
    wr = (net > 0).mean() * 100
    bps = net.mean() * 1e4
    return dict(n=n, wr=wr, bps=bps)

print("Loading 1h native data for normal static OOS test...")
with open(f'{DIR}/metadata.json') as f: meta = json.load(f)
fe, params = meta['features'], meta['params']
df = load(DATA, RET); X = Xmat(df, fe)

tr = df['ym'] <= TRAIN_END
va = df['ym'] == VAL_MONTH
te = df['ym'] >= TEST_START

print(f"Train rows: {tr.sum():,} | Val rows: {va.sum():,} | Test rows: {te.sum():,}")

print("Training Long Model...")
bl = xgb.train(params, fitdm(X[tr], df[RET].values[tr], df['Query_ID'].values[tr], False), 500,
               evals=[(fitdm(X[va], df[RET].values[va], df['Query_ID'].values[va], False),'v')], early_stopping_rounds=50, verbose_eval=False)

print("Training Short Model...")
bs = xgb.train(params, fitdm(X[tr], df[RET].values[tr], df['Query_ID'].values[tr], True), 500,
               evals=[(fitdm(X[va], df[RET].values[va], df['Query_ID'].values[va], True),'v')], early_stopping_rounds=50, verbose_eval=False)

print("Generating predictions on OOS Test Set...")
sub = df[te].copy()
sub['sL'] = bl.predict(xgb.DMatrix(X[te]))
sub['sS'] = bs.predict(xgb.DMatrix(X[te]))

sub['posL'] = sub.groupby('dt')['sL'].rank(ascending=False, method='first')
sub['posS'] = sub.groupby('dt')['sS'].rank(ascending=False, method='first')

print("Computing mean of top 15 for each dt...")
mean_15_L = sub[sub['posL'] <= 15].groupby('dt')['sL'].mean().rename('mean15_L')
mean_15_S = sub[sub['posS'] <= 15].groupby('dt')['sS'].mean().rename('mean15_S')

sub = sub.join(mean_15_L, on='dt')
sub = sub.join(mean_15_S, on='dt')

print("\n============================================================")
print(f"  STATIC OOS TEST: {TEST_START} to 2026-06 (10 Months)")
print("============================================================")

for K in [1, 3]:
    print(f"\n--- TRADING TOP {K} (K={K}) ---")
    for thr in [0, 10, 20, 30, 40, 50, 75]:
        print(f"\n  [ Threshold: +{thr}% over Mean Top 15 ]")
        
        # LONG
        sub_L = sub[(sub['posL'] <= K) & (sub['sL'] > sub['mean15_L'] * (1 + thr/100.0))]
        net_L = sub_L['Next_Hour_Return'].values * 1 - COST
        st_L = stat(net_L)
        if st_L:
            print(f"  LONG  | Trades: {st_L['n']:>5} | WR: {st_L['wr']:>5.1f}% | Net Bps: {st_L['bps']:>+7.1f}")
        else:
            print("  LONG  | Trades:     0")
            
        # SHORT
        sub_S = sub[(sub['posS'] <= K) & (sub['sS'] > sub['mean15_S'] * (1 + thr/100.0))]
        net_S = sub_S['Next_Hour_Return'].values * (-1) - COST
        st_S = stat(net_S)
        if st_S:
            print(f"  SHORT | Trades: {st_S['n']:>5} | WR: {st_S['wr']:>5.1f}% | Net Bps: {st_S['bps']:>+7.1f}")
        else:
            print("  SHORT | Trades:     0")

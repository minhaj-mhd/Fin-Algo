import os, sys, json, warnings, gc
import numpy as np, pandas as pd, xgboost as xgb
from scipy.stats import rankdata
warnings.filterwarnings('ignore'); sys.path.append(os.getcwd())

DIR, DATA, RET = 'models/v10_native_1h', 'data/ranking_data_upstox_1h_v3_3y.csv', 'Next_Hour_Return'
H_TEST, MIN_TRAIN, COST = 4, 18, 10/1e4
RNG = np.random.default_rng(0)

def load(p, ret):
    df = pd.concat([c for c in pd.read_csv(p, chunksize=200_000)], ignore_index=True)
    df['dt'] = pd.to_datetime(df['DateTime']); df['ym'] = df['DateTime'].str[:7]; df['hour'] = df['dt'].dt.strftime('%H:%M')
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
    d = xgb.DMatrix(X, label=iranks(y, q, inv)); d.set_group(pd.Series(q).groupby(q).size().values); return d
def stat(net, fold):
    n = len(net)
    if n < 20: return None
    t = net.mean()/(net.std()/np.sqrt(n)) if net.std() > 0 else 0
    bs = [net[RNG.integers(0,n,n)].mean()*1e4 for _ in range(1500)]
    fm = [net[fold==fi].mean() for fi in np.unique(fold)]
    sig = '***' if abs(t)>2.58 else ('**' if abs(t)>1.96 else ('*' if abs(t)>1.64 else ''))
    return dict(n=n, wr=(net>0).mean()*100, bps=net.mean()*1e4, ci=(np.percentile(bs,2.5),np.percentile(bs,97.5)), t=t, sig=sig, pos=sum(1 for x in fm if x>0), nf=len(fm))

print("Loading 1h native...")
with open(f'{DIR}/metadata.json') as f: meta = json.load(f)
fe, params = meta['features'], meta['params']
df = load(DATA, RET); X = Xmat(df, fe)
months = sorted(df['ym'].unique())
folds = []; i = MIN_TRAIN + 1
while i + 1 <= len(months):
    folds.append((months[:i-1], months[i-1], months[i:i+H_TEST])); i += H_TEST
print(f"  {len(folds)} folds OOS {folds[0][2][0]} -> {folds[-1][2][-1]}")

rows = []
for fi, (tr_m, val_m, te_m) in enumerate(folds, 1):
    tr = df['ym'].isin(tr_m).values; va = df['ym'].isin([val_m]).values; te = df['ym'].isin(te_m).values
    bl = xgb.train(params, fitdm(X[tr], df[RET].values[tr], df['Query_ID'].values[tr], False), 500,
                   evals=[(fitdm(X[va], df[RET].values[va], df['Query_ID'].values[va], False),'v')], early_stopping_rounds=50, verbose_eval=False)
    bs = xgb.train(params, fitdm(X[tr], df[RET].values[tr], df['Query_ID'].values[tr], True), 500,
                   evals=[(fitdm(X[va], df[RET].values[va], df['Query_ID'].values[va], True),'v')], early_stopping_rounds=50, verbose_eval=False)
    sub = df[te].copy(); sub['sL'] = bl.predict(xgb.DMatrix(X[te])); sub['sS'] = bs.predict(xgb.DMatrix(X[te]))
    sub['pctL'] = sub.groupby('dt')['sL'].rank(pct=True); sub['pctS'] = sub.groupby('dt')['sS'].rank(pct=True)
    sub['posL'] = sub.groupby('dt')['sL'].rank(ascending=False, method='first')
    sub['posS'] = sub.groupby('dt')['sS'].rank(ascending=False, method='first')
    sub['fold'] = fi
    rows.append(sub[['fold','dt','hour','Next_Hour_Return','posL','posS','pctL','pctS','sL','sS']])
    print(f"  fold {fi}/{len(folds)} {te_m[0]}->{te_m[-1]} done")
    del sub; gc.collect()
P = pd.concat(rows, ignore_index=True)
print(f"  pooled OOS rows: {len(P):,}\n")

print("Computing mean of top 15 for each dt...")
# We need the mean of sL for posL <= 15 per dt, and sS for posS <= 15
mean_15_L = P[P['posL'] <= 15].groupby('dt')['sL'].mean().rename('mean15_L')
mean_15_S = P[P['posS'] <= 15].groupby('dt')['sS'].mean().rename('mean15_S')

P = P.join(mean_15_L, on='dt')
P = P.join(mean_15_S, on='dt')

print("="*92)
print("  Walk-Forward pooled OOS — Relative Conviction Thresholds (K1 & K3 vs Top 15 Mean)")
print("="*92)

for K in [1, 3]:
    print(f"\n--- TRADING TOP {K} (K={K}) ---")
    for thr in [0, 10, 20, 30, 40, 50, 75]:
        print(f"\n  [ Threshold: +{thr}% over Mean Top 15 ]")
        print(f"  {'dir':>5} {'N':>6} {'WR%':>6} {'NetBps':>8} {'95% CI':>16} {'t':>6} {'+folds':>7}")
        
        # LONG
        sub_L = P[(P['posL'] <= K) & (P['sL'] > P['mean15_L'] * (1 + thr/100.0))]
        net_L = sub_L['Next_Hour_Return'].values*1 - COST
        st_L = stat(net_L, sub_L['fold'].values)
        if st_L:
            print(f"  {'LONG':>5} {st_L['n']:>6} {st_L['wr']:>5.1f}% {st_L['bps']:>+7.1f} [{st_L['ci'][0]:>+5.1f},{st_L['ci'][1]:>+5.1f}] {st_L['t']:>5.1f}{st_L['sig']:<3} {st_L['pos']:>2}/{st_L['nf']:<2}")
        else:
            print(f"  LONG: not enough trades")
            
        # SHORT
        sub_S = P[(P['posS'] <= K) & (P['sS'] > P['mean15_S'] * (1 + thr/100.0))]
        net_S = sub_S['Next_Hour_Return'].values*(-1) - COST
        st_S = stat(net_S, sub_S['fold'].values)
        if st_S:
            print(f"  {'SHORT':>5} {st_S['n']:>6} {st_S['wr']:>5.1f}% {st_S['bps']:>+7.1f} [{st_S['ci'][0]:>+5.1f},{st_S['ci'][1]:>+5.1f}] {st_S['t']:>5.1f}{st_S['sig']:<3} {st_S['pos']:>2}/{st_S['nf']:<2}")
        else:
            print(f"  SHORT: not enough trades")

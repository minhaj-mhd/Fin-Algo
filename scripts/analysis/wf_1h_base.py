"""
Does the 1h BASE model (native v10, standalone, no 15m) have ANY tradable edge?
Walk-forward (1h's own 2022+ history), pooled OOS. Sweeps:
  (1) conviction by top-K {1,3,5,10}
  (2) conviction by ABSOLUTE rank-percentile threshold {p90,p95,p99} (pool all bars above)
  (3) per-hour breakdown (top-3) — rules out any single-hour edge (the old '2 PM' claim)
Both long & short, 10 bps, bootstrap CI + t-stat.
"""
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
    rows.append(sub[['fold','ym','hour','Next_Hour_Return','posL','posS','pctL','pctS']])
    print(f"  fold {fi}/{len(folds)} {te_m[0]}->{te_m[-1]} done")
    del sub; gc.collect()
P = pd.concat(rows, ignore_index=True)
print(f"  pooled OOS rows: {len(P):,}\n")

print("="*92)
print("  1h BASE MODEL (standalone, no 15m) — walk-forward pooled OOS, 10 bps")
print("="*92)
print("\n  (1) CONVICTION by top-K:")
print(f"  {'dir':>5} {'topK':>5} {'N':>6} {'WR%':>6} {'NetBps':>8} {'95% CI':>16} {'t':>6} {'+folds':>7}")
for d, sgn, poscol in [('LONG', 1, 'posL'), ('SHORT', -1, 'posS')]:
    for K in [1, 3, 5, 10]:
        sub = P[P[poscol] <= K]; net = sub['Next_Hour_Return'].values*sgn - COST
        st = stat(net, sub['fold'].values)
        if st: print(f"  {d:>5} {K:>5} {st['n']:>6} {st['wr']:>5.1f}% {st['bps']:>+7.1f} [{st['ci'][0]:>+5.1f},{st['ci'][1]:>+5.1f}] {st['t']:>5.1f}{st['sig']:<3} {st['pos']:>2}/{st['nf']:<2}")

print("\n  (2) CONVICTION by absolute rank-percentile (pool ALL bars above threshold):")
print(f"  {'dir':>5} {'gate':>6} {'N':>7} {'WR%':>6} {'NetBps':>8} {'95% CI':>16} {'t':>6} {'+folds':>7}")
for d, sgn, pctcol in [('LONG', 1, 'pctL'), ('SHORT', -1, 'pctS')]:
    for thr in [0.90, 0.95, 0.99]:
        sub = P[P[pctcol] > thr]; net = sub['Next_Hour_Return'].values*sgn - COST
        st = stat(net, sub['fold'].values)
        if st: print(f"  {d:>5} {'p'+str(int(thr*100)):>6} {st['n']:>7} {st['wr']:>5.1f}% {st['bps']:>+7.1f} [{st['ci'][0]:>+5.1f},{st['ci'][1]:>+5.1f}] {st['t']:>5.1f}{st['sig']:<3} {st['pos']:>2}/{st['nf']:<2}")

print("\n  (3) PER-HOUR (top-3) — does any single hour work? (old '2PM' claim):")
print(f"  {'dir':>5} {'hour':>6} {'N':>6} {'WR%':>6} {'NetBps':>8} {'t':>6}")
for d, sgn, poscol in [('LONG', 1, 'posL'), ('SHORT', -1, 'posS')]:
    for h in sorted(P['hour'].unique()):
        sub = P[(P[poscol] <= 3) & (P['hour'] == h)]; net = sub['Next_Hour_Return'].values*sgn - COST
        st = stat(net, sub['fold'].values)
        if st: print(f"  {d:>5} {h:>6} {st['n']:>6} {st['wr']:>5.1f}% {st['bps']:>+7.1f} {st['t']:>5.1f}{st['sig']}")
print("\n" + "="*92 + "\n  CI spanning 0 => not significant. *p<.10 **p<.05 ***p<.01\n" + "="*92)
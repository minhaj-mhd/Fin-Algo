"""
Walk-forward dual-TF GRID over CONVICTION (top-K) x CALIBRATION percentile thresholds,
for BOTH long and short. Each fold trains 1h+15m (long+short) strictly before its test block,
builds a pooled OOS candidate table, then evaluates the full grid. 10 bps, bootstrap CI.

Answers: at ANY conviction level (top-1..top-10) and ANY 15m confirmation percentile
(none..p99), does the dual-TF turn net-positive after costs?
"""
import os, sys, json, warnings, gc
import numpy as np, pandas as pd, xgboost as xgb
from scipy.stats import rankdata
warnings.filterwarnings('ignore'); sys.path.append(os.getcwd())

M15 = dict(dir='models/v3_15min_clean', data='data/ranking_data_upstox_15min_3y_clean.csv', ret='Next_15Min_Return')
H1  = dict(dir='models/v10_native_1h', data='data/ranking_data_upstox_1h_v3_3y.csv', ret='Next_Hour_Return')
H_TEST, MIN_TRAIN, COST = 4, 18, 10/1e4
RNG = np.random.default_rng(0); C45 = pd.Timedelta(minutes=45)

def load(path, ret):
    df = pd.concat([c for c in pd.read_csv(path, chunksize=200_000)], ignore_index=True)
    df['dt'] = pd.to_datetime(df['DateTime']); df['ym'] = df['DateTime'].str[:7]
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
def fit(X, y, q, params, inv):
    d = xgb.DMatrix(X, label=iranks(y, q, inv)); d.set_group(pd.Series(q).groupby(q).size().values); return d

print("Loading...")
with open(f'{M15["dir"]}/metadata.json') as f: m15m = json.load(f)
with open(f'{H1["dir"]}/metadata.json') as f: m1m = json.load(f)
f15, p15 = m15m['features'], m15m['params']; f1, p1 = m1m['features'], m1m['params']
d15 = load(M15['data'], M15['ret']); X15 = Xmat(d15, f15)
d1 = load(H1['data'], H1['ret']); X1 = Xmat(d1, f1)
months = sorted(d15['ym'].unique())
folds = []; i = MIN_TRAIN + 1
while i + 1 <= len(months):
    folds.append((months[:i-1], months[i-1], months[i:i+H_TEST])); i += H_TEST
print(f"  {len(folds)} folds OOS {folds[0][2][0]} -> {folds[-1][2][-1]}")

def fit_predict(d, X, fe, ret, params, tr_m, val_m, te_m):
    tr = d['ym'].isin(tr_m).values; va = d['ym'].isin([val_m]).values; te = d['ym'].isin(te_m).values
    bl = xgb.train(params, fit(X[tr], d[ret].values[tr], d['Query_ID'].values[tr], params, False), 500,
                   evals=[(fit(X[va], d[ret].values[va], d['Query_ID'].values[va], params, False),'v')], early_stopping_rounds=50, verbose_eval=False)
    bs = xgb.train(params, fit(X[tr], d[ret].values[tr], d['Query_ID'].values[tr], params, True), 500,
                   evals=[(fit(X[va], d[ret].values[va], d['Query_ID'].values[va], params, True),'v')], early_stopping_rounds=50, verbose_eval=False)
    sub = d[te].copy(); sub['s_long'] = bl.predict(xgb.DMatrix(X[te])); sub['s_short'] = bs.predict(xgb.DMatrix(X[te]))
    return sub

cand = []  # pooled candidate table across folds
for fi, (tr_m, val_m, te_m) in enumerate(folds, 1):
    print(f"  Fold {fi}/{len(folds)} test {te_m[0]}->{te_m[-1]}")
    s15 = fit_predict(d15, X15, f15, M15['ret'], p15, tr_m, val_m, te_m)
    s15['rkL'] = s15.groupby('dt')['s_long'].rank(pct=True); s15['rkS'] = s15.groupby('dt')['s_short'].rank(pct=True)
    look = {(r.Ticker, r.dt): (r.rkL, r.rkS) for r in s15.itertuples(index=False)}
    tr1 = [m for m in sorted(d1['ym'].unique()) if m < val_m]
    s1 = fit_predict(d1, X1, f1, H1['ret'], p1, tr1, val_m, te_m)
    for dt1, grp in s1.groupby('dt'):
        gl = grp.sort_values('s_long', ascending=False).reset_index(drop=True)
        posL = {t: i+1 for i, t in enumerate(gl['Ticker'])}
        gs = grp.sort_values('s_short', ascending=False).reset_index(drop=True)
        posS = {t: i+1 for i, t in enumerate(gs['Ticker'])}
        for _, r in grp.iterrows():
            conf = look.get((r['Ticker'], dt1 + C45))
            if conf is None: continue
            cand.append((fi, r['ym'], r['Ticker'], r['Next_Hour_Return'],
                         posL[r['Ticker']], posS[r['Ticker']], conf[0], conf[1]))
    del s15, s1, look; gc.collect()

cdf = pd.DataFrame(cand, columns=['fold', 'ym', 'tk', 'ret', 'posL', 'posS', 'crkL', 'crkS'])
print(f"\n  pooled candidates: {len(cdf):,}")

def stats(net, fold):
    n = len(net)
    if n < 20: return None
    t = net.mean()/(net.std()/np.sqrt(n)) if net.std() > 0 else 0
    bs = [net[RNG.integers(0,n,n)].mean()*1e4 for _ in range(1500)]
    fm = [net[fold==fi].mean() for fi in np.unique(fold)]
    return dict(n=n, wr=(net>0).mean()*100, bps=net.mean()*1e4,
                ci=(np.percentile(bs,2.5), np.percentile(bs,97.5)), t=t,
                pos=sum(1 for x in fm if x>0), nf=len(fm))

print(f"\n{'='*104}")
print("  CONVICTION x CALIBRATION GRID (walk-forward pooled OOS, 10 bps)")
print(f"{'='*104}")
KS = [1, 3, 5, 10]
THRS = [None, 0.85, 0.90, 0.95, 0.99]
for direction in ['LONG', 'SHORT']:
    print(f"\n  ===== {direction} =====  (conviction = 1h top-K; calibration gate = 15m own-rank percentile)")
    print(f"  {'topK':>5} {'gate':>6} {'N':>6} {'WR%':>6} {'NetBps':>8} {'95% CI':>16} {'t':>6} {'+folds':>7}")
    for K in KS:
        pos = cdf['posL'] if direction == 'LONG' else cdf['posS']
        crk = cdf['crkL'] if direction == 'LONG' else cdf['crkS']
        sgn = 1.0 if direction == 'LONG' else -1.0
        for thr in THRS:
            m = (pos <= K) if thr is None else ((pos <= K) & (crk > thr))
            sub = cdf[m]
            if len(sub) < 20: continue
            net = (sub['ret'].values * sgn) - COST
            st = stats(net, sub['fold'].values)
            if st is None: continue
            sig = '***' if abs(st['t'])>2.58 else ('**' if abs(st['t'])>1.96 else ('*' if abs(st['t'])>1.64 else ''))
            gate = 'none' if thr is None else f'p{int(thr*100)}'
            print(f"  {K:>5} {gate:>6} {st['n']:>6} {st['wr']:>5.1f}% {st['bps']:>+7.1f} [{st['ci'][0]:>+5.1f},{st['ci'][1]:>+5.1f}] {st['t']:>5.1f}{sig:<3} {st['pos']:>2}/{st['nf']:<2}")
print(f"\n{'='*104}\n  CI spanning 0 => not significant.  *p<.10 **p<.05 ***p<.01\n{'='*104}")
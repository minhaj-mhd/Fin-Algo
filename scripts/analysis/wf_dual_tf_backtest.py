"""
MULTI-FOLD WALK-FORWARD dual-TF backtest (the powered, decisive test).

For each fold: train 1h (native v10 cfg) + 15m (clean v3 cfg), long+short, on months STRICTLY
before the test block; predict on the unseen block; run the dual-TF strategy; pool ALL trades
across folds -> ~2 years of genuinely-OOS trades for real statistical power.

Expanding train, non-overlapping test blocks (H months). Entry-confirmation P&L = 1h
Next_Hour_Return; gate = that fold's holdout-15m ranks (self-contained per timestamp).
10 bps friction. Bootstrap 95% CI + t-stat, pooled and per-fold.
"""
import os, sys, json, warnings, gc
import numpy as np, pandas as pd, xgboost as xgb
from scipy.stats import rankdata
warnings.filterwarnings('ignore'); sys.path.append(os.getcwd())

M15 = dict(dir='models/v3_15min_clean', data='data/ranking_data_upstox_15min_3y_clean.csv', ret='Next_15Min_Return')
H1  = dict(dir='models/v10_native_1h', data='data/ranking_data_upstox_1h_v3_3y.csv', ret='Next_Hour_Return')
H_TEST = 4          # months per non-overlapping test block
MIN_TRAIN = 18      # initial train months (by the 15m calendar, the binding constraint)
COST = 10/1e4
RNG = np.random.default_rng(0)
C45 = pd.Timedelta(minutes=45)

def load(path, ret):
    ch = [c for c in pd.read_csv(path, chunksize=200_000)]
    df = pd.concat(ch, ignore_index=True)
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

def train_side(X, y, q, params, inv):
    d = xgb.DMatrix(X, label=iranks(y, q, inv)); d.set_group(pd.Series(q).groupby(q).size().values)
    return d

print("Loading full datasets...")
with open(f'{M15["dir"]}/metadata.json') as f: m15meta = json.load(f)
with open(f'{H1["dir"]}/metadata.json') as f: m1meta = json.load(f)
f15, p15 = m15meta['features'], m15meta['params']
f1, p1 = m1meta['features'], m1meta['params']
d15 = load(M15['data'], M15['ret']); X15 = Xmat(d15, f15)
d1 = load(H1['data'], H1['ret']);   X1 = Xmat(d1, f1)
print(f"  15m {len(d15):,} rows | 1h {len(d1):,} rows")

months = sorted(d15['ym'].unique())   # 15m is binding (shorter history)
folds = []
i = MIN_TRAIN + 1
while i + 1 <= len(months):
    val_m = months[i-1]; test_m = months[i:i+H_TEST]
    if not test_m: break
    folds.append((months[:i-1], val_m, test_m)); i += H_TEST
print(f"  {len(folds)} folds, H={H_TEST}mo, OOS {folds[0][2][0]} -> {folds[-1][2][-1]}")

def fit_predict(d, X, fe, ret, params, train_m, val_m, test_m):
    tr = d['ym'].isin(train_m).values; va = d['ym'].isin([val_m]).values; te = d['ym'].isin(test_m).values
    dl = train_side(X[tr], d[ret].values[tr], d['Query_ID'].values[tr], params, False)
    ds = train_side(X[tr], d[ret].values[tr], d['Query_ID'].values[tr], params, True)
    dvl = train_side(X[va], d[ret].values[va], d['Query_ID'].values[va], params, False)
    dvs = train_side(X[va], d[ret].values[va], d['Query_ID'].values[va], params, True)
    bl = xgb.train(params, dl, 500, evals=[(dvl,'v')], early_stopping_rounds=50, verbose_eval=False)
    bs = xgb.train(params, ds, 500, evals=[(dvs,'v')], early_stopping_rounds=50, verbose_eval=False)
    sub = d[te].copy(); Xte = X[te]
    sub['s_long'] = bl.predict(xgb.DMatrix(Xte)); sub['s_short'] = bs.predict(xgb.DMatrix(Xte))
    return sub

CONFIGS = [
    ('SHORT baseline',         'short', {}),
    ('SHORT ShortConfirm p90', 'short', {'short':0.90}),
    ('SHORT LongAvoid p15',    'short', {'long':0.15}),
    ('LONG  baseline',         'long',  {}),
    ('LONG  LongConfirm p90',  'long',  {'long':0.90}),
    ('LONG  ShortAvoid p15',   'long',  {'short':0.15}),
]
trades = {lbl: [] for lbl, _, _ in CONFIGS}

for fi, (train_m, val_m, test_m) in enumerate(folds, 1):
    print(f"  Fold {fi}/{len(folds)}: train<= {train_m[-1]} | val {val_m} | test {test_m[0]}->{test_m[-1]}")
    s15 = fit_predict(d15, X15, f15, M15['ret'], p15, train_m, val_m, test_m)
    s15['rk_long'] = s15.groupby('dt')['s_long'].rank(pct=True)
    s15['rk_short'] = s15.groupby('dt')['s_short'].rank(pct=True)
    look = {}
    for r in s15.itertuples(index=False): look[(r.Ticker, r.dt)] = (r.rk_long, r.rk_short)
    # 1h: train on its own (longer) history before the same test window
    tr1 = [m for m in sorted(d1['ym'].unique()) if m < val_m]
    s1 = fit_predict(d1, X1, f1, H1['ret'], p1, tr1, val_m, test_m)
    for lbl, direction, gate in CONFIGS:
        score = 's_long' if direction == 'long' else 's_short'
        for dt1, grp in s1.groupby('dt'):
            for _, r in grp.nlargest(3, score).iterrows():
                conf = look.get((r['Ticker'], dt1 + C45))
                if conf is None: continue
                rkl, rks = conf
                if direction == 'long':
                    if gate.get('long') is not None and not (rkl > gate['long']): continue
                    if gate.get('short') is not None and not (rks < gate['short']): continue
                    g = r['Next_Hour_Return']
                else:
                    if gate.get('short') is not None and not (rks > gate['short']): continue
                    if gate.get('long') is not None and not (rkl < gate['long']): continue
                    g = -r['Next_Hour_Return']
                trades[lbl].append((g, fi, r['ym']))
    del s15, s1, look; gc.collect()

print(f"\n{'='*100}")
print(f"  WALK-FORWARD DUAL-TF (pooled OOS, {len(folds)} folds, ~{len(folds)*H_TEST} months, 10 bps)")
print(f"{'='*100}")
print(f"  {'Config':<24} {'N':>6} {'NetWR%':>7} {'NetBps':>8} {'95% CI':>16} {'t':>6} {'+folds':>8} {'TotRet%':>9}")
print(f"  {'-'*98}")
summary = {}
for lbl, _, _ in CONFIGS:
    arr = trades[lbl]
    if not arr: print(f"  {lbl:<24} no trades"); continue
    g = np.array([a[0] for a in arr]); fold_id = np.array([a[1] for a in arr])
    net = g - COST; n = len(net)
    t = net.mean()/(net.std()/np.sqrt(n)) if net.std() > 0 else 0
    bs = [net[RNG.integers(0,n,n)].mean()*1e4 for _ in range(2000)]
    ci = (np.percentile(bs,2.5), np.percentile(bs,97.5))
    # per-fold positivity
    fold_means = [net[fold_id==fi].mean() for fi in range(1, len(folds)+1) if (fold_id==fi).sum() > 0]
    pos = sum(1 for fm in fold_means if fm > 0)
    tot = (np.prod(1+net)-1)*100
    sig = '***' if abs(t)>2.58 else ('**' if abs(t)>1.96 else ('*' if abs(t)>1.64 else ''))
    print(f"  {lbl:<24} {n:>6} {(net>0).mean()*100:>6.1f}% {net.mean()*1e4:>+7.1f} [{ci[0]:>+5.1f},{ci[1]:>+5.1f}] {t:>5.1f}{sig:<3} {pos:>3}/{len(fold_means):<3} {tot:>+8.1f}%")
    summary[lbl] = dict(n=n, net_bps=net.mean()*1e4, ci=ci, t=t, pos=pos, nf=len(fold_means))

# per-fold table for the key confirm configs
print(f"\n  Per-fold net bps (10 bps) — consistency check:")
print(f"  {'Fold':<6} " + " ".join(f"{f[2][0]:>9}" for f in folds))
for lbl in ['SHORT ShortConfirm p90', 'LONG  LongConfirm p90', 'SHORT LongAvoid p15']:
    arr = trades[lbl]
    if not arr: continue
    g = np.array([a[0] for a in arr]); fid = np.array([a[1] for a in arr])
    cells = []
    for fi in range(1, len(folds)+1):
        m = fid == fi
        cells.append(f"{(g[m].mean()-COST)*1e4:>+9.1f}" if m.sum() > 0 else f"{'—':>9}")
    print(f"  {lbl[:18]:<18} " + " ".join(cells))
print(f"\n{'='*100}")
print("  CI spanning 0 => not significant.  *p<.10 **p<.05 ***p<.01")
print(f"{'='*100}")
"""
DATA-SIZE ABLATION (exploratory, no verdict): is the 15m ranker's 1h-horizon edge
driven by DATA QUANTITY or by 15m-resolution FEATURES?

The 15m model (3.1M rows) beat the 1h model (320K rows) at the 1h horizon (IC 0.029 vs
0.027). 3.1M rows = same 3 years sampled 4x finer = ~4x more cross-sectional snapshots
(queries), NOT more regimes. This shrinks the 15m TRAINING data by subsampling timestamps
(keeping full cross-sections so rank:pairwise stays valid; TEST set untouched) and
re-measures the 1h-horizon IC.

  frac 1.00 -> full data (baseline, ~0.029)
  frac 0.25 -> ~matches the 1h model's query/row count
  frac 0.10 -> well below it
If IC falls with fraction toward ~0.026, DATA SIZE is the driver (hypothesis confirmed).
If IC holds ~0.029 even at 1h-size, it's 15m RESOLUTION/features, not quantity.
Writes data/research/15m_datasize_ablation_result.txt.
"""
import os, sys, json, warnings, gc
import numpy as np, pandas as pd, xgboost as xgb
from scipy.stats import rankdata, spearmanr
warnings.filterwarnings('ignore'); sys.path.append(os.getcwd())

M = dict(dir='models/v3_15min_clean', data='data/ranking_data_upstox_15min_3y_clean.csv', ret='Next_15Min_Return')
H_TEST, MIN_TRAIN = 4, 18
FRACS = [1.0, 0.5, 0.25, 0.10]
RNG = np.random.default_rng(7)
OUT = 'data/research/15m_datasize_ablation_result.txt'

def load(p, ret):
    df = pd.concat([c for c in pd.read_csv(p, chunksize=200_000)], ignore_index=True)
    df['dt'] = pd.to_datetime(df['DateTime']); df['ym'] = df['DateTime'].str[:7]; df['date'] = df['dt'].dt.date
    return df.dropna(subset=[ret]).reset_index(drop=True)
def Xfill(X):
    for ci in range(X.shape[1]):
        c = X[:, ci]; b = np.isnan(c) | np.isinf(c)
        if b.any(): X[b, ci] = float(np.nanmean(c[~b])) if (~b).any() else 0.0
    return X
def iranks(y, q, inv=False):
    out = np.zeros_like(y, dtype=int)
    for qid in np.unique(q):
        m = q == qid; v = -y[m] if inv else y[m]; out[m] = rankdata(v, method='ordinal') - 1
    return out
def fitdm(X, y, q, inv):
    d = xgb.DMatrix(X, label=iranks(y, q, inv)); d.set_group(pd.Series(q).groupby(q).size().values); return d
def perq_ic(score, targ, dts):
    ok = np.isfinite(targ); rhos = []
    for d in np.unique(dts[ok]):
        m = (dts == d) & ok
        if m.sum() < 5: continue
        r = spearmanr(score[m], targ[m]).correlation
        if np.isfinite(r): rhos.append(r)
    return float(np.mean(rhos))

print("Loading 15m dataset...")
with open(f'{M["dir"]}/metadata.json') as f: meta = json.load(f)
feats, params = meta['features'], dict(meta['params']); params['device'] = 'cpu'
df = load(M['data'], M['ret'])
g = df.groupby(['Ticker', 'date'])[M['ret']]
r0 = df[M['ret']]; r1 = g.shift(-1); r2 = g.shift(-2); r3 = g.shift(-3)
df['fwd1h'] = (1 + r0) * (1 + r1) * (1 + r2) * (1 + r3) - 1
X = Xfill(df[feats].values.astype(float))
ret = df[M['ret']].values; qid = df['Query_ID'].values; ymv = df['ym'].values; dts = df['dt'].values
fwd = df['fwd1h'].values
months = sorted(df['ym'].unique()); folds = []; i = MIN_TRAIN + 1
while i + 1 <= len(months):
    folds.append((months[:i-1], months[i-1], months[i:i+H_TEST])); i += H_TEST
print(f"  {len(folds)} folds; total rows {len(df):,}, unique timestamps {df['dt'].nunique():,}")

results = {}
for frac in FRACS:
    l1h, s1h, rows_used = [], [], []
    for fi, (tr_m, val_m, te_m) in enumerate(folds, 1):
        tr = np.isin(ymv, tr_m); va = np.isin(ymv, [val_m]); te = np.isin(ymv, te_m)
        if frac < 1.0:                      # subsample TRAIN timestamps (keep full cross-sections)
            tr_dts = np.unique(dts[tr]); keep = set(RNG.choice(tr_dts, int(len(tr_dts) * frac), replace=False))
            tr = tr & np.array([d in keep for d in dts])
        rows_used.append(int(tr.sum()))
        bl = xgb.train(params, fitdm(X[tr], ret[tr], qid[tr], False), 500,
                       evals=[(fitdm(X[va], ret[va], qid[va], False), 'v')], early_stopping_rounds=50, verbose_eval=False)
        bs = xgb.train(params, fitdm(X[tr], ret[tr], qid[tr], True), 500,
                       evals=[(fitdm(X[va], ret[va], qid[va], True), 'v')], early_stopping_rounds=50, verbose_eval=False)
        pl = bl.predict(xgb.DMatrix(X[te])); ps = bs.predict(xgb.DMatrix(X[te]))
        l1h.append(perq_ic(pl, fwd[te], dts[te])); s1h.append(perq_ic(ps, -fwd[te], dts[te]))
        del bl, bs; gc.collect()
    results[frac] = (np.mean(l1h), np.mean(s1h), int(np.mean(rows_used)))
    print(f"  frac={frac:.2f}  ~{results[frac][2]:>8,} train rows/fold  1h IC L={results[frac][0]:+.4f} S={results[frac][1]:+.4f}")

lines = ["DATA-SIZE ABLATION — 15m ranker 1h-horizon IC vs training-data fraction (exploratory)",
         "incumbent 1h ranker: long +0.0275 short +0.0254", ""]
for frac in FRACS:
    l, s, n = results[frac]
    lines.append(f"  frac={frac:.2f}  ~{n:>8,} rows/fold   1h IC  long={l:+.4f}  short={s:+.4f}")
lines += ["",
          "READ: if IC falls toward ~0.026 as fraction shrinks (esp. frac~0.25 ~ 1h model size),",
          "DATA QUANTITY is the driver. If IC holds ~0.029 even at small fractions, it's the 15m",
          "RESOLUTION/features, not the row count."]
os.makedirs('data/research', exist_ok=True)
with open(OUT, 'w') as f: f.write("\n".join(lines) + "\n")
print("\n".join(lines)); print(f"\nsaved -> {OUT}")

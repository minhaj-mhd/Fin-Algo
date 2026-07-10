"""
FEED 1-HOUR FEATURE-VECTORS TO THE 15m MODEL, PREDICT 1h (exploratory, no verdict).

The user's literal question: train the 15m ranker (on 15-min bars), then SCORE 1-HOUR
feature-vectors with it and predict the 1-hour-forward return. Legitimate because the
model eats CROSS-SECTIONALLY Z-SCORED features (scale mismatch removed) — the open
question is whether the 15m-learned cross-sectional relationships transfer to 1h-bar inputs.

CLEAN walk-forward (no in-sample leak): per fold, train 15m long/short rankers on 15-min
bars from train-months, then apply to 1h feature-vectors from test-months; IC vs
Next_Hour_Return (per 1h-timestamp cross-sectional Spearman).

Compare to:  dedicated 1h ranker IC +0.0275/+0.0254  ;  15m-model-held-1h +0.0287/+0.0307.
Memory-lean (usecols + float32) so it can run alongside the 5m fetch. No verdict authority.
"""
import os, sys, json, warnings, gc
import numpy as np, pandas as pd, xgboost as xgb
from scipy.stats import rankdata, spearmanr
warnings.filterwarnings('ignore'); sys.path.append(os.getcwd())

M15 = 'models/v3_15min_clean'
D15 = 'data/ranking_data_upstox_15min_3y_clean.csv'   # TRAIN source (15-min bars)
D1  = 'data/ranking_data_upstox_1h_v3_3y.csv'         # TEST source (feed 1h vectors to 15m model)
H_TEST, MIN_TRAIN = 4, 18
OUT = 'data/research/1h_data_on_15m_model_result.txt'

with open(f'{M15}/metadata.json') as f: feats = json.load(f)['features']
params = dict(json.load(open(f'{M15}/metadata.json'))['params']); params['device'] = 'cpu'

def load(path, ret):
    cols = feats + [ret, 'Query_ID', 'DateTime']
    dt = {c: 'float32' for c in feats}
    df = pd.concat([c for c in pd.read_csv(path, usecols=lambda x: x in cols, dtype=dt, chunksize=200_000)],
                   ignore_index=True)
    df['ym'] = df['DateTime'].str[:7]; df['dt'] = pd.to_datetime(df['DateTime'])
    return df.dropna(subset=[ret]).reset_index(drop=True)

print("Loading 15m (train) + 1h (test) ...", flush=True)
d15 = load(D15, 'Next_15Min_Return')
d1 = load(D1, 'Next_Hour_Return')
miss = [f for f in feats if f not in d1.columns]
print(f"  15m rows {len(d15):,} | 1h rows {len(d1):,} | features missing in 1h: {miss if miss else 'none'}", flush=True)

def fill(X):
    for ci in range(X.shape[1]):
        c = X[:, ci]; b = np.isnan(c) | np.isinf(c)
        if b.any(): X[b, ci] = float(np.nanmean(c[~b])) if (~b).any() else 0.0
    return X
X15 = fill(d15[feats].values.astype(np.float32)); X1 = fill(d1[feats].values.astype(np.float32))
def iranks(y, q, inv=False):
    out = np.zeros_like(y, dtype=int)
    for qid in np.unique(q):
        m = q == qid; v = -y[m] if inv else y[m]; out[m] = rankdata(v, method='ordinal') - 1
    return out
def fitdm(X, y, q, inv):
    d = xgb.DMatrix(X, label=iranks(y, q, inv)); d.set_group(pd.Series(q).groupby(q).size().values); return d
def perq_ic(score, targ, dts):
    rhos = []
    for d in np.unique(dts):
        m = dts == d
        if m.sum() < 5: continue
        r = spearmanr(score[m], targ[m]).correlation
        if np.isfinite(r): rhos.append(r)
    return float(np.mean(rhos))

ym15, q15, r15 = d15['ym'].values, d15['Query_ID'].values, d15['Next_15Min_Return'].values
ym1, dt1, r1 = d1['ym'].values, d1['dt'].values, d1['Next_Hour_Return'].values
months = sorted(set(d15['ym'].unique()) & set(d1['ym'].unique()))
folds = []; i = MIN_TRAIN + 1
while i + 1 <= len(months):
    folds.append((months[:i-1], months[i:i+H_TEST])); i += H_TEST
print(f"  {len(folds)} folds, OOS {folds[0][1][0]}->{folds[-1][1][-1]}", flush=True)

L, S = [], []
for fi, (tr_m, te_m) in enumerate(folds, 1):
    tr = np.isin(ym15, tr_m)                          # train on 15m bars
    te = np.isin(ym1, te_m)                           # test on 1h vectors
    bl = xgb.train(params, fitdm(X15[tr], r15[tr], q15[tr], False), 400, verbose_eval=False)
    bs = xgb.train(params, fitdm(X15[tr], r15[tr], q15[tr], True), 400, verbose_eval=False)
    pl = bl.predict(xgb.DMatrix(X1[te])); ps = bs.predict(xgb.DMatrix(X1[te]))
    L.append(perq_ic(pl, r1[te], dt1[te])); S.append(perq_ic(ps, -r1[te], dt1[te]))
    print(f"  fold {fi}/{len(folds)} {te_m[0]}->{te_m[-1]}  1h IC L={L[-1]:+.4f} S={S[-1]:+.4f}", flush=True)
    del bl, bs; gc.collect()

lines = [
    "FEED 1h FEATURE-VECTORS TO THE 15m MODEL -> predict 1h (clean WF, exploratory)",
    f"  15m-model scoring 1h vectors:  long {np.mean(L):+.4f}  short {np.mean(S):+.4f}",
    f"  dedicated 1h ranker:           long +0.0275      short +0.0254",
    f"  15m-model on 15m vectors, held 1h: long +0.0287  short +0.0307",
    "",
    "READ: if 15m-model-on-1h-vectors >= the dedicated 1h ranker, the 15m-learned",
    "relationships transfer to 1h inputs (cross-sectional z-scoring makes inputs comparable).",
    "If well below, the 15m model's structure is tied to 15m-bar dynamics. IC = ranking skill",
    "only; all sub-cost.",
]
os.makedirs('data/research', exist_ok=True)
with open(OUT, 'w') as f: f.write("\n".join(lines) + "\n")
print("\n" + "\n".join(lines)); print(f"\nsaved -> {OUT}")

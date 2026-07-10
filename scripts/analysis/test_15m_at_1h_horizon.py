"""
DOES THE 15m RANKER'S SKILL SURVIVE AT A 1-HOUR HORIZON? (exploratory, no verdict)

The 15m ranker has higher native IC (~0.06) than the 1h ranker (~0.026). But that IC
is vs 15-MIN-forward returns. This tests whether the SAME 15m model's scores predict
1-HOUR-forward returns — i.e., can we harvest 15m skill at the 1h horizon?

Clean WF (retrain 15m long+short per fold, score OOS test). On OOS predictions compute
per-timestamp cross-sectional Spearman IC vs:
  (a) native 15m-forward target (Next_15Min_Return)  -> should reproduce ~0.06
  (b) 1h-forward return = compound of the next 4 Next_15Min_Return, SESSION-MASKED
      (all 4 bars same trading day -> no overnight leak)
Compare the 1h-horizon IC to the incumbent 1h model (~0.026). Higher => 15m skill
transfers and there's something to chase; <= => no benefit over the existing 1h ranker.
Writes data/research/15m_at_1h_horizon_result.txt.
"""
import os, sys, json, warnings, gc
import numpy as np, pandas as pd, xgboost as xgb
from scipy.stats import rankdata, spearmanr
warnings.filterwarnings('ignore'); sys.path.append(os.getcwd())

M = dict(dir='models/v3_15min_clean', data='data/ranking_data_upstox_15min_3y_clean.csv', ret='Next_15Min_Return')
H_TEST, MIN_TRAIN = 4, 18
OUT = 'data/research/15m_at_1h_horizon_result.txt'

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
    ok = np.isfinite(targ)
    rhos = []
    for d in np.unique(dts[ok]):
        m = (dts == d) & ok
        if m.sum() < 5: continue
        r = spearmanr(score[m], targ[m]).correlation
        if np.isfinite(r): rhos.append(r)
    return float(np.mean(rhos)), len(rhos)

print("Loading 15m dataset...")
with open(f'{M["dir"]}/metadata.json') as f: meta = json.load(f)
feats, params = meta['features'], dict(meta['params']); params['device'] = 'cpu'
df = load(M['data'], M['ret'])

# 1h-forward target = compound of next 4 Next_15Min_Return within the SAME trading day (per ticker)
print("Building session-masked 1h-forward target...")
df = df.sort_values(['Ticker', 'dt']).reset_index(drop=True)
n15 = df[M['ret']].values
fwd1h = np.full(len(df), np.nan)
for _, idx in df.groupby(['Ticker', 'date']).indices.items():
    idx = np.sort(idx)
    for j in range(len(idx) - 3):           # need 4 consecutive same-day bars
        seg = n15[idx[j:j+4]]
        fwd1h[idx[j]] = np.prod(1 + seg) - 1
df['fwd1h'] = fwd1h
X = Xfill(df[feats].values.astype(float))
ret = df[M['ret']].values; qid = df['Query_ID'].values; ymv = df['ym'].values; dts = df['dt'].values
months = sorted(df['ym'].unique()); folds = []; i = MIN_TRAIN + 1
while i + 1 <= len(months):
    folds.append((months[:i-1], months[i-1], months[i:i+H_TEST])); i += H_TEST
print(f"  {len(folds)} folds OOS {folds[0][2][0]} -> {folds[-1][2][-1]}")

agg = {'l15': [], 's15': [], 'l1h': [], 's1h': []}
for fi, (tr_m, val_m, te_m) in enumerate(folds, 1):
    tr = np.isin(ymv, tr_m); va = np.isin(ymv, [val_m]); te = np.isin(ymv, te_m)
    bl = xgb.train(params, fitdm(X[tr], ret[tr], qid[tr], False), 500,
                   evals=[(fitdm(X[va], ret[va], qid[va], False), 'v')], early_stopping_rounds=50, verbose_eval=False)
    bs = xgb.train(params, fitdm(X[tr], ret[tr], qid[tr], True), 500,
                   evals=[(fitdm(X[va], ret[va], qid[va], True), 'v')], early_stopping_rounds=50, verbose_eval=False)
    pl = bl.predict(xgb.DMatrix(X[te])); ps = bs.predict(xgb.DMatrix(X[te]))
    l15, _ = perq_ic(pl, ret[te], dts[te]);   s15, _ = perq_ic(ps, -ret[te], dts[te])
    l1h, n = perq_ic(pl, df['fwd1h'].values[te], dts[te]); s1h, _ = perq_ic(ps, -df['fwd1h'].values[te], dts[te])
    for k, v in zip(agg, (l15, s15, l1h, s1h)): agg[k].append(v)
    print(f"  fold {fi}/{len(folds)} {te_m[0]}->{te_m[-1]}  15m IC L{l15:+.4f}/S{s15:+.4f}   1h IC L{l1h:+.4f}/S{s1h:+.4f}")
    del bl, bs; gc.collect()

m = {k: np.mean(v) for k, v in agg.items()}
lines = [
    "DOES 15m RANKER SKILL SURVIVE AT 1H HORIZON? (exploratory, no verdict)",
    f"15m model, clean {len(folds)}-fold WF OOS, full-universe cross-sectional IC:",
    f"  vs NATIVE 15m-forward return:  long {m['l15']:+.4f}  short {m['s15']:+.4f}   (expect ~0.06)",
    f"  vs 1h-forward return (masked): long {m['l1h']:+.4f}  short {m['s1h']:+.4f}",
    f"  incumbent dedicated 1h ranker:  long +0.0275  short +0.0254",
    "",
    "VERDICT: if 1h-horizon IC > ~0.03 (clears the 1h ranker), 15m skill transfers and is worth",
    "chasing; if <= ~0.026, the 15m edge is horizon-specific (decays by 1h) and offers no lift",
    "over the existing 1h model. (IC alone is not post-cost edge — both rankers are sub-cost.)",
]
os.makedirs('data/research', exist_ok=True)
with open(OUT, 'w') as f: f.write("\n".join(lines) + "\n")
print("\n".join(lines)); print(f"\nsaved -> {OUT}")

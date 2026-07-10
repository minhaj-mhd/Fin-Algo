"""
RESOLUTION x HORIZON MATRIX (exploratory, no verdict authority).

Tests the user hypothesis "finer resolution = clearer patterns, generalizes" and the
refinement "a model's resolution must match the signal's natural timescale."

For each intraday ranker {15m, 30min, 1h}, clean WF (retrain per fold, OOS), measure
full-universe cross-sectional IC at multiple FORWARD horizons (session-masked compound
of native fwd-returns within ticker/day -> no overnight leak):
  15m  model: horizons 15m / 1h / 2h   (1 / 4 / 8 native bars)
  30min model: horizons 30m / 1h / 2h  (1 / 2 / 4 native bars)
  1h   model: horizons 1h / 2h         (1 / 2 native bars)

Datasets loaded SEQUENTIALLY (one in memory at a time + gc) to avoid the concurrent-load
OOM that crashed an earlier run. Reads each model's own metadata feature list/params.
Expected pattern if the hypothesis holds: each model's IC peaks at/near its native horizon
and decays at longer ones; finer models lead at the shared 1h horizon. (IC is ranking skill,
NOT post-cost edge — all intraday rankers are sub-cost; see signal_economics notes.)
Writes data/research/resolution_horizon_matrix_result.txt.
"""
import os, sys, json, warnings, gc
import numpy as np, pandas as pd, xgboost as xgb
from scipy.stats import rankdata, spearmanr
warnings.filterwarnings('ignore'); sys.path.append(os.getcwd())

MODELS = [
    dict(name='15m',  dir='models/v3_15min_clean',  data='data/ranking_data_upstox_15min_3y_clean.csv',
         ret='Next_15Min_Return', horizons={'15m': 1, '1h': 4, '2h': 8}),
    dict(name='30min', dir='models/v2_30min_v3_3y',  data='data/ranking_data_upstox_30min_v3_3y.csv',
         ret='Next_30Min_Return', horizons={'30m': 1, '1h': 2, '2h': 4}),
    dict(name='1h',   dir='models/v10_native_1h',    data='data/ranking_data_upstox_1h_v3_3y.csv',
         ret='Next_Hour_Return', horizons={'1h': 1, '2h': 2}),
]
H_TEST, MIN_TRAIN = 4, 18
OUT = 'data/research/resolution_horizon_matrix_result.txt'

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
    return float(np.mean(rhos)) if rhos else np.nan

matrix = {}
for cfg in MODELS:
    print(f"\n=== {cfg['name']} model ===")
    with open(f"{cfg['dir']}/metadata.json") as f: meta = json.load(f)
    feats, params = meta['features'], dict(meta['params']); params['device'] = 'cpu'
    df = load(cfg['data'], cfg['ret'])
    # multi-horizon session-masked forward returns
    g = df.groupby(['Ticker', 'date'])[cfg['ret']]
    shifts = {k: g.shift(-k) for k in range(max(cfg['horizons'].values()))}
    fwd = {}
    for hname, hbars in cfg['horizons'].items():
        prod = np.ones(len(df))
        for k in range(hbars): prod = prod * (1 + shifts[k].values)
        fwd[hname] = prod - 1
    X = Xfill(df[feats].values.astype(float))
    ret = df[cfg['ret']].values; qid = df['Query_ID'].values; ymv = df['ym'].values; dts = df['dt'].values
    months = sorted(df['ym'].unique()); folds = []; i = MIN_TRAIN + 1
    while i + 1 <= len(months):
        folds.append((months[:i-1], months[i-1], months[i:i+H_TEST])); i += H_TEST
    print(f"  {len(folds)} folds OOS {folds[0][2][0]}->{folds[-1][2][-1]}, {len(df):,} rows")
    acc = {h: {'L': [], 'S': []} for h in cfg['horizons']}
    for fi, (tr_m, val_m, te_m) in enumerate(folds, 1):
        tr = np.isin(ymv, tr_m); va = np.isin(ymv, [val_m]); te = np.isin(ymv, te_m)
        bl = xgb.train(params, fitdm(X[tr], ret[tr], qid[tr], False), 500,
                       evals=[(fitdm(X[va], ret[va], qid[va], False), 'v')], early_stopping_rounds=50, verbose_eval=False)
        bs = xgb.train(params, fitdm(X[tr], ret[tr], qid[tr], True), 500,
                       evals=[(fitdm(X[va], ret[va], qid[va], True), 'v')], early_stopping_rounds=50, verbose_eval=False)
        pl = bl.predict(xgb.DMatrix(X[te])); ps = bs.predict(xgb.DMatrix(X[te]))
        for h in cfg['horizons']:
            acc[h]['L'].append(perq_ic(pl, fwd[h][te], dts[te]))
            acc[h]['S'].append(perq_ic(ps, -fwd[h][te], dts[te]))
        print(f"    fold {fi}/{len(folds)} done")
        del bl, bs; gc.collect()
    matrix[cfg['name']] = {h: (np.nanmean(acc[h]['L']), np.nanmean(acc[h]['S'])) for h in cfg['horizons']}
    for h in cfg['horizons']:
        print(f"  {cfg['name']} @ {h:>4}: IC L={matrix[cfg['name']][h][0]:+.4f} S={matrix[cfg['name']][h][1]:+.4f}")
    del df, X, fwd, shifts, g; gc.collect()

lines = ["RESOLUTION x HORIZON IC MATRIX (exploratory, no verdict) — long / short",
         "(each model retrained clean WF; IC = full-universe cross-sectional Spearman; session-masked)", ""]
allh = ['15m', '30m', '1h', '2h']
lines.append(f"  {'model':<7}" + "".join(f"{h:>16}" for h in allh))
for name in ['15m', '30min', '1h']:
    row = f"  {name:<7}"
    for h in allh:
        if h in matrix[name]:
            l, s = matrix[name][h]; row += f"  {l:+.4f}/{s:+.4f}"
        else:
            row += f"{'—':>16}"
    lines.append(row)
lines += ["",
          "READ: if each model's IC peaks near its NATIVE horizon and decays at longer ones,",
          "and finer models lead at the shared 1h column, the 'match resolution to signal",
          "timescale' picture holds. NOTE: IC is ranking skill only — all intraday rankers are",
          "sub-cost; profitability comes from the HORIZON axis (3-day daily sleeve), not finer bars."]
os.makedirs('data/research', exist_ok=True)
with open(OUT, 'w') as f: f.write("\n".join(lines) + "\n")
print("\n" + "\n".join(lines)); print(f"\nsaved -> {OUT}")

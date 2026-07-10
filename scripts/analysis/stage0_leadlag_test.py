"""
CST STAGE-0 FALSIFICATION (exploratory, no verdict authority) — the pre-registered
cheapest test of "can the 1h ranker be improved by NEW (cross-asset/lead-lag) info?"
Spec: [[Cross-Sectional Transformer Architecture Proposal]] §8 Stage 0.

Incumbent baseline (from data/model_analysis/v10_v18_independent/walkforward_preds.npz):
  long rho = +0.0275, short rho = +0.0254  -> the bar to beat "meaningfully".

The 1h dataset already carries CONTEMPORANEOUS Market_Mean_Return / Relative_Return,
but NO lagged market/breadth/dispersion. This adds exactly that lead-lag block:
  mkt_lag{1,2,4}      : universe-mean return at the 1/2/4 PRIOR bars
  disp, disp_lag1     : cross-sectional std of returns (now + prior bar)
  breadth, breadth_l1 : fraction of universe with positive return (now + prior bar)
  n500_lag{1,2,4}     : external Nifty500 index return at prior 1/2/4 bars
  mkt_accel           : mkt_lag1 - mkt_lag2
All features known at bar close T; label Next_Hour_Return is the NEXT hour -> no look-ahead.

Retrains the v10 spec under purged WF on BASE vs BASE+lead-lag and compares per-query
Spearman (long & short). If aug-rho does not move meaningfully above baseline, the
cross-name signal is NOT present at 1h granularity -> CST dead on arrival; redirect to
order-flow/microstructure. Writes data/research/stage0_leadlag_result.txt.
"""
import os, sys, json, warnings, gc
import numpy as np, pandas as pd, xgboost as xgb
from scipy.stats import rankdata, spearmanr
warnings.filterwarnings('ignore'); sys.path.append(os.getcwd())

H1 = dict(dir='models/v10_native_1h', data='data/ranking_data_upstox_1h_v3_3y.csv', ret='Next_Hour_Return')
N500 = 'data/raw_index_cache/nifty500_1h.csv'
H_TEST, MIN_TRAIN = 4, 18
OUT = 'data/research/stage0_leadlag_result.txt'

def load(p, ret):
    df = pd.concat([c for c in pd.read_csv(p, chunksize=200_000)], ignore_index=True)
    df['dt'] = pd.to_datetime(df['DateTime']); df['ym'] = df['DateTime'].str[:7]
    return df.dropna(subset=[ret]).reset_index(drop=True)
def iranks(y, q, inv=False):
    out = np.zeros_like(y, dtype=int)
    for qid in np.unique(q):
        m = q == qid; v = -y[m] if inv else y[m]; out[m] = rankdata(v, method='ordinal') - 1
    return out
def Xfill(X):
    for ci in range(X.shape[1]):
        c = X[:, ci]; b = np.isnan(c) | np.isinf(c)
        if b.any(): X[b, ci] = float(np.nanmean(c[~b])) if (~b).any() else 0.0
    return X
def fitdm(X, y, q, inv):
    d = xgb.DMatrix(X, label=iranks(y, q, inv)); d.set_group(pd.Series(q).groupby(q).size().values); return d
def perq_rho(score, targ, q):
    rhos = []
    for qid in np.unique(q):
        m = q == qid
        if m.sum() < 5: continue
        r = spearmanr(score[m], targ[m]).correlation
        if np.isfinite(r): rhos.append(r)
    return float(np.mean(rhos))

print("Loading 1h dataset + market cache...")
with open(f'{H1["dir"]}/metadata.json') as f: meta = json.load(f)
base_feats, params = meta['features'], dict(meta['params']); params['device'] = 'cpu'
df = load(H1['data'], H1['ret'])

# --- build lead-lag block from the universe itself (per-timestamp series, lagged) ---
g = df.groupby('dt')['Return']
mkt = g.mean(); disp = g.std(); breadth = g.apply(lambda s: (s > 0).mean())
ser = pd.DataFrame({'mkt': mkt, 'disp': disp, 'breadth': breadth}).sort_index()
for L in (1, 2, 4): ser[f'mkt_lag{L}'] = ser['mkt'].shift(L)
ser['disp_lag1'] = ser['disp'].shift(1); ser['breadth_lag1'] = ser['breadth'].shift(1)
ser['mkt_accel'] = ser['mkt_lag1'] - ser['mkt_lag2']
# --- external Nifty500 index lead-lag ---
n5 = pd.read_csv(N500, parse_dates=['timestamp']).set_index('timestamp').sort_index()
n5ret = n5['close'].pct_change()
for L in (1, 2, 4): ser[f'n500_lag{L}'] = n5ret.reindex(ser.index).shift(L)
new_feats = ['mkt_lag1','mkt_lag2','mkt_lag4','disp','disp_lag1','breadth','breadth_lag1',
             'mkt_accel','n500_lag1','n500_lag2','n500_lag4']
df = df.merge(ser[new_feats], left_on='dt', right_index=True, how='left')

Xb = Xfill(df[base_feats].values.astype(float))
Xa = Xfill(df[base_feats + new_feats].values.astype(float))
ret = df[H1['ret']].values; qid = df['Query_ID'].values; ymv = df['ym'].values
months = sorted(df['ym'].unique()); folds = []; i = MIN_TRAIN + 1
while i + 1 <= len(months):
    folds.append((months[:i-1], months[i-1], months[i:i+H_TEST])); i += H_TEST
print(f"  {len(folds)} folds OOS {folds[0][2][0]} -> {folds[-1][2][-1]}; +{len(new_feats)} lead-lag feats")

def run(X, tag):
    SL = {'rho_l': [], 'rho_s': []}; importances = np.zeros(X.shape[1])
    for fi, (tr_m, val_m, te_m) in enumerate(folds, 1):
        tr = np.isin(ymv, tr_m); va = np.isin(ymv, [val_m]); te = np.isin(ymv, te_m)
        bl = xgb.train(params, fitdm(X[tr], ret[tr], qid[tr], False), 500,
                       evals=[(fitdm(X[va], ret[va], qid[va], False), 'v')], early_stopping_rounds=50, verbose_eval=False)
        bs = xgb.train(params, fitdm(X[tr], ret[tr], qid[tr], True), 500,
                       evals=[(fitdm(X[va], ret[va], qid[va], True), 'v')], early_stopping_rounds=50, verbose_eval=False)
        pl = bl.predict(xgb.DMatrix(X[te])); ps = bs.predict(xgb.DMatrix(X[te]))
        SL['rho_l'].append(perq_rho(pl, ret[te], qid[te]))
        SL['rho_s'].append(perq_rho(ps, -ret[te], qid[te]))
        print(f"    [{tag}] fold {fi}/{len(folds)} {te_m[0]}->{te_m[-1]}  rho_l={SL['rho_l'][-1]:+.4f} rho_s={SL['rho_s'][-1]:+.4f}")
    return np.mean(SL['rho_l']), np.mean(SL['rho_s'])

print("BASE (incumbent feature set):"); bl_, bs_ = run(Xb, 'BASE')
print("AUG (+ lead-lag block):");      al_, as_ = run(Xa, 'AUG')

lines = [
    "CST STAGE-0 FALSIFICATION — 1h ranker + cross-asset/lead-lag features (exploratory, no verdict)",
    f"Incumbent npz baseline:  long +0.0275  short +0.0254",
    f"BASE re-run (this harness): long {bl_:+.4f}  short {bs_:+.4f}",
    f"AUG (+{len(new_feats)} lead-lag): long {al_:+.4f}  short {as_:+.4f}",
    f"DELTA (AUG - BASE):       long {al_-bl_:+.4f}  short {as_-bs_:+.4f}",
    "",
    "VERDICT: lead-lag info is PRESENT at 1h granularity only if delta is materially positive",
    "(rule of thumb >= +0.01 rho AND aug clears ~0.035). Else CST dead on arrival -> redirect",
    "to order-flow/microstructure data roadmap (the only remaining new-information lever).",
]
os.makedirs('data/research', exist_ok=True)
with open(OUT, 'w') as f: f.write("\n".join(lines) + "\n")
print("\n".join(lines)); print(f"\nsaved -> {OUT}")

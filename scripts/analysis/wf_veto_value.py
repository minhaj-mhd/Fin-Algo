"""
VETO VALUE: as a filter (not an alpha), does the model reliably separate good trades from bad?
Host = 1h top-3 candidates (both directions). The 15m model acts as the veto signal.

For each direction we split host candidates by the 15m signal into PASS vs VETO and measure
the SPREAD (PASS - VETO) with a two-sample test. A veto adds value if VETO trades are
significantly worse than PASS trades (regardless of absolute level — veto doesn't pay the cost).

Two veto logics tested:
  OWN-agree : LONG keeps high 15m long-rank / SHORT keeps high 15m short-rank.
  CROSS-veto: LONG vetoed if 15m short-rank high (model says it'll fall); SHORT vice-versa.

Walk-forward pooled OOS. Reports gross spread (veto value is pre-cost separation) + net@10.
Saves the pooled candidate table for reuse.
"""
import os, sys, json, warnings, gc
import numpy as np, pandas as pd, xgboost as xgb
from scipy.stats import rankdata, ttest_ind
warnings.filterwarnings('ignore'); sys.path.append(os.getcwd())

M15 = dict(dir='models/v3_15min_clean', data='data/ranking_data_upstox_15min_3y_clean.csv', ret='Next_15Min_Return')
H1  = dict(dir='models/v10_native_1h', data='data/ranking_data_upstox_1h_v3_3y.csv', ret='Next_Hour_Return')
H_TEST, MIN_TRAIN, COST = 4, 18, 10/1e4
OUT = 'data/dual_tf_veto_candidates.csv'
RNG = np.random.default_rng(0); C45 = pd.Timedelta(minutes=45)

def load(p, ret):
    df = pd.concat([c for c in pd.read_csv(p, chunksize=200_000)], ignore_index=True)
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
def fitdm(X, y, q, inv):
    d = xgb.DMatrix(X, label=iranks(y, q, inv)); d.set_group(pd.Series(q).groupby(q).size().values); return d
def fit_predict(d, X, fe, ret, params, tr_m, val_m, te_m):
    tr = d['ym'].isin(tr_m).values; va = d['ym'].isin([val_m]).values; te = d['ym'].isin(te_m).values
    bl = xgb.train(params, fitdm(X[tr], d[ret].values[tr], d['Query_ID'].values[tr], False), 500,
                   evals=[(fitdm(X[va], d[ret].values[va], d['Query_ID'].values[va], False),'v')], early_stopping_rounds=50, verbose_eval=False)
    bs = xgb.train(params, fitdm(X[tr], d[ret].values[tr], d['Query_ID'].values[tr], True), 500,
                   evals=[(fitdm(X[va], d[ret].values[va], d['Query_ID'].values[va], True),'v')], early_stopping_rounds=50, verbose_eval=False)
    sub = d[te].copy(); sub['sL'] = bl.predict(xgb.DMatrix(X[te])); sub['sS'] = bs.predict(xgb.DMatrix(X[te]))
    return sub

print("Loading...")
with open(f'{M15["dir"]}/metadata.json') as f: m15m = json.load(f)
with open(f'{H1["dir"]}/metadata.json') as f: m1m = json.load(f)
f15, p15 = m15m['features'], m15m['params']; f1, p1 = m1m['features'], m1m['params']
d15 = load(M15['data'], M15['ret']); X15 = Xmat(d15, f15)
d1 = load(H1['data'], H1['ret']); X1 = Xmat(d1, f1)
months = sorted(d15['ym'].unique()); folds = []; i = MIN_TRAIN+1
while i+1 <= len(months):
    folds.append((months[:i-1], months[i-1], months[i:i+H_TEST])); i += H_TEST
print(f"  {len(folds)} folds OOS {folds[0][2][0]} -> {folds[-1][2][-1]}")

rows = []
for fi, (tr_m, val_m, te_m) in enumerate(folds, 1):
    print(f"  fold {fi}/{len(folds)} {te_m[0]}->{te_m[-1]}")
    s15 = fit_predict(d15, X15, f15, M15['ret'], p15, tr_m, val_m, te_m)
    s15['rkL'] = s15.groupby('dt')['sL'].rank(pct=True); s15['rkS'] = s15.groupby('dt')['sS'].rank(pct=True)
    look = {(r.Ticker, r.dt): (r.rkL, r.rkS) for r in s15.itertuples(index=False)}
    tr1 = [m for m in sorted(d1['ym'].unique()) if m < val_m]
    s1 = fit_predict(d1, X1, f1, H1['ret'], p1, tr1, val_m, te_m)
    for dt1, grp in s1.groupby('dt'):
        gl = grp.nlargest(3, 'sL'); gs = grp.nlargest(3, 'sS')
        for direction, gg in [('long', gl), ('short', gs)]:
            for _, r in gg.iterrows():
                conf = look.get((r['Ticker'], dt1 + C45))
                if conf is None: continue
                rkL, rkS = conf
                rows.append((fi, r['ym'], direction, r['Next_Hour_Return'], rkL, rkS))
    del s15, s1, look; gc.collect()
C = pd.DataFrame(rows, columns=['fold','ym','dir','ret','rkL','rkS'])
C.to_csv(OUT, index=False)
print(f"  pooled candidates {len(C):,} saved -> {OUT}\n")

def grp_stats(g, sgn):
    net = g['ret'].values*sgn - COST; gross = g['ret'].values*sgn
    return len(g), (net>0).mean()*100, gross.mean()*1e4, net.mean()*1e4

print("="*92)
print("  VETO VALUE — host = 1h top-3; 15m model as veto signal (WF pooled OOS)")
print("="*92)
for direction, sgn, own, cross in [('LONG', 1, 'rkL', 'rkS'), ('SHORT', -1, 'rkS', 'rkL')]:
    sub = C[C['dir'] == direction].copy()
    print(f"\n  ===== {direction} candidates (N={len(sub)}) =====")
    # baseline
    n, wr, g, net = grp_stats(sub, sgn)
    print(f"  {'ALL host':<28} N={n:>5} WR={wr:>5.1f}% gross={g:>+6.1f} net@10={net:>+6.1f}")
    # OWN-agree veto: PASS = own-rank top tercile, VETO = bottom tercile
    for logic, col, hi_is_pass in [('OWN-agree', own, True), ('CROSS-veto', cross, False)]:
        q33, q67 = sub[col].quantile(0.33), sub[col].quantile(0.67)
        if hi_is_pass:
            passg = sub[sub[col] >= q67]; vetog = sub[sub[col] <= q33]
        else:  # cross: high opposite-rank = veto, so PASS = low opposite-rank
            passg = sub[sub[col] <= q33]; vetog = sub[sub[col] >= q67]
        pn, pwr, pg, pnet = grp_stats(passg, sgn); vn, vwr, vg, vnet = grp_stats(vetog, sgn)
        # two-sample t on gross (separation power, cost-agnostic)
        t, p = ttest_ind(passg['ret'].values*sgn, vetog['ret'].values*sgn, equal_var=False)
        sig = '***' if p<.001 else ('**' if p<.01 else ('*' if p<.05 else 'ns'))
        print(f"  [{logic}]  PASS N={pn:>5} WR={pwr:>5.1f}% gross={pg:>+6.1f} net@10={pnet:>+6.1f}")
        print(f"  {'':>10}  VETO N={vn:>5} WR={vwr:>5.1f}% gross={vg:>+6.1f} net@10={vnet:>+6.1f}")
        print(f"  {'':>10}  -> SEPARATION (PASS-VETO) gross spread = {pg-vg:>+5.1f} bps | WR spread = {pwr-vwr:>+4.1f}pp | p={p:.1e} {sig}")
print("\n" + "="*92)
print("  Veto value = PASS-VETO separation (cost-agnostic). Significant spread => usable filter.")
print("="*92)

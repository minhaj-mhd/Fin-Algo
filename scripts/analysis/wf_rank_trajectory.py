"""
RANK-TRAJECTORY VETO (exploratory — no verdict authority).

Hypothesis: as a 1h top-3 signal fires, the MOMENTUM of the 15m cross-sectional
conviction rank over the 4 candles leading into entry tells us whether to act.
  THRIVING  (rank slope rising)  -> take the 1h shot.
  DIMINISHING(rank slope falling) -> block.

This EXTENDS scripts/analysis/wf_veto_value.py (which only looked at the single
entry bar) to the 4-candle trajectory. Same validated machinery:
  - real cross-sectional percentile ranks (groupby('dt').rank(pct=True)),
  - 1h<->15m alignment via the +45min confirm bar (forward NHR is the NEXT hour,
    so all 4 trajectory bars at dt1+{0,15,30,45} are strictly BEFORE entry -> no look-ahead),
  - purged walk-forward, pooled OOS, bootstrap/t-stats, cost @ 10 bps.

CRITICAL CONTROL: rising 15m long-conviction is mechanically tied to the stock
having just risen (IBS/Buy_Pressure microstructure). So we also split on raw
PRICE momentum (cumulative own Return over the same 4 candles). If rank-slope
separates no better than price-momentum, it carries no NEW information.

Two questions, both must pass for "take the shot":
  (1) SEPARATION: THRIVING beats DIMINISHING on gross return (t-test).
  (2) DEPLOYABILITY: the THRIVING bucket is net-POSITIVE after 10 bps.
"""
import os, sys, json, warnings, gc
import numpy as np, pandas as pd, xgboost as xgb
from scipy.stats import rankdata, ttest_ind
warnings.filterwarnings('ignore'); sys.path.append(os.getcwd())

M15 = dict(dir='models/v3_15min_clean', data='data/ranking_data_upstox_15min_3y_clean.csv', ret='Next_15Min_Return')
H1  = dict(dir='models/v10_native_1h',  data='data/ranking_data_upstox_1h_v3_3y.csv',        ret='Next_Hour_Return')
H_TEST, MIN_TRAIN, COST = 4, 18, 10/1e4
OFFSETS = [0, 15, 30, 45]                      # 4 completed 15m bars leading into entry
OUT = 'data/rank_trajectory_candidates.csv'

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
def fit_predict(d, X, ret, params, tr_m, val_m, te_m):
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
for p in (p15, p1): p['device'] = 'cpu'   # exploratory: run on CPU
d15 = load(M15['data'], M15['ret']); X15 = Xmat(d15, f15)
d1  = load(H1['data'],  H1['ret']);  X1  = Xmat(d1, f1)
months = sorted(d15['ym'].unique()); folds = []; i = MIN_TRAIN+1
while i+1 <= len(months):
    folds.append((months[:i-1], months[i-1], months[i:i+H_TEST])); i += H_TEST
print(f"  {len(folds)} folds OOS {folds[0][2][0]} -> {folds[-1][2][-1]}")

OFF = [pd.Timedelta(minutes=m) for m in OFFSETS]
rows = []
for fi, (tr_m, val_m, te_m) in enumerate(folds, 1):
    print(f"  fold {fi}/{len(folds)} {te_m[0]}->{te_m[-1]}")
    s15 = fit_predict(d15, X15, M15['ret'], p15, tr_m, val_m, te_m)
    s15['rkL'] = s15.groupby('dt')['sL'].rank(pct=True)
    s15['rkS'] = s15.groupby('dt')['sS'].rank(pct=True)
    # lookup: (Ticker, dt) -> (rkL, rkS, own_return_of_candle)
    look = {(r.Ticker, r.dt): (r.rkL, r.rkS, r.Return) for r in s15.itertuples(index=False)}
    tr1 = [m for m in sorted(d1['ym'].unique()) if m < val_m]
    s1 = fit_predict(d1, X1, H1['ret'], p1, tr1, val_m, te_m)
    for dt1, grp in s1.groupby('dt'):
        gl = grp.nlargest(3, 'sL'); gs = grp.nlargest(3, 'sS')
        for direction, gg in [('long', gl), ('short', gs)]:
            for _, r in gg.iterrows():
                traj = [look.get((r['Ticker'], dt1 + o)) for o in OFF]
                if any(t is None for t in traj): continue
                rkL = np.array([t[0] for t in traj]); rkS = np.array([t[1] for t in traj])
                rets = np.array([t[2] for t in traj])           # own per-candle returns
                rk = rkL if direction == 'long' else rkS         # own-side conviction track
                x = np.arange(len(rk))
                slope = np.polyfit(x, rk, 1)[0]                  # rank momentum
                pmom = float(np.nansum(rets))                    # price momentum control
                rows.append((fi, r['ym'], direction, r['Next_Hour_Return'],
                             rk[-1], slope, rk[-1]-rk[0], pmom))
    del s15, s1, look; gc.collect()

C = pd.DataFrame(rows, columns=['fold','ym','dir','ret','rk_last','slope','delta','pmom'])
C.to_csv(OUT, index=False)
print(f"  pooled candidates {len(C):,} saved -> {OUT}\n")

def stats(g, sgn):
    if len(g) == 0: return (0, np.nan, np.nan, np.nan)
    net = g['ret'].values*sgn - COST; gross = g['ret'].values*sgn
    return len(g), (net>0).mean()*100, gross.mean()*1e4, net.mean()*1e4

def split_report(sub, sgn, col, label):
    """Split candidates by sign of `col` (THRIVING > median vs DIMINISHING < median)."""
    med = sub[col].median()
    up = sub[sub[col] > med]; dn = sub[sub[col] <= med]
    un, uwr, ug, unet = stats(up, sgn); dnn, dwr, dg, dnet = stats(dn, sgn)
    t, p = ttest_ind(up['ret'].values*sgn, dn['ret'].values*sgn, equal_var=False)
    sig = '***' if p<.001 else ('**' if p<.01 else ('*' if p<.05 else 'ns'))
    print(f"  [{label}]")
    print(f"     THRIVING  N={un:>5} WR={uwr:>5.1f}% gross={ug:>+6.1f} net@10={unet:>+6.1f}  <- deployable iff net@10>0")
    print(f"     DIMINISH  N={dnn:>5} WR={dwr:>5.1f}% gross={dg:>+6.1f} net@10={dnet:>+6.1f}")
    print(f"     -> SEPARATION gross={ug-dg:>+5.1f}bps  WR={uwr-dwr:>+4.1f}pp  t={t:>+5.2f} p={p:.1e} {sig}")

print("="*94)
print("  RANK-TRAJECTORY veto (host=1h top-3; 15m 4-candle rank slope). EXPLORATORY — no verdict.")
print("="*94)
for direction, sgn in [('long', 1), ('short', -1)]:
    sub = C[C['dir'] == direction].copy()
    n, wr, g, net = stats(sub, sgn)
    print(f"\n  ===== {direction.upper()} candidates (N={len(sub)}) =====")
    print(f"  ALL host                 N={n:>5} WR={wr:>5.1f}% gross={g:>+6.1f} net@10={net:>+6.1f}")
    split_report(sub, sgn, 'slope', 'RANK-SLOPE  (your hypothesis: rising conviction)')
    split_report(sub, sgn, 'pmom',  'PRICE-MOM   (control: did it just go up)')
    # does rank-slope add anything beyond price momentum? correlation of the two splits
    rho = sub[['slope','pmom']].corr().iloc[0,1]
    print(f"  corr(rank_slope, price_mom) = {rho:+.3f}  (high => rank-slope is just price-momentum repackaged)")
print("\n" + "="*94)
print("  TAKE-THE-SHOT requires BOTH: significant separation AND THRIVING net@10 > 0.")
print("  If RANK-SLOPE separation ~ PRICE-MOM separation, the 15m model adds no new info.")
print("="*94)

"""
EARLY-EXIT overlay via 15m conviction decay (exploratory — no verdict authority).

Idea: after entering a 1h top-3 trade, watch the 15m own-side conviction rank
during the hold. If it decays, EXIT early to cut the loss instead of holding the
full hour. Cost is identical (one round-trip either way), so this is a pure
timing overlay — the only question is whether it realizes a better return.

Timeline (validated alignment): 1h bar left-labeled at dt1 closes at dt1+60 =
ENTRY. Next_Hour_Return covers dt1+60..dt1+120 = 4 fifteen-min sub-periods whose
returns are N15 at dt1+{60,75,90,105}. Decision checkpoints (conviction observed
at the bar END, return-so-far already realized -> no look-ahead):
   after sub1 (dt1+75), after sub2 (dt1+90), after sub3 (dt1+105).
If an exit triggers at checkpoint k, realize the compounded return of the first
k sub-periods; otherwise hold the full hour.

Policies compared (all pay the same 10 bps round-trip):
  FULL-HOLD     : baseline, hold the whole hour.
  CONV-LEVEL    : exit at first checkpoint where own-side rank < 0.5 (model no longer favors it).
  CONV-MOMENTUM : exit at first checkpoint where own-side rank < the previous reading (decaying).
  PRICE-STOP    : control — exit at first checkpoint where signed return-so-far < 0 (dumb stop).

If CONV-* does not beat PRICE-STOP, the 15m model adds nothing to exits.
"""
import os, sys, json, warnings, gc
import numpy as np, pandas as pd, xgboost as xgb
from scipy.stats import rankdata
warnings.filterwarnings('ignore'); sys.path.append(os.getcwd())

M15 = dict(dir='models/v3_15min_clean', data='data/ranking_data_upstox_15min_3y_clean.csv', ret='Next_15Min_Return')
H1  = dict(dir='models/v10_native_1h',  data='data/ranking_data_upstox_1h_v3_3y.csv',        ret='Next_Hour_Return')
H_TEST, MIN_TRAIN, COST = 4, 18, 10/1e4
HOLD = [60, 75, 90, 105]                      # offsets (min) of the 4 in-hold sub-period N15 returns
RANK_THR = 0.5
OUT = 'data/early_exit_candidates.csv'

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
for p in (p15, p1): p['device'] = 'cpu'
d15 = load(M15['data'], M15['ret']); X15 = Xmat(d15, f15)
d1  = load(H1['data'],  H1['ret']);  X1  = Xmat(d1, f1)
months = sorted(d15['ym'].unique()); folds = []; i = MIN_TRAIN+1
while i+1 <= len(months):
    folds.append((months[:i-1], months[i-1], months[i:i+H_TEST])); i += H_TEST
print(f"  {len(folds)} folds OOS {folds[0][2][0]} -> {folds[-1][2][-1]}")

HOFF = [pd.Timedelta(minutes=m) for m in HOLD]
rows = []
for fi, (tr_m, val_m, te_m) in enumerate(folds, 1):
    print(f"  fold {fi}/{len(folds)} {te_m[0]}->{te_m[-1]}")
    s15 = fit_predict(d15, X15, M15['ret'], p15, tr_m, val_m, te_m)
    s15['rkL'] = s15.groupby('dt')['sL'].rank(pct=True)
    s15['rkS'] = s15.groupby('dt')['sS'].rank(pct=True)
    look = {(r.Ticker, r.dt): (r.rkL, r.rkS, r.Next_15Min_Return) for r in s15.itertuples(index=False)}
    tr1 = [m for m in sorted(d1['ym'].unique()) if m < val_m]
    s1 = fit_predict(d1, X1, H1['ret'], p1, tr1, val_m, te_m)
    for dt1, grp in s1.groupby('dt'):
        for direction, gg in [('long', grp.nlargest(3, 'sL')), ('short', grp.nlargest(3, 'sS'))]:
            for _, r in gg.iterrows():
                hold = [look.get((r['Ticker'], dt1 + o)) for o in HOFF]
                if any(h is None for h in hold): continue        # need all 4 in-hold bars (clean apples-to-apples)
                rk = [(h[0] if direction == 'long' else h[1]) for h in hold]  # own-side ranks at +60,+75,+90,+105
                n15 = [h[2] for h in hold]                                    # sub-period returns
                rows.append((fi, r['ym'], direction, n15[0], n15[1], n15[2], n15[3],
                             rk[0], rk[1], rk[2], rk[3]))
    del s15, s1, look; gc.collect()

cols = ['fold','ym','dir','n0','n1','n2','n3','rk0','rk1','rk2','rk3']
C = pd.DataFrame(rows, columns=cols); C.to_csv(OUT, index=False)
print(f"  pooled candidates {len(C):,} saved -> {OUT}\n")

def simulate(row, sgn, policy):
    """Return net return (signed, after one round-trip cost) for a policy."""
    n = [row.n0, row.n1, row.n2, row.n3]
    rk = [row.rk0, row.rk1, row.rk2, row.rk3]      # rank at +60(entry),+75,+90,+105
    # checkpoints after sub k=1,2,3 -> observed rank rk[k], realized cum of first k sub-returns
    exit_k = 4                                      # default = full hold (all 4 sub-periods)
    for k in (1, 2, 3):
        cum_signed = (np.prod([1+x for x in n[:k]]) - 1) * sgn
        if policy == 'conv_level'    and rk[k] < RANK_THR:   exit_k = k; break
        if policy == 'conv_momentum' and rk[k] < rk[k-1]:    exit_k = k; break
        if policy == 'price_stop'    and cum_signed < 0:     exit_k = k; break
    cum_signed = (np.prod([1+x for x in n[:exit_k]]) - 1) * sgn
    return cum_signed - COST

print("="*96)
print("  EARLY-EXIT overlay — 15m conviction decay vs full-hold vs dumb price-stop. EXPLORATORY.")
print("="*96)
for direction, sgn in [('long', 1), ('short', -1)]:
    sub = C[C['dir'] == direction]
    print(f"\n  ===== {direction.upper()} (N={len(sub)}) =====")
    res = {}
    for pol in ['full_hold','conv_level','conv_momentum','price_stop']:
        if pol == 'full_hold':
            net = ((np.prod([sub[f'n{k}']+1 for k in range(4)], axis=0)-1)*sgn - COST)
        else:
            net = sub.apply(lambda r: simulate(r, sgn, pol), axis=1).values
        res[pol] = net
        print(f"  {pol:<14} mean_net={net.mean()*1e4:>+6.1f}bps  WR={(net>0).mean()*100:>5.1f}%  median={np.median(net)*1e4:>+5.1f}")
    # loss-cutting on the trades that LOSE under full hold
    fh = res['full_hold']; losers = fh < 0
    print(f"  -- on full-hold LOSERS (N={losers.sum()}): full_hold avg={fh[losers].mean()*1e4:+.1f}bps", end="")
    for pol in ['conv_level','conv_momentum','price_stop']:
        print(f" | {pol}={res[pol][losers].mean()*1e4:+.1f}", end="")
    print()
print("\n" + "="*96)
print("  WIN: a CONV-* policy raises mean_net above full_hold AND beats price_stop.")
print("="*96)

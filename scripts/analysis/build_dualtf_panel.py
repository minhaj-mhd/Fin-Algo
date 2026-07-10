"""
DUAL-TF TRADE PANEL builder (research asset — no verdict authority).

Emits ONE rich per-trade table so future entry/exit research never needs to
retrain models. For every 1h top-3 candidate (both directions, walk-forward OOS)
it records the full 15m context:

  identity : fold, ym, dt1, ticker, dir, sL (1h long score), sS (1h short score)
  PRE-ENTRY trajectory  : rkL/rkS at dt1+{0,15,30,45}  (the 4 candles before entry)
  IN-HOLD path          : rkL/rkS at dt1+{60,75,90,105} (during the hour held)
  IN-HOLD sub-returns    : sub_{60,75,90,105} = N15 of each in-hold 15m bar
  outcome               : nhr = Next_Hour_Return (full-hold return, dt1+60..120)

Missing bars are stored as NaN (not dropped) so researchers choose their own
filters. With this panel you can test ANY entry gate, block rule, or early-exit
policy in seconds (see scripts/analysis/wf_rank_trajectory.py and wf_early_exit.py
for example consumers). Cross-sectional ranks are real (groupby('dt').rank(pct=True));
alignment validated (1h closes at dt1+60=entry; nhr covers dt1+60..120; all
pre-entry bars strictly before entry -> no look-ahead).
"""
import os, sys, json, warnings, gc
import numpy as np, pandas as pd, xgboost as xgb
from scipy.stats import rankdata
warnings.filterwarnings('ignore'); sys.path.append(os.getcwd())

M15 = dict(dir='models/v3_15min_clean', data='data/ranking_data_upstox_15min_3y_clean.csv', ret='Next_15Min_Return')
H1  = dict(dir='models/v10_native_1h',  data='data/ranking_data_upstox_1h_v3_3y.csv',        ret='Next_Hour_Return')
H_TEST, MIN_TRAIN = 4, 18
PRE  = [0, 15, 30, 45]          # pre-entry trajectory offsets (min from dt1)
HOLD = [60, 75, 90, 105]        # in-hold sub-period offsets
OUTDIR = 'data/research/entry_exit'
OUT = f'{OUTDIR}/dualtf_trade_panel.csv'

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
os.makedirs(OUTDIR, exist_ok=True)
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

ALL = PRE + HOLD
OFF = {m: pd.Timedelta(minutes=m) for m in ALL}
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
                rec = {'fold': fi, 'ym': r['ym'], 'dt1': dt1.isoformat(), 'ticker': r['Ticker'],
                       'dir': direction, 'sL': r['sL'], 'sS': r['sS'], 'nhr': r['Next_Hour_Return']}
                for m in ALL:
                    h = look.get((r['Ticker'], dt1 + OFF[m]))
                    rec[f'rkL_{m}'] = h[0] if h else np.nan
                    rec[f'rkS_{m}'] = h[1] if h else np.nan
                    if m in HOLD:
                        rec[f'sub_{m}'] = h[2] if h else np.nan
                rows.append(rec)
    del s15, s1, look; gc.collect()

P = pd.DataFrame(rows); P.to_csv(OUT, index=False)
print(f"\n  panel {len(P):,} trades x {P.shape[1]} cols saved -> {OUT}")
print(f"  coverage: pre-entry full {P[[f'rkL_{m}' for m in PRE]].notna().all(1).mean():.0%} | "
      f"in-hold full {P[[f'sub_{m}' for m in HOLD]].notna().all(1).mean():.0%}")

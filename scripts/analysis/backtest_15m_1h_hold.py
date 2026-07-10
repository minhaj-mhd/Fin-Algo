"""
P&L BACKTEST: trade the 15m ranker's top-K, hold 1 HOUR (exploratory, no verdict).

Direct test of "can the 15m model provide tradeable results at a 1h horizon?" — the
configuration the dual-TF studies never ran standalone (they used 15m only as an
overlay on 1h-model picks). Clean WF (retrain 15m per fold, trade OOS), per-trade net.

Per OOS timestamp: take top-3 by long score (long) and top-3 by short score (short);
hold 1 hour = session-masked compound of next 4 Next_15Min_Return; net = signed - cost.
Per-trade mean is overlap-unbiased (overlap affects portfolio variance, not per-trade EV).
Reports gross + net@{10,6,4,2} bps, overall and by entry hour. Incumbent 1h top-3 was
~-8 bps net@10 (full-hold) — see data/research/entry_exit/results/signal_economics_2026-06-11.txt.
"""
import os, sys, json, warnings, gc
import numpy as np, pandas as pd, xgboost as xgb
from scipy.stats import rankdata
warnings.filterwarnings('ignore'); sys.path.append(os.getcwd())

M = dict(dir='models/v3_15min_clean', data='data/ranking_data_upstox_15min_3y_clean.csv', ret='Next_15Min_Return')
H_TEST, MIN_TRAIN, K = 4, 18, 3
OUT = 'data/research/15m_1h_hold_backtest_result.txt'

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

print("Loading 15m dataset...")
with open(f'{M["dir"]}/metadata.json') as f: meta = json.load(f)
feats, params = meta['features'], dict(meta['params']); params['device'] = 'cpu'
df = load(M['data'], M['ret'])

# vectorized session-masked 1h-forward return (compound next 4 fwd-15m within ticker/day)
g = df.groupby(['Ticker', 'date'])[M['ret']]
r0 = df[M['ret']]; r1 = g.shift(-1); r2 = g.shift(-2); r3 = g.shift(-3)
df['fwd1h'] = (1 + r0) * (1 + r1) * (1 + r2) * (1 + r3) - 1
df['hour'] = df['dt'].dt.hour
X = Xfill(df[feats].values.astype(float))
ret = df[M['ret']].values; qid = df['Query_ID'].values; ymv = df['ym'].values
months = sorted(df['ym'].unique()); folds = []; i = MIN_TRAIN + 1
while i + 1 <= len(months):
    folds.append((months[:i-1], months[i-1], months[i:i+H_TEST])); i += H_TEST
print(f"  {len(folds)} folds OOS {folds[0][2][0]} -> {folds[-1][2][-1]}")

trades = []   # (hour, signed_fwd1h)
for fi, (tr_m, val_m, te_m) in enumerate(folds, 1):
    tr = np.isin(ymv, tr_m); va = np.isin(ymv, [val_m]); te = np.isin(ymv, te_m)
    bl = xgb.train(params, fitdm(X[tr], ret[tr], qid[tr], False), 500,
                   evals=[(fitdm(X[va], ret[va], qid[va], False), 'v')], early_stopping_rounds=50, verbose_eval=False)
    bs = xgb.train(params, fitdm(X[tr], ret[tr], qid[tr], True), 500,
                   evals=[(fitdm(X[va], ret[va], qid[va], True), 'v')], early_stopping_rounds=50, verbose_eval=False)
    sub = df[te].copy()
    sub['sL'] = bl.predict(xgb.DMatrix(X[te])); sub['sS'] = bs.predict(xgb.DMatrix(X[te]))
    sub = sub.dropna(subset=['fwd1h'])
    for _, grp in sub.groupby('dt'):
        for col, sgn, dname in [('sL', 1, 'long'), ('sS', -1, 'short')]:
            for _, r in grp.nlargest(K, col).iterrows():
                trades.append((r['hour'], dname, r['fwd1h'] * sgn))
    print(f"  fold {fi}/{len(folds)} {te_m[0]}->{te_m[-1]} done ({len(trades)} trades cum)")
    del bl, bs, sub; gc.collect()

T = pd.DataFrame(trades, columns=['hour', 'dir', 'sg'])
T.to_csv('data/research/15m_1h_hold_trades.csv', index=False)   # per-trade for future analysis
sg = T['sg'].values
g_all = sg.mean() * 1e4
def wr(arr, c): return ((arr - c/1e4) > 0).mean() * 100   # net win rate at cost c (bps)
lines = [
    "P&L BACKTEST — 15m ranker top-3, hold 1 HOUR, WF OOS (exploratory, no verdict)",
    f"N trades = {len(T)}",
    f"GROSS (signed, per trade): {g_all:+.2f} bps",
    "MEAN NET by round-trip cost:  " + "   ".join(f"@{c}={g_all-c:+.1f}" for c in (10, 6, 4, 2)),
    "",
    "NET WIN RATE (% of trades with return > cost):",
    f"  raw (cost 0): {wr(sg,0):.1f}%   @2bps: {wr(sg,2):.1f}%   @4bps: {wr(sg,4):.1f}%   "
    f"@6bps: {wr(sg,6):.1f}%   @10bps: {wr(sg,10):.1f}%",
    f"  LONG  raw {wr(T[T['dir']=='long']['sg'].values,0):.1f}%  net@10 {wr(T[T['dir']=='long']['sg'].values,10):.1f}%",
    f"  SHORT raw {wr(T[T['dir']=='short']['sg'].values,0):.1f}%  net@10 {wr(T[T['dir']=='short']['sg'].values,10):.1f}%",
    "",
    "By entry hour (gross | net@10 | net@6 | net@4 | net@2 | rawWR | WR@4):",
]
for h, gg in T.groupby('hour'):
    a = gg['sg'].values; gr = a.mean() * 1e4
    lines.append(f"  hour {h:02d}  N={len(gg):>5}  gross={gr:>+6.2f} | " +
                 " ".join(f"{gr-c:>+6.1f}" for c in (10, 6, 4, 2)) +
                 f" | rawWR {wr(a,0):.0f}% WR@4 {wr(a,4):.0f}%")
lines += [
    "",
    "Compare: incumbent 1h top-3 full-hold ~ -8 bps net@10. If 15m-picks-held-1h is also",
    "net-negative at realistic cost, the marginally-higher IC does NOT translate to tradeable",
    "edge -> confirms the signal is sub-cost regardless of which ranker selects.",
]
os.makedirs('data/research', exist_ok=True)
with open(OUT, 'w') as f: f.write("\n".join(lines) + "\n")
print("\n".join(lines)); print(f"\nsaved -> {OUT}")

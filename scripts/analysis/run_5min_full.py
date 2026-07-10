"""
5-MINUTE RANKER — full test vs the 1h ranker (GPU, exploratory, no verdict).
Runs once data/ranking_data_upstox_5min_v3_3y.csv exists (built by collect_upstox_5min_v3.py).

Clean WF (retrain 5m long/short per fold, OOS), device=cuda. Produces:
  (A) IC matrix: 5m model scored at horizons 5m / 1h / 2h (session-masked compound of native fwd)
  (B) P&L backtest: 5m-ranker top-3 held 1 HOUR -> gross + net@{10,6,4,2} + win rate (raw & by cost)
Directly comparable to: 15m-held-1h (IC 0.029, gross +3.37, net@10 -6.6, raw WR 53.9%) and the
1h ranker (IC 0.027). Memory-lean (usecols, float32). Falls back to cpu if cuda errors.
"""
import os, sys, json, warnings, gc
import numpy as np, pandas as pd, xgboost as xgb
from scipy.stats import rankdata, spearmanr
warnings.filterwarnings('ignore'); sys.path.append(os.getcwd())

DATA = 'data/ranking_data_upstox_5min_v3_3y.csv'
RET  = 'Next_5Min_Return'
H_TEST, MIN_TRAIN, K, COST = 4, 18, 3, 10/1e4
HBARS = {'5m': 1, '1h': 12, '2h': 24}          # 12 five-min bars = 1 hour
OUT = 'data/research/5min_ranker_full_result.txt'
feats = json.load(open('models/v3_15min_clean/metadata.json'))['features']  # same 86-feature family
params = dict(json.load(open('models/v3_15min_clean/metadata.json'))['params'])
params.update(device='cuda', tree_method='hist')

if not os.path.exists(DATA):
    print(f"[WAIT] {DATA} not built yet."); sys.exit(0)

def load():
    cols = feats + [RET, 'Query_ID', 'DateTime', 'Ticker']
    dt = {c: 'float32' for c in feats}
    df = pd.concat([c for c in pd.read_csv(DATA, usecols=lambda x: x in cols, dtype=dt, chunksize=200_000)],
                   ignore_index=True)
    df['ym'] = df['DateTime'].str[:7]; df['dt'] = pd.to_datetime(df['DateTime']); df['date'] = df['dt'].dt.date
    df['hour'] = df['dt'].dt.hour
    return df.dropna(subset=[RET]).reset_index(drop=True)
def fill(X):
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

print("Loading 5m dataset...", flush=True)
df = load()
print(f"  rows {len(df):,}  span {df['ym'].min()}->{df['ym'].max()}", flush=True)
g = df.groupby(['Ticker', 'date'])[RET]
shifts = {k: g.shift(-k).values for k in range(max(HBARS.values()))}
fwd = {}
for h, nb in HBARS.items():
    p = np.ones(len(df))
    for k in range(nb): p = p * (1 + shifts[k])
    fwd[h] = p - 1
X = fill(df[feats].values.astype(np.float32))
ret = df[RET].values; qid = df['Query_ID'].values; ymv = df['ym'].values; dts = df['dt'].values; hrs = df['hour'].values
months = sorted(df['ym'].unique()); folds = []; i = MIN_TRAIN + 1
while i + 1 <= len(months):
    folds.append((months[:i-1], months[i:i+H_TEST])); i += H_TEST
print(f"  {len(folds)} folds OOS {folds[0][1][0]}->{folds[-1][1][-1]}", flush=True)

ic = {h: {'L': [], 'S': []} for h in HBARS}
trades = []   # (hour, signed_1h_fwd) for the P&L backtest
for fi, (tr_m, te_m) in enumerate(folds, 1):
    tr = np.isin(ymv, tr_m); te = np.isin(ymv, te_m)
    try:
        bl = xgb.train(params, fitdm(X[tr], ret[tr], qid[tr], False), 400, verbose_eval=False)
        bs = xgb.train(params, fitdm(X[tr], ret[tr], qid[tr], True), 400, verbose_eval=False)
    except Exception as e:
        print(f"  [cuda failed -> cpu] {str(e)[:80]}"); params['device'] = 'cpu'
        bl = xgb.train(params, fitdm(X[tr], ret[tr], qid[tr], False), 400, verbose_eval=False)
        bs = xgb.train(params, fitdm(X[tr], ret[tr], qid[tr], True), 400, verbose_eval=False)
    pl = bl.predict(xgb.DMatrix(X[te])); ps = bs.predict(xgb.DMatrix(X[te]))
    for h in HBARS:
        ic[h]['L'].append(perq_ic(pl, fwd[h][te], dts[te])); ic[h]['S'].append(perq_ic(ps, -fwd[h][te], dts[te]))
    # P&L: top-K per timestamp, hold 1h
    sub = pd.DataFrame({'dt': dts[te], 'hour': hrs[te], 'sL': pl, 'sS': ps, 'f1': fwd['1h'][te]}).dropna(subset=['f1'])
    for _, grp in sub.groupby('dt'):
        for col, sgn in [('sL', 1), ('sS', -1)]:
            for _, r in grp.nlargest(K, col).iterrows():
                trades.append((r['hour'], r['f1'] * sgn))
    print(f"  fold {fi}/{len(folds)} {te_m[0]}->{te_m[-1]} done", flush=True)
    del bl, bs, sub; gc.collect()

T = pd.DataFrame(trades, columns=['hour', 'sg']); sg = T['sg'].values; gross = sg.mean() * 1e4
def wr(a, c): return ((a - c/1e4) > 0).mean() * 100
lines = [
    "5-MINUTE RANKER — full WF test (GPU, exploratory, no verdict)",
    "(A) IC by horizon (long / short):",
    "  vs 1h ranker +0.0275/+0.0254  ;  15m-held-1h +0.0287/+0.0307",
]
for h in HBARS:
    lines.append(f"    5m model @ {h:>3}: L={np.nanmean(ic[h]['L']):+.4f}  S={np.nanmean(ic[h]['S']):+.4f}")
lines += ["",
          f"(B) P&L: 5m-ranker top-3 HELD 1 HOUR  (N={len(T)})",
          f"  GROSS {gross:+.2f} bps | net@10 {gross-10:+.1f}  @6 {gross-6:+.1f}  @4 {gross-4:+.1f}  @2 {gross-2:+.1f}",
          f"  WIN RATE raw {wr(sg,0):.1f}%  @4 {wr(sg,4):.1f}%  @10 {wr(sg,10):.1f}%",
          "  (compare 15m-held-1h: gross +3.37, net@10 -6.6, raw WR 53.9%)"]
os.makedirs('data/research', exist_ok=True)
with open(OUT, 'w') as f: f.write("\n".join(lines) + "\n")
print("\n" + "\n".join(lines)); print(f"\nsaved -> {OUT}")

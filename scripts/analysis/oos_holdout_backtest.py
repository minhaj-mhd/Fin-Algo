"""
GENUINELY OUT-OF-SAMPLE dual-TF backtest + audit.

Fixes the validity bug: the production models were trained on all-but-last-month, so the
3-month backtest window was in-sample. Here we train HOLDOUT models on data STRICTLY BEFORE
the test window, then backtest on the fully-unseen last 3 months.

  Train  : months[:-4]      (everything up to 4 months ago)
  Val    : months[-4]       (early stopping; before OOS)
  OOS    : months[-3:]      (2026-04/05/06 — never seen by the models)

Models trained fresh: native 1h (long+short) and clean 15m (long+short).
Audit: fee sensitivity (0/5/10/15/20 bps), per-month consistency, bootstrap 95% CI + t-stat.
Entry-confirmation P&L = 1h Next_Hour_Return; gate = holdout-15m ranks. 09:15-aligned.
"""
import os, sys, json, warnings, gc
import numpy as np, pandas as pd, xgboost as xgb
from scipy.stats import rankdata
warnings.filterwarnings('ignore'); sys.path.append(os.getcwd())

H1_DIR, M15_DIR = 'models/v10_native_1h', 'models/v3_15min_clean'
H1_DATA  = 'data/ranking_data_upstox_1h_v3_3y.csv'
M15_DATA = 'data/ranking_data_upstox_15min_3y_clean.csv'
OOS_MONTHS = 3
FEES = [0, 5, 10, 15, 20]
RNG = np.random.default_rng(0)

def load(path, ret):
    chunks = []
    for ch in pd.read_csv(path, chunksize=200_000):
        chunks.append(ch)
    df = pd.concat(chunks, ignore_index=True)
    df['dt'] = pd.to_datetime(df['DateTime']); df['ym'] = df['DateTime'].str[:7]
    return df.dropna(subset=[ret])

def featX(df, fe):
    X = df[fe].values.astype(float)
    for ci in range(X.shape[1]):
        c = X[:, ci]; b = np.isnan(c) | np.isinf(c)
        if b.any(): X[b, ci] = float(np.nanmean(c[~b])) if (~b).any() else 0.0
    return X

def int_ranks(y, q, inv=False):
    out = np.zeros_like(y, dtype=int)
    for qid in np.unique(q):
        m = q == qid; v = -y[m] if inv else y[m]; out[m] = rankdata(v, method='ordinal')-1
    return out

def train_pair(df, fe, ret, params, train_m, val_m):
    tr = df['ym'].isin(train_m).values; va = df['ym'].isin(val_m).values
    Xtr, Xva = featX(df[tr], fe), featX(df[va], fe)
    ytr, yva = df[tr][ret].values, df[va][ret].values
    qtr, qva = df[tr]['Query_ID'].values, df[va]['Query_ID'].values
    gtr = pd.Series(qtr).groupby(qtr).size().values; gva = pd.Series(qva).groupby(qva).size().values
    out = {}
    for side, inv in [('long', False), ('short', True)]:
        dtr = xgb.DMatrix(Xtr, label=int_ranks(ytr, qtr, inv)); dtr.set_group(gtr)
        dva = xgb.DMatrix(Xva, label=int_ranks(yva, qva, inv)); dva.set_group(gva)
        out[side] = xgb.train(params, dtr, 500, evals=[(dva, 'v')], early_stopping_rounds=50, verbose_eval=False)
    return out

# ---- 15m: train holdout, predict OOS, build rank lookup ----
print("Loading 15m clean (full)...")
with open(f'{M15_DIR}/metadata.json') as f: m15 = json.load(f)
f15, p15 = m15['features'], m15['params']
df15 = load(M15_DATA, 'Next_15Min_Return')
months15 = sorted(df15['ym'].unique())
oos = months15[-OOS_MONTHS:]; val15 = [months15[-(OOS_MONTHS+1)]]; tr15 = months15[:-(OOS_MONTHS+1)]
print(f"  15m train {tr15[0]}->{tr15[-1]} | val {val15[0]} | OOS {oos}")
print("  Training holdout 15m long+short...")
m15h = train_pair(df15, f15, 'Next_15Min_Return', p15, tr15, val15)
df15o = df15[df15['ym'].isin(oos)].copy()
Xo = featX(df15o, f15)
df15o['rk_long'] = pd.Series(m15h['long'].predict(xgb.DMatrix(Xo)), index=df15o.index)
df15o['rk_short'] = pd.Series(m15h['short'].predict(xgb.DMatrix(Xo)), index=df15o.index)
df15o['rk_long'] = df15o.groupby('dt')['rk_long'].rank(pct=True)
df15o['rk_short'] = df15o.groupby('dt')['rk_short'].rank(pct=True)
look = {}
for r in df15o.itertuples(index=False): look[(r.Ticker, r.dt)] = (r.rk_long, r.rk_short)
del df15, df15o, Xo; gc.collect()
print("  15m holdout ranks ready.")

# ---- 1h native: train holdout, predict OOS ----
print("Loading 1h native (full)...")
with open(f'{H1_DIR}/metadata.json') as f: m1 = json.load(f)
f1, p1 = m1['features'], m1['params']
df1 = load(H1_DATA, 'Next_Hour_Return')
months1 = sorted(df1['ym'].unique())
oos1 = months1[-OOS_MONTHS:]; val1 = [months1[-(OOS_MONTHS+1)]]; tr1 = months1[:-(OOS_MONTHS+1)]
print(f"  1h train {tr1[0]}->{tr1[-1]} | val {val1[0]} | OOS {oos1}")
assert oos1 == oos, "OOS month mismatch between 1h and 15m"
print("  Training holdout 1h long+short...")
m1h = train_pair(df1, f1, 'Next_Hour_Return', p1, tr1, val1)
df1o = df1[df1['ym'].isin(oos)].copy()
Xo1 = featX(df1o, f1)
df1o['s_long'] = m1h['long'].predict(xgb.DMatrix(Xo1))
df1o['s_short'] = m1h['short'].predict(xgb.DMatrix(Xo1))
df1o['ret'] = df1o['Next_Hour_Return']
del df1; gc.collect()
print(f"  1h holdout OOS preds ready. OOS rows={len(df1o):,}\n")

C45 = pd.Timedelta(minutes=45)
def collect(direction, gate):
    score = 's_long' if direction == 'long' else 's_short'
    rows = []
    for dt1, grp in df1o.groupby('dt'):
        for _, r in grp.nlargest(3, score).iterrows():
            conf = look.get((r['Ticker'], dt1 + C45))
            if conf is None: continue
            rkl, rks = conf
            if direction == 'long':
                if gate.get('long') is not None and not (rkl > gate['long']): continue
                if gate.get('short') is not None and not (rks < gate['short']): continue
                g = r['ret']
            else:
                if gate.get('short') is not None and not (rks > gate['short']): continue
                if gate.get('long') is not None and not (rkl < gate['long']): continue
                g = -r['ret']
            rows.append((g, r['ym']))
    return pd.DataFrame(rows, columns=['gross', 'ym'])

configs = [
    ('SHORT baseline',         'short', {}),
    ('SHORT ShortConfirm p90', 'short', {'short':0.90}),
    ('SHORT LongAvoid p15',    'short', {'long':0.15}),
    ('LONG  baseline',         'long',  {}),
    ('LONG  LongConfirm p90',  'long',  {'long':0.90}),
    ('LONG  ShortAvoid p15',   'long',  {'short':0.15}),
]
print("="*100)
print(f"  GENUINE OOS DUAL-TF AUDIT (holdout models, OOS={oos}, never trained on these months)")
print("="*100)
for label, direction, gate in configs:
    t = collect(direction, gate)
    if len(t) == 0: print(f"  {label}: no trades"); continue
    g = t['gross'].values; n = len(g); gross = g.mean()*1e4
    net10 = g - 10/1e4; tstat = net10.mean()/(net10.std()/np.sqrt(n)) if net10.std()>0 else 0
    bs = [net10[RNG.integers(0,n,n)].mean()*1e4 for _ in range(1500)]
    ci = (np.percentile(bs,2.5), np.percentile(bs,97.5))
    fee_str = "  ".join(f"{fb}:{(gross-fb):+.1f}/{(g>fb/1e4).mean()*100:.0f}%" for fb in FEES)
    pm = " | ".join(f"{ym}:{(t[t['ym']==ym]['gross'].mean()-10/1e4)*1e4:+.1f}bps/{(t[t['ym']==ym]['gross']>10/1e4).mean()*100:.0f}%(n{len(t[t['ym']==ym])})" for ym in oos if len(t[t['ym']==ym])>0)
    print(f"\n  {label}  (N={n})")
    print(f"    gross={gross:+.1f}bps | net@10={net10.mean()*1e4:+.1f}bps CI[{ci[0]:+.0f},{ci[1]:+.0f}] t={tstat:.1f} | breakeven={gross:.1f}bps")
    print(f"    fee(bps):netbps/WR  {fee_str}")
    print(f"    per-month net@10:   {pm}")
print("\n" + "="*100)
print("  NOTE: these models NEVER saw the OOS months (trained up to "+tr1[-1]+", val "+val1[0]+").")
print("="*100)

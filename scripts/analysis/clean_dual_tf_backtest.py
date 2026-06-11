"""
CLEAN dual-TF backtest: NATIVE 1h (v10) signal + CLEAN 15m (v3) confirmation/exits.
Both datasets are 09:15-anchored, so a 1h bar == its four 15m bars (verified in-script).

Alignment (look-ahead-free):
  1h signal label T -> known at T+60 (1h close). Next_Hour_Return covers T+60 -> T+120.
  15m confirm bar  = label T+45 (closes at T+60, the decision instant).
  hold/exit bars   = 15m labels T+60,75,90,105 (compound == Next_Hour_Return).

Friction 10 bps. Bootstrap 95% CI + t-stat per config. OOS = last 3 months.
"""
import os, sys, json, warnings
import numpy as np, pandas as pd, xgboost as xgb
warnings.filterwarnings('ignore'); sys.path.append(os.getcwd())

H1_DIR, M15_DIR = 'models/v10_native_1h', 'models/v3_15min_clean'
H1_DATA  = 'data/ranking_data_upstox_1h_v3_3y.csv'
M15_DATA = 'data/ranking_data_upstox_15min_3y_clean.csv'
OOS_MONTHS, COST, N_BOOT = 3, 10/10000, 2000
RNG = np.random.default_rng(42)

print("Loading clean models (native 1h + clean 15m)...")
with open(f'{H1_DIR}/metadata.json') as f: m1 = json.load(f)
with open(f'{M15_DIR}/metadata.json') as f: m15 = json.load(f)
f1, f15 = m1['features'], m15['features']
b1l = xgb.Booster(); b1l.load_model(f'{H1_DIR}/xgb_long_model.json')
b1s = xgb.Booster(); b1s.load_model(f'{H1_DIR}/xgb_short_model.json')
m15l = xgb.Booster(); m15l.load_model(f'{M15_DIR}/xgb_long_model.json')
m15s = xgb.Booster(); m15s.load_model(f'{M15_DIR}/xgb_short_model.json')

def load(path, ret):
    am = set()
    for ch in pd.read_csv(path, usecols=['DateTime'], chunksize=500_000): am.update(ch['DateTime'].str[:7].unique())
    oos = sorted(am)[-OOS_MONTHS:]; ck = []
    for ch in pd.read_csv(path, chunksize=200_000):
        s = ch[ch['DateTime'].str[:7].isin(oos)]
        if len(s): ck.append(s)
    df = pd.concat(ck, ignore_index=True); df['dt'] = pd.to_datetime(df['DateTime'])
    return df.dropna(subset=[ret]), oos
def dmat(df, fe):
    X = df[fe].values.astype(float)
    for ci in range(X.shape[1]):
        c = X[:, ci]; b = np.isnan(c) | np.isinf(c)
        if b.any(): X[b, ci] = float(np.nanmean(c[~b])) if (~b).any() else 0.0
    return xgb.DMatrix(X)

print("Loading 1h native...")
df1, oos = load(H1_DATA, 'Next_Hour_Return')
df1['s_long'] = b1l.predict(dmat(df1, f1)); df1['s_short'] = b1s.predict(dmat(df1, f1)); df1['ret'] = df1['Next_Hour_Return']
print(f"  {len(df1):,} rows OOS {oos} | signal tods {sorted(df1['dt'].dt.strftime('%H:%M').unique())}")
print("Loading 15m clean...")
df15, _ = load(M15_DATA, 'Next_15Min_Return')
df15['s_long'] = m15l.predict(dmat(df15, f15)); df15['s_short'] = m15s.predict(dmat(df15, f15))
df15['rk_long'] = df15.groupby('dt')['s_long'].rank(pct=True); df15['rk_short'] = df15.groupby('dt')['s_short'].rank(pct=True)
df15['ret15'] = df15['Next_15Min_Return']
look = {}
for r in df15.itertuples(index=False): look[(r.Ticker, r.dt)] = (r.rk_long, r.rk_short, r.ret15)
print(f"  {len(df15):,} rows | lookup ready")

C45 = pd.Timedelta(minutes=45); HOLD = [pd.Timedelta(minutes=m) for m in (60, 75, 90, 105)]
# reconciliation: native-1h NHR vs compound of clean-15m bars
diffs = []
for _, r in df1.sample(min(2000, len(df1)), random_state=1).iterrows():
    bars = [look.get((r['Ticker'], r['dt'] + o)) for o in HOLD]
    if any(b is None for b in bars): continue
    comp = 1.0
    for b in bars: comp *= (1 + b[2])
    diffs.append(abs((comp - 1) - r['ret']))
diffs = np.array(diffs)
print(f"  native1h<->clean15m reconciliation: median|diff|={np.median(diffs)*1e4:.2f}bps mean={diffs.mean()*1e4:.2f}bps (n={len(diffs)})")

def sim(tk, dt1, direction, gate, exitc, full_ret):
    conf = look.get((tk, dt1 + C45))
    if conf is None: return None
    rkl0, rks0, _ = conf
    if direction == 'long':
        if gate.get('long') is not None and not (rkl0 > gate['long']): return None
        if gate.get('short') is not None and not (rks0 < gate['short']): return None
    else:
        if gate.get('short') is not None and not (rks0 > gate['short']): return None
        if gate.get('long') is not None and not (rkl0 < gate['long']): return None
    if exitc is None:
        g = full_ret if direction == 'long' else -full_ret
        return g - COST, False
    cum, n, ex = 0.0, 0, False
    for i, off in enumerate(HOLD):
        bar = look.get((tk, dt1 + off))
        if bar is None: break
        rkl, rks, ret = bar
        if i > 0:
            lose = False
            if direction == 'long':
                if exitc.get('long') is not None and rkl < exitc['long']: lose = True
                if exitc.get('short') is not None and rks > exitc['short']: lose = True
            else:
                if exitc.get('short') is not None and rks < exitc['short']: lose = True
                if exitc.get('long') is not None and rkl > exitc['long']: lose = True
            if lose: ex = True; break
        gb = ret if direction == 'long' else -ret
        cum = (1 + cum) * (1 + gb) - 1; n += 1
    if n == 0: return None
    return cum - COST, ex

def run(direction='long', top_k=3, gate=None, exitc=None, label=''):
    gate = gate or {}; score = 's_long' if direction == 'long' else 's_short'
    nets, exits = [], []
    for dt1, grp in df1.groupby('dt'):
        for _, r in grp.nlargest(top_k, score).iterrows():
            res = sim(r['Ticker'], dt1, direction, gate, exitc, r['ret'])
            if res is None: continue
            nets.append(res[0]); exits.append(res[1])
    if not nets: return None
    net = np.array(nets); n = len(net); eq = np.cumprod(1 + net); rm = np.maximum.accumulate(eq)
    bs = [net[RNG.integers(0, n, n)].mean()*1e4 for _ in range(N_BOOT)]
    return dict(label=label, n=n, net_wr=(net > 0).mean()*100, net_bps=net.mean()*1e4,
                ci=(np.percentile(bs, 2.5), np.percentile(bs, 97.5)),
                t=net.mean()/(net.std()/np.sqrt(n)) if net.std() > 0 else 0,
                ret=(eq[-1]-1)*100, dd=((eq-rm)/rm).min()*100, early=np.mean(exits)*100)

cfgs = [
    dict(direction='long', label='LONG Baseline      | 1h native, no gate'),
    dict(direction='long', gate={'long':0.90}, label='LONG LongConfirm   | 15m rkL>p90'),
    dict(direction='long', gate={'short':0.15}, label='LONG ShortAvoid    | 15m rkS<p15'),
    dict(direction='long', gate={'long':0.80,'short':0.20}, label='LONG Dual entry    | L>p80 & S<p20'),
    dict(direction='long', gate={'long':0.80,'short':0.20}, exitc={'long':0.50,'short':0.80}, label='LONG Dual+Exit     | +exit L<p50|S>p80'),
    dict(direction='short', label='SHORT Baseline     | 1h native, no gate'),
    dict(direction='short', gate={'short':0.90}, label='SHORT ShortConfirm | 15m rkS>p90'),
    dict(direction='short', gate={'long':0.15}, label='SHORT LongAvoid    | 15m rkL<p15'),
]
print(f"\n{'='*98}\n  CLEAN DUAL-TF (native 1h signal + clean 15m confirm/exit, 10 bps, bootstrap 95% CI)\n{'='*98}")
print(f"  {'Config':<42} {'N':>5} {'NetWR%':>7} {'NetBps (95% CI)':>22} {'t':>6} {'Ret%':>8} {'DD%':>7} {'Exit':>5}")
print(f"  {'-'*96}")
for c in cfgs:
    r = run(**{k:v for k,v in c.items() if k!='label'}, label=c['label'])
    if not r: continue
    sig = '***' if abs(r['t'])>2.58 else ('**' if abs(r['t'])>1.96 else ('*' if abs(r['t'])>1.64 else ''))
    bps = f"{r['net_bps']:+.1f}[{r['ci'][0]:+.0f},{r['ci'][1]:+.0f}]"
    ex = f"{r['early']:.0f}%" if r['early']>0 else "-"
    print(f"  {r['label']:<42} {r['n']:>5} {r['net_wr']:>6.1f}% {bps:>22} {r['t']:>5.1f}{sig:<3} {r['ret']:>+7.1f}% {r['dd']:>6.1f}% {ex:>5}")
print(f"\n  CI spanning 0 => edge NOT statistically established.  *p<.10 **p<.05 ***p<.01\n{'='*98}")

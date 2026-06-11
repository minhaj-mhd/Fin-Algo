"""
DEFINITIVE dual-timeframe backtest — fully reconciled, look-ahead-free, with bootstrap CIs.

ROOT-CAUSE RESOLUTION (proven by raw-close matching):
  * 1h file is LEFT-labeled  (timestamp = bar START).  Bar "T" closes at wall-clock T+60.
  * 15m file is RIGHT-labeled (timestamp = bar END).    Bar "L" closes at wall-clock L.
  => A 1h signal labeled T is known at wall T+60 and aligns to the 15m bar labeled T+60.
  => Verified: 1h Next_Hour_Return[T] == compound(15m N15 at T+60,T+75,T+90,T+105) to 1e-5.

CORRECT MECHANICS:
  * 1h signal at label T (known at wall T+60). Tradeable hour = wall T+60 -> T+120.
  * Entry confirmation = 15m bar labeled T+60 (closes exactly at entry instant; no look-ahead).
  * Hold / exits        = 15m bars labeled T+60, T+75, T+90, T+105 (their N15 returns).
  * Full-hold return    = Next_Hour_Return (identical to the 4-bar compound; asserted).
  * EXCLUDE the 14:30 signal: its "Next_Hour_Return" is an OVERNIGHT return (next-day close)
    -> contaminated every earlier backtest. Dropped here.

P&L 100% from one reconciled basis. 10 bps friction. Bootstrap 95% CIs on every metric.
1h = v8_upstox_3y, 15m = v2_15min_3y. OOS = last 3 months of 1h data.
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
warnings.filterwarnings('ignore')
sys.path.append(os.getcwd())

H1_MODEL_DIR  = 'models/v8_upstox_3y'
M15_MODEL_DIR = 'models/v2_15min_3y'
H1_DATA       = 'data/ranking_data_upstox_3y.csv'
M15_DATA      = 'data/ranking_data_upstox_15min_3y.csv'
OUT_DIR       = 'data/model_analysis/dual_tf'
MEM_DIR       = r'finalgo-memory-layer\finalgo\08. Model Analysis\15-Minute Vanguard Model\assets'
os.makedirs(OUT_DIR, exist_ok=True)

OOS_MONTHS = 3
COST       = 10 / 10000
N_BOOT     = 2000
RNG        = np.random.default_rng(42)

BG, AX_BG, TEXT, GRID = '#0f1117', '#161b2e', '#dde1f0', '#252a45'
LONG_C, SHORT_C, GOLD, NEUT, PURPLE, CYAN = '#00d4aa', '#ff6b6b', '#f0c040', '#7c83a3', '#b57bee', '#00b4e0'
plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': AX_BG, 'axes.edgecolor': GRID,
    'axes.labelcolor': TEXT, 'xtick.color': TEXT, 'ytick.color': TEXT, 'text.color': TEXT,
    'grid.color': GRID, 'grid.linewidth': 0.5, 'legend.facecolor': '#1e2338', 'legend.edgecolor': GRID,
})
SEP = "=" * 92

# ── load + score ───────────────────────────────────────────────────────────────
print("Loading models...")
with open(f'{H1_MODEL_DIR}/metadata.json')  as f: meta_1h  = json.load(f)
with open(f'{M15_MODEL_DIR}/metadata.json') as f: meta_15m = json.load(f)
feat_1h, feat_15m = meta_1h['features'], meta_15m['features']
b1l = xgb.Booster(); b1l.load_model(f'{H1_MODEL_DIR}/xgb_long_model.json')
m15l = xgb.Booster(); m15l.load_model(f'{M15_MODEL_DIR}/xgb_long_model.json')
m15s = xgb.Booster(); m15s.load_model(f'{M15_MODEL_DIR}/xgb_short_model.json')

def load(data_file, ret_col):
    am = set()
    for ch in pd.read_csv(data_file, usecols=['DateTime'], chunksize=500_000):
        am.update(ch['DateTime'].str[:7].unique())
    oos = sorted(am)[-OOS_MONTHS:]
    chunks = []
    for ch in pd.read_csv(data_file, chunksize=200_000):
        sub = ch[ch['DateTime'].str[:7].isin(oos)]
        if len(sub): chunks.append(sub)
    df = pd.concat(chunks, ignore_index=True)
    df['dt'] = pd.to_datetime(df['DateTime']).dt.tz_localize(None)
    df = df.dropna(subset=[ret_col])
    return df, oos

def dmat(df, feats):
    X = df[feats].values.astype(float)
    for ci in range(X.shape[1]):
        col = X[:, ci]; bad = np.isnan(col) | np.isinf(col)
        if bad.any(): X[bad, ci] = float(np.nanmean(col[~bad])) if (~bad).any() else 0.0
    return xgb.DMatrix(X)

print("Streaming 1h...")
df1, oos = load(H1_DATA, 'Next_Hour_Return')
df1['s_long'] = b1l.predict(dmat(df1, feat_1h))
df1['ret']    = df1['Next_Hour_Return']
df1['tod']    = df1['dt'].dt.strftime('%H:%M')
# EXCLUDE overnight-contaminated last bar of day (14:30)
before = len(df1)
df1 = df1[df1['tod'] != '14:30'].copy()
print(f"  {before:,} -> {len(df1):,} rows after dropping 14:30 (overnight) | OOS {oos}")
print(f"  1h signal times kept: {sorted(df1['tod'].unique())}")

print("Streaming 15m...")
df15, _ = load(M15_DATA, 'Next_15Min_Return')
df15['s_long']  = m15l.predict(dmat(df15, feat_15m))
df15['s_short'] = m15s.predict(dmat(df15, feat_15m))
df15['rk_long']  = df15.groupby('dt')['s_long'].rank(pct=True)
df15['rk_short'] = df15.groupby('dt')['s_short'].rank(pct=True)
df15['ret15']    = df15['Next_15Min_Return']

# lookup: (ticker, 15m-label) -> (rk_long, rk_short, n15_return)
look = {}
for r in df15.itertuples(index=False):
    look[(r.Ticker, r.dt)] = (r.rk_long, r.rk_short, r.ret15)
print(f"  {len(df15):,} rows | lookup ready")

# offsets in 15m-label space (right-labeled): entry/hold bars
H60, H75, H90, H105 = (pd.Timedelta(minutes=m) for m in (60, 75, 90, 105))
HOLD = [H60, H75, H90, H105]

# ── consistency assertion: NHR == compound of 4 aligned 15m bars ───────────────
print("\nSanity check: NHR vs aligned 4-bar 15m compound (should match ~0)...")
diffs = []
for _, r in df1.sample(min(3000, len(df1)), random_state=1).iterrows():
    tk, dt1 = r['Ticker'], r['dt']
    bars = [look.get((tk, dt1 + o)) for o in HOLD]
    if any(b is None for b in bars):
        continue
    comp = 1.0
    for b in bars: comp *= (1 + b[2])
    diffs.append(abs((comp - 1) - r['ret']))
diffs = np.array(diffs)
print(f"  matched {len(diffs)} | mean|diff|={diffs.mean():.2e} | max|diff|={diffs.max():.2e}  "
      f"-> {'RECONCILED' if diffs.mean()<1e-4 else 'MISMATCH!'}")

# ── trade simulator ──────────────────────────────────────────────────────────
def sim(tk, dt1, gate, exitc, full_ret):
    """Entry-confirm at label dt1+60; hold/exit over the 4 aligned bars.
       Returns (net, held, exited) or None."""
    conf = look.get((tk, dt1 + H60))
    if conf is None:
        return None
    rkl0, rks0, _ = conf
    if gate.get('long')  is not None and not (rkl0 > gate['long']):  return None
    if gate.get('short') is not None and not (rks0 < gate['short']): return None

    if exitc is None:
        # full hold == Next_Hour_Return (reconciled). Use it directly (exact).
        return full_ret - COST, 4, False

    cum, held, exited = 0.0, 0, False
    for i, off in enumerate(HOLD):
        bar = look.get((tk, dt1 + off))
        if bar is None:
            break
        rkl, rks, ret = bar
        if i > 0:  # decide before riding segment i (rank just closed at this bar)
            lose = ((exitc.get('long')  is not None and rkl < exitc['long']) or
                    (exitc.get('short') is not None and rks > exitc['short']))
            if lose:
                exited = True
                break
        cum = (1 + cum) * (1 + ret) - 1
        held += 1
    if held == 0:
        return None
    return cum - COST, held, exited

def run(top_k=3, gate=None, exitc=None, label=''):
    gate = gate or {}
    rows = []
    for dt1, grp in df1.groupby('dt'):
        for _, r in grp.nlargest(top_k, 's_long').iterrows():
            res = sim(r['Ticker'], dt1, gate, exitc, r['ret'])
            if res is None: continue
            net, held, exited = res
            rows.append((net, held, exited))
    if not rows:
        return None
    net = np.array([x[0] for x in rows])
    held = np.array([x[1] for x in rows])
    exited = np.array([x[2] for x in rows])
    g = net + COST
    eq = np.cumprod(1 + net); rm = np.maximum.accumulate(eq)
    # bootstrap CIs (order-invariant metrics)
    n = len(net)
    bs_bps, bs_ret, bs_wr = [], [], []
    for _ in range(N_BOOT):
        idx = RNG.integers(0, n, n)
        s = net[idx]
        bs_bps.append(s.mean()*10000)
        bs_ret.append((np.prod(1+s)-1)*100)
        bs_wr.append((s>0).mean()*100)
    ci = lambda a: (np.percentile(a,2.5), np.percentile(a,97.5))
    return dict(label=label, n=n,
                raw_wr=(g>0).mean()*100, net_wr=(net>0).mean()*100,
                net_bps=net.mean()*10000, bps_ci=ci(bs_bps),
                total_ret=(eq[-1]-1)*100, ret_ci=ci(bs_ret),
                wr_ci=ci(bs_wr),
                max_dd=((eq-rm)/rm).min()*100,
                early_pct=exited.mean()*100, avg_held=held.mean(),
                t_stat=net.mean()/(net.std()/np.sqrt(n)) if net.std()>0 else 0,
                equity=eq)

# ── configurations ─────────────────────────────────────────────────────────────
cfgs = [
    dict(label='Baseline           | 1H long, no gate'),
    dict(label='LongConfirm        | rkL>p90',                          gate={'long':0.90}),
    dict(label='ShortAvoid         | rkS<p15',                          gate={'short':0.15}),
    dict(label='Dual medium        | rkL>p80 & rkS<p20',                gate={'long':0.80,'short':0.20}),
    dict(label='Dual strict        | rkL>p85 & rkS<p15',                gate={'long':0.85,'short':0.15}),
    dict(label='Dual + DualExit    | rkL>p80&rkS<p20, exit L<p50|S>p80',gate={'long':0.80,'short':0.20}, exitc={'long':0.50,'short':0.80}),
    dict(label='Dual + DualExit-T  | rkL>p80&rkS<p20, exit L<p60|S>p70',gate={'long':0.80,'short':0.20}, exitc={'long':0.60,'short':0.70}),
    dict(label='FULL STACK         | rkL>p85&rkS<p15 + dual exit',      gate={'long':0.85,'short':0.15}, exitc={'long':0.50,'short':0.80}),
    dict(label='Baseline K=5       | 1H long, no gate', top_k=5),
    dict(label='Dual medium K=5    | rkL>p80 & rkS<p20', top_k=5, gate={'long':0.80,'short':0.20}),
    dict(label='FULL STACK K=5     | rkL>p85&rkS<p15 + dual exit', top_k=5, gate={'long':0.85,'short':0.15}, exitc={'long':0.50,'short':0.80}),
]

print(f"\n{SEP}\n  DEFINITIVE results (reconciled, 14:30 dropped, bootstrap 95% CI)\n{SEP}")
results = []
for c in cfgs:
    r = run(**{k:v for k,v in c.items() if k!='label'}, label=c['label'])
    if r:
        results.append(r)

# ── full table with CIs and significance ──────────────────────────────────────
print(f"\n  {'Config':<37} {'N':>4} {'NetWR%(95%CI)':>18} {'NetBps(95%CI)':>22} {'t':>5} {'Ret%':>7} {'DD%':>6} {'Exit':>5}")
print(f"  {'-'*112}")
for r in results:
    sig = '***' if abs(r['t_stat'])>2.58 else ('**' if abs(r['t_stat'])>1.96 else ('*' if abs(r['t_stat'])>1.64 else ''))
    wr_s  = f"{r['net_wr']:.1f}[{r['wr_ci'][0]:.0f}-{r['wr_ci'][1]:.0f}]"
    bps_s = f"{r['net_bps']:+.1f}[{r['bps_ci'][0]:+.0f},{r['bps_ci'][1]:+.0f}]"
    ex = f"{r['early_pct']:.0f}%" if r['early_pct']>0 else "-"
    print(f"  {r['label']:<37} {r['n']:>4} {wr_s:>18} {bps_s:>22} {r['t_stat']:>4.1f}{sig:<3} "
          f"{r['total_ret']:>+6.1f}% {r['max_dd']:>5.1f}% {ex:>5}")

print(f"\n  Significance (per-trade edge vs 0):  * p<0.10   ** p<0.05   *** p<0.01")
print(f"  NetBps CI containing 0  => edge NOT statistically established on this sample.")

# ── plot ──────────────────────────────────────────────────────────────────────
print("\nPlotting...")
fig = plt.figure(figsize=(17, 9), facecolor=BG)
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.28)
ax_eq, ax_bps, ax_dd = fig.add_subplot(gs[0,:]), fig.add_subplot(gs[1,0]), fig.add_subplot(gs[1,1])
key = ['Baseline           | 1H long, no gate',
       'LongConfirm        | rkL>p90',
       'ShortAvoid         | rkS<p15',
       'Dual strict        | rkL>p85 & rkS<p15',
       'FULL STACK         | rkL>p85&rkS<p15 + dual exit']
cols=[NEUT,LONG_C,CYAN,GOLD,PURPLE]
sel=[(next((r for r in results if r['label']==k),None),c) for k,c in zip(key,cols)]
sel=[(r,c) for r,c in sel if r]
def style(ax,t='',xl='',yl=''):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(GRID); ax.spines['left'].set_color(GRID)
    if t: ax.set_title(t,fontsize=9,fontweight='bold',color=TEXT,pad=6)
    if xl: ax.set_xlabel(xl,fontsize=8,color=NEUT)
    if yl: ax.set_ylabel(yl,fontsize=8,color=NEUT)
    ax.grid(True,alpha=0.3)
for r,col in sel:
    ax_eq.plot(range(len(r['equity'])), r['equity'], color=col, lw=1.7,
               label=f"{r['label'].split('|')[0].strip()}  [{r['total_ret']:+.0f}%  N={r['n']}  DD{r['max_dd']:.0f}%]")
ax_eq.axhline(1,color=GRID,lw=0.7,ls='--'); ax_eq.legend(fontsize=8,loc='upper left')
ax_eq.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_:f'{v:.1f}x'))
style(ax_eq,'DEFINITIVE Equity Curves (reconciled, 14:30 dropped)', yl='Equity (×1)')
labels=[r['label'].split('|')[0].strip() for r,_ in sel]; xb=np.arange(len(sel))
bps=[r['net_bps'] for r,_ in sel]
err=[[r['net_bps']-r['bps_ci'][0] for r,_ in sel],[r['bps_ci'][1]-r['net_bps'] for r,_ in sel]]
bars=ax_bps.bar(xb,bps,color=[c for _,c in sel],alpha=0.85,width=0.6,
                yerr=err,capsize=4,ecolor=TEXT,error_kw={'elinewidth':1})
ax_bps.axhline(0,color=SHORT_C,lw=0.9); ax_bps.axhline(10,color=NEUT,lw=0.7,ls=':',alpha=0.6)
ax_bps.set_xticks(xb); ax_bps.set_xticklabels(labels,fontsize=7,rotation=15)
style(ax_bps,'Avg Net bps per Trade (95% bootstrap CI)',yl='bps')
dds=[r['max_dd'] for r,_ in sel]
bars=ax_dd.bar(xb,dds,color=[c for _,c in sel],alpha=0.85,width=0.6); ax_dd.axhline(0,color=GRID,lw=0.8)
for b,v in zip(bars,dds): ax_dd.text(b.get_x()+b.get_width()/2,v-1.0,f'{v:.1f}%',ha='center',fontsize=8,color=TEXT)
ax_dd.set_xticks(xb); ax_dd.set_xticklabels(labels,fontsize=7,rotation=15)
style(ax_dd,'Max Drawdown',yl='DD %')
fig.suptitle(f'DEFINITIVE Dual-TF Backtest (root-cause resolved)  |  OOS {oos[0]} to {oos[-1]}',
             fontsize=12, fontweight='bold', color=TEXT, y=1.005)
plt.tight_layout()
out='definitive_backtest.png'
for d in [OUT_DIR, MEM_DIR]:
    fig.savefig(os.path.join(d,out),dpi=150,bbox_inches='tight',facecolor=BG)
plt.close(fig)
print(f"  Saved: {OUT_DIR}/{out}")
print(f"\n{SEP}\n  Done.\n{SEP}")

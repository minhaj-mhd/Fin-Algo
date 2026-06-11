"""
Using the 15m SHORT model's D1 (low short rank = 'won't fall') as a LONG filter.

Compares long-entry confirmation schemes on 1h top-K long picks:
  Baseline      no 15m filter
  LongConfirm   require 15m rk_long  > X   (our prior winner)
  ShortAvoid    require 15m rk_short < Y   (short model vetoes the fall -> D1 signal)
  DualAgree     require rk_long > X AND rk_short < Y
  Composite     rank by (rk_long + (1 - rk_short)); keep top by composite

1h = v8_upstox_3y, 15m = v2_15min_3y. 10 bps friction. OOS = last 3 months 1h.
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

BG, AX_BG, TEXT, GRID = '#0f1117', '#161b2e', '#dde1f0', '#252a45'
LONG_C, SHORT_C, GOLD, NEUT, PURPLE, CYAN = '#00d4aa', '#ff6b6b', '#f0c040', '#7c83a3', '#b57bee', '#00b4e0'
plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': AX_BG,
    'axes.edgecolor': GRID, 'axes.labelcolor': TEXT,
    'xtick.color': TEXT, 'ytick.color': TEXT, 'text.color': TEXT,
    'grid.color': GRID, 'grid.linewidth': 0.5,
    'legend.facecolor': '#1e2338', 'legend.edgecolor': GRID,
})
SEP = "=" * 80

# ── load + score ────────────────────────────────────────────────────────────────
print("Loading models...")
with open(f'{H1_MODEL_DIR}/metadata.json')  as f: meta_1h  = json.load(f)
with open(f'{M15_MODEL_DIR}/metadata.json') as f: meta_15m = json.load(f)
feat_1h, feat_15m = meta_1h['features'], meta_15m['features']

b1l = xgb.Booster(); b1l.load_model(f'{H1_MODEL_DIR}/xgb_long_model.json')
b1s = xgb.Booster(); b1s.load_model(f'{H1_MODEL_DIR}/xgb_short_model.json')
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

def feat_matrix(df, feats):
    X = df[feats].values.astype(float)
    for ci in range(X.shape[1]):
        col = X[:, ci]; bad = np.isnan(col) | np.isinf(col)
        if bad.any(): X[bad, ci] = float(np.nanmean(col[~bad])) if (~bad).any() else 0.0
    return xgb.DMatrix(X)

print("Streaming 1h...")
df1, oos = load(H1_DATA, 'Next_Hour_Return')
dm1 = feat_matrix(df1, feat_1h)
df1['s_long']  = b1l.predict(dm1)
df1['s_short'] = b1s.predict(dm1)
df1['ret']     = df1['Next_Hour_Return']
print(f"  {len(df1):,} rows | OOS {oos}")

print("Streaming 15m...")
df15, _ = load(M15_DATA, 'Next_15Min_Return')
dm15 = feat_matrix(df15, feat_15m)
df15['s_long']  = m15l.predict(dm15)
df15['s_short'] = m15s.predict(dm15)
df15['rk_long']  = df15.groupby('dt')['s_long'].rank(pct=True)
df15['rk_short'] = df15.groupby('dt')['s_short'].rank(pct=True)
print(f"  {len(df15):,} rows")

# lookup at entry (T+45)
TD45 = pd.Timedelta('45min')
idx15 = df15.set_index(['Ticker', 'dt'])[['rk_long', 'rk_short']]
def ranks_at(tk, ts):
    try:
        row = idx15.loc[(tk, ts)]
        if isinstance(row, pd.DataFrame): row = row.iloc[0]
        return float(row['rk_long']), float(row['rk_short'])
    except KeyError:
        return np.nan, np.nan

# ── backtest engine ───────────────────────────────────────────────────────────
def run(scheme, top_k=3, x_long=None, y_short=None, comp_top=None, label=''):
    """
    scheme: 'baseline' | 'longconfirm' | 'shortavoid' | 'dualagree' | 'composite'
    x_long : rk_long threshold (require rk_long > x_long)
    y_short: rk_short threshold (require rk_short < y_short)
    comp_top: for composite, keep stocks with composite >= this percentile within the 1h top pool
    """
    trades = []
    for dt1, grp in df1.groupby('dt'):
        top = grp.nlargest(top_k if scheme != 'composite' else max(top_k*4, 12), 's_long')
        cand = []
        for _, row in top.iterrows():
            rkl, rks = ranks_at(row['Ticker'], dt1 + TD45)
            if np.isnan(rkl):  # no 15m data
                continue
            ok = False
            if scheme == 'baseline':
                ok = True
            elif scheme == 'longconfirm':
                ok = rkl > x_long
            elif scheme == 'shortavoid':
                ok = rks < y_short
            elif scheme == 'dualagree':
                ok = (rkl > x_long) and (rks < y_short)
            elif scheme == 'composite':
                ok = True  # filter later
            if ok:
                comp = rkl + (1 - rks)
                cand.append((row['Ticker'], row['ret'], rkl, rks, comp))

        if scheme == 'composite':
            cand.sort(key=lambda c: c[4], reverse=True)
            cand = cand[:top_k]
        elif scheme == 'baseline':
            cand = cand[:top_k]

        for tk, ret, rkl, rks, comp in cand:
            trades.append({'net': ret - COST, 'dt': dt1, 'tk': tk})

    if not trades:
        return None
    t = pd.DataFrame(trades).sort_values('dt').reset_index(drop=True)
    g = t['net'] + COST
    eq = (1 + t['net']).cumprod()
    rm = eq.cummax()
    return dict(label=label, n=len(t),
                raw_wr=(g > 0).mean()*100, net_wr=(t['net'] > 0).mean()*100,
                raw_bps=g.mean()*10000, net_bps=t['net'].mean()*10000,
                total_ret=(eq.iloc[-1]-1)*100, max_dd=((eq-rm)/rm).min()*100,
                equity=eq)

# ── configs ─────────────────────────────────────────────────────────────────────
cfgs = [
    dict(scheme='baseline',                                     label='Baseline        | 1H long K=3'),
    dict(scheme='longconfirm', x_long=0.90,                     label='LongConfirm     | rk_long>p90'),
    dict(scheme='shortavoid',  y_short=0.25,                    label='ShortAvoid      | rk_short<p25'),
    dict(scheme='shortavoid',  y_short=0.15,                    label='ShortAvoid      | rk_short<p15'),
    dict(scheme='shortavoid',  y_short=0.10,                    label='ShortAvoid (D1) | rk_short<p10'),
    dict(scheme='dualagree',   x_long=0.80, y_short=0.20,       label='DualAgree       | L>p80 & S<p20'),
    dict(scheme='dualagree',   x_long=0.85, y_short=0.15,       label='DualAgree strict | L>p85 & S<p15'),
    dict(scheme='composite',                                    label='Composite       | top by L+(1-S)'),
    # K=5 variants of the best ideas
    dict(scheme='baseline',    top_k=5,                         label='Baseline        | 1H long K=5'),
    dict(scheme='shortavoid',  top_k=5, y_short=0.15,           label='ShortAvoid      | rk_short<p15 K=5'),
    dict(scheme='dualagree',   top_k=5, x_long=0.80, y_short=0.20, label='DualAgree     | L>p80 & S<p20 K=5'),
    dict(scheme='composite',   top_k=5,                         label='Composite       | top by L+(1-S) K=5'),
]

print(f"\n{SEP}\n  Running {len(cfgs)} long-filter configurations...\n{SEP}")
results = []
for c in cfgs:
    r = run(**{k: v for k, v in c.items() if k != 'label'}, label=c['label'])
    if r:
        results.append(r)
        print(f"  {r['label']:<40}  N={r['n']:>5,}  Raw={r['raw_wr']:.1f}%  "
              f"Net={r['net_wr']:.1f}%  Bps={r['net_bps']:+.1f}  "
              f"Ret={r['total_ret']:+.1f}%  DD={r['max_dd']:.1f}%")

# ── summary table ─────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print(f"  {'Config':<42} {'N':>5}  {'RawWR':>7} {'NetWR':>7}  {'NetBps':>8}  {'TotRet%':>9}  {'MaxDD%':>8}")
print(f"  {'-'*90}")
for r in results:
    print(f"  {r['label']:<42} {r['n']:>5,}  {r['raw_wr']:>6.1f}% {r['net_wr']:>6.1f}%  "
          f"{r['net_bps']:>+7.2f}  {r['total_ret']:>+8.1f}%  {r['max_dd']:>7.1f}%")

# ── plot ──────────────────────────────────────────────────────────────────────
print("\nPlotting...")
fig = plt.figure(figsize=(17, 10), facecolor=BG)
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.28)
ax_eq  = fig.add_subplot(gs[0, :])
ax_bps = fig.add_subplot(gs[1, 0])
ax_dd  = fig.add_subplot(gs[1, 1])

def style(ax, title='', xl='', yl=''):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(GRID); ax.spines['left'].set_color(GRID)
    if title: ax.set_title(title, fontsize=9, fontweight='bold', color=TEXT, pad=6)
    if xl: ax.set_xlabel(xl, fontsize=8, color=NEUT)
    if yl: ax.set_ylabel(yl, fontsize=8, color=NEUT)
    ax.grid(True, alpha=0.3)

key = ['Baseline        | 1H long K=3',
       'LongConfirm     | rk_long>p90',
       'ShortAvoid      | rk_short<p15',
       'DualAgree       | L>p80 & S<p20',
       'Composite       | top by L+(1-S)']
cols = [NEUT, LONG_C, GOLD, PURPLE, CYAN]
sel = [r for r in results if r['label'] in key]
for r, col in zip(sel, cols):
    eq = r['equity'].reset_index(drop=True)
    ax_eq.plot(eq.index, eq.values, color=col, lw=1.7,
               label=f"{r['label'].split('|')[0].strip()}  [{r['total_ret']:+.0f}%  N={r['n']}]")
ax_eq.axhline(1, color=GRID, lw=0.7, ls='--')
ax_eq.legend(fontsize=8, loc='upper left')
ax_eq.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.1f}x'))
style(ax_eq, 'Long-Side Equity: Short-Model-as-Filter Schemes (K=3)', yl='Equity (×1)')

labels = [r['label'].split('|')[0].strip() for r in sel]
xb = np.arange(len(sel))
bps = [r['net_bps'] for r in sel]
bars = ax_bps.bar(xb, bps, color=cols, alpha=0.85, width=0.6)
ax_bps.axhline(0, color=GRID, lw=0.8); ax_bps.axhline(10, color=SHORT_C, lw=0.7, ls=':', alpha=0.6)
for b, v in zip(bars, bps):
    ax_bps.text(b.get_x()+b.get_width()/2, v+0.5, f'{v:+.1f}', ha='center', fontsize=8, color=TEXT)
ax_bps.set_xticks(xb); ax_bps.set_xticklabels(labels, fontsize=7, rotation=15)
style(ax_bps, 'Avg Net bps per Trade', yl='bps')

dds = [r['max_dd'] for r in sel]
bars = ax_dd.bar(xb, dds, color=cols, alpha=0.85, width=0.6)
ax_dd.axhline(0, color=GRID, lw=0.8)
for b, v in zip(bars, dds):
    ax_dd.text(b.get_x()+b.get_width()/2, v-1.2, f'{v:.1f}%', ha='center', fontsize=8, color=TEXT)
ax_dd.set_xticks(xb); ax_dd.set_xticklabels(labels, fontsize=7, rotation=15)
style(ax_dd, 'Max Drawdown', yl='DD %')

fig.suptitle(f'Short-Model-as-Long-Filter  |  1H + 15M  |  OOS {oos[0]} to {oos[-1]}',
             fontsize=12, fontweight='bold', color=TEXT, y=1.005)
plt.tight_layout()
out = 'long_filter_comparison.png'
for d in [OUT_DIR, MEM_DIR]:
    fig.savefig(os.path.join(d, out), dpi=150, bbox_inches='tight', facecolor=BG)
plt.close(fig)
print(f"  Saved: {OUT_DIR}/{out}")
print(f"\n{SEP}\n  Done.\n{SEP}")

"""
Dual-Timeframe Backtest: v8_upstox_3y (1h) confirmed by v2_15min_3y (15m).
Runs Long side, Short side, and Combined portfolio.

Entry  : 1h model top-K → confirmed by 15m model at T+45min (last bar before entry).
Hold   : Up to 4 x 15-min bars (= 1 hour). Base return = Next_Hour_Return.
Exit   : Early if 15m rank drops during hold; else force-exit after 1 full hour.
Costs  : 10 bps flat per round trip.
OOS    : Last 3 months of 1h data (Mar–May 2026).
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
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
LONG_C, SHORT_C, GOLD, NEUT, PURPLE = '#00d4aa', '#ff6b6b', '#f0c040', '#7c83a3', '#b57bee'
plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': AX_BG,
    'axes.edgecolor': GRID, 'axes.labelcolor': TEXT,
    'xtick.color': TEXT, 'ytick.color': TEXT, 'text.color': TEXT,
    'grid.color': GRID, 'grid.linewidth': 0.5,
    'legend.facecolor': '#1e2338', 'legend.edgecolor': GRID,
})
SEP = "=" * 76

# ── load models ────────────────────────────────────────────────────────────────
print("Loading models...")
with open(f'{H1_MODEL_DIR}/metadata.json')  as f: meta_1h  = json.load(f)
with open(f'{M15_MODEL_DIR}/metadata.json') as f: meta_15m = json.load(f)
feat_1h  = meta_1h['features']
feat_15m = meta_15m['features']

bst_1h_long  = xgb.Booster(); bst_1h_long.load_model(f'{H1_MODEL_DIR}/xgb_long_model.json')
bst_1h_short = xgb.Booster(); bst_1h_short.load_model(f'{H1_MODEL_DIR}/xgb_short_model.json')
bst_15m_long = xgb.Booster(); bst_15m_long.load_model(f'{M15_MODEL_DIR}/xgb_long_model.json')
bst_15m_short= xgb.Booster(); bst_15m_short.load_model(f'{M15_MODEL_DIR}/xgb_short_model.json')

# ── load & score 1h OOS ───────────────────────────────────────────────────────
print("Streaming 1h OOS data...")
all_m = set()
for ch in pd.read_csv(H1_DATA, usecols=['DateTime'], chunksize=500_000):
    all_m.update(ch['DateTime'].str[:7].unique())
oos_m = sorted(all_m)[-OOS_MONTHS:]
print(f"  OOS: {oos_m}")

chunks = []
for ch in pd.read_csv(H1_DATA, chunksize=200_000):
    sub = ch[ch['DateTime'].str[:7].isin(oos_m)]
    if len(sub): chunks.append(sub)
df1 = pd.concat(chunks, ignore_index=True)
df1['dt'] = pd.to_datetime(df1['DateTime']).dt.tz_localize(None)
df1 = df1.dropna(subset=['Next_Hour_Return'])
print(f"  {len(df1):,} rows | {df1['Ticker'].nunique()} tickers")

X1 = df1[feat_1h].values.astype(float)
for ci in range(X1.shape[1]):
    col = X1[:, ci]; bad = np.isnan(col) | np.isinf(col)
    if bad.any(): X1[bad, ci] = float(np.nanmean(col[~bad])) if (~bad).any() else 0.0

df1['score_1h_long']  = bst_1h_long.predict(xgb.DMatrix(X1))
df1['score_1h_short'] = bst_1h_short.predict(xgb.DMatrix(X1))
df1['rank_1h_long']   = df1.groupby('dt')['score_1h_long'].rank(pct=True)
df1['rank_1h_short']  = df1.groupby('dt')['score_1h_short'].rank(pct=True)
print("  1h scored.")

# ── load & score 15m OOS ──────────────────────────────────────────────────────
print("Streaming 15m OOS data...")
chunks = []
for ch in pd.read_csv(M15_DATA, chunksize=200_000):
    sub = ch[ch['DateTime'].str[:7].isin(oos_m)]
    if len(sub): chunks.append(sub)
df15 = pd.concat(chunks, ignore_index=True)
df15['dt'] = pd.to_datetime(df15['DateTime']).dt.tz_localize(None)
print(f"  {len(df15):,} rows")

X15 = df15[feat_15m].values.astype(float)
for ci in range(X15.shape[1]):
    col = X15[:, ci]; bad = np.isnan(col) | np.isinf(col)
    if bad.any(): X15[bad, ci] = float(np.nanmean(col[~bad])) if (~bad).any() else 0.0

df15['score_15m_long']  = bst_15m_long.predict(xgb.DMatrix(X15))
df15['score_15m_short'] = bst_15m_short.predict(xgb.DMatrix(X15))
df15['rank_15m_long']   = df15.groupby('dt')['score_15m_long'].rank(pct=True)
df15['rank_15m_short']  = df15.groupby('dt')['score_15m_short'].rank(pct=True)
print("  15m scored.")

# ── build 15m lookup ─────────────────────────────────────────────────────────
df15_idx = df15.set_index(['Ticker', 'dt'])

def get15(ticker, ts, field):
    try:
        v = df15_idx.loc[(ticker, ts), field]
        return float(v) if not isinstance(v, pd.Series) else float(v.iloc[0])
    except KeyError:
        return np.nan

TD45  = pd.Timedelta('45min')
TD60  = pd.Timedelta('60min')
TD75  = pd.Timedelta('75min')
TD90  = pd.Timedelta('90min')
TD105 = pd.Timedelta('105min')

# ── backtest engine ───────────────────────────────────────────────────────────
def run(direction, top_k=3, conf_pct=None, exit_pct=None, label=''):
    """
    direction : 'long' or 'short'
    conf_pct  : minimum rank_15m percentile required at entry (None = no filter)
    exit_pct  : rank floor during hold; exit if rank drops below (None = no early exit)
    """
    score_col = 'score_1h_long' if direction == 'long' else 'score_1h_short'
    rank_15m  = 'rank_15m_long' if direction == 'long' else 'rank_15m_short'

    trades = []
    for dt1, grp in df1.groupby('dt'):
        top = grp.nlargest(top_k, score_col)
        for _, row in top.iterrows():
            tk  = row['Ticker']
            r1h = row['Next_Hour_Return']

            # ── 15m entry confirmation ────────────────────────────────────────
            if conf_pct is not None:
                rk = get15(tk, dt1 + TD45, rank_15m)
                if np.isnan(rk) or rk < conf_pct:
                    continue

            # ── early exit monitoring ─────────────────────────────────────────
            if exit_pct is not None:
                bars = {off: {'rank': get15(tk, dt1 + off, rank_15m),
                               'ret':  get15(tk, dt1 + off, 'Next_15Min_Return')}
                        for off in [TD60, TD75, TD90, TD105]}

                cumret = 0.0
                exited = False
                for off, nxt in [(TD60, TD75), (TD75, TD90), (TD90, TD105)]:
                    br = bars[off]['ret']
                    if np.isnan(br): break
                    gross_bar = br if direction == 'long' else -br
                    cumret = (1 + cumret) * (1 + gross_bar) - 1
                    nxt_rk = bars[nxt]['rank']
                    if not np.isnan(nxt_rk) and nxt_rk < exit_pct:
                        trades.append({'net': cumret - COST, 'dt': dt1,
                                       'tk': tk, 'exit': 'early'})
                        exited = True; break
                if not exited:
                    gross = r1h if direction == 'long' else -r1h
                    trades.append({'net': gross - COST, 'dt': dt1,
                                   'tk': tk, 'exit': 'full'})
            else:
                gross = r1h if direction == 'long' else -r1h
                trades.append({'net': gross - COST, 'dt': dt1,
                               'tk': tk, 'exit': 'full'})

    if not trades:
        return None
    t = pd.DataFrame(trades).sort_values('dt').reset_index(drop=True)
    gross_arr = t['net'] + COST
    net_arr   = t['net']
    equity    = (1 + net_arr).cumprod()
    roll_max  = equity.cummax()
    return dict(
        label     = label,
        direction = direction,
        n         = len(t),
        raw_wr    = (gross_arr > 0).mean() * 100,
        net_wr    = (net_arr   > 0).mean() * 100,
        raw_bps   = gross_arr.mean() * 10000,
        net_bps   = net_arr.mean()   * 10000,
        total_ret = (equity.iloc[-1] - 1) * 100,
        max_dd    = ((equity - roll_max) / roll_max).min() * 100,
        early_pct = (t['exit'] == 'early').mean() * 100 if 'exit' in t.columns else 0,
        trades    = t,
        equity    = equity,
    )

# ── configurations ─────────────────────────────────────────────────────────────
cfgs = [
    # ── LONG ──
    dict(direction='long',  top_k=3, conf_pct=None, exit_pct=None, label='Baseline LONG  | 1H only  K=3'),
    dict(direction='long',  top_k=3, conf_pct=0.85, exit_pct=None, label='Dual-TF  LONG  | conf p85 K=3'),
    dict(direction='long',  top_k=3, conf_pct=0.90, exit_pct=None, label='Dual-TF  LONG  | conf p90 K=3'),
    dict(direction='long',  top_k=3, conf_pct=0.95, exit_pct=None, label='Dual-TF  LONG  | conf p95 K=3'),
    dict(direction='long',  top_k=3, conf_pct=0.90, exit_pct=0.40, label='Dual+Exit LONG  | p90 exit<p40'),
    dict(direction='long',  top_k=5, conf_pct=0.90, exit_pct=None, label='Dual-TF  LONG  | conf p90 K=5'),
    # ── SHORT ──
    dict(direction='short', top_k=3, conf_pct=None, exit_pct=None, label='Baseline SHORT | 1H only  K=3'),
    dict(direction='short', top_k=3, conf_pct=0.85, exit_pct=None, label='Dual-TF  SHORT | conf p85 K=3'),
    dict(direction='short', top_k=3, conf_pct=0.90, exit_pct=None, label='Dual-TF  SHORT | conf p90 K=3'),
    dict(direction='short', top_k=3, conf_pct=0.95, exit_pct=None, label='Dual-TF  SHORT | conf p95 K=3'),
    dict(direction='short', top_k=3, conf_pct=0.90, exit_pct=0.40, label='Dual+Exit SHORT | p90 exit<p40'),
    dict(direction='short', top_k=5, conf_pct=0.90, exit_pct=None, label='Dual-TF  SHORT | conf p90 K=5'),
]

print(f"\n{SEP}\n  Running {len(cfgs)} configurations...\n{SEP}")
results = []
for cfg in cfgs:
    r = run(**cfg)
    if r:
        results.append(r)
        ep = f"  early={r['early_pct']:.0f}%" if r['early_pct'] > 0 else ""
        print(f"  {r['label']:<42}  N={r['n']:>5,}  Raw={r['raw_wr']:.1f}%  "
              f"Net={r['net_wr']:.1f}%  Bps={r['net_bps']:+.1f}  "
              f"Ret={r['total_ret']:+.1f}%  DD={r['max_dd']:.1f}%{ep}")

# ── combined portfolio for best config ────────────────────────────────────────
def combined_portfolio(long_res, short_res, label='Combined'):
    """Pool all trades from long + short, sort by date, compound."""
    tl = long_res['trades'].copy();  tl['side'] = 'long'
    ts = short_res['trades'].copy(); ts['side'] = 'short'
    all_t = pd.concat([tl, ts], ignore_index=True).sort_values('dt').reset_index(drop=True)
    net   = all_t['net']
    eq    = (1 + net).cumprod()
    rm    = eq.cummax()
    return dict(
        label     = label,
        direction = 'combined',
        n         = len(all_t),
        raw_wr    = ((net + COST) > 0).mean() * 100,
        net_wr    = (net > 0).mean() * 100,
        raw_bps   = (net + COST).mean() * 10000,
        net_bps   = net.mean() * 10000,
        total_ret = (eq.iloc[-1] - 1) * 100,
        max_dd    = ((eq - rm) / rm).min() * 100,
        early_pct = (all_t['exit'] == 'early').mean() * 100 if 'exit' in all_t else 0,
        trades    = all_t,
        equity    = eq,
    )

# build combined portfolios for key matching pairs
combos = [
    ('Baseline LONG  | 1H only  K=3',   'Baseline SHORT | 1H only  K=3',   'Combined Baseline  | K=3'),
    ('Dual-TF  LONG  | conf p90 K=3',   'Dual-TF  SHORT | conf p90 K=3',   'Combined Dual-TF   | p90 K=3'),
    ('Dual-TF  LONG  | conf p95 K=3',   'Dual-TF  SHORT | conf p95 K=3',   'Combined Dual-TF   | p95 K=3'),
    ('Dual+Exit LONG  | p90 exit<p40',  'Dual+Exit SHORT | p90 exit<p40',  'Combined Dual+Exit | p90 exit<p40'),
    ('Dual-TF  LONG  | conf p90 K=5',   'Dual-TF  SHORT | conf p90 K=5',   'Combined Dual-TF   | p90 K=5'),
]

res_by_label = {r['label']: r for r in results}
combined_results = []
for ll, sl, cl in combos:
    if ll in res_by_label and sl in res_by_label:
        c = combined_portfolio(res_by_label[ll], res_by_label[sl], cl)
        combined_results.append(c)

# ── print summary tables ──────────────────────────────────────────────────────
for section, section_res in [('LONG', [r for r in results if r['direction'] == 'long']),
                               ('SHORT', [r for r in results if r['direction'] == 'short']),
                               ('COMBINED PORTFOLIO', combined_results)]:
    print(f"\n{SEP}")
    print(f"  {section}")
    print(SEP)
    print(f"  {'Config':<44} {'N':>5}  {'RawWR':>7} {'NetWR':>7}  {'NetBps':>8}  {'TotRet%':>9}  {'MaxDD%':>8}")
    print(f"  {'-'*90}")
    for r in section_res:
        ep = f"  e={r['early_pct']:.0f}%" if r.get('early_pct', 0) > 0 else ""
        print(f"  {r['label']:<44} {r['n']:>5,}  {r['raw_wr']:>6.1f}% {r['net_wr']:>6.1f}%  "
              f"{r['net_bps']:>+7.2f}  {r['total_ret']:>+8.1f}%  {r['max_dd']:>7.1f}%{ep}")

# ── PLOTS ─────────────────────────────────────────────────────────────────────
print(f"\nGenerating plots...")

def equity_plot(ax, res_list, colors, title):
    for r, col in zip(res_list, colors):
        eq = r['equity'].reset_index(drop=True)
        ax.plot(eq.index, eq.values, color=col, lw=1.6,
                label=f"{r['label'].split('|')[1].strip()}  [{r['total_ret']:+.0f}%  N={r['n']}]")
    ax.axhline(1, color=GRID, lw=0.7, ls='--')
    ax.set_title(title, fontsize=9, fontweight='bold', color=TEXT, pad=6)
    ax.set_ylabel('Equity (×1)', fontsize=8, color=NEUT)
    ax.legend(fontsize=7, loc='upper left')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.1f}x'))
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(GRID); ax.spines['left'].set_color(GRID)
    ax.grid(True, alpha=0.3)

fig = plt.figure(figsize=(18, 14), facecolor=BG)
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.28)

ax_long  = fig.add_subplot(gs[0, :])   # long equity (full width)
ax_short = fig.add_subplot(gs[1, :])   # short equity (full width)
ax_comb  = fig.add_subplot(gs[2, :])   # combined equity (full width)

long_plot  = [res_by_label[k] for k in [
    'Baseline LONG  | 1H only  K=3',
    'Dual-TF  LONG  | conf p90 K=3',
    'Dual-TF  LONG  | conf p95 K=3',
    'Dual+Exit LONG  | p90 exit<p40',
] if k in res_by_label]

short_plot = [res_by_label[k] for k in [
    'Baseline SHORT | 1H only  K=3',
    'Dual-TF  SHORT | conf p90 K=3',
    'Dual-TF  SHORT | conf p95 K=3',
    'Dual+Exit SHORT | p90 exit<p40',
] if k in res_by_label]

equity_plot(ax_long,  long_plot,  [NEUT, LONG_C, GOLD, PURPLE],
            'LONG Side — Baseline vs Dual-TF Configurations')
equity_plot(ax_short, short_plot, [NEUT, SHORT_C, GOLD, PURPLE],
            'SHORT Side — Baseline vs Dual-TF Configurations')
equity_plot(ax_comb,  combined_results[:4], [NEUT, LONG_C, GOLD, PURPLE],
            'COMBINED Portfolio (Long + Short) — Baseline vs Dual-TF')

fig.suptitle(
    f'Dual-TF Backtest: v8_upstox_3y (1H) + v2_15min_3y (15M)  |  OOS {oos_m[0]} to {oos_m[-1]}',
    fontsize=12, fontweight='bold', color=TEXT, y=1.005)
plt.tight_layout()

out = 'dual_tf_full.png'
for d in [OUT_DIR, MEM_DIR]:
    fig.savefig(os.path.join(d, out), dpi=150, bbox_inches='tight', facecolor=BG)
plt.close(fig)
print(f"  Saved: {OUT_DIR}/{out}")
print(f"\n{SEP}\n  Done.\n{SEP}")

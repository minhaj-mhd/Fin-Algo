"""
Hour-by-hour breakdown of signal quality for each tier threshold.
No time filter imposed — let the data show where the edge lives.
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

MODEL_DIR  = 'models/v2_15min_3y'
META_PATH  = f'{MODEL_DIR}/metadata.json'
LONG_PATH  = f'{MODEL_DIR}/xgb_long_model.json'
DATA_FILE  = 'data/ranking_data_upstox_15min_3y.csv'
OUT_DIR    = 'data/model_analysis/v2_15min_3y'
MEM_DIR    = r'finalgo-memory-layer\finalgo\08. Model Analysis\15-Minute Vanguard Model\assets'
os.makedirs(OUT_DIR, exist_ok=True)

COST = 10 / 10000
OOS_MONTHS = 3

BG, AX_BG, TEXT, GRID = '#0f1117', '#161b2e', '#dde1f0', '#252a45'
LONG_C, SHORT_C, GOLD, NEUT = '#00d4aa', '#ff6b6b', '#f0c040', '#7c83a3'
plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': AX_BG,
    'axes.edgecolor': GRID, 'axes.labelcolor': TEXT,
    'xtick.color': TEXT, 'ytick.color': TEXT, 'text.color': TEXT,
    'grid.color': GRID, 'grid.linewidth': 0.5,
    'legend.facecolor': '#1e2338', 'legend.edgecolor': GRID,
})

print("Loading...")
with open(META_PATH) as f: meta = json.load(f)
feature_cols = meta['features']
bst_long = xgb.Booster(); bst_long.load_model(LONG_PATH)

all_months = set()
for ch in pd.read_csv(DATA_FILE, usecols=['DateTime'], chunksize=500_000):
    all_months.update(ch['DateTime'].str[:7].unique())
oos_m = sorted(all_months)[-OOS_MONTHS:]

chunks = []
for ch in pd.read_csv(DATA_FILE, chunksize=200_000):
    sub = ch[ch['DateTime'].str[:7].isin(oos_m)]
    if len(sub): chunks.append(sub)
df = pd.concat(chunks, ignore_index=True)

X = df[feature_cols].values.astype(float)
for ci in range(X.shape[1]):
    col = X[:, ci]; bad = np.isnan(col) | np.isinf(col)
    if bad.any():
        X[bad, ci] = float(np.nanmean(col[~bad])) if (~bad).any() else 0.0

df['long_score'] = bst_long.predict(xgb.DMatrix(X))
df['dt']         = pd.to_datetime(df['DateTime'])
df['hour']       = df['dt'].dt.hour
df['ret']        = df['Next_15Min_Return']

hours = sorted(df['hour'].unique())

thresholds = [
    ('Tier 1  L>0.0829 (p99)', 0.0829),
    ('Tier 2  L>0.0629 (p95)', 0.0629),
    ('Tier 3  L>0.0514 (p90)', 0.0514),
    ('No filter (all L scores)', None),   # entire universe at each hour
]

SEP = "=" * 80

for label, thr in thresholds:
    print(f"\n{SEP}")
    print(f"  {label}")
    print(SEP)
    print(f"  {'Hour':>5}  {'N':>6}  {'Raw WR':>8}  {'Net WR':>8}  {'Raw bps':>9}  {'Net bps':>9}")
    print(f"  {'-'*60}")

    for h in hours:
        if thr is not None:
            mask = (df['long_score'] > thr) & (df['hour'] == h)
        else:
            mask = (df['hour'] == h)
        sub = df[mask]
        if len(sub) < 5:
            continue
        gross = sub['ret'].values
        net   = gross - COST
        raw_wr  = (gross > 0).mean() * 100
        net_wr  = (net   > 0).mean() * 100
        raw_bps = gross.mean() * 10000
        net_bps = net.mean()   * 10000
        marker = " <-- sniper" if h == 15 else ""
        print(f"  {h:>5}h  {len(sub):>6,}  {raw_wr:>7.1f}%  {net_wr:>7.1f}%  "
              f"{raw_bps:>+8.2f}  {net_bps:>+8.2f}{marker}")

# ── PLOT: net bps by hour for each tier ──────────────────────────────────────
print(f"\nGenerating plot...")

fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor=BG)
axes = axes.flatten()

plot_configs = [
    ('Tier 1  L>0.0829 (p99)', 0.0829, LONG_C),
    ('Tier 2  L>0.0629 (p95)', 0.0629, '#00b4e0'),
    ('Tier 3  L>0.0514 (p90)', 0.0514, GOLD),
    ('No filter — full universe', None,  NEUT),
]

for ax, (label, thr, color) in zip(axes, plot_configs):
    net_by_hour = []
    raw_by_hour = []
    ns = []
    for h in hours:
        if thr is not None:
            mask = (df['long_score'] > thr) & (df['hour'] == h)
        else:
            mask = (df['hour'] == h)
        sub = df[mask]
        if len(sub) < 5:
            net_by_hour.append(np.nan); raw_by_hour.append(np.nan); ns.append(0)
            continue
        gross = sub['ret'].values
        net   = gross - COST
        net_by_hour.append(net.mean() * 10000)
        raw_by_hour.append(gross.mean() * 10000)
        ns.append(len(sub))

    x = np.arange(len(hours))
    bar_colors = [LONG_C if v is not None and not np.isnan(v) and v >= 0 else SHORT_C
                  for v in net_by_hour]
    bars = ax.bar(x, net_by_hour, color=bar_colors, alpha=0.8, width=0.6, label='Net bps (10bp friction)')
    ax.plot(x, raw_by_hour, color=GOLD, lw=1.5, ls='--', marker='o', markersize=4, label='Raw bps')
    ax.axhline(0, color=GRID, lw=1.0)
    ax.axhline(10, color=color, lw=0.7, ls=':', alpha=0.5, label='+10 bps (fee breakeven)')

    # annotate N on each bar
    for i, (bar, n) in enumerate(zip(bars, ns)):
        if n > 0 and not np.isnan(net_by_hour[i]):
            ax.text(bar.get_x() + bar.get_width()/2,
                    net_by_hour[i] + (1.5 if net_by_hour[i] >= 0 else -3),
                    f'n={n}', ha='center', fontsize=6, color=TEXT, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{h}h' for h in hours], fontsize=8)
    ax.set_title(label, fontsize=9, fontweight='bold', color=TEXT, pad=6)
    ax.set_ylabel('bps per trade', fontsize=8, color=NEUT)
    ax.legend(fontsize=7, loc='upper left')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(GRID); ax.spines['left'].set_color(GRID)
    ax.grid(True, alpha=0.3, axis='y')

fig.suptitle('Signal Quality by Hour of Day — v2_15min_3y OOS (Apr–Jun 2026)',
             fontsize=11, fontweight='bold', color=TEXT, y=1.01)
plt.tight_layout()

out_name = '13_hourly_breakdown.png'
for d in [OUT_DIR, MEM_DIR]:
    fig.savefig(os.path.join(d, out_name), dpi=150, bbox_inches='tight', facecolor=BG)
plt.close(fig)
print(f"  Saved: {OUT_DIR}/{out_name}")

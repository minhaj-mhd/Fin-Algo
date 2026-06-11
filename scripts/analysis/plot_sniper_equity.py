"""
Equity curve + drawdown for Tier 1 sniper: L > 0.0829 at hour 15.
Two-panel chart: top = cumulative equity, bottom = drawdown.
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
warnings.filterwarnings('ignore')
sys.path.append(os.getcwd())

MODEL_DIR  = 'models/v2_15min_3y'
META_PATH  = f'{MODEL_DIR}/metadata.json'
LONG_PATH  = f'{MODEL_DIR}/xgb_long_model.json'
SHORT_PATH = f'{MODEL_DIR}/xgb_short_model.json'
DATA_FILE  = 'data/ranking_data_upstox_15min_3y.csv'
OUT_DIR    = 'data/model_analysis/v2_15min_3y'
MEM_DIR    = r'finalgo-memory-layer\finalgo\08. Model Analysis\15-Minute Vanguard Model\assets'
os.makedirs(OUT_DIR, exist_ok=True)

COST_BPS   = 10
COST       = COST_BPS / 10000
OOS_MONTHS = 3

# Tier 1 filter
THR_LONG   = 0.0829
SNIPER_HOUR = 15

# ── palette ────────────────────────────────────────────────────────────────────
BG, AX_BG, TEXT, GRID = '#0f1117', '#161b2e', '#dde1f0', '#252a45'
LONG_C, SHORT_C, GOLD, NEUT = '#00d4aa', '#ff6b6b', '#f0c040', '#7c83a3'
plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': AX_BG,
    'axes.edgecolor': GRID, 'axes.labelcolor': TEXT,
    'xtick.color': TEXT, 'ytick.color': TEXT, 'text.color': TEXT,
    'grid.color': GRID, 'grid.linewidth': 0.5,
    'legend.facecolor': '#1e2338', 'legend.edgecolor': GRID,
})

print("Loading models...")
with open(META_PATH) as f:
    meta = json.load(f)
feature_cols = meta['features']
bst_long  = xgb.Booster(); bst_long.load_model(LONG_PATH)
bst_short = xgb.Booster(); bst_short.load_model(SHORT_PATH)

print("Streaming OOS data...")
all_months = set()
for ch in pd.read_csv(DATA_FILE, usecols=['DateTime'], chunksize=500_000):
    all_months.update(ch['DateTime'].str[:7].unique())
oos_m = sorted(all_months)[-OOS_MONTHS:]
print(f"  OOS months: {oos_m}")

chunks = []
for ch in pd.read_csv(DATA_FILE, chunksize=200_000):
    sub = ch[ch['DateTime'].str[:7].isin(oos_m)]
    if len(sub): chunks.append(sub)
df = pd.concat(chunks, ignore_index=True)

print(f"  Loaded {len(df):,} rows")

X = df[feature_cols].values.astype(float)
for ci in range(X.shape[1]):
    col = X[:, ci]; bad = np.isnan(col) | np.isinf(col)
    if bad.any():
        X[bad, ci] = float(np.nanmean(col[~bad])) if (~bad).any() else 0.0

df['long_score'] = bst_long.predict(xgb.DMatrix(X))
df['dt']         = pd.to_datetime(df['DateTime'])
df['hour']       = df['dt'].dt.hour
df['ret']        = df['Next_15Min_Return']

# ── filter to Tier 1 sniper ───────────────────────────────────────────────────
mask = (df['long_score'] > THR_LONG) & (df['hour'] == SNIPER_HOUR)
trades = df[mask].copy()
trades = trades.sort_values('dt').reset_index(drop=True)

print(f"\nTier 1 sniper: L>{THR_LONG} @ {SNIPER_HOUR}h")
print(f"  Trades : {len(trades):,}")

# net return per trade
trades['net'] = trades['ret'] - COST

# cumulative equity (starting at 1.0)
trades['equity'] = (1 + trades['net']).cumprod()

# drawdown
trades['roll_max'] = trades['equity'].cummax()
trades['dd']       = (trades['equity'] - trades['roll_max']) / trades['roll_max'] * 100

# stats for annotations
total_ret = (trades['equity'].iloc[-1] - 1) * 100
max_dd    = trades['dd'].min()
wr        = (trades['net'] > 0).mean() * 100
avg_bps   = trades['net'].mean() * 10000
n_trades  = len(trades)

# also compute equal-weight market benchmark per trade (unrestricted universe, same dates)
# market = average return of all stocks at each 15h bar
mkt = df[df['hour'] == SNIPER_HOUR].groupby('dt')['ret'].mean().reset_index()
mkt = mkt.rename(columns={'ret': 'mkt_ret'})
mkt['mkt_equity'] = (1 + mkt['mkt_ret']).cumprod()
mkt = mkt.sort_values('dt').reset_index(drop=True)

# ── build trade-date x-axis (one point per trade) ────────────────────────────
x_trades = trades['dt'].values

# ── PLOT ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 9), facecolor=BG)
gs  = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.08)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)

# ── top panel: equity curve ──────────────────────────────────────────────────
# market benchmark aligned to sniper trade dates
mkt_at_trades = mkt.set_index('dt')['mkt_equity'].reindex(
    trades['dt'], method='nearest').values / mkt['mkt_equity'].iloc[0]

ax1.plot(x_trades, trades['equity'].values, color=LONG_C, lw=1.8, label='Tier 1 Sniper (L>0.0829 @ 15h)')
ax1.plot(x_trades, mkt_at_trades, color=NEUT, lw=1.2, ls='--', alpha=0.7, label='Market @ 15h (equal-weight)')
ax1.fill_between(x_trades, trades['equity'].values, mkt_at_trades,
                 where=trades['equity'].values >= mkt_at_trades,
                 color=LONG_C, alpha=0.12, label='Alpha region')

# high water mark line
ax1.plot(x_trades, trades['roll_max'].values, color=GOLD, lw=0.8, ls=':', alpha=0.6, label='High-water mark')

# annotation box
stats_text = (
    f"N = {n_trades:,} trades\n"
    f"WR = {wr:.1f}%\n"
    f"Avg = +{avg_bps:.1f} bps\n"
    f"Total = +{total_ret:.1f}%\n"
    f"Max DD = {max_dd:.1f}%"
)
ax1.text(0.02, 0.97, stats_text, transform=ax1.transAxes,
         fontsize=9, va='top', ha='left', color=TEXT,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#1e2338', edgecolor=GRID, alpha=0.9))

ax1.set_title(f'Tier 1 Sniper Equity Curve — v2_15min_3y  |  L > {THR_LONG} @ {SNIPER_HOUR}h  |  10 bps friction',
              fontsize=11, fontweight='bold', color=TEXT, pad=10)
ax1.set_ylabel('Equity (×1)', fontsize=9, color=NEUT)
ax1.legend(loc='lower right', fontsize=8)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.1f}x'))
ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_color(GRID); ax1.spines['left'].set_color(GRID)
ax1.grid(True, alpha=0.3)
plt.setp(ax1.get_xticklabels(), visible=False)

# ── bottom panel: drawdown ───────────────────────────────────────────────────
ax2.fill_between(x_trades, trades['dd'].values, 0, color=SHORT_C, alpha=0.45)
ax2.plot(x_trades, trades['dd'].values, color=SHORT_C, lw=1.0)
ax2.axhline(0, color=GRID, lw=0.8)
ax2.axhline(max_dd, color=GOLD, lw=0.8, ls='--', alpha=0.7)
ax2.text(x_trades[-1], max_dd + 0.3, f'Max DD {max_dd:.1f}%',
         color=GOLD, fontsize=8, ha='right', va='bottom')

ax2.set_ylabel('Drawdown %', fontsize=9, color=NEUT)
ax2.set_xlabel('Date', fontsize=9, color=NEUT)
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_color(GRID); ax2.spines['left'].set_color(GRID)
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
plt.setp(ax2.get_xticklabels(), rotation=30, ha='right', fontsize=8)

plt.tight_layout()

out_name = '11_tier1_equity_curve.png'
for d in [OUT_DIR, MEM_DIR]:
    fig.savefig(os.path.join(d, out_name), dpi=150, bbox_inches='tight', facecolor=BG)
plt.close(fig)

print(f"\nSaved: {OUT_DIR}/{out_name}")
print(f"\n{'='*55}")
print(f"  Tier 1 Sniper Summary")
print(f"  Signal  : L > {THR_LONG}  at hour {SNIPER_HOUR}")
print(f"  Trades  : {n_trades:,}")
print(f"  Win Rate: {wr:.1f}%")
print(f"  Avg bps : +{avg_bps:.2f} (net of {COST_BPS} bps)")
print(f"  Total   : +{total_ret:.1f}%")
print(f"  Max DD  : {max_dd:.1f}%")
print(f"{'='*55}")

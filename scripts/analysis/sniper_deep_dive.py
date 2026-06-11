"""
Deep-dive breakdown of Tier 1 sniper: L > 0.0829 @ 15h
Monthly, weekly, daily stats + trade distribution + streaks + risk metrics.
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
from scipy import stats
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

COST     = 10 / 10000
THR_LONG = 0.0829
SNIPER_H = 15
OOS_MONTHS = 3

BG, AX_BG, TEXT, GRID = '#0f1117', '#161b2e', '#dde1f0', '#252a45'
LONG_C, SHORT_C, GOLD, NEUT, PURPLE = '#00d4aa', '#ff6b6b', '#f0c040', '#7c83a3', '#b57bee'
plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': AX_BG,
    'axes.edgecolor': GRID, 'axes.labelcolor': TEXT,
    'xtick.color': TEXT, 'ytick.color': TEXT, 'text.color': TEXT,
    'grid.color': GRID, 'grid.linewidth': 0.5,
    'legend.facecolor': '#1e2338', 'legend.edgecolor': GRID,
})

# ── load ──────────────────────────────────────────────────────────────────────
print("Loading models...")
with open(META_PATH) as f: meta = json.load(f)
feature_cols = meta['features']
bst_long = xgb.Booster(); bst_long.load_model(LONG_PATH)

print("Streaming OOS data...")
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

# ── filter ────────────────────────────────────────────────────────────────────
mask   = (df['long_score'] > THR_LONG) & (df['hour'] == SNIPER_H)
trades = df[mask].copy().sort_values('dt').reset_index(drop=True)
trades['net']      = trades['ret'] - COST
trades['gross']    = trades['ret']
trades['net_bps']  = trades['net'] * 10000
trades['gross_bps']= trades['gross'] * 10000
trades['equity']   = (1 + trades['net']).cumprod()
trades['roll_max'] = trades['equity'].cummax()
trades['dd']       = (trades['equity'] - trades['roll_max']) / trades['roll_max'] * 100
trades['date']     = trades['dt'].dt.date
trades['week']     = trades['dt'].dt.to_period('W')
trades['month']    = trades['dt'].dt.to_period('M')
trades['win']      = (trades['net'] > 0).astype(int)

SEP = "=" * 68

# ── SECTION 1: top-level summary ──────────────────────────────────────────────
print(f"\n{SEP}")
print(f"  TIER 1 SNIPER DEEP DIVE  |  L > {THR_LONG} @ {SNIPER_H}h  |  10 bps friction")
print(SEP)

total_ret   = (trades['equity'].iloc[-1] - 1) * 100
max_dd      = trades['dd'].min()
wr_raw      = (trades['gross'] > 0).mean() * 100
wr_net      = (trades['net'] > 0).mean() * 100
avg_g       = trades['gross_bps'].mean()
avg_n       = trades['net_bps'].mean()
std_n       = trades['net_bps'].std()
best_trade  = trades['net_bps'].max()
worst_trade = trades['net_bps'].min()
n_trades    = len(trades)
n_days      = trades['date'].nunique()
tpd         = n_trades / n_days   # trades per day

# Sharpe (annualised, assuming ~252 trading days, ~8 trades/day → ~2016 periods/yr)
periods_per_year = n_trades / (n_days / 252)
sharpe = (trades['net'].mean() / trades['net'].std()) * np.sqrt(periods_per_year)

# Sortino
downside = trades['net'][trades['net'] < 0].std()
sortino  = (trades['net'].mean() / downside) * np.sqrt(periods_per_year) if downside > 0 else np.inf

# Calmar
calmar = (total_ret / abs(max_dd)) if max_dd != 0 else np.inf

print(f"\n  Total return      : +{total_ret:.1f}%")
print(f"  Max drawdown      :  {max_dd:.1f}%")
print(f"  Calmar ratio      :  {calmar:.2f}x  (return / max DD)")
print(f"  Sharpe ratio      :  {sharpe:.2f}  (annualised)")
print(f"  Sortino ratio     :  {sortino:.2f}  (annualised)")
print(f"  Total trades      :  {n_trades:,}  over {n_days} trading days")
print(f"  Avg trades / day  :  {tpd:.1f}")
print(f"  Raw WR            :  {wr_raw:.1f}%")
print(f"  Net WR (10 bps)   :  {wr_net:.1f}%")
print(f"  Avg gross / trade :  +{avg_g:.2f} bps")
print(f"  Avg net  / trade  :  +{avg_n:.2f} bps")
print(f"  Std (net)         :   {std_n:.2f} bps")
print(f"  Best trade        :  +{best_trade:.2f} bps")
print(f"  Worst trade       :   {worst_trade:.2f} bps")

# ── SECTION 2: monthly breakdown ─────────────────────────────────────────────
print(f"\n{SEP}")
print(f"  MONTHLY BREAKDOWN")
print(SEP)
print(f"  {'Month':<10} {'Trades':>7} {'Raw WR':>8} {'Net WR':>8} {'Avg bps':>9} {'Month Ret%':>11}")
print(f"  {'-'*54}")

monthly = []
for m, g in trades.groupby('month'):
    mret = (1 + g['net']).prod() - 1
    monthly.append({
        'month': str(m),
        'n': len(g),
        'raw_wr': (g['gross'] > 0).mean() * 100,
        'net_wr': (g['net'] > 0).mean() * 100,
        'avg_bps': g['net_bps'].mean(),
        'month_ret': mret * 100,
    })
    print(f"  {str(m):<10} {len(g):>7,} {(g['gross']>0).mean()*100:>7.1f}% "
          f"{(g['net']>0).mean()*100:>7.1f}% {g['net_bps'].mean():>+8.2f}  {mret*100:>+9.1f}%")

# ── SECTION 3: weekly breakdown ───────────────────────────────────────────────
print(f"\n{SEP}")
print(f"  WEEKLY BREAKDOWN")
print(SEP)
print(f"  {'Week':<14} {'Trades':>7} {'Net WR':>8} {'Avg bps':>9} {'Week Ret%':>10} {'Equity':>8}")

equity_start = 1.0
for w, g in trades.groupby('week'):
    wret = (1 + g['net']).prod() - 1
    equity_end = equity_start * (1 + wret)
    arrow = "+" if wret >= 0 else "-"
    print(f"  {str(w):<14} {len(g):>7,} {(g['net']>0).mean()*100:>7.1f}% "
          f"{g['net_bps'].mean():>+8.2f}  {wret*100:>+8.1f}%  {equity_end:.3f}x")
    equity_start = equity_end

# ── SECTION 4: daily stats ────────────────────────────────────────────────────
daily = trades.groupby('date').apply(lambda g: pd.Series({
    'n': len(g),
    'net_ret': (1 + g['net']).prod() - 1,
    'wr': (g['net'] > 0).mean(),
})).reset_index()
daily['net_ret_pct'] = daily['net_ret'] * 100

print(f"\n{SEP}")
print(f"  DAILY STATS")
print(SEP)
print(f"  Trading days         : {len(daily)}")
print(f"  Profitable days      : {(daily['net_ret'] > 0).sum()}  ({(daily['net_ret'] > 0).mean()*100:.1f}%)")
print(f"  Loss days            : {(daily['net_ret'] < 0).sum()}  ({(daily['net_ret'] < 0).mean()*100:.1f}%)")
print(f"  Avg day return       : {daily['net_ret_pct'].mean():+.3f}%")
print(f"  Best day             : {daily['net_ret_pct'].max():+.3f}%")
print(f"  Worst day            : {daily['net_ret_pct'].min():+.3f}%")
print(f"  Avg trades/day       : {daily['n'].mean():.1f}  (min={daily['n'].min()}, max={daily['n'].max()})")

# ── SECTION 5: win/loss streaks ───────────────────────────────────────────────
print(f"\n{SEP}")
print(f"  WIN / LOSS STREAKS")
print(SEP)

max_win_streak = max_loss_streak = cur_win = cur_loss = 0
for w in trades['win']:
    if w:
        cur_win += 1; cur_loss = 0
        max_win_streak = max(max_win_streak, cur_win)
    else:
        cur_loss += 1; cur_win = 0
        max_loss_streak = max(max_loss_streak, cur_loss)

print(f"  Max consecutive wins   : {max_win_streak}")
print(f"  Max consecutive losses : {max_loss_streak}")

# ── SECTION 6: return distribution ───────────────────────────────────────────
print(f"\n{SEP}")
print(f"  RETURN DISTRIBUTION (net bps)")
print(SEP)
pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
vals = np.percentile(trades['net_bps'], pcts)
for p, v in zip(pcts, vals):
    print(f"  p{p:<3} : {v:+.2f} bps")
skew = stats.skew(trades['net_bps'])
kurt = stats.kurtosis(trades['net_bps'])
print(f"\n  Skewness : {skew:.3f}  {'(right-skewed, fat right tail)' if skew > 0.3 else '(left-skewed)' if skew < -0.3 else '(near-symmetric)'}")
print(f"  Kurtosis : {kurt:.3f}  {'(fat-tailed)' if kurt > 1 else '(thin-tailed)'}")

# ── PLOT ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12), facecolor=BG)
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

ax_eq  = fig.add_subplot(gs[0, :2])   # equity curve (wide)
ax_dd  = fig.add_subplot(gs[1, :2])   # drawdown
ax_dist= fig.add_subplot(gs[0, 2])    # return distribution
ax_mon = fig.add_subplot(gs[1, 2])    # monthly returns bar
ax_day = fig.add_subplot(gs[2, :2])   # daily P&L bars
ax_wk  = fig.add_subplot(gs[2, 2])    # trades per day histogram

def style(ax, title='', xl='', yl=''):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(GRID); ax.spines['left'].set_color(GRID)
    if title: ax.set_title(title, fontsize=9, fontweight='bold', color=TEXT, pad=6)
    if xl: ax.set_xlabel(xl, fontsize=8, color=NEUT)
    if yl: ax.set_ylabel(yl, fontsize=8, color=NEUT)
    ax.grid(True, alpha=0.3)

# 1. equity
x = trades['dt'].values
ax_eq.plot(x, trades['equity'].values, color=LONG_C, lw=1.8)
ax_eq.plot(x, trades['roll_max'].values, color=GOLD, lw=0.8, ls=':', alpha=0.6)
ax_eq.fill_between(x, trades['equity'].values, 1, alpha=0.1, color=LONG_C)
ax_eq.text(0.02, 0.95, f"+{total_ret:.1f}% total\nWR {wr_net:.1f}% | {n_trades} trades",
           transform=ax_eq.transAxes, va='top', fontsize=9, color=TEXT,
           bbox=dict(boxstyle='round,pad=0.4', facecolor='#1e2338', edgecolor=GRID))
style(ax_eq, f'Equity Curve  —  L > {THR_LONG} @ {SNIPER_H}h  (10 bps friction)', yl='Equity (×1)')
ax_eq.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.1f}x'))
ax_eq.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.setp(ax_eq.get_xticklabels(), rotation=20, fontsize=7)

# 2. drawdown
ax_dd.fill_between(x, trades['dd'].values, 0, color=SHORT_C, alpha=0.45)
ax_dd.plot(x, trades['dd'].values, color=SHORT_C, lw=0.8)
ax_dd.axhline(0, color=GRID, lw=0.6)
ax_dd.axhline(max_dd, color=GOLD, lw=0.8, ls='--', alpha=0.7)
ax_dd.text(0.98, 0.05, f'Max DD {max_dd:.1f}%', transform=ax_dd.transAxes,
           ha='right', fontsize=8, color=GOLD)
style(ax_dd, 'Drawdown %', yl='DD %')
ax_dd.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.setp(ax_dd.get_xticklabels(), rotation=20, fontsize=7)

# 3. return distribution
n_bins = 40
ax_dist.hist(trades['net_bps'], bins=n_bins, color=LONG_C, alpha=0.7, edgecolor='none')
ax_dist.axvline(0, color=GRID, lw=1.0)
ax_dist.axvline(trades['net_bps'].mean(), color=GOLD, lw=1.2, ls='--',
                label=f'Mean {trades["net_bps"].mean():+.1f}bps')
ax_dist.axvline(-10, color=SHORT_C, lw=0.8, ls=':', alpha=0.7, label='-10 bps (cost)')
ax_dist.legend(fontsize=7)
style(ax_dist, 'Return Distribution', xl='Net bps', yl='Count')

# 4. monthly bar
months_labels = [m['month'] for m in monthly]
months_rets   = [m['month_ret'] for m in monthly]
colors_m      = [LONG_C if r >= 0 else SHORT_C for r in months_rets]
bars = ax_mon.bar(range(len(monthly)), months_rets, color=colors_m, alpha=0.8, width=0.6)
ax_mon.axhline(0, color=GRID, lw=0.8)
ax_mon.set_xticks(range(len(monthly)))
ax_mon.set_xticklabels(months_labels, fontsize=7, rotation=20)
for bar, val in zip(bars, months_rets):
    ax_mon.text(bar.get_x() + bar.get_width()/2, val + (1 if val >= 0 else -3),
                f'{val:+.0f}%', ha='center', fontsize=7, color=TEXT)
style(ax_mon, 'Monthly Returns', yl='Return %')

# 5. daily P&L bars
daily_sorted = daily.sort_values('date')
day_x = range(len(daily_sorted))
colors_d = [LONG_C if r >= 0 else SHORT_C for r in daily_sorted['net_ret_pct']]
ax_day.bar(day_x, daily_sorted['net_ret_pct'], color=colors_d, alpha=0.75, width=0.8)
ax_day.axhline(0, color=GRID, lw=0.8)
ax_day.set_xlabel('Trading Day', fontsize=8, color=NEUT)
# mark every 10th day with date label
ticks = list(range(0, len(daily_sorted), 10))
ax_day.set_xticks(ticks)
ax_day.set_xticklabels([str(daily_sorted['date'].iloc[i]) for i in ticks], fontsize=7, rotation=30)
style(ax_day, 'Daily P&L %', yl='Return %')

# 6. trades per day histogram
ax_wk.hist(daily['n'], bins=range(1, int(daily['n'].max()) + 2), color=PURPLE, alpha=0.75, edgecolor='none', align='left')
ax_wk.axvline(daily['n'].mean(), color=GOLD, lw=1.2, ls='--', label=f'Mean {daily["n"].mean():.1f}')
ax_wk.legend(fontsize=7)
style(ax_wk, 'Trades Per Day Dist.', xl='# Trades', yl='# Days')

fig.suptitle(f'Tier 1 Sniper Deep Dive  |  v2_15min_3y  |  OOS {oos_m[0]} to {oos_m[-1]}',
             fontsize=11, fontweight='bold', color=TEXT, y=0.99)

out_name = '12_sniper_deep_dive.png'
for d in [OUT_DIR, MEM_DIR]:
    fig.savefig(os.path.join(d, out_name), dpi=150, bbox_inches='tight', facecolor=BG)
plt.close(fig)

print(f"\n{SEP}")
print(f"  Saved: {OUT_DIR}/{out_name}")
print(SEP)

"""
Sniper trade discovery for v2_15min_3y.
Sweeps: Direct Long/Short, Inverted Long/Short, Dual-Lock Long/Short
Across score thresholds AND hours of day.
All results net of 10 bps friction. Min 20 trades required.
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr
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
MIN_TRADES = 20
OOS_MONTHS = 3

# ── palette ──────────────────────────────────────────────────────────────────
BG, AX_BG, TEXT, GRID = '#0f1117', '#161b2e', '#dde1f0', '#252a45'
LONG_C, SHORT_C, GOLD, NEUT, PURPLE = '#00d4aa', '#ff6b6b', '#f0c040', '#7c83a3', '#b57bee'
plt.rcParams.update({'figure.facecolor': BG, 'axes.facecolor': AX_BG,
                     'axes.edgecolor': GRID, 'axes.labelcolor': TEXT,
                     'xtick.color': TEXT, 'ytick.color': TEXT, 'text.color': TEXT,
                     'grid.color': GRID, 'grid.linewidth': 0.5,
                     'legend.facecolor': '#1e2338', 'legend.edgecolor': GRID})

SEP = "=" * 72

def hdr(t): print(f"\n{SEP}\n  {t}\n{SEP}")

def ax_style(ax, title='', xl='', yl=''):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(GRID); ax.spines['left'].set_color(GRID)
    if title: ax.set_title(title, fontsize=10, fontweight='bold', color=TEXT, pad=7)
    if xl: ax.set_xlabel(xl, fontsize=9, color=NEUT)
    if yl: ax.set_ylabel(yl, fontsize=9, color=NEUT)
    ax.grid(True, alpha=0.35)

def save(fig, name):
    for d in [OUT_DIR, MEM_DIR]:
        p = os.path.join(d, name)
        fig.savefig(p, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f"  Saved → {os.path.join(OUT_DIR, name)}")

# ── load ──────────────────────────────────────────────────────────────────────
hdr("Loading models & OOS data")
with open(META_PATH) as f: meta = json.load(f)
feature_cols = meta['features']
bst_long  = xgb.Booster(); bst_long.load_model(LONG_PATH)
bst_short = xgb.Booster(); bst_short.load_model(SHORT_PATH)

all_months = set()
for ch in pd.read_csv(DATA_FILE, usecols=['DateTime'], chunksize=500_000):
    all_months.update(ch['DateTime'].str[:7].unique())
oos_m = sorted(all_months)[-OOS_MONTHS:]
print(f"  OOS: {oos_m}")

chunks = []
for ch in pd.read_csv(DATA_FILE, chunksize=200_000):
    sub = ch[ch['DateTime'].str[:7].isin(oos_m)]
    if len(sub): chunks.append(sub)
df = pd.concat(chunks, ignore_index=True)
df['YearMonth'] = df['DateTime'].str[:7]

X = df[feature_cols].values.astype(float)
for ci in range(X.shape[1]):
    col = X[:, ci]; bad = np.isnan(col) | np.isinf(col)
    if bad.any():
        X[bad, ci] = float(np.nanmean(col[~bad])) if (~bad).any() else 0.0

df['long_score']  = bst_long.predict(xgb.DMatrix(X))
df['short_score'] = bst_short.predict(xgb.DMatrix(X))
df['dt']          = pd.to_datetime(df['DateTime'])
df['hour']        = df['dt'].dt.hour
df['ret']         = df['Next_15Min_Return']
print(f"  Rows: {len(df):,}  |  Queries: {df['Query_ID'].nunique():,}")

# ── score distribution ────────────────────────────────────────────────────────
hdr("Score Distribution Profile")
for sc, label in [('long_score', 'LONG'), ('short_score', 'SHORT')]:
    s = df[sc]
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"\n  {label} score  min={s.min():.4f}  max={s.max():.4f}  "
          f"mean={s.mean():.4f}  std={s.std():.4f}")
    print(f"  Percentiles: " + "  ".join(f"p{p}={np.percentile(s,p):.4f}" for p in pcts))

# ── threshold grid (absolute + percentile-anchored) ───────────────────────────
def auto_thresholds(scores, n=14):
    pcts = np.linspace(70, 99, n)
    return sorted(set(round(float(np.percentile(scores, p)), 4) for p in pcts))

L_POS = auto_thresholds(df['long_score'])
S_POS = auto_thresholds(df['short_score'])
L_NEG = sorted(set(round(-t, 4) for t in auto_thresholds(-df['long_score'])))
S_NEG = sorted(set(round(-t, 4) for t in auto_thresholds(-df['short_score'])))

HOURS    = sorted(df['hour'].unique())
ALL_DAYS = -1   # sentinel for "all hours"

# ── core evaluator ────────────────────────────────────────────────────────────
def evaluate(mask, direction):
    """direction: 'long' → net = ret-cost | 'short' → net = -ret-cost"""
    sub = df[mask]
    if len(sub) < MIN_TRADES:
        return None
    if direction == 'long':
        net = sub['ret'] - COST
    else:
        net = -sub['ret'] - COST
    n       = len(net)
    wr      = (net > 0).mean()
    avg_bps = net.mean() * 10000
    tot_ret = (1 + net).prod() - 1
    # max drawdown on cumulative
    cum = (1 + net).cumprod()
    roll_max = cum.cummax()
    dd = ((cum - roll_max) / roll_max).min()
    return dict(n=n, wr=wr, avg_bps=avg_bps, tot_ret_pct=tot_ret*100,
                max_dd_pct=dd*100, ev=wr*avg_bps)

# ── SWEEP ────────────────────────────────────────────────────────────────────
hdr("Running signal sweeps (6 types × thresholds × hours)")
records = []

for hour_filter in [ALL_DAYS] + HOURS:
    h_mask = (df['hour'] == hour_filter) if hour_filter != ALL_DAYS else pd.Series([True]*len(df))

    # 1. Direct Long
    for thr in L_POS:
        mask = h_mask & (df['long_score'] > thr)
        r = evaluate(mask, 'long')
        if r:
            records.append({**r, 'signal':'Direct Long',
                            'threshold':f'L>{thr:.4f}', 'hour':hour_filter, 'thr_val':thr})

    # 2. Direct Short
    for thr in S_POS:
        mask = h_mask & (df['short_score'] > thr)
        r = evaluate(mask, 'short')
        if r:
            records.append({**r, 'signal':'Direct Short',
                            'threshold':f'S>{thr:.4f}', 'hour':hour_filter, 'thr_val':thr})

    # 3. Inverted Long → Short  (like 1-hr model sniper)
    for thr in L_NEG:
        mask = h_mask & (df['long_score'] < thr)
        r = evaluate(mask, 'short')
        if r:
            records.append({**r, 'signal':'Inverted Long→Short',
                            'threshold':f'L<{thr:.4f}', 'hour':hour_filter, 'thr_val':abs(thr)})

    # 4. Inverted Short → Long
    for thr in S_NEG:
        mask = h_mask & (df['short_score'] < thr)
        r = evaluate(mask, 'long')
        if r:
            records.append({**r, 'signal':'Inverted Short→Long',
                            'threshold':f'S<{thr:.4f}', 'hour':hour_filter, 'thr_val':abs(thr)})

    # 5. Dual-Lock Long  (high long_score + low short_score)
    for lt in L_POS[4:]:   # top half of thresholds only
        for st in S_NEG[4:]:
            mask = h_mask & (df['long_score'] > lt) & (df['short_score'] < st)
            r = evaluate(mask, 'long')
            if r:
                records.append({**r, 'signal':'Dual-Lock Long',
                                'threshold':f'L>{lt:.4f} S<{st:.4f}',
                                'hour':hour_filter, 'thr_val':(lt+abs(st))/2})

    # 6. Dual-Lock Short  (high short_score + low long_score)
    for st in S_POS[4:]:
        for lt in L_NEG[4:]:
            mask = h_mask & (df['short_score'] > st) & (df['long_score'] < lt)
            r = evaluate(mask, 'short')
            if r:
                records.append({**r, 'signal':'Dual-Lock Short',
                                'threshold':f'S>{st:.4f} L<{lt:.4f}',
                                'hour':hour_filter, 'thr_val':(st+abs(lt))/2})

df_r = pd.DataFrame(records)
print(f"  Total valid configurations: {len(df_r):,}")

# ── RESULTS ───────────────────────────────────────────────────────────────────
hdr("TOP 30 SNIPER CONFIGURATIONS  (sorted by WinRate × AvgBps)")
df_r['ev'] = df_r['wr'] * df_r['avg_bps']
top = df_r.sort_values('ev', ascending=False).head(30).reset_index(drop=True)

print(f"\n  {'#':>3}  {'Signal':<22}  {'Threshold':<26}  {'Hr':>4}  "
      f"{'N':>5}  {'WR%':>6}  {'Avg bps':>8}  {'Tot%':>7}  {'MaxDD%':>8}  {'EV':>7}")
print(f"  {'─'*3}  {'─'*22}  {'─'*26}  {'─'*4}  "
      f"{'─'*5}  {'─'*6}  {'─'*8}  {'─'*7}  {'─'*8}  {'─'*7}")

for i, row in top.iterrows():
    hr = "ALL" if row['hour'] == ALL_DAYS else f"{int(row['hour']):02d}h"
    print(f"  {i+1:>3}  {row['signal']:<22}  {row['threshold']:<26}  {hr:>4}  "
          f"{int(row['n']):>5}  {row['wr']*100:>5.1f}%  {row['avg_bps']:>+8.2f}  "
          f"{row['tot_ret_pct']:>+7.2f}%  {row['max_dd_pct']:>+8.2f}%  {row['ev']:>+7.3f}")

# ── TOP by signal type ─────────────────────────────────────────────────────────
hdr("BEST CONFIGURATION PER SIGNAL TYPE")
for sig in df_r['signal'].unique():
    sub = df_r[df_r['signal'] == sig].sort_values('ev', ascending=False).head(1)
    if sub.empty: continue
    row = sub.iloc[0]
    hr = "ALL" if row['hour'] == ALL_DAYS else f"{int(row['hour']):02d}h"
    print(f"\n  {row['signal']}")
    print(f"    Threshold : {row['threshold']}")
    print(f"    Hour      : {hr}")
    print(f"    N trades  : {int(row['n'])}")
    print(f"    Win Rate  : {row['wr']*100:.1f}%")
    print(f"    Avg bps   : {row['avg_bps']:+.2f}")
    print(f"    Total ret : {row['tot_ret_pct']:+.2f}%")
    print(f"    Max DD    : {row['max_dd_pct']:+.2f}%")
    print(f"    EV        : {row['ev']:+.3f}")

# ── SNIPER TIERS  (WR > 60%, min 20 trades) ──────────────────────────────────
hdr("SNIPER TIERS  (WR >= 60%, all-hours and time-gated)")
sniper = df_r[df_r['wr'] >= 0.60].sort_values('ev', ascending=False)
if len(sniper) == 0:
    print("  No configuration reached 60% WR. Showing top at 55%+:")
    sniper = df_r[df_r['wr'] >= 0.55].sort_values('ev', ascending=False)

print(f"\n  {'#':>3}  {'Signal':<22}  {'Threshold':<26}  {'Hr':>4}  "
      f"{'N':>5}  {'WR%':>6}  {'Avg bps':>8}  {'Tot%':>7}  {'MaxDD%':>8}")
print(f"  {'─'*3}  {'─'*22}  {'─'*26}  {'─'*4}  "
      f"{'─'*5}  {'─'*6}  {'─'*8}  {'─'*7}  {'─'*8}")

for i, (_, row) in enumerate(sniper.head(20).iterrows()):
    hr = "ALL" if row['hour'] == ALL_DAYS else f"{int(row['hour']):02d}h"
    print(f"  {i+1:>3}  {row['signal']:<22}  {row['threshold']:<26}  {hr:>4}  "
          f"{int(row['n']):>5}  {row['wr']*100:>5.1f}%  {row['avg_bps']:>+8.2f}  "
          f"{row['tot_ret_pct']:>+7.2f}%  {row['max_dd_pct']:>+8.2f}%")

# ── TIME-OF-DAY HEATMAP ───────────────────────────────────────────────────────
hdr("Building time-of-day heatmaps")

def hour_sweep(score_col, direction, thr_list, thr_type='pos'):
    """Return DataFrame: rows=hours, cols=thresholds, values=WR"""
    rows = {}
    for h in HOURS:
        rows[h] = {}
        for thr in thr_list:
            if thr_type == 'pos':
                mask = (df['hour'] == h) & (df[score_col] > thr)
            else:
                mask = (df['hour'] == h) & (df[score_col] < thr)
            r = evaluate(mask, direction)
            rows[h][thr] = r['wr'] if r else np.nan
    return pd.DataFrame(rows).T  # hours as rows, thresholds as cols

# use subset of thresholds for readability
L_heat = L_POS[::2][:8]
S_heat = S_POS[::2][:8]
L_inv  = L_NEG[::2][:8]

fig = plt.figure(figsize=(18, 14), facecolor=BG)
fig.suptitle('Sniper Heatmaps  |  v2_15min_3y  |  Win Rate by Hour × Threshold  (OOS Apr-Jun 2026)',
             fontsize=13, fontweight='bold', color=TEXT)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

def plot_heatmap(ax, data_df, title, cmap='RdYlGn', vmin=0.40, vmax=0.70):
    clean = data_df.copy()
    im = ax.imshow(clean.values, aspect='auto', cmap=cmap,
                   vmin=vmin, vmax=vmax, origin='upper')
    ax.set_yticks(range(len(clean.index)))
    ax.set_yticklabels([f'{int(h):02d}:00' for h in clean.index], fontsize=8, color=TEXT)
    ax.set_xticks(range(len(clean.columns)))
    ax.set_xticklabels([f'{v:.4f}' for v in clean.columns],
                       rotation=45, ha='right', fontsize=7, color=TEXT)
    for i in range(len(clean.index)):
        for j in range(len(clean.columns)):
            v = clean.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f'{v*100:.0f}%', ha='center', va='center',
                        fontsize=7.5, color='black' if 0.45 < v < 0.65 else TEXT,
                        fontweight='bold')
    cb = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.8)
    cb.set_label('Win Rate', color=TEXT, fontsize=8)
    cb.ax.yaxis.set_tick_params(color=TEXT, labelsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT)
    ax.set_title(title, fontsize=10, fontweight='bold', color=TEXT, pad=8)
    ax.set_xlabel('Score Threshold', fontsize=8, color=NEUT)
    ax.set_ylabel('Hour (IST)', fontsize=8, color=NEUT)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(GRID); ax.spines['left'].set_color(GRID)

print("  Computing hour × threshold WR matrices...")
hm_dl = hour_sweep('long_score', 'long',  L_heat, 'pos')
hm_ds = hour_sweep('short_score','short', S_heat, 'pos')
hm_il = hour_sweep('long_score', 'short', L_inv,  'neg')

plot_heatmap(fig.add_subplot(gs[0,0]), hm_dl, 'Direct Long  (L > thr)')
plot_heatmap(fig.add_subplot(gs[0,1]), hm_ds, 'Direct Short (S > thr)')
plot_heatmap(fig.add_subplot(gs[0,2]), hm_il, 'Inverted Long→Short (L < thr)')

# Bottom row: avg bps versions
def hour_sweep_bps(score_col, direction, thr_list, thr_type='pos'):
    rows = {}
    for h in HOURS:
        rows[h] = {}
        for thr in thr_list:
            if thr_type == 'pos':
                mask = (df['hour'] == h) & (df[score_col] > thr)
            else:
                mask = (df['hour'] == h) & (df[score_col] < thr)
            r = evaluate(mask, direction)
            rows[h][thr] = r['avg_bps'] if r else np.nan
    return pd.DataFrame(rows).T

hm_dl_bps = hour_sweep_bps('long_score', 'long',  L_heat, 'pos')
hm_ds_bps = hour_sweep_bps('short_score','short', S_heat, 'pos')
hm_il_bps = hour_sweep_bps('long_score', 'short', L_inv,  'neg')

plot_heatmap(fig.add_subplot(gs[1,0]), hm_dl_bps,
             'Direct Long  — Avg Bps', cmap='RdYlGn', vmin=-10, vmax=30)
plot_heatmap(fig.add_subplot(gs[1,1]), hm_ds_bps,
             'Direct Short — Avg Bps', cmap='RdYlGn', vmin=-10, vmax=30)
plot_heatmap(fig.add_subplot(gs[1,2]), hm_il_bps,
             'Inverted Long→Short — Avg Bps', cmap='RdYlGn', vmin=-10, vmax=30)

save(fig, '09_sniper_heatmaps.png')

# ── SNIPER cumulative P&L chart ───────────────────────────────────────────────
hdr("Building sniper cumulative P&L curves")

# Pick top 4 sniper configs (highest EV with sufficient trades)
top4 = df_r.sort_values('ev', ascending=False).head(8)
top4 = top4[top4['n'] >= 30].head(4).reset_index(drop=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=BG)
fig.suptitle('Top Sniper Configurations — Cumulative P&L  |  v2_15min_3y  |  OOS Apr-Jun 2026',
             fontsize=13, fontweight='bold', color=TEXT)
axes = axes.flatten()
colors = [LONG_C, SHORT_C, GOLD, PURPLE]

mkt_cum = (1 + (df.groupby('Query_ID')['ret'].mean().reset_index(drop=True))).cumprod() - 1

for idx, (_, row) in enumerate(top4.iterrows()):
    ax = axes[idx]
    signal = row['signal']
    thr_str = row['threshold']
    hour_filter = row['hour']

    # rebuild mask
    parts = thr_str.split()
    if len(parts) == 1:
        op  = thr_str[1]
        sc  = 'long_score' if thr_str[0] == 'L' else 'short_score'
        thr = float(thr_str[2:])
        mask = (df[sc] > thr) if op == '>' else (df[sc] < thr)
    else:
        p1, p2 = parts[0], parts[1]
        sc1 = 'long_score' if p1[0]=='L' else 'short_score'
        op1, v1 = p1[1], float(p1[2:])
        sc2 = 'long_score' if p2[0]=='L' else 'short_score'
        op2, v2 = p2[1], float(p2[2:])
        m1 = (df[sc1] > v1) if op1=='>' else (df[sc1] < v1)
        m2 = (df[sc2] > v2) if op2=='>' else (df[sc2] < v2)
        mask = m1 & m2

    if hour_filter != ALL_DAYS:
        mask = mask & (df['hour'] == hour_filter)

    direction = 'long' if 'Long' in signal else 'short'
    sub = df[mask].copy()
    if len(sub) < 5:
        ax.set_visible(False); continue

    net = (sub['ret'] - COST) if direction == 'long' else (-sub['ret'] - COST)
    sub = sub.copy(); sub['net'] = net.values
    sub_sorted = sub.sort_values('DateTime')
    cum = (1 + sub_sorted['net']).cumprod() - 1
    xs = range(len(cum))

    hr_label = "All Hours" if hour_filter == ALL_DAYS else f"{int(hour_filter):02d}:xx"
    ax.plot(xs, cum.values * 100, color=colors[idx], lw=2.0, alpha=0.95, label='Strategy')
    ax.fill_between(xs, cum.values * 100, 0,
                    where=(cum.values >= 0), alpha=0.12, color=colors[idx], interpolate=True)
    ax.fill_between(xs, cum.values * 100, 0,
                    where=(cum.values < 0),  alpha=0.12, color=SHORT_C, interpolate=True)
    ax.axhline(0, color=NEUT, lw=0.7, alpha=0.4)

    final = cum.values[-1] * 100
    n_t   = len(sub_sorted)
    wr    = (sub_sorted['net'] > 0).mean()
    bps   = sub_sorted['net'].mean() * 10000

    ax.text(0.98, 0.04,
            f"N={n_t}  WR={wr*100:.1f}%  Avg={bps:+.1f}bps\nReturn={final:+.2f}%",
            transform=ax.transAxes, fontsize=8.5, color=TEXT, ha='right', va='bottom',
            bbox=dict(facecolor='#1e2338', edgecolor=GRID, boxstyle='round,pad=0.3'))

    title = f"{signal}\n{thr_str}  @{hr_label}"
    ax_style(ax, title=title, xl='Trade #', yl='Cumulative Return (%)')

save(fig, '10_sniper_pnl.png')

# ── FINAL SUMMARY TABLE ───────────────────────────────────────────────────────
hdr("FINAL SNIPER SUMMARY — TIERED EXECUTION SCHEDULE")

# Best per signal type per hour (all + specific hours)
best_overall = df_r.sort_values('ev', ascending=False).head(1).iloc[0]
print(f"\n  Overall best single config:")
print(f"    {best_overall['signal']}  |  {best_overall['threshold']}  |  "
      f"Hr={'ALL' if best_overall['hour']==ALL_DAYS else int(best_overall['hour'])}  |  "
      f"N={int(best_overall['n'])}  WR={best_overall['wr']*100:.1f}%  "
      f"Avg={best_overall['avg_bps']:+.2f}bps  EV={best_overall['ev']:+.3f}")

print(f"\n  Per-signal type champions (highest EV, ≥ 20 trades):")
for sig in ['Direct Long','Direct Short','Inverted Long→Short',
            'Inverted Short→Long','Dual-Lock Long','Dual-Lock Short']:
    sub = df_r[df_r['signal']==sig].sort_values('ev', ascending=False)
    if sub.empty: continue
    row = sub.iloc[0]
    hr = "ALL" if row['hour']==ALL_DAYS else f"{int(row['hour']):02d}h"
    print(f"  {sig:<24}  {row['threshold']:<26}  {hr:>4}  "
          f"N={int(row['n']):<5}  WR={row['wr']*100:.1f}%  "
          f"Avg={row['avg_bps']:+.2f}bps  EV={row['ev']:+.3f}")

# save results JSON
results = {
    'period': f"{oos_m[0]} to {oos_m[-1]}",
    'cost_bps': COST_BPS,
    'top_configs': top.to_dict(orient='records'),
    'sniper_tiers': sniper.head(20).to_dict(orient='records'),
}
json_path = 'data/sniper_15min_results.json'
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n  Results saved → {json_path}")
print(f"\n{SEP}")

"""
SHORT vs LONG model audit — why does 15m confirmation help long but not short?

Tests 4 hypotheses:
  H1  Market regime headwind   — did the universe drift up (shorts fight the tide)?
  H2  Cross-sectional IC       — does each model rank future winners/losers? (Spearman)
  H3  Bucket monotonicity      — does score decile align with realized return?
  H4  Confirmation alpha       — does the 15m rank concentrate forward return at entry?

Runs for both 1h (v8_upstox_3y) and 15m (v2_15min_3y), long & short.
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
SEP = "=" * 78

def load_score(model_dir, data_file, ret_col):
    with open(f'{model_dir}/metadata.json') as f:
        meta = json.load(f)
    feats = meta['features']
    bl = xgb.Booster(); bl.load_model(f'{model_dir}/xgb_long_model.json')
    bs = xgb.Booster(); bs.load_model(f'{model_dir}/xgb_short_model.json')

    all_m = set()
    for ch in pd.read_csv(data_file, usecols=['DateTime'], chunksize=500_000):
        all_m.update(ch['DateTime'].str[:7].unique())
    oos = sorted(all_m)[-OOS_MONTHS:]

    chunks = []
    for ch in pd.read_csv(data_file, chunksize=200_000):
        sub = ch[ch['DateTime'].str[:7].isin(oos)]
        if len(sub): chunks.append(sub)
    df = pd.concat(chunks, ignore_index=True)
    df['dt'] = pd.to_datetime(df['DateTime']).dt.tz_localize(None)
    df['ym'] = df['DateTime'].str[:7]
    df = df.dropna(subset=[ret_col])

    X = df[feats].values.astype(float)
    for ci in range(X.shape[1]):
        col = X[:, ci]; bad = np.isnan(col) | np.isinf(col)
        if bad.any(): X[bad, ci] = float(np.nanmean(col[~bad])) if (~bad).any() else 0.0
    dm = xgb.DMatrix(X)
    df['s_long']  = bl.predict(dm)
    df['s_short'] = bs.predict(dm)
    df['ret']     = df[ret_col]
    return df, oos

# ────────────────────────────────────────────────────────────────────────────────
print("Loading + scoring 1h ...")
df1, oos = load_score(H1_MODEL_DIR, H1_DATA, 'Next_Hour_Return')
print("Loading + scoring 15m ...")
df15, _  = load_score(M15_MODEL_DIR, M15_DATA, 'Next_15Min_Return')
print(f"  1h rows={len(df1):,}  15m rows={len(df15):,}  OOS={oos}\n")

# ════════════════════════════════════════════════════════════════════════════════
# H1 — MARKET REGIME
# ════════════════════════════════════════════════════════════════════════════════
print(SEP); print("  H1 — MARKET REGIME (is the universe drifting up?)"); print(SEP)
print(f"  {'Frame':<6} {'Month':<9} {'MeanRet%':>9} {'%PosBars':>9} {'MedRet%':>9}")
print(f"  {'-'*48}")
for frame, d in [('1h', df1), ('15m', df15)]:
    for ym, g in d.groupby('ym'):
        mean_r = g['ret'].mean() * 100
        pos    = (g['ret'] > 0).mean() * 100
        med_r  = g['ret'].median() * 100
        print(f"  {frame:<6} {ym:<9} {mean_r:>+8.4f}  {pos:>7.1f}%  {med_r:>+8.4f}")
    ov_mean = d['ret'].mean() * 100
    ov_pos  = (d['ret'] > 0).mean() * 100
    print(f"  {frame:<6} {'OVERALL':<9} {ov_mean:>+8.4f}  {ov_pos:>7.1f}%")
    print(f"  {'-'*48}")

# ════════════════════════════════════════════════════════════════════════════════
# H2 — CROSS-SECTIONAL IC (Spearman per query, averaged)
# ════════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}"); print("  H2 — CROSS-SECTIONAL IC (mean per-query Spearman vs forward return)"); print(SEP)
print("  For LONG : corr(s_long, ret)   should be > 0")
print("  For SHORT: corr(s_short, -ret) should be > 0  (high short score -> falls)\n")
print(f"  {'Frame':<6} {'Side':<6} {'MeanIC':>9} {'StdIC':>8} {'%QueriesPos':>12} {'t-stat':>8}")
print(f"  {'-'*54}")

def query_ic(d, score, target_sign):
    ics = []
    for _, g in d.groupby('dt'):
        if len(g) < 5: continue
        tgt = g['ret'] * target_sign
        rho, _ = spearmanr(g[score], tgt)
        if not np.isnan(rho): ics.append(rho)
    ics = np.array(ics)
    t = ics.mean() / (ics.std() / np.sqrt(len(ics))) if ics.std() > 0 else 0
    return ics.mean(), ics.std(), (ics > 0).mean() * 100, t, len(ics)

for frame, d in [('1h', df1), ('15m', df15)]:
    for side, score, sgn in [('LONG', 's_long', +1), ('SHORT', 's_short', -1)]:
        m, s, pq, t, nq = query_ic(d, score, sgn)
        print(f"  {frame:<6} {side:<6} {m:>+8.4f}  {s:>7.4f}  {pq:>10.1f}%  {t:>7.2f}")

# ════════════════════════════════════════════════════════════════════════════════
# H3 — BUCKET MONOTONICITY (score decile -> realized return)
# ════════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}"); print("  H3 — BUCKET MONOTONICITY (decile of score -> mean forward return %)"); print(SEP)

def buckets(d, score, side):
    d = d.copy()
    d['dec'] = pd.qcut(d[score].rank(method='first'), 10, labels=False) + 1
    rows = []
    for dec, g in d.groupby('dec'):
        # for short, the 'edge' is -ret (profit when stock falls)
        edge = (-g['ret'] if side == 'SHORT' else g['ret']).mean() * 100
        rows.append((dec, g['ret'].mean()*100, edge))
    arr = pd.DataFrame(rows, columns=['dec', 'mean_ret', 'edge'])
    rho, _ = spearmanr(arr['dec'], arr['edge'])
    return arr, rho

bucket_store = {}
for frame, d in [('1h', df1), ('15m', df15)]:
    for side, score in [('LONG', 's_long'), ('SHORT', 's_short')]:
        arr, rho = buckets(d, score, side)
        bucket_store[(frame, side)] = (arr, rho)
        d10 = arr[arr['dec']==10]['edge'].values[0]
        d1  = arr[arr['dec']==1]['edge'].values[0]
        spread = d10 - d1
        verdict = "GOOD" if rho > 0.7 and spread > 0 else ("WEAK" if rho > 0.3 else "BROKEN")
        print(f"  {frame:<4} {side:<6}  BucketRho={rho:>+.4f}  D10edge={d10:>+.4f}%  "
              f"D1edge={d1:>+.4f}%  spread={spread:>+.4f}%  [{verdict}]")

# ════════════════════════════════════════════════════════════════════════════════
# H4 — CONFIRMATION ALPHA (does 15m rank at entry concentrate forward return?)
# ════════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}"); print("  H4 — 15m CONFIRMATION ALPHA on 1h trades"); print(SEP)
print("  When 1h picks top-3, bucket by 15m rank at T+45 -> mean 1h forward edge")

TD45 = pd.Timedelta('45min')
df15_idx = df15.set_index(['Ticker', 'dt'])

def get15(tk, ts, fld):
    try:
        v = df15_idx.loc[(tk, ts), fld]
        return float(v) if not isinstance(v, pd.Series) else float(v.iloc[0])
    except KeyError:
        return np.nan

# precompute 15m ranks
df15['rk_long']  = df15.groupby('dt')['s_long'].rank(pct=True)
df15['rk_short'] = df15.groupby('dt')['s_short'].rank(pct=True)
df15_idx = df15.set_index(['Ticker', 'dt'])

conf_store = {}
for side, score, rkfld, sgn in [('LONG', 's_long', 'rk_long', +1),
                                  ('SHORT', 's_short', 'rk_short', -1)]:
    recs = []
    for dt1, grp in df1.groupby('dt'):
        top = grp.nlargest(3, score)
        for _, row in top.iterrows():
            rk = get15(row['Ticker'], dt1 + TD45, rkfld)
            if np.isnan(rk): continue
            edge = row['ret'] * sgn   # forward 1h edge in the trade direction
            recs.append((rk, edge))
    r = pd.DataFrame(recs, columns=['rk15', 'edge'])
    conf_store[side] = r
    print(f"\n  {side}  (n with 15m match = {len(r):,})")
    print(f"    {'15m rank bin':<14} {'N':>6} {'MeanEdge%':>11} {'NetWR%':>8}")
    r['bin'] = pd.cut(r['rk15'], [0,.5,.7,.85,.95,1.0],
                      labels=['0-50','50-70','70-85','85-95','95-100'])
    for b, g in r.groupby('bin'):
        edge = g['edge'].mean()*100
        wr   = ((g['edge'] - COST) > 0).mean()*100
        print(f"    {str(b):<14} {len(g):>6,} {edge:>+10.4f}  {wr:>7.1f}%")
    # correlation between 15m rank and forward edge
    rho, _ = spearmanr(r['rk15'], r['edge'])
    print(f"    -> Spearman(15m rank, forward edge) = {rho:+.4f}")

# ════════════════════════════════════════════════════════════════════════════════
# PLOT
# ════════════════════════════════════════════════════════════════════════════════
print("\nGenerating audit plot...")
fig = plt.figure(figsize=(17, 11), facecolor=BG)
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.30)

def style(ax, title='', xl='', yl=''):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(GRID); ax.spines['left'].set_color(GRID)
    if title: ax.set_title(title, fontsize=9, fontweight='bold', color=TEXT, pad=6)
    if xl: ax.set_xlabel(xl, fontsize=8, color=NEUT)
    if yl: ax.set_ylabel(yl, fontsize=8, color=NEUT)
    ax.grid(True, alpha=0.3)

# panel 1: regime (monthly mean ret)
ax = fig.add_subplot(gs[0, 0])
for frame, d, col in [('1h', df1, NEUT), ('15m', df15, PURPLE)]:
    mm = d.groupby('ym')['ret'].mean()*100
    ax.plot(range(len(mm)), mm.values, marker='o', color=col, lw=1.5, label=frame)
    ax.set_xticks(range(len(mm))); ax.set_xticklabels(mm.index, fontsize=7, rotation=20)
ax.axhline(0, color=SHORT_C, lw=0.8, ls='--')
ax.legend(fontsize=8)
style(ax, 'H1: Universe Mean Return by Month', yl='Mean Ret %')

# panel 2+3: bucket monotonicity for 1h and 15m
for i, frame in enumerate(['1h', '15m']):
    ax = fig.add_subplot(gs[0, 1+i])
    arrL, rhoL = bucket_store[(frame, 'LONG')]
    arrS, rhoS = bucket_store[(frame, 'SHORT')]
    x = arrL['dec'].values
    ax.plot(x, arrL['edge'].values, marker='o', color=LONG_C, lw=1.5,
            label=f'LONG  rho={rhoL:+.2f}')
    ax.plot(x, arrS['edge'].values, marker='s', color=SHORT_C, lw=1.5,
            label=f'SHORT rho={rhoS:+.2f}')
    ax.axhline(0, color=GRID, lw=0.8)
    ax.legend(fontsize=8)
    style(ax, f'H3: {frame} Bucket Edge by Score Decile', xl='Score Decile (1-10)', yl='Edge %')

# panel 4: H4 confirmation alpha — long vs short
ax = fig.add_subplot(gs[1, :2])
width = 0.35
binlabels = ['0-50','50-70','70-85','85-95','95-100']
for j, (side, col) in enumerate([('LONG', LONG_C), ('SHORT', SHORT_C)]):
    r = conf_store[side]
    r['bin'] = pd.cut(r['rk15'], [0,.5,.7,.85,.95,1.0], labels=binlabels)
    means = [r[r['bin']==b]['edge'].mean()*100 for b in binlabels]
    xpos = np.arange(len(binlabels)) + (j-0.5)*width
    bars = ax.bar(xpos, means, width, color=col, alpha=0.8, label=side)
    for xp, mv in zip(xpos, means):
        ax.text(xp, mv + (0.002 if mv>=0 else -0.006), f'{mv:+.3f}', ha='center', fontsize=6, color=TEXT)
ax.axhline(0, color=GRID, lw=0.8)
ax.set_xticks(range(len(binlabels))); ax.set_xticklabels(binlabels, fontsize=8)
ax.legend(fontsize=9)
style(ax, 'H4: Forward Edge of 1h Trades by 15m Confirmation Rank (higher rank should -> higher edge)',
      xl='15m rank percentile bin at entry', yl='Mean forward edge %')

# panel 5: IC summary bars
ax = fig.add_subplot(gs[1, 2])
labels, vals, cols = [], [], []
for frame, d in [('1h', df1), ('15m', df15)]:
    for side, score, sgn, c in [('L', 's_long', +1, LONG_C), ('S', 's_short', -1, SHORT_C)]:
        m, *_ = query_ic(d, score, sgn)
        labels.append(f'{frame}\n{side}'); vals.append(m); cols.append(c)
bars = ax.bar(range(len(vals)), vals, color=cols, alpha=0.85)
ax.axhline(0, color=GRID, lw=0.8)
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8)
for i, v in enumerate(vals):
    ax.text(i, v + (0.001 if v>=0 else -0.003), f'{v:+.3f}', ha='center', fontsize=7, color=TEXT)
style(ax, 'H2: Mean Cross-Sectional IC', yl='Spearman IC')

fig.suptitle(f'SHORT vs LONG Model Audit  |  OOS {oos[0]} to {oos[-1]}',
             fontsize=12, fontweight='bold', color=TEXT, y=1.005)
plt.tight_layout()
out = 'short_model_audit.png'
for dd in [OUT_DIR, MEM_DIR]:
    fig.savefig(os.path.join(dd, out), dpi=150, bbox_inches='tight', facecolor=BG)
plt.close(fig)
print(f"  Saved: {OUT_DIR}/{out}")
print(f"\n{SEP}\n  Done.\n{SEP}")

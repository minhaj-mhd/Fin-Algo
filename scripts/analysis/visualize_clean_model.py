"""
8-plot diagnostic suite for the CLEAN/native ranking models, parametrized.
  python scripts/analysis/visualize_clean_model.py --model v10_native_1h
  python scripts/analysis/visualize_clean_model.py --model v3_15min_clean
Mirrors the original visualize_15min_v2.py suite exactly, on the rebuilt datasets.
"""
import os, sys, json, argparse, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mtick
from scipy.stats import spearmanr, rankdata, pearsonr, norm as sp_norm
warnings.filterwarnings('ignore')
sys.path.append(os.getcwd())
try:
    import shap; SHAP_OK = True
except ImportError:
    SHAP_OK = False; print("[WARN] shap not installed -> SHAP plots skipped")

CFG = {
    'v10_native_1h': dict(model_dir='models/v10_native_1h', data='data/ranking_data_upstox_1h_v3_3y.csv',
                          ret='Next_Hour_Return', out='data/model_analysis/v10_native_1h', label='v10_native_1h (1-hour)'),
    'v3_15min_clean': dict(model_dir='models/v3_15min_clean', data='data/ranking_data_upstox_15min_3y_clean.csv',
                           ret='Next_15Min_Return', out='data/model_analysis/v3_15min_clean', label='v3_15min_clean (15-min)'),
    'v9_clean_1h': dict(model_dir='models/v9_clean_1h', data='data/ranking_data_upstox_1h_3y_clean.csv',
                        ret='Next_Hour_Return', out='data/model_analysis/v9_clean_1h', label='v9_clean_1h (1-hour resampled)'),
}
ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True, choices=list(CFG))
A = ap.parse_args(); C = CFG[A.model]
MODEL_DIR, DATA_FILE, RET, OUT_DIR, LABEL = C['model_dir'], C['data'], C['ret'], C['out'], C['label']
os.makedirs(OUT_DIR, exist_ok=True)
OOS_MONTHS, LC_TRAIN_MO, SHAP_SAMPLE, DPI = 3, 6, 3000, 150

BG, AX_BG, TEXT, GRID = '#0f1117', '#161b2e', '#dde1f0', '#252a45'
LONG_C, SHORT_C, NEUT, GOLD, PURPLE = '#00d4aa', '#ff6b6b', '#7c83a3', '#f0c040', '#b57bee'
plt.rcParams.update({'figure.facecolor': BG, 'axes.facecolor': AX_BG, 'axes.edgecolor': GRID,
    'axes.labelcolor': TEXT, 'xtick.color': TEXT, 'ytick.color': TEXT, 'text.color': TEXT,
    'grid.color': GRID, 'grid.linewidth': 0.5, 'legend.facecolor': '#1e2338', 'legend.edgecolor': GRID,
    'font.family': 'DejaVu Sans'})

def ax_style(ax, title='', xl='', yl='', grid=True):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(GRID); ax.spines['left'].set_color(GRID)
    if title: ax.set_title(title, fontsize=11, fontweight='bold', color=TEXT, pad=8)
    if xl: ax.set_xlabel(xl, fontsize=9, color=NEUT)
    if yl: ax.set_ylabel(yl, fontsize=9, color=NEUT)
    if grid: ax.grid(True, alpha=0.4)

def save(fig, name, tight=True):
    p = os.path.join(OUT_DIR, name)
    if tight: fig.tight_layout()
    fig.savefig(p, dpi=DPI, bbox_inches='tight', facecolor=BG); plt.close(fig)
    print(f"  Saved -> {p}")

SEP = "-" * 60
print(f"\n{SEP}\n  Visualizing {LABEL}\n{SEP}")
with open(f'{MODEL_DIR}/metadata.json') as f: meta = json.load(f)
fc = meta['features']
bst_long = xgb.Booster(); bst_long.load_model(f'{MODEL_DIR}/xgb_long_model.json')
bst_short = xgb.Booster(); bst_short.load_model(f'{MODEL_DIR}/xgb_short_model.json')
print(f"  Features {len(fc)} | training rows {meta['total_rows']:,}")

all_months = set()
for ch in pd.read_csv(DATA_FILE, usecols=['DateTime'], chunksize=500_000):
    all_months.update(ch['DateTime'].str[:7].unique())
all_months = sorted(all_months)
oos_m = all_months[-OOS_MONTHS:]
lc_tr_m = all_months[-(OOS_MONTHS+LC_TRAIN_MO+1):-(OOS_MONTHS+1)]
lc_val_m = [all_months[-(OOS_MONTHS+1)]]
load_set = set(oos_m) | set(lc_tr_m) | set(lc_val_m)
print(f"  OOS {oos_m} | LC train {lc_tr_m[0]}->{lc_tr_m[-1]} val {lc_val_m[0]}")
chunks = []
for ch in pd.read_csv(DATA_FILE, chunksize=200_000):
    s = ch[ch['DateTime'].str[:7].isin(load_set)]
    if len(s): chunks.append(s)
df_all = pd.concat(chunks, ignore_index=True); df_all['YearMonth'] = df_all['DateTime'].str[:7]
df_oos = df_all[df_all['YearMonth'].isin(oos_m)].copy().reset_index(drop=True)
df_lc_tr = df_all[df_all['YearMonth'].isin(lc_tr_m)].copy().reset_index(drop=True)
df_lc_val = df_all[df_all['YearMonth'].isin(lc_val_m)].copy().reset_index(drop=True)
print(f"  OOS {len(df_oos):,} rows / {df_oos['Query_ID'].nunique()} queries")

def prep(d):
    X = d[fc].values.astype(float)
    for ci in range(X.shape[1]):
        c = X[:, ci]; b = np.isnan(c) | np.isinf(c)
        if b.any(): X[b, ci] = float(np.nanmean(c[~b])) if (~b).any() else 0.0
    return X
X_oos = prep(df_oos)
df_oos['long_score'] = bst_long.predict(xgb.DMatrix(X_oos))
df_oos['short_score'] = bst_short.predict(xgb.DMatrix(X_oos))
df_oos = df_oos.sort_values('DateTime').reset_index(drop=True)

def int_ranks(y, q, inv=False):
    out = np.zeros_like(y, dtype=int)
    for qid in np.unique(q):
        m = q == qid; v = -y[m] if inv else y[m]; out[m] = rankdata(v, method='ordinal')-1
    return out

# 1 FEATURE IMPORTANCE
print("  Plot 1/8 Feature Importance")
tl, ts = meta['top_features_long'], meta['top_features_short']
allf = sorted(set(tl)|set(ts), key=lambda x: -(tl.get(x,0)+ts.get(x,0)))[:20]; allf.reverse()
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 7), facecolor=BG)
fig.suptitle(f'Feature Importance | {LABEL} (Gain, Top 20)', fontsize=13, fontweight='bold', color=TEXT, y=1.01)
for ax, fd, col, t in [(a1, tl, LONG_C, 'LONG'), (a2, ts, SHORT_C, 'SHORT')]:
    vals = [fd.get(f, 0) for f in allf]
    ax.barh(allf, vals, color=col, alpha=0.85)
    for i, v in enumerate(vals):
        if v > 0: ax.text(v+max(vals)*0.01, i, f'{v:.1f}', va='center', fontsize=7.5, color=TEXT)
    ax_style(ax, title=t, xl='Gain', grid=False); ax.tick_params(labelsize=8.5)
save(fig, '01_feature_importance.png')

# 2 SHAP SUMMARY
if SHAP_OK:
    print("  Plot 2/8 SHAP Summary")
    rng = np.random.default_rng(42); idx = rng.choice(len(X_oos), min(SHAP_SAMPLE, len(X_oos)), replace=False)
    Xs = X_oos[idx]
    expl = shap.TreeExplainer(bst_long); sv = expl.shap_values(Xs)
    fig, ax = plt.subplots(figsize=(11, 8), facecolor=BG)
    fig.suptitle(f'SHAP Summary | Long | {LABEL}', fontsize=13, fontweight='bold', color=TEXT)
    plt.sca(ax)
    shap.summary_plot(sv, Xs, feature_names=fc, max_display=20, show=False, plot_type='dot')
    ax.set_facecolor(AX_BG); ax.tick_params(colors=TEXT, labelsize=9); ax.xaxis.label.set_color(TEXT)
    for sp in ax.spines.values(): sp.set_color(GRID)
    save(fig, '02_shap_summary.png', tight=False)

    print("  Plot 3/8 SHAP Dependence")
    deps = [f for f in ['IBS', 'Buy_Pressure'] if f in fc]
    inter = {'IBS': 'Buy_Pressure', 'Buy_Pressure': 'IBS'}
    if len(deps) == 2:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
        fig.suptitle(f'SHAP Dependence | Long | {LABEL}', fontsize=13, fontweight='bold', color=TEXT)
        for ax, feat in zip(axes, deps):
            fi = fc.index(feat); ii = fc.index(inter[feat])
            sc = ax.scatter(Xs[:, fi], sv[:, fi], c=Xs[:, ii], cmap='RdYlGn', alpha=0.4, s=8, edgecolors='none')
            cb = plt.colorbar(sc, ax=ax, pad=0.02); cb.set_label(inter[feat], color=TEXT, fontsize=8)
            plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT); cb.ax.yaxis.set_tick_params(color=TEXT, labelsize=7.5)
            ax.axhline(0, color=NEUT, lw=0.8, ls='--'); ax_style(ax, title=f'SHAP({feat})', xl=feat, yl='SHAP value')
        save(fig, '03_shap_dependence.png')

# 4 LEARNING CURVE
print("  Plot 4/8 Learning Curve")
Xt, qt = prep(df_lc_tr), df_lc_tr['Query_ID'].values
Xv, qv = prep(df_lc_val), df_lc_val['Query_ID'].values
yt, yv = df_lc_tr[RET].values, df_lc_val[RET].values
def dm(X, y, q, inv):
    d = xgb.DMatrix(X, label=int_ranks(y, q, inv)); d.set_group(pd.Series(q).groupby(q).size().values); return d
lcp = dict(meta['params'])
er_l, er_s = {}, {}
xgb.train(lcp, dm(Xt, yt, qt, False), 500, evals=[(dm(Xt, yt, qt, False), 'train'), (dm(Xv, yv, qv, False), 'val')],
          evals_result=er_l, early_stopping_rounds=50, verbose_eval=False)
xgb.train(lcp, dm(Xt, yt, qt, True), 500, evals=[(dm(Xt, yt, qt, True), 'train'), (dm(Xv, yv, qv, True), 'val')],
          evals_result=er_s, early_stopping_rounds=50, verbose_eval=False)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
fig.suptitle(f'Learning Curve | NDCG@3 | {LABEL} | Train {lc_tr_m[0]}->{lc_tr_m[-1]} Val {lc_val_m[0]}',
             fontsize=12, fontweight='bold', color=TEXT)
for ax, er, t in [(a1, er_l, 'Long'), (a2, er_s, 'Short')]:
    r = range(1, len(er['train']['ndcg@3'])+1)
    ax.plot(r, er['train']['ndcg@3'], color=LONG_C, lw=1.5, label='Train')
    ax.plot(r, er['val']['ndcg@3'], color=SHORT_C, lw=1.5, label='Validation')
    br = int(np.argmax(er['val']['ndcg@3']))+1; bv = max(er['val']['ndcg@3'])
    ax.axvline(br, color=GOLD, lw=1, ls='--', alpha=0.7); ax.scatter([br], [bv], color=GOLD, s=60, zorder=5)
    ax.annotate(f'Best r{br}\nNDCG={bv:.4f}', xy=(br, bv), xytext=(br+max(r)*0.05, bv-0.002),
                color=GOLD, fontsize=8, arrowprops=dict(arrowstyle='->', color=GOLD, lw=0.8))
    ax_style(ax, title=t, xl='Boosting Round', yl='NDCG@3'); ax.legend(fontsize=9)
save(fig, '04_learning_curve.png')

# 5 PREDICTION BUCKET
print("  Plot 5/8 Prediction Bucket")
NB = 10
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
fig.suptitle(f'Prediction Bucket | Avg Return by Score Decile | {LABEL} OOS', fontsize=12, fontweight='bold', color=TEXT)
for ax, sc, col, t, inv in [(a1, 'long_score', LONG_C, 'Long', False), (a2, 'short_score', SHORT_C, 'Short', True)]:
    df_oos['bk'] = pd.qcut(df_oos[sc], NB, labels=False, duplicates='drop')
    g = df_oos.groupby('bk')[RET]
    means = g.mean().values*((-1) if inv else 1)*100; sems = g.sem().values*100; cnt = g.count().values
    xs = np.arange(len(means))
    ax.bar(xs, means, color=[LONG_C if v >= 0 else SHORT_C for v in means], alpha=0.85)
    ax.errorbar(xs, means, yerr=sems*1.96, fmt='none', color=TEXT, capsize=3, elinewidth=1, alpha=0.6)
    mkt = df_oos[RET].mean()*((-1) if inv else 1)*100
    ax.axhline(mkt, color=GOLD, lw=1.5, ls='--', label=f'Market avg {mkt:.4f}%'); ax.axhline(0, color=NEUT, lw=0.8, alpha=0.5)
    ax.set_xticks(xs); ax.set_xticklabels([f'D{i+1}' for i in xs], fontsize=8.5)
    rho, _ = spearmanr(xs, means)
    ax.text(0.02, 0.97, f'Bucket Rho={rho:.3f}', transform=ax.transAxes, fontsize=9, color=GOLD, va='top', fontweight='bold')
    ax_style(ax, title=t, xl='Score Decile', yl='Avg Return (%)'); ax.legend(fontsize=8.5)
save(fig, '05_prediction_bucket.png')

# 6 CUMULATIVE RETURN
print("  Plot 6/8 Cumulative Return")
def cum(df_, sc, k=3, inv=False):
    rows = []
    for qid, q in df_.groupby('Query_ID'):
        if len(q) < k+1: continue
        ret = q[RET].values; s = q[sc].values
        idx = np.argsort(s)[::-1][:k]
        rows.append({'dt': q['DateTime'].iloc[0], 'top': (-ret[idx] if inv else ret[idx]).mean(), 'mkt': ret.mean()})
    d = pd.DataFrame(rows).sort_values('dt').reset_index(drop=True)
    d['cs'] = (1+d['top']).cumprod()-1; d['cm'] = (1+d['mkt']).cumprod()-1; return d
cl, cs = cum(df_oos, 'long_score', 3, False), cum(df_oos, 'short_score', 3, True)
fig = plt.figure(figsize=(14, 6), facecolor=BG)
fig.suptitle(f'Cumulative Return | Top-3 vs Market | {LABEL} OOS {oos_m[0]}->{oos_m[-1]}', fontsize=12, fontweight='bold', color=TEXT)
gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.12)
for ax, dc, col, t in [(fig.add_subplot(gs[0]), cl, LONG_C, 'Long Top-3'), (fig.add_subplot(gs[1]), cs, SHORT_C, 'Short Top-3')]:
    ax.plot(range(len(dc)), dc['cs']*100, color=col, lw=1.8, label='Strategy')
    ax.plot(range(len(dc)), dc['cm']*100, color=NEUT, lw=1.2, ls='--', label='Market (EW)')
    ax.fill_between(range(len(dc)), dc['cs']*100, dc['cm']*100, where=dc['cs'] >= dc['cm'], alpha=0.12, color=col, interpolate=True)
    ax.axhline(0, color=NEUT, lw=0.7, alpha=0.4)
    fs, fm = dc['cs'].iloc[-1]*100, dc['cm'].iloc[-1]*100
    ax.text(0.98, 0.04, f'Strat {fs:+.2f}%\nMkt {fm:+.2f}%\nExcess {fs-fm:+.2f}%', transform=ax.transAxes, fontsize=9,
            color=TEXT, ha='right', va='bottom', bbox=dict(facecolor='#1e2338', edgecolor=GRID, boxstyle='round,pad=0.4'))
    ax_style(ax, title=t, xl='Query Index', yl='Cumulative Return (%)'); ax.legend(fontsize=9, loc='upper left')
save(fig, '06_cumulative_return.png')

# 7 CALIBRATION
print("  Plot 7/8 Calibration")
NC = 10
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
fig.suptitle(f'Calibration | Predicted Rank Pct vs Realized Return | {LABEL} OOS', fontsize=12, fontweight='bold', color=TEXT)
for ax, sc, sign, col, t in [(a1, 'long_score', 1, LONG_C, 'Long'), (a2, 'short_score', -1, SHORT_C, 'Short')]:
    pp, ar = [], []
    for qid, q in df_oos.groupby('Query_ID'):
        if len(q) < 4: continue
        pp.extend((rankdata(q[sc].values, method='average')/len(q)).tolist())
        ar.extend((q[RET].values*sign).tolist())
    pp, ar = np.array(pp), np.array(ar); bins = np.linspace(0, 1, NC+1); bi = np.digitize(pp, bins[1:-1])
    bx, by, be = [], [], []
    for b in range(NC):
        m = bi == b
        if m.sum() > 10:
            bx.append(bins[b]+0.05); by.append(ar[m].mean()*100); be.append(ar[m].std()/np.sqrt(m.sum())*100*1.96)
    bx, by, be = np.array(bx), np.array(by), np.array(be)
    ax.bar(bx, by, width=0.08, color=[col if v >= 0 else SHORT_C for v in by], alpha=0.8)
    ax.errorbar(bx, by, yerr=be, fmt='none', color=TEXT, capsize=3, elinewidth=1, alpha=0.6)
    z = np.polyfit(bx, by, 1); xl = np.linspace(0, 1, 100)
    ax.plot(xl, np.poly1d(z)(xl), color=GOLD, lw=1.5, ls='--', label=f'slope={z[0]:.4f}')
    ax.axhline(0, color=NEUT, lw=0.8, alpha=0.5)
    rho, _ = spearmanr(bx, by)
    ax.text(0.03, 0.97, f'Bin Rho={rho:.3f}', transform=ax.transAxes, fontsize=9.5, color=GOLD, va='top', fontweight='bold')
    ax_style(ax, title=t, xl='Predicted Rank Percentile', yl='Avg Realized Return (%)'); ax.legend(fontsize=8.5); ax.set_xlim(0, 1)
save(fig, '07_calibration.png')

# 8 RESIDUAL
print("  Plot 8/8 Residual Analysis")
rl, rs = [], []
for qid, q in df_oos.groupby('Query_ID'):
    if len(q) < 4: continue
    n = len(q)
    rl.extend((rankdata(q['long_score'].values, method='ordinal')/n - rankdata(q[RET].values, method='ordinal')/n).tolist())
    rs.extend((rankdata(q['short_score'].values, method='ordinal')/n - rankdata(-q[RET].values, method='ordinal')/n).tolist())
rl, rs = np.array(rl), np.array(rs)
fig = plt.figure(figsize=(14, 10), facecolor=BG)
fig.suptitle(f'Residual Analysis | {LABEL} | OOS', fontsize=13, fontweight='bold', color=TEXT)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.25)
xr = np.linspace(-1, 1, 200)
for pos, resid, col, t in [((0, 0), rl, LONG_C, 'Long'), ((0, 1), rs, SHORT_C, 'Short')]:
    ax = fig.add_subplot(gs[pos])
    ax.hist(resid, bins=80, color=col, alpha=0.75, density=True)
    ax.plot(xr, sp_norm.pdf(xr, resid.mean(), resid.std()), color=GOLD, lw=1.5,
            label='mu=%.4f sigma=%.4f' % (resid.mean(), resid.std()))
    ax.axvline(0, color=NEUT, lw=0.8, alpha=0.5)
    ax_style(ax, title=f'{t} Residuals (Pred-True Rank)', xl='Residual', yl='Density'); ax.legend(fontsize=8)
ax_bl = fig.add_subplot(gs[1, 0])
pr, tr = [], []
for qid, q in list(df_oos.groupby('Query_ID'))[:200]:
    if len(q) < 4: continue
    pr.extend((rankdata(q['long_score'].values, method='ordinal')/len(q)).tolist())
    tr.extend((rankdata(q[RET].values, method='ordinal')/len(q)).tolist())
pr, tr = np.array(pr[:3000]), np.array(tr[:3000])
ax_bl.scatter(pr, tr, alpha=0.07, s=4, color=LONG_C); ax_bl.plot([0, 1], [0, 1], color=GOLD, lw=1.2, ls='--', label='Perfect')
rv, _ = pearsonr(pr, tr)
ax_bl.text(0.03, 0.97, f'Pearson r={rv:.4f}', transform=ax_bl.transAxes, fontsize=9, color=GOLD, va='top', fontweight='bold')
ax_style(ax_bl, title='Long: Pred vs True Rank', xl='Predicted Rank Pct', yl='True Rank Pct'); ax_bl.legend(fontsize=8)
ax_br = fig.add_subplot(gs[1, 1])
dc = df_oos.copy(); dc['hour'] = pd.to_datetime(dc['DateTime']).dt.hour
hours = sorted(dc['hour'].unique()); hr = []
for h in hours:
    qh = dc[dc['hour'] == h]; rr = []
    for qid, q in qh.groupby('Query_ID'):
        if len(q) < 4: continue
        rr.extend((rankdata(q['long_score'].values, method='ordinal')/len(q) - rankdata(q[RET].values, method='ordinal')/len(q)).tolist())
    hr.append(np.mean(np.abs(rr)) if rr else 0.0)
ax_br.bar(hours, hr, color=[LONG_C if v < np.median(hr) else SHORT_C for v in hr], alpha=0.85)
ax_br.axhline(np.mean(hr), color=GOLD, lw=1.2, ls='--', label=f'Mean MAE={np.mean(hr):.4f}')
ax_style(ax_br, title='Mean Abs Rank Error by Hour', xl='Hour (IST)', yl='Mean |Residual|'); ax_br.legend(fontsize=8.5)
save(fig, '08_residual_analysis.png')

# DASHBOARD
print("  Building dashboard")
from PIL import Image
pf = ['01_feature_importance.png', '04_learning_curve.png', '05_prediction_bucket.png',
      '06_cumulative_return.png', '07_calibration.png', '08_residual_analysis.png']
if SHAP_OK: pf = ['02_shap_summary.png', '03_shap_dependence.png'] + pf
paths = [os.path.join(OUT_DIR, f) for f in pf if os.path.exists(os.path.join(OUT_DIR, f))]
if len(paths) >= 2:
    imgs = [Image.open(p) for p in paths]; W = max(i.width for i in imgs); H = max(i.height for i in imgs)
    nrows = (len(imgs)+1)//2; canvas = Image.new('RGB', (W*2, H*nrows), (15, 17, 23))
    for i, im in enumerate(imgs):
        r, c = divmod(i, 2); canvas.paste(im, (c*W+(W-im.width)//2, r*H+(H-im.height)//2))
    canvas.save(os.path.join(OUT_DIR, '00_dashboard.png'), dpi=(DPI, DPI))
    print(f"  Dashboard -> {OUT_DIR}/00_dashboard.png")
print(f"\n  Done. Plots in {OUT_DIR}/")

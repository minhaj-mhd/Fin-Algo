"""
8 diagnostic visualizations for v2_15min_3y (15-min XGBoost ranking model).
Saves PNGs to data/model_analysis/v2_15min_3y/
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mtick
from scipy.stats import spearmanr, rankdata, pearsonr
warnings.filterwarnings('ignore')
sys.path.append(os.getcwd())

try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False
    print("[WARN] pip install shap  →  SHAP plots will be skipped")

# ── CONFIG ──────────────────────────────────────────────────────────────────
MODEL_DIR   = 'models/v2_15min_3y'
META_PATH   = f'{MODEL_DIR}/metadata.json'
LONG_PATH   = f'{MODEL_DIR}/xgb_long_model.json'
SHORT_PATH  = f'{MODEL_DIR}/xgb_short_model.json'
DATA_FILE   = 'data/ranking_data_upstox_15min_3y.csv'
OUT_DIR     = 'data/model_analysis/v2_15min_3y'
os.makedirs(OUT_DIR, exist_ok=True)

OOS_MONTHS      = 3
LC_TRAIN_MO     = 6     # months for learning-curve mini-retrain
SHAP_SAMPLE     = 3000  # rows sampled for SHAP (speed)
DPI             = 150

# ── PALETTE ──────────────────────────────────────────────────────────────────
BG      = '#0f1117'
AX_BG   = '#161b2e'
TEXT    = '#dde1f0'
GRID    = '#252a45'
LONG_C  = '#00d4aa'
SHORT_C = '#ff6b6b'
NEUT    = '#7c83a3'
GOLD    = '#f0c040'
PURPLE  = '#b57bee'

plt.rcParams.update({
    'figure.facecolor':  BG,
    'axes.facecolor':    AX_BG,
    'axes.edgecolor':    GRID,
    'axes.labelcolor':   TEXT,
    'xtick.color':       TEXT,
    'ytick.color':       TEXT,
    'text.color':        TEXT,
    'grid.color':        GRID,
    'grid.linewidth':    0.5,
    'legend.facecolor':  '#1e2338',
    'legend.edgecolor':  GRID,
    'font.family':       'DejaVu Sans',
})

def ax_style(ax, title='', xl='', yl='', grid=True):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.spines['left'].set_color(GRID)
    if title: ax.set_title(title, fontsize=11, fontweight='bold', color=TEXT, pad=8)
    if xl:    ax.set_xlabel(xl, fontsize=9, color=NEUT)
    if yl:    ax.set_ylabel(yl, fontsize=9, color=NEUT)
    if grid:  ax.grid(True, alpha=0.4)

def save(fig, name, tight=True):
    path = os.path.join(OUT_DIR, name)
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f"  Saved → {path}")

SEP = "─" * 60

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD METADATA & MODELS
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}\n  Loading models\n{SEP}")
with open(META_PATH) as f:
    meta = json.load(f)
feature_cols = meta['features']

bst_long  = xgb.Booster(); bst_long.load_model(LONG_PATH)
bst_short = xgb.Booster(); bst_short.load_model(SHORT_PATH)
print(f"  Features: {len(feature_cols)}  |  Total training rows: {meta['total_rows']:,}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. STREAM DATA  (OOS + LC train/val)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}\n  Streaming data\n{SEP}")
all_months = set()
for chunk in pd.read_csv(DATA_FILE, usecols=['DateTime'], chunksize=500_000):
    all_months.update(chunk['DateTime'].str[:7].unique())
all_months = sorted(all_months)
print(f"  Dataset: {all_months[0]} → {all_months[-1]}  ({len(all_months)} months)")

oos_m    = all_months[-OOS_MONTHS:]
lc_tr_m  = all_months[-(OOS_MONTHS + LC_TRAIN_MO + 1) : -(OOS_MONTHS + 1)]
lc_val_m = [all_months[-(OOS_MONTHS + 1)]]
load_set = set(oos_m) | set(lc_tr_m) | set(lc_val_m)

print(f"  OOS months : {oos_m}")
print(f"  LC  train  : {lc_tr_m}")
print(f"  LC  val    : {lc_val_m}")
print("  Loading rows...")

chunks = []
for chunk in pd.read_csv(DATA_FILE, chunksize=200_000):
    sub = chunk[chunk['DateTime'].str[:7].isin(load_set)]
    if len(sub): chunks.append(sub)
df_all = pd.concat(chunks, ignore_index=True)
df_all['YearMonth'] = df_all['DateTime'].str[:7]

df_oos    = df_all[df_all['YearMonth'].isin(oos_m)].copy().reset_index(drop=True)
df_lc_tr  = df_all[df_all['YearMonth'].isin(lc_tr_m)].copy().reset_index(drop=True)
df_lc_val = df_all[df_all['YearMonth'].isin(lc_val_m)].copy().reset_index(drop=True)
print(f"  OOS:       {len(df_oos):,} rows | {df_oos['Query_ID'].nunique():,} queries")
print(f"  LC train:  {len(df_lc_tr):,}  | LC val: {len(df_lc_val):,}")

def prep(df_):
    X = df_[feature_cols].values.astype(float)
    for ci in range(X.shape[1]):
        col = X[:, ci]
        bad = np.isnan(col) | np.isinf(col)
        if bad.any():
            X[bad, ci] = float(np.nanmean(col[~bad])) if (~bad).any() else 0.0
    return X

X_oos = prep(df_oos)
lp = bst_long.predict(xgb.DMatrix(X_oos))
sp = bst_short.predict(xgb.DMatrix(X_oos))
df_oos['long_score']  = lp
df_oos['short_score'] = sp
df_oos = df_oos.sort_values('DateTime').reset_index(drop=True)
print("  OOS predictions ready.\n")

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: integer rank labels
# ─────────────────────────────────────────────────────────────────────────────
def int_ranks(y, qids, invert=False):
    out = np.zeros_like(y, dtype=int)
    for qid in np.unique(qids):
        mask = qids == qid
        vals = -y[mask] if invert else y[mask]
        out[mask] = rankdata(vals, method='ordinal') - 1
    return out

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1  –  FEATURE IMPORTANCE (Long & Short side-by-side)
# ─────────────────────────────────────────────────────────────────────────────
print(f"{SEP}\n  Plot 1 / 8 – Feature Importance\n{SEP}")

top_long  = meta['top_features_long']
top_short = meta['top_features_short']
all_f     = sorted(set(top_long) | set(top_short),
                   key=lambda x: -(top_long.get(x, 0) + top_short.get(x, 0)))[:20]
all_f.reverse()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), facecolor=BG)
fig.suptitle('Feature Importance  |  v2_15min_3y  (Gain, Top 20)',
             fontsize=13, fontweight='bold', color=TEXT, y=1.01)

for ax, feat_dict, color, title in [
    (ax1, top_long,  LONG_C,  'LONG Model'),
    (ax2, top_short, SHORT_C, 'SHORT Model'),
]:
    vals  = [feat_dict.get(f, 0) for f in all_f]
    bars  = ax.barh(all_f, vals, color=color, alpha=0.85, edgecolor='none')
    for bar, val in zip(bars, vals):
        if val > 0:
            ax.text(val + max(vals)*0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}', va='center', fontsize=7.5, color=TEXT)
    ax_style(ax, title=title, xl='Gain Score', grid=False)
    ax.tick_params(labelsize=8.5)

save(fig, '01_feature_importance.png')

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2  –  SHAP SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print(f"{SEP}\n  Plot 2 / 8 – SHAP Summary\n{SEP}")

if SHAP_OK:
    rng = np.random.default_rng(42)
    idx_s = rng.choice(len(X_oos), min(SHAP_SAMPLE, len(X_oos)), replace=False)
    X_shap = X_oos[idx_s]
    df_shap = df_oos.iloc[idx_s].reset_index(drop=True)

    print(f"  Computing SHAP values on {len(X_shap):,} samples...")
    explainer  = shap.TreeExplainer(bst_long)
    shap_vals  = explainer.shap_values(X_shap)

    fig, ax = plt.subplots(figsize=(11, 8), facecolor=BG)
    fig.suptitle('SHAP Summary  |  Long Model  |  v2_15min_3y',
                 fontsize=13, fontweight='bold', color=TEXT)
    plt.sca(ax)
    shap.summary_plot(shap_vals, X_shap, feature_names=feature_cols,
                      max_display=20, show=False, plot_type='dot',
                      color_bar_label='Feature value (normalized)')
    ax.set_facecolor(AX_BG)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    save(fig, '02_shap_summary.png', tight=False)
else:
    print("  SHAP not available – skipping.")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3  –  SHAP DEPENDENCE  (IBS  +  Buy_Pressure)
# ─────────────────────────────────────────────────────────────────────────────
print(f"{SEP}\n  Plot 3 / 8 – SHAP Dependence\n{SEP}")

if SHAP_OK:
    dep_features = ['IBS', 'Buy_Pressure']
    inter_feat   = {'IBS': 'Buy_Pressure', 'Buy_Pressure': 'IBS'}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
    fig.suptitle('SHAP Dependence Plots  |  Long Model  |  v2_15min_3y',
                 fontsize=13, fontweight='bold', color=TEXT)

    for ax, feat in zip(axes, dep_features):
        fidx  = feature_cols.index(feat)
        iidx  = feature_cols.index(inter_feat[feat])
        x_val = X_shap[:, fidx]
        s_val = shap_vals[:, fidx]
        c_val = X_shap[:, iidx]

        sc = ax.scatter(x_val, s_val, c=c_val, cmap='RdYlGn',
                        alpha=0.4, s=8, edgecolors='none')
        cb = plt.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label(inter_feat[feat], color=TEXT, fontsize=8)
        cb.ax.yaxis.set_tick_params(color=TEXT, labelsize=7.5)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT)
        ax.axhline(0, color=NEUT, linewidth=0.8, linestyle='--')
        ax_style(ax, title=f'SHAP({feat})', xl=feat, yl=f'SHAP value')
        ax.tick_params(labelsize=8.5)

    save(fig, '03_shap_dependence.png')
else:
    print("  SHAP not available – skipping.")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4  –  LEARNING CURVE  (mini-retrain on LC window)
# ─────────────────────────────────────────────────────────────────────────────
print(f"{SEP}\n  Plot 4 / 8 – Learning Curve  (mini-retrain on {LC_TRAIN_MO}+1 months)\n{SEP}")

X_tr  = prep(df_lc_tr);  qids_tr  = df_lc_tr['Query_ID'].values
X_val = prep(df_lc_val); qids_val = df_lc_val['Query_ID'].values
y_tr  = df_lc_tr['Next_15Min_Return'].values
y_val = df_lc_val['Next_15Min_Return'].values

y_l_tr  = int_ranks(y_tr,  qids_tr,  invert=False)
y_l_val = int_ranks(y_val, qids_val, invert=False)
y_s_tr  = int_ranks(y_tr,  qids_tr,  invert=True)
y_s_val = int_ranks(y_val, qids_val, invert=True)

grp_tr  = pd.Series(qids_tr).groupby(qids_tr).size().values
grp_val = pd.Series(qids_val).groupby(qids_val).size().values

lc_params = dict(meta['params'])

def build_dm(X, y, grp):
    d = xgb.DMatrix(X, label=y)
    d.set_group(grp)
    return d

dtr_l = build_dm(X_tr, y_l_tr, grp_tr);   dvl_l = build_dm(X_val, y_l_val, grp_val)
dtr_s = build_dm(X_tr, y_s_tr, grp_tr);   dvl_s = build_dm(X_val, y_s_val, grp_val)

print("  Training Long LC model...")
er_l = {}
xgb.train(lc_params, dtr_l, num_boost_round=500,
          evals=[(dtr_l, 'train'), (dvl_l, 'val')],
          evals_result=er_l, early_stopping_rounds=50, verbose_eval=False)

print("  Training Short LC model...")
er_s = {}
xgb.train(lc_params, dtr_s, num_boost_round=500,
          evals=[(dtr_s, 'train'), (dvl_s, 'val')],
          evals_result=er_s, early_stopping_rounds=50, verbose_eval=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
fig.suptitle(f'Learning Curve  |  NDCG@3  |  Train={lc_tr_m[0]}→{lc_tr_m[-1]}  Val={lc_val_m[0]}',
             fontsize=12, fontweight='bold', color=TEXT)

for ax, er, title in [(ax1, er_l, 'Long Model'), (ax2, er_s, 'Short Model')]:
    rounds = range(1, len(er['train']['ndcg@3']) + 1)
    ax.plot(rounds, er['train']['ndcg@3'], color=LONG_C, lw=1.5, label='Train', alpha=0.9)
    ax.plot(rounds, er['val']['ndcg@3'],   color=SHORT_C, lw=1.5, label='Validation', alpha=0.9)
    best_r = int(np.argmax(er['val']['ndcg@3'])) + 1
    best_v = max(er['val']['ndcg@3'])
    ax.axvline(best_r, color=GOLD, lw=1, linestyle='--', alpha=0.7)
    ax.scatter([best_r], [best_v], color=GOLD, s=60, zorder=5)
    ax.annotate(f'Best: round {best_r}\nNDCG={best_v:.4f}',
                xy=(best_r, best_v), xytext=(best_r + max(rounds)*0.05, best_v - 0.002),
                color=GOLD, fontsize=8, arrowprops=dict(arrowstyle='->', color=GOLD, lw=0.8))
    ax_style(ax, title=title, xl='Boosting Round', yl='NDCG@3')
    ax.legend(fontsize=9)

save(fig, '04_learning_curve.png')

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 5  –  PREDICTION BUCKET ANALYSIS  (return by score decile)
# ─────────────────────────────────────────────────────────────────────────────
print(f"{SEP}\n  Plot 5 / 8 – Prediction Bucket Analysis\n{SEP}")

N_BUCKETS = 10
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
fig.suptitle('Prediction Bucket Analysis  |  Avg Return by Score Decile  |  OOS (3 months)',
             fontsize=12, fontweight='bold', color=TEXT)

for ax, score_col, color, title, invert in [
    (ax1, 'long_score',  LONG_C,  'Long Model', False),
    (ax2, 'short_score', SHORT_C, 'Short Model', True),
]:
    df_oos['bucket'] = pd.qcut(df_oos[score_col], N_BUCKETS, labels=False)
    grp = df_oos.groupby('bucket')['Next_15Min_Return']
    means  = grp.mean().values * ((-1) if invert else 1) * 100
    sems   = grp.sem().values * 100
    counts = grp.count().values
    xs     = np.arange(N_BUCKETS)

    bar_colors = [LONG_C if v >= 0 else SHORT_C for v in means]
    bars = ax.bar(xs, means, color=bar_colors, alpha=0.85, edgecolor='none')
    ax.errorbar(xs, means, yerr=sems*1.96, fmt='none',
                color=TEXT, capsize=3, capthick=1, elinewidth=1, alpha=0.6)

    mkt_avg = df_oos['Next_15Min_Return'].mean() * ((-1) if invert else 1) * 100
    ax.axhline(mkt_avg, color=GOLD, lw=1.5, linestyle='--', alpha=0.9, label=f'Market avg: {mkt_avg:.4f}%')
    ax.axhline(0, color=NEUT, lw=0.8, alpha=0.5)

    for i, (bar, cnt) in enumerate(zip(bars, counts)):
        ax.text(i, bar.get_height() + sems[i]*1.96 + abs(means).max()*0.03,
                f'n={cnt//1000:.0f}k', ha='center', va='bottom', fontsize=7, color=NEUT)

    ax.set_xticks(xs)
    ax.set_xticklabels([f'D{i+1}' for i in xs], fontsize=8.5)
    ax_style(ax, title=title, xl='Score Decile (D1=lowest, D10=highest)', yl='Avg Return (%)')
    ax.legend(fontsize=8.5)

    # Spearman rho on bucket means
    rho, _ = spearmanr(xs, means)
    ax.text(0.02, 0.97, f'Bucket Rho={rho:.3f}', transform=ax.transAxes,
            fontsize=9, color=GOLD, va='top', fontweight='bold')

save(fig, '05_prediction_bucket.png')

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 6  –  CUMULATIVE RETURN CURVE
# ─────────────────────────────────────────────────────────────────────────────
print(f"{SEP}\n  Plot 6 / 8 – Cumulative Return Curve\n{SEP}")

def cum_returns(df_, score_col, k=3, invert=False):
    rows = []
    for qid in df_['Query_ID'].unique():
        q = df_[df_['Query_ID'] == qid]
        if len(q) < k + 1: continue
        ret = q['Next_15Min_Return'].values
        sc  = q[score_col].values
        dt  = q['DateTime'].iloc[0]
        idx = np.argsort(sc)[::-1][:k]
        top_ret = (-ret[idx] if invert else ret[idx]).mean()
        mkt_ret = ret.mean()
        rows.append({'dt': dt, 'top': top_ret, 'mkt': mkt_ret})
    df_r = pd.DataFrame(rows).sort_values('dt').reset_index(drop=True)
    df_r['cum_strat'] = (1 + df_r['top']).cumprod() - 1
    df_r['cum_mkt']   = (1 + df_r['mkt']).cumprod() - 1
    return df_r

print("  Computing cumulative curves...")
cl = cum_returns(df_oos, 'long_score',  k=3, invert=False)
cs = cum_returns(df_oos, 'short_score', k=3, invert=True)

fig = plt.figure(figsize=(14, 6), facecolor=BG)
fig.suptitle('Cumulative Return Curve  |  Top-3 Strategy vs Market  |  OOS Apr–Jun 2026',
             fontsize=12, fontweight='bold', color=TEXT)
gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.12)
ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

for ax, df_c, s_color, title in [
    (ax1, cl, LONG_C,  'Long Model  –  Top-3 Long'),
    (ax2, cs, SHORT_C, 'Short Model  –  Top-3 Short'),
]:
    ax.plot(range(len(df_c)), df_c['cum_strat'] * 100,
            color=s_color, lw=1.8, label='Strategy', alpha=0.95)
    ax.plot(range(len(df_c)), df_c['cum_mkt'] * 100,
            color=NEUT, lw=1.2, linestyle='--', label='Market (EW)', alpha=0.8)
    ax.fill_between(range(len(df_c)),
                    df_c['cum_strat'] * 100, df_c['cum_mkt'] * 100,
                    where=df_c['cum_strat'] >= df_c['cum_mkt'],
                    alpha=0.12, color=s_color, interpolate=True)
    ax.axhline(0, color=NEUT, lw=0.7, alpha=0.4)

    final_s = df_c['cum_strat'].iloc[-1] * 100
    final_m = df_c['cum_mkt'].iloc[-1] * 100
    excess  = final_s - final_m
    ax.text(0.98, 0.04, f'Strategy: {final_s:+.2f}%\nMarket: {final_m:+.2f}%\nExcess: {excess:+.2f}%',
            transform=ax.transAxes, fontsize=9, color=TEXT, ha='right', va='bottom',
            bbox=dict(facecolor='#1e2338', edgecolor=GRID, boxstyle='round,pad=0.4'))

    ax_style(ax, title=title, xl='Query Index (15-min bars)', yl='Cumulative Return (%)')
    ax.legend(fontsize=9, loc='upper left')

    n_queries = len(df_c)
    tick_step = max(1, n_queries // 6)
    ax.set_xticks(range(0, n_queries, tick_step))
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{int(x):,}'))

save(fig, '06_cumulative_return.png')

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 7  –  CALIBRATION PLOT  (predicted rank percentile vs actual)
# ─────────────────────────────────────────────────────────────────────────────
print(f"{SEP}\n  Plot 7 / 8 – Calibration Plot\n{SEP}")

N_CAL = 10
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
fig.suptitle('Calibration Plot  |  Predicted Rank Percentile vs Realized Return  |  OOS',
             fontsize=12, fontweight='bold', color=TEXT)

for ax, score_col, ret_sign, color, title in [
    (ax1, 'long_score',   1,  LONG_C,  'Long Model'),
    (ax2, 'short_score', -1, SHORT_C, 'Short Model'),
]:
    pred_pct, act_ret = [], []
    for qid in df_oos['Query_ID'].unique():
        q = df_oos[df_oos['Query_ID'] == qid]
        if len(q) < 4: continue
        scores = q[score_col].values
        rets   = q['Next_15Min_Return'].values * ret_sign
        pct    = rankdata(scores, method='average') / len(scores)
        pred_pct.extend(pct.tolist())
        act_ret.extend(rets.tolist())

    pred_arr = np.array(pred_pct)
    ret_arr  = np.array(act_ret)
    bins     = np.linspace(0, 1, N_CAL + 1)
    bin_idx  = np.digitize(pred_arr, bins[1:-1])

    bin_x, bin_y, bin_e, bin_n = [], [], [], []
    for b in range(N_CAL):
        mask = bin_idx == b
        if mask.sum() > 10:
            bin_x.append(bins[b] + 0.05)
            bin_y.append(ret_arr[mask].mean() * 100)
            bin_e.append(ret_arr[mask].std() / np.sqrt(mask.sum()) * 100 * 1.96)
            bin_n.append(mask.sum())

    bin_x, bin_y, bin_e = np.array(bin_x), np.array(bin_y), np.array(bin_e)
    bar_colors = [color if v >= 0 else SHORT_C for v in bin_y]
    ax.bar(bin_x, bin_y, width=0.08, color=bar_colors, alpha=0.8, edgecolor='none')
    ax.errorbar(bin_x, bin_y, yerr=bin_e, fmt='none', color=TEXT,
                capsize=3, capthick=1, elinewidth=1, alpha=0.6)

    z = np.polyfit(bin_x, bin_y, 1)
    xline = np.linspace(0, 1, 100)
    ax.plot(xline, np.poly1d(z)(xline), color=GOLD, lw=1.5, linestyle='--',
            alpha=0.8, label=f'Trend (slope={z[0]:.4f})')
    ax.axhline(0, color=NEUT, lw=0.8, alpha=0.5)

    rho, _ = spearmanr(bin_x, bin_y)
    ax.text(0.03, 0.97, f'Bin Rho={rho:.3f}', transform=ax.transAxes,
            fontsize=9.5, color=GOLD, va='top', fontweight='bold')
    ax_style(ax, title=title, xl='Predicted Rank Percentile', yl='Avg Realized Return (%)')
    ax.legend(fontsize=8.5)
    ax.set_xlim(0, 1)

save(fig, '07_calibration.png')

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 8  –  RESIDUAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print(f"{SEP}\n  Plot 8 / 8 – Residual Analysis\n{SEP}")

resid_l, resid_s = [], []
for qid in df_oos['Query_ID'].unique():
    q = df_oos[df_oos['Query_ID'] == qid]
    if len(q) < 4: continue
    n = len(q)
    pred_rank = rankdata(q['long_score'].values,  method='ordinal') / n
    true_rank = rankdata(q['Next_15Min_Return'].values, method='ordinal') / n
    resid_l.extend((pred_rank - true_rank).tolist())

    pred_rank_s = rankdata(q['short_score'].values, method='ordinal') / n
    true_rank_s = rankdata(-q['Next_15Min_Return'].values, method='ordinal') / n
    resid_s.extend((pred_rank_s - true_rank_s).tolist())

resid_l = np.array(resid_l)
resid_s = np.array(resid_s)

fig = plt.figure(figsize=(14, 10), facecolor=BG)
fig.suptitle('Residual Analysis  |  v2_15min_3y  |  OOS (3 months)',
             fontsize=13, fontweight='bold', color=TEXT)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.25)

# Top-left: Long residual histogram
ax_tl = fig.add_subplot(gs[0, 0])
ax_tl.hist(resid_l, bins=80, color=LONG_C, alpha=0.75, edgecolor='none', density=True)
xr = np.linspace(-1, 1, 200)
from scipy.stats import norm as sp_norm
ax_tl.plot(xr, sp_norm.pdf(xr, resid_l.mean(), resid_l.std()),
           color=GOLD, lw=1.5, label=f'Normal fit\n$\mu$={resid_l.mean():.4f}, $\sigma$={resid_l.std():.4f}')
ax_tl.axvline(0, color=NEUT, lw=0.8, alpha=0.5)
ax_style(ax_tl, title='Long Residuals  (Pred Rank – True Rank)', xl='Residual', yl='Density')
ax_tl.legend(fontsize=8)

# Top-right: Short residual histogram
ax_tr = fig.add_subplot(gs[0, 1])
ax_tr.hist(resid_s, bins=80, color=SHORT_C, alpha=0.75, edgecolor='none', density=True)
ax_tr.plot(xr, sp_norm.pdf(xr, resid_s.mean(), resid_s.std()),
           color=GOLD, lw=1.5, label=f'Normal fit\n$\mu$={resid_s.mean():.4f}, $\sigma$={resid_s.std():.4f}')
ax_tr.axvline(0, color=NEUT, lw=0.8, alpha=0.5)
ax_style(ax_tr, title='Short Residuals  (Pred Rank – True Rank)', xl='Residual', yl='Density')
ax_tr.legend(fontsize=8)

# Bottom-left: Long predicted vs true rank scatter (sample)
ax_bl = fig.add_subplot(gs[1, 0])
sample_n = min(3000, len(resid_l))
si = np.random.default_rng(0).choice(len(resid_l), sample_n, replace=False)
qids_u = list(df_oos['Query_ID'].unique())[:sample_n]
pred_r_all, true_r_all = [], []
for qid in qids_u[:200]:
    q = df_oos[df_oos['Query_ID'] == qid]
    if len(q) < 4: continue
    n = len(q)
    pred_r_all.extend((rankdata(q['long_score'].values, method='ordinal') / n).tolist())
    true_r_all.extend((rankdata(q['Next_15Min_Return'].values, method='ordinal') / n).tolist())

pr = np.array(pred_r_all[:3000]); tr = np.array(true_r_all[:3000])
ax_bl.scatter(pr, tr, alpha=0.07, s=4, color=LONG_C, edgecolors='none')
ax_bl.plot([0, 1], [0, 1], color=GOLD, lw=1.2, linestyle='--', alpha=0.8, label='Perfect rank')
r_val, _ = pearsonr(pr, tr)
ax_bl.text(0.03, 0.97, f'Pearson r={r_val:.4f}', transform=ax_bl.transAxes,
           fontsize=9, color=GOLD, va='top', fontweight='bold')
ax_style(ax_bl, title='Long: Predicted Rank vs True Rank', xl='Predicted Rank Pct', yl='True Rank Pct')
ax_bl.legend(fontsize=8)

# Bottom-right: Residual by hour-of-day
ax_br = fig.add_subplot(gs[1, 1])
df_oos_copy = df_oos.copy()
df_oos_copy['hour'] = pd.to_datetime(df_oos_copy['DateTime']).dt.hour
hour_resid = []
hours = sorted(df_oos_copy['hour'].unique())
for h in hours:
    mask_h = df_oos_copy['hour'] == h
    q_h = df_oos_copy[mask_h]
    r_h = []
    for qid in q_h['Query_ID'].unique():
        q = q_h[q_h['Query_ID'] == qid]
        if len(q) < 4: continue
        pr_h = rankdata(q['long_score'].values, method='ordinal') / len(q)
        tr_h = rankdata(q['Next_15Min_Return'].values, method='ordinal') / len(q)
        r_h.extend((pr_h - tr_h).tolist())
    hour_resid.append(np.mean(np.abs(r_h)) if r_h else 0.0)

bar_c = [LONG_C if v < np.median(hour_resid) else SHORT_C for v in hour_resid]
ax_br.bar(hours, hour_resid, color=bar_c, alpha=0.85, edgecolor='none')
ax_br.axhline(np.mean(hour_resid), color=GOLD, lw=1.2, linestyle='--',
              label=f'Mean MAE={np.mean(hour_resid):.4f}')
ax_style(ax_br, title='Mean Absolute Rank Error by Hour', xl='Hour of Day (IST)', yl='Mean |Residual|')
ax_br.legend(fontsize=8.5)

save(fig, '08_residual_analysis.png')

# ─────────────────────────────────────────────────────────────────────────────
# COMBINED DASHBOARD  (4×2 grid)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}\n  Building combined dashboard\n{SEP}")

from PIL import Image

plot_files = [
    '01_feature_importance.png',
    '04_learning_curve.png',
    '05_prediction_bucket.png',
    '06_cumulative_return.png',
    '07_calibration.png',
    '08_residual_analysis.png',
]
if SHAP_OK:
    plot_files = ['02_shap_summary.png', '03_shap_dependence.png'] + plot_files

imgs_paths = [os.path.join(OUT_DIR, f) for f in plot_files if os.path.exists(os.path.join(OUT_DIR, f))]

if len(imgs_paths) >= 2:
    imgs = [Image.open(p) for p in imgs_paths]
    n = len(imgs)
    ncols = 2
    nrows = (n + 1) // 2

    W = max(img.width for img in imgs)
    H = max(img.height for img in imgs)
    canvas = Image.new('RGB', (W * ncols, H * nrows), color=(15, 17, 23))

    for i, img in enumerate(imgs):
        r, c = divmod(i, ncols)
        x = c * W + (W - img.width) // 2
        y = r * H + (H - img.height) // 2
        canvas.paste(img, (x, y))

    dash_path = os.path.join(OUT_DIR, '00_dashboard.png')
    canvas.save(dash_path, dpi=(DPI, DPI))
    print(f"  Dashboard saved → {dash_path}")
else:
    print("  Not enough plots for dashboard.")

print(f"\n{'=' * 60}")
print(f"  All plots saved to: {OUT_DIR}/")
print(f"{'=' * 60}\n")

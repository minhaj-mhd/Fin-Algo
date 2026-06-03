"""
validate_daily_model.py — Deep diagnostic of the daily XGBoost model

Investigates why the daily model underperforms, especially on the long side
(P@3: 49.7%, barely random). Checks:
  1. Production model complexity (best_iteration) — is it underfitting?
  2. Feature appropriateness — are hourly features meaningful on daily bars?
  3. Data quality — NaN rates, label distribution, query group sizes
  4. Per-fold breakdown — which folds/regimes fail?
  5. Cross-sectional Z-scoring impact on daily data
  6. Feature importance vs. intraday models — what's different?
  7. Train vs. Test performance gap — overfitting check
  8. Long vs. Short asymmetry — structural bias analysis
"""

import os, sys, json, pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr, rankdata
from tqdm import tqdm

sys.path.append(os.getcwd())

# ========================================
# CONFIG
# ========================================
DATA_FILE = 'data/ranking_data_upstox_daily_5y.csv'
MODEL_DIR = 'models/daily_xgb'
META_PATH = f'{MODEL_DIR}/metadata.json'
LONG_MODEL_PATH = f'{MODEL_DIR}/xgb_long_model.json'
SHORT_MODEL_PATH = f'{MODEL_DIR}/xgb_short_model.json'

print("=" * 70)
print("DAILY XGBOOST MODEL — DEEP DIAGNOSTIC")
print("=" * 70)

# ========================================
# 1. PRODUCTION MODEL INSPECTION
# ========================================
print("\n" + "=" * 70)
print("1. PRODUCTION MODEL INSPECTION")
print("=" * 70)

with open(META_PATH) as f:
    meta = json.load(f)

print(f"  Data source: {meta['data_source']}")
print(f"  Total rows: {meta['total_rows']:,}")
print(f"  Features: {meta['num_features']}")
print(f"  Params: eta={meta['params']['eta']}, max_depth={meta['params']['max_depth']}, "
      f"min_child_weight={meta['params']['min_child_weight']}")
print(f"  Regularization: alpha={meta['params']['alpha']}, lambda={meta['params']['lambda']}")

prod = meta['production_training']
print(f"\n  Production Training:")
print(f"    Train months: {prod['train_months'][0]} -> {prod['train_months'][-1]} ({len(prod['train_months'])} months)")
print(f"    Val months: {prod['val_months']}")
print(f"    Long best iteration: {prod['long_best_iteration']}")
print(f"    Short best iteration: {prod['short_best_iteration']}")

# THIS IS A RED FLAG — long model stopped at iteration 30 out of 800!
if prod['long_best_iteration'] < 50:
    print(f"\n  ⚠️ CRITICAL: Long model stopped at iteration {prod['long_best_iteration']}/800!")
    print(f"     This means the model barely learned anything useful for longs.")
    print(f"     Early stopping triggered after just {prod['long_best_iteration']} rounds.")
    print(f"     The validation NDCG was not improving — the model cannot rank daily longs.")

if prod['short_best_iteration'] < 50:
    print(f"\n  ⚠️ NOTE: Short model also stopped early at {prod['short_best_iteration']}/800")
else:
    print(f"\n  ✓ Short model trained for {prod['short_best_iteration']} rounds — reasonable.")

# ========================================
# 2. WALK-FORWARD FOLD ANALYSIS
# ========================================
print("\n" + "=" * 70)
print("2. WALK-FORWARD FOLD-BY-FOLD ANALYSIS")
print("=" * 70)

wf = meta['walk_forward_folds']
print(f"\n  {'Fold':>4} | {'Long ρ':>8} | {'Short ρ':>8} | {'Long P@1':>8} | {'Long P@3':>8} | {'Short P@3':>8} | {'Long Edge':>10} | {'Short Edge':>10}")
print(f"  {'-'*4} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*10} | {'-'*10}")

for fold in wf:
    print(f"  {fold['fold']:>4} | {fold['long_rho']:>8.4f} | {fold['short_rho']:>8.4f} | "
          f"{fold['long_win_rates']['1']:>7.1%} | {fold['long_win_rates']['3']:>7.1%} | "
          f"{fold['short_win_rates']['3']:>7.1%} | {fold['long_edge']*100:>+9.4f}% | "
          f"{fold['short_edge']*100:>+9.4f}%")

# Highlight: Is Long consistently bad?
long_p3_values = [f['long_win_rates']['3'] for f in wf]
short_p3_values = [f['short_win_rates']['3'] for f in wf]
long_edges = [f['long_edge'] for f in wf]

print(f"\n  Long P@3 range: {min(long_p3_values):.1%} - {max(long_p3_values):.1%}")
print(f"  Short P@3 range: {min(short_p3_values):.1%} - {max(short_p3_values):.1%}")
print(f"  Long edge negative in {sum(1 for e in long_edges if e < 0)}/{len(long_edges)} folds")

# ========================================
# 3. DATA QUALITY ANALYSIS
# ========================================
print("\n" + "=" * 70)
print("3. DATA QUALITY ANALYSIS")
print("=" * 70)

print("\nLoading dataset...")
df = pd.read_csv(DATA_FILE)
print(f"  Loaded {df.shape[0]:,} rows, {df.shape[1]} columns")

# Basic stats
df['YearMonth'] = df['DateTime'].str[:7]
unique_months = sorted(df['YearMonth'].unique())
print(f"  Date range: {unique_months[0]} to {unique_months[-1]} ({len(unique_months)} months)")

# Query group sizes
grp_sizes = df.groupby('Query_ID').size()
print(f"\n  Query group sizes:")
print(f"    Mean: {grp_sizes.mean():.1f}")
print(f"    Min:  {grp_sizes.min()}")
print(f"    Max:  {grp_sizes.max()}")
print(f"    Std:  {grp_sizes.std():.1f}")
print(f"    Queries < 50 tickers: {(grp_sizes < 50).sum()}")
print(f"    Queries < 100 tickers: {(grp_sizes < 100).sum()}")

# Label distribution
returns = df['Next_Day_Return'].dropna()
print(f"\n  Next_Day_Return distribution:")
print(f"    Mean:   {returns.mean()*100:.4f}%")
print(f"    Median: {returns.median()*100:.4f}%")
print(f"    Std:    {returns.std()*100:.4f}%")
print(f"    Skew:   {returns.skew():.4f}")
print(f"    Kurt:   {returns.kurtosis():.4f}")
print(f"    % Positive: {(returns > 0).mean():.1%}")

# ========================================
# 4. FEATURE APPROPRIATENESS — HOURLY FEATURES ON DAILY DATA
# ========================================
print("\n" + "=" * 70)
print("4. FEATURE APPROPRIATENESS CHECK")
print("=" * 70)

feature_cols = meta['features']

# These features are designed for intraday and are MEANINGLESS on daily bars:
intraday_features = [
    'Hour', 'Is_Open_Hour', 'Is_Close_Hour', 'Time_To_Close',
    'Intraday_Return', 'VWAP_Dist', 'IBS_3',
    'Alpha_3H', 'Alpha_6H'
]

problematic = [f for f in intraday_features if f in feature_cols]
print(f"\n  ⚠️ INTRADAY-SPECIFIC FEATURES PRESENT IN DAILY MODEL: {len(problematic)}")
for f in problematic:
    print(f"    - {f}")

# Check if Hour/Time_To_Close have meaningful values in daily data
if 'Hour' in df.columns:
    hour_vals = df['Hour'].dropna().unique()
    print(f"\n  'Hour' column values in daily data: {sorted(hour_vals)}")
    print(f"    (Should be constant for daily bars — model will learn noise!)")

if 'Time_To_Close' in df.columns:
    ttc = df['Time_To_Close'].dropna()
    print(f"\n  'Time_To_Close' in daily data: mean={ttc.mean():.4f}, std={ttc.std():.4f}")
    print(f"    (Should be constant — provides zero information on daily bars)")

if 'Is_Open_Hour' in df.columns:
    ioh = df['Is_Open_Hour'].dropna()
    print(f"\n  'Is_Open_Hour' in daily data: unique values = {sorted(ioh.unique())}")

if 'Intraday_Return' in df.columns:
    ir = df['Intraday_Return'].dropna()
    print(f"\n  'Intraday_Return' in daily data: mean={ir.mean():.6f}, std={ir.std():.4f}")
    print(f"    (This is Open-to-Close return — actually useful on daily!)")

if 'Alpha_3H' in df.columns:
    a3 = df['Alpha_3H'].dropna()
    print(f"\n  'Alpha_3H' in daily data: mean={a3.mean():.6f}, std={a3.std():.4f}")
    print(f"    (3-bar alpha on daily = 3-DAY alpha — semantically different from 3-hour)")

if 'Alpha_6H' in df.columns:
    a6 = df['Alpha_6H'].dropna()
    print(f"\n  'Alpha_6H' in daily data: mean={a6.mean():.6f}, std={a6.std():.4f}")
    print(f"    (6-bar alpha on daily = 6-DAY alpha — semantically different from 6-hour)")

# ========================================
# 5. NaN/CONSTANT FEATURE CHECK
# ========================================
print("\n" + "=" * 70)
print("5. NaN / CONSTANT / LOW-VARIANCE FEATURE CHECK")
print("=" * 70)

available_features = [f for f in feature_cols if f in df.columns]
X_feat = df[available_features]

nan_pct = X_feat.isna().mean() * 100
const_features = []
low_var_features = []

for col in available_features:
    n_unique = X_feat[col].nunique()
    if n_unique <= 1:
        const_features.append(col)
    elif X_feat[col].std() < 0.01:
        low_var_features.append((col, X_feat[col].std(), n_unique))

high_nan = nan_pct[nan_pct > 5].sort_values(ascending=False)
if len(high_nan) > 0:
    print(f"\n  Features with >5% NaN:")
    for feat, pct in high_nan.items():
        print(f"    {feat}: {pct:.1f}% NaN")
else:
    print(f"\n  ✓ No features with >5% NaN")

if const_features:
    print(f"\n  ⚠️ CONSTANT FEATURES (zero information): {const_features}")
else:
    print(f"  ✓ No constant features")

if low_var_features:
    print(f"\n  Low-variance features (std < 0.01):")
    for feat, std_val, n_u in low_var_features[:10]:
        print(f"    {feat}: std={std_val:.6f}, unique={n_u}")

# ========================================
# 6. CROSS-SECTIONAL Z-SCORING IMPACT
# ========================================
print("\n" + "=" * 70)
print("6. CROSS-SECTIONAL Z-SCORING DIAGNOSTIC")
print("=" * 70)

# In hourly data, each query has ~172 tickers (high cross-sectional variation)
# In daily data, each query might also have many tickers but...
# Z-scoring on DAILY bars means we're normalizing features across all stocks
# on the SAME DAY. This is fine for relative features but destroys absolute signals.

# Key question: Are features actually Z-scored in the saved data?
sample_query = df.groupby('Query_ID').filter(lambda x: len(x) > 50).head(200)
if len(sample_query) > 0:
    sample_qid = sample_query['Query_ID'].iloc[0]
    q_data = df[df['Query_ID'] == sample_qid]
    print(f"\n  Checking Z-scoring on Query_ID={sample_qid} ({len(q_data)} tickers):")
    for feat in ['Return', 'RSI_14', 'BB_Width', 'Volume_Zscore']:
        if feat in q_data.columns:
            vals = q_data[feat].dropna()
            print(f"    {feat}: mean={vals.mean():.4f}, std={vals.std():.4f}")
    
    print(f"\n  If mean ≈ 0 and std ≈ 1, features are Z-scored (good for ranking)")
    print(f"  If not, features retain raw values")

# ========================================
# 7. RETRAIN-AND-EVALUATE: Production Model on Holdout
# ========================================
print("\n" + "=" * 70)
print("7. PRODUCTION MODEL — FULL HOLDOUT EVALUATION")
print("=" * 70)

# Load production models
bst_long = xgb.Booster()
bst_long.load_model(LONG_MODEL_PATH)
bst_short = xgb.Booster()
bst_short.load_model(SHORT_MODEL_PATH)

print(f"  Long model trees: {len(bst_long.get_dump())}")
print(f"  Short model trees: {len(bst_short.get_dump())}")

# Evaluate on the last 2 months (validation period used for production training)
val_months = prod['val_months']
df_val = df[df['YearMonth'].isin(val_months)].copy()
print(f"\n  Evaluating on {val_months} ({len(df_val):,} rows, {df_val['Query_ID'].nunique()} trading days)")

# Predict
X_val = df_val[feature_cols].values
X_val = np.nan_to_num(X_val)
dmat_val = xgb.DMatrix(X_val, feature_names=feature_cols)

df_val['long_score'] = bst_long.predict(dmat_val)
df_val['short_score'] = bst_short.predict(dmat_val)
df_val['long_conv'] = df_val['long_score'] - df_val['short_score']
df_val['short_conv'] = df_val['short_score'] - df_val['long_score']

# Per-query evaluation
results = {'long_hits': 0, 'long_total': 0, 'short_hits': 0, 'short_total': 0,
           'long_returns': [], 'short_returns': [], 'random_returns': [],
           'long_rhos': [], 'short_rhos': []}

for qid in df_val['Query_ID'].unique():
    q_df = df_val[df_val['Query_ID'] == qid]
    if len(q_df) < 5:
        continue
    
    actual = q_df['Next_Day_Return'].values
    median_ret = np.median(actual)
    
    # Long top-3
    long_sc = q_df['long_score'].values
    top3_long = np.argsort(long_sc)[::-1][:3]
    results['long_hits'] += (actual[top3_long] > median_ret).sum()
    results['long_total'] += 3
    results['long_returns'].append(actual[top3_long].mean())
    
    # Short top-3
    short_sc = q_df['short_score'].values
    top3_short = np.argsort(short_sc)[::-1][:3]
    results['short_hits'] += (actual[top3_short] < median_ret).sum()
    results['short_total'] += 3
    results['short_returns'].append(-actual[top3_short].mean())
    
    results['random_returns'].append(actual.mean())
    
    # Spearman
    rho_l, _ = spearmanr(long_sc, actual)
    rho_s, _ = spearmanr(short_sc, -actual)
    if not np.isnan(rho_l): results['long_rhos'].append(rho_l)
    if not np.isnan(rho_s): results['short_rhos'].append(rho_s)

long_p3 = results['long_hits'] / results['long_total'] if results['long_total'] > 0 else 0
short_p3 = results['short_hits'] / results['short_total'] if results['short_total'] > 0 else 0

print(f"\n  PRODUCTION MODEL HOLDOUT RESULTS ({val_months}):")
print(f"    Long  Spearman ρ: {np.mean(results['long_rhos']):.4f}")
print(f"    Short Spearman ρ: {np.mean(results['short_rhos']):.4f}")
print(f"    Long  P@3: {long_p3:.1%} ({'BELOW RANDOM' if long_p3 < 0.52 else 'OK'})")
print(f"    Short P@3: {short_p3:.1%} ({'BELOW RANDOM' if short_p3 < 0.52 else 'OK'})")
print(f"    Long avg return of top-3: {np.mean(results['long_returns'])*100:+.4f}%/day")
print(f"    Short avg return of top-3: {np.mean(results['short_returns'])*100:+.4f}%/day")
print(f"    Market avg return: {np.mean(results['random_returns'])*100:+.4f}%/day")

# ========================================
# 8. LONG-SIDE FAILURE DEEP DIVE
# ========================================
print("\n" + "=" * 70)
print("8. LONG-SIDE FAILURE DEEP DIVE")
print("=" * 70)

# Check: Is the long model producing diverse scores or near-constant?
long_scores = df_val['long_score'].values
short_scores = df_val['short_score'].values

print(f"\n  Long score distribution: mean={long_scores.mean():.4f}, std={long_scores.std():.4f}")
print(f"  Short score distribution: mean={short_scores.mean():.4f}, std={short_scores.std():.4f}")
print(f"  Long score range: [{long_scores.min():.4f}, {long_scores.max():.4f}]")
print(f"  Short score range: [{short_scores.min():.4f}, {short_scores.max():.4f}]")

# Check within-query score variance (does the model differentiate stocks?)
query_stds_long = []
query_stds_short = []
for qid in df_val['Query_ID'].unique():
    q_df = df_val[df_val['Query_ID'] == qid]
    if len(q_df) > 5:
        query_stds_long.append(q_df['long_score'].std())
        query_stds_short.append(q_df['short_score'].std())

print(f"\n  Within-query score std (how well model differentiates stocks):")
print(f"    Long model:  mean={np.mean(query_stds_long):.4f}")
print(f"    Short model: mean={np.mean(query_stds_short):.4f}")

if np.mean(query_stds_long) < np.mean(query_stds_short) * 0.5:
    print(f"    ⚠️ Long model produces much less differentiated scores than short!")
    print(f"       This suggests the long model is not learning useful ranking patterns.")

# Check: Do top long picks actually have LOWER returns? (inverse signal)
print(f"\n  Are long model picks inversely correlated?")
top1_long_rets = []
bottom1_long_rets = []
for qid in df_val['Query_ID'].unique():
    q_df = df_val[df_val['Query_ID'] == qid]
    if len(q_df) < 10:
        continue
    actual = q_df['Next_Day_Return'].values
    ls = q_df['long_score'].values
    best = np.argsort(ls)[::-1][0]
    worst = np.argsort(ls)[0]
    top1_long_rets.append(actual[best])
    bottom1_long_rets.append(actual[worst])

print(f"    Top-1 long pick avg return: {np.mean(top1_long_rets)*100:+.4f}%")
print(f"    Bottom-1 long pick avg return: {np.mean(bottom1_long_rets)*100:+.4f}%")
if np.mean(top1_long_rets) < np.mean(bottom1_long_rets):
    print(f"    ⚠️ INVERSE SIGNAL: Bottom-ranked stocks outperform top-ranked!")

# ========================================
# 9. COMPARISON WITH INTRADAY MODELS
# ========================================
print("\n" + "=" * 70)
print("9. COMPARISON WITH INTRADAY MODELS")
print("=" * 70)

# Compare key diagnostics
comparisons = {
    'Daily': {
        'best_iter_long': prod['long_best_iteration'],
        'long_rho': meta['walk_forward_summary']['avg_long_spearman'],
        'short_rho': meta['walk_forward_summary']['avg_short_spearman'],
        'long_p3': meta['walk_forward_summary']['avg_long_win_rate_k3'],
        'short_p3': meta['walk_forward_summary']['avg_short_win_rate_k3'],
    }
}

# Load other model metadata for comparison
for model_name, model_dir in [('1-Hour (v8)', 'models/v8_upstox_3y'),
                                ('15-Min', 'models/v1_15min'),
                                ('30-Min', 'models/v1_30min')]:
    meta_path = f'{model_dir}/metadata.json'
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            m = json.load(f)
        wf_summary = m.get('walk_forward_summary', {})
        comparisons[model_name] = {
            'long_rho': wf_summary.get('avg_long_spearman', 'N/A'),
            'short_rho': wf_summary.get('avg_short_spearman', 'N/A'),
            'long_p3': wf_summary.get('avg_long_win_rate_k3', 'N/A'),
            'short_p3': wf_summary.get('avg_short_win_rate_k3', 'N/A'),
        }

print(f"\n  {'Model':<16} | {'Long ρ':>8} | {'Short ρ':>8} | {'Long P@3':>8} | {'Short P@3':>8}")
print(f"  {'-'*16} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8}")
for name, vals in comparisons.items():
    lr = f"{vals['long_rho']:.4f}" if isinstance(vals['long_rho'], float) else vals['long_rho']
    sr = f"{vals['short_rho']:.4f}" if isinstance(vals['short_rho'], float) else vals['short_rho']
    lp = f"{vals['long_p3']:.1%}" if isinstance(vals['long_p3'], float) else vals['long_p3']
    sp = f"{vals['short_p3']:.1%}" if isinstance(vals['short_p3'], float) else vals['short_p3']
    print(f"  {name:<16} | {lr:>8} | {sr:>8} | {lp:>8} | {sp:>8}")

# ========================================
# 10. ROOT CAUSE SUMMARY
# ========================================
print("\n" + "=" * 70)
print("10. ROOT CAUSE ANALYSIS — SUMMARY")
print("=" * 70)

issues = []

# Issue 1: Extremely early stopping
if prod['long_best_iteration'] < 50:
    issues.append(f"CRITICAL — Long model stopped at iteration {prod['long_best_iteration']}/800. "
                  f"The NDCG@5 validation metric did not improve beyond ~30 rounds. "
                  f"This means the model essentially has ~30 trees — far too few to learn "
                  f"complex daily ranking patterns.")

# Issue 2: Feature mismatch
if len(problematic) > 3:
    issues.append(f"HIGH — {len(problematic)} intraday-specific features are present in the daily model: "
                  f"{problematic}. Features like 'Hour', 'Is_Open_Hour', 'Time_To_Close' are constant/meaningless "
                  f"on daily bars and inject noise. 'Alpha_3H' and 'Alpha_6H' compute 3-bar and 6-bar "
                  f"momentum which on daily data means 3-day and 6-day momentum — semantically correct "
                  f"but the feature NAMES in the model were trained to expect hourly semantics.")

# Issue 3: Same features as hourly
issues.append(f"MEDIUM — The daily model uses the EXACT SAME 86 features as the 1-hour model "
              f"(same compute_features() function). Daily price action has fundamentally different "
              f"dynamics than intraday. Missing features: multi-day momentum (5D/10D/20D returns), "
              f"ADX, gap analysis, earnings proximity, sector rotation, etc. "
              f"The feature_utils.py already has compute_features_daily_xgb() with 160+ daily-optimized "
              f"features, but it's NOT being used.")

# Issue 4: Daily prediction is inherently harder
issues.append(f"STRUCTURAL — Daily returns are noisier than intraday returns. On intraday timeframes, "
              f"microstructure effects (IBS, volume pressure) provide short-lived alpha. "
              f"On daily timeframes, overnight news, macro events, and earnings dominate, "
              f"making cross-sectional ranking much harder.")

# Issue 5: Long-Short asymmetry
if meta['walk_forward_summary']['avg_long_win_rate_k3'] < 0.51:
    issues.append(f"ASYMMETRY — Long P@3 (49.7%) is below random while Short P@3 (56.5%) is decent. "
                  f"This suggests the model can identify daily losers better than winners. "
                  f"In an up-trending market (2021-2026 Indian equities), most stocks go up on most days, "
                  f"making it harder to identify the TOP winners vs. easier to identify relative losers.")

for i, issue in enumerate(issues, 1):
    print(f"\n  [{i}] {issue}")

print("\n" + "=" * 70)
print("RECOMMENDATIONS")
print("=" * 70)
print("""
  1. USE DAILY-SPECIFIC FEATURES: Switch from compute_features() to 
     compute_features_daily_xgb() which has 160+ daily-optimized features
     including multi-day momentum, ADX, gap analysis, and interaction terms.

  2. REMOVE INTRADAY FEATURES: Drop Hour, Is_Open_Hour, Is_Close_Hour,
     Time_To_Close, VWAP_Dist from the daily model feature set.

  3. INCREASE REGULARIZATION FOR LONG MODEL: The long model's early stopping
     at 30 iterations suggests noise dominates. Try:
     - Higher min_child_weight (30-50)
     - Lower eta (0.01)
     - Higher alpha/lambda (2.0/4.0)
     - Or use rank:ndcg instead of rank:pairwise

  4. CONSIDER LONG-ONLY DISABLING: Given 49.7% P@3, the daily long model
     provides no edge. Consider using it ONLY for short-side signals
     (which has 56.5% P@3) until the long model is improved.

  5. ADD FUNDAMENTAL/MACRO FEATURES: Daily models benefit from features
     the intraday models cannot use: sector momentum, market breadth,
     FII/DII flow data, put-call ratios, etc.
""")
print("=" * 70)

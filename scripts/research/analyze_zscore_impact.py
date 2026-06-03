"""
Analyze the impact of cross-sectional Z-scoring on feature predictiveness.

Goal: Quantify how much signal is destroyed by Z-scoring ALL features
within each hourly query group.

Method:
1. Load the raw Upstox 3y data (pre-Z-scored)
2. Compute per-feature Spearman correlation with Next_Hour_Return
   a) Using RAW feature values
   b) Using cross-sectionally Z-scored values
3. Compare: which features GAIN and which LOSE predictive power from Z-scoring
4. Estimate potential Spearman improvement from dual-feature approach
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm

sys.path.append(os.getcwd())

# ============================================================
# STEP 1: Load data and reconstruct raw features
# ============================================================
# We need to load per-ticker raw data from the cache and recompute
# features WITHOUT Z-scoring to measure the raw signal

print("=" * 70)
print("Z-SCORE IMPACT ANALYSIS")
print("=" * 70)

# The 3y dataset is already Z-scored. We need to reconstruct raw features.
# Use the raw cache files instead.
RAW_CACHE_DIR = "data/raw_upstox_cache"

from scripts.tickers import TICKERS
from scripts.feature_utils import compute_features

# Sample: use a representative subset of tickers for speed
SAMPLE_TICKERS = TICKERS[:30]  # First 30 tickers

print(f"\nLoading raw data for {len(SAMPLE_TICKERS)} tickers...")

all_raw = []
for ticker in tqdm(SAMPLE_TICKERS, desc="Loading raw cache"):
    cache_file = os.path.join(RAW_CACHE_DIR, f"{ticker.replace('.NS','')}.csv")
    if not os.path.exists(cache_file):
        continue
    
    try:
        df = pd.read_csv(cache_file, parse_dates=['timestamp'])
        if len(df) < 100:
            continue
        
        # Resample 30min -> 1hr
        df = df.set_index('timestamp')
        df_1h = df.resample('1h', origin='start_day').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum'
        }).dropna(subset=['open', 'close'])
        
        # Filter market hours
        if df_1h.index.tz is not None:
            df_1h.index = df_1h.index.tz_convert('Asia/Kolkata')
        df_1h = df_1h[(df_1h.index.hour >= 9) & (df_1h.index.hour < 16)]
        
        if len(df_1h) < 50:
            continue
        
        # Rename columns
        df_1h = df_1h.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'volume': 'Volume'
        })
        
        # Compute features (raw, un-Z-scored)
        df_feat = compute_features(df_1h, legacy=False)
        df_feat['Next_Hour_Return'] = df_feat['Close'].shift(-1) / df_feat['Close'] - 1
        df_feat['Ticker'] = ticker
        df_feat['DateTime_Hour'] = df_feat.index.floor('h')
        
        all_raw.append(df_feat)
    except Exception as e:
        continue

if not all_raw:
    print("[FATAL] No raw data loaded. Check raw_upstox_cache directory.")
    sys.exit(1)

df_raw = pd.concat(all_raw, ignore_index=False)
df_raw = df_raw.dropna(subset=['Next_Hour_Return'])

# Create Query_ID
df_raw['Query_ID'] = df_raw.groupby('DateTime_Hour').ngroup()

# Filter queries with >= 5 tickers
query_sizes = df_raw.groupby('Query_ID').size()
valid_queries = query_sizes[query_sizes >= 5].index
df_raw = df_raw[df_raw['Query_ID'].isin(valid_queries)].copy()

print(f"\nDataset: {len(df_raw):,} rows, {df_raw['Query_ID'].nunique():,} queries, "
      f"{df_raw['Ticker'].nunique()} tickers")

# ============================================================
# STEP 2: Define feature columns
# ============================================================
exclude_cols = {
    'DateTime', 'DateTime_Hour', 'Query_ID', 'Ticker', 'Next_Hour_Return',
    'Open', 'High', 'Low', 'Close', 'Volume',
    'Market_Mean_Return', 'Relative_Return',
    'Market_Mean_Volatility', 'Relative_Volatility',
}
feature_cols = [c for c in df_raw.columns if c not in exclude_cols 
                and df_raw[c].dtype in ['float64', 'float32', 'int64', 'int32']]

print(f"Feature columns: {len(feature_cols)}")

# ============================================================
# STEP 3: Compute Spearman IC for RAW features
# ============================================================
print("\n" + "=" * 70)
print("COMPUTING RAW FEATURE INFORMATION COEFFICIENTS")
print("=" * 70)

def compute_ic_per_query(df, feature_col, target_col='Next_Hour_Return'):
    """Compute average Spearman IC across all queries for a single feature."""
    ics = []
    for qid, group in df.groupby('Query_ID'):
        if len(group) < 5:
            continue
        feat_vals = group[feature_col].values
        target_vals = group[target_col].values
        
        # Skip if constant
        if np.std(feat_vals) < 1e-10 or np.std(target_vals) < 1e-10:
            continue
        
        # Skip if too many NaN/Inf
        valid = np.isfinite(feat_vals) & np.isfinite(target_vals)
        if valid.sum() < 5:
            continue
        
        corr, _ = spearmanr(feat_vals[valid], target_vals[valid])
        if not np.isnan(corr):
            ics.append(corr)
    
    if not ics:
        return 0.0, 0.0, 0
    return np.mean(ics), np.std(ics), len(ics)

raw_ics = {}
for col in tqdm(feature_cols, desc="Raw IC"):
    mean_ic, std_ic, count = compute_ic_per_query(df_raw, col)
    raw_ics[col] = {'mean_ic': mean_ic, 'std_ic': std_ic, 'count': count}

# ============================================================
# STEP 4: Z-score features and compute IC again
# ============================================================
print("\n" + "=" * 70)
print("Z-SCORING FEATURES AND RECOMPUTING IC")
print("=" * 70)

df_zscored = df_raw.copy()
# Apply the same Z-scoring as the training pipeline
for col in tqdm(feature_cols, desc="Z-Scoring"):
    if col in ['Hour', 'DayOfWeek', 'Is_Open_Hour', 'Is_Close_Hour', 'Time_To_Close']:
        continue  # Skip time features as the pipeline does
    grp_mean = df_zscored.groupby('Query_ID')[col].transform('mean')
    grp_std = df_zscored.groupby('Query_ID')[col].transform('std')
    df_zscored[col] = (df_zscored[col] - grp_mean) / (grp_std + 1e-8)

zscored_ics = {}
for col in tqdm(feature_cols, desc="Z-Scored IC"):
    mean_ic, std_ic, count = compute_ic_per_query(df_zscored, col)
    zscored_ics[col] = {'mean_ic': mean_ic, 'std_ic': std_ic, 'count': count}

# ============================================================
# STEP 5: Compare and report
# ============================================================
print("\n" + "=" * 70)
print("RESULTS: RAW vs Z-SCORED INFORMATION COEFFICIENTS")
print("=" * 70)

results = []
for col in feature_cols:
    raw_ic = raw_ics[col]['mean_ic']
    z_ic = zscored_ics[col]['mean_ic']
    delta = z_ic - raw_ic
    abs_delta = abs(z_ic) - abs(raw_ic)  # Did Z-scoring increase or decrease |IC|?
    results.append({
        'Feature': col,
        'Raw_IC': raw_ic,
        'ZScored_IC': z_ic,
        'Delta_IC': delta,
        'Abs_Delta': abs_delta,
        'Signal_Change': 'IMPROVED' if abs_delta > 0.001 else ('DESTROYED' if abs_delta < -0.001 else 'UNCHANGED')
    })

results_df = pd.DataFrame(results).sort_values('Abs_Delta')

# Features where Z-scoring DESTROYED signal
destroyed = results_df[results_df['Signal_Change'] == 'DESTROYED'].sort_values('Abs_Delta')
improved = results_df[results_df['Signal_Change'] == 'IMPROVED'].sort_values('Abs_Delta', ascending=False)
unchanged = results_df[results_df['Signal_Change'] == 'UNCHANGED']

print(f"\n{'=' * 70}")
print(f"SIGNAL DESTROYED BY Z-SCORING ({len(destroyed)} features)")
print(f"{'=' * 70}")
print(f"{'Feature':<30} {'Raw |IC|':>10} {'ZScored |IC|':>12} {'Lost':>10}")
print("-" * 65)
for _, row in destroyed.iterrows():
    print(f"{row['Feature']:<30} {abs(row['Raw_IC']):>10.4f} {abs(row['ZScored_IC']):>12.4f} {row['Abs_Delta']:>10.4f}")

print(f"\n{'=' * 70}")
print(f"SIGNAL IMPROVED BY Z-SCORING ({len(improved)} features)")
print(f"{'=' * 70}")
print(f"{'Feature':<30} {'Raw |IC|':>10} {'ZScored |IC|':>12} {'Gained':>10}")
print("-" * 65)
for _, row in improved.iterrows():
    print(f"{row['Feature']:<30} {abs(row['Raw_IC']):>10.4f} {abs(row['ZScored_IC']):>12.4f} {row['Abs_Delta']:>10.4f}")

print(f"\n{'=' * 70}")
print(f"SIGNAL UNCHANGED ({len(unchanged)} features)")
print(f"{'=' * 70}")

# ============================================================
# STEP 6: Estimate combined IC potential
# ============================================================
print(f"\n{'=' * 70}")
print("ESTIMATED IMPROVEMENT POTENTIAL")
print(f"{'=' * 70}")

# For each feature, take the BEST IC (raw or z-scored)
total_raw_signal = sum(abs(raw_ics[c]['mean_ic']) for c in feature_cols)
total_z_signal = sum(abs(zscored_ics[c]['mean_ic']) for c in feature_cols)
total_best_signal = sum(max(abs(raw_ics[c]['mean_ic']), abs(zscored_ics[c]['mean_ic'])) for c in feature_cols)
total_lost = sum(abs(row['Abs_Delta']) for _, row in destroyed.iterrows())

print(f"  Total |IC| (raw features only):       {total_raw_signal:.4f}")
print(f"  Total |IC| (Z-scored features only):   {total_z_signal:.4f}")
print(f"  Total |IC| (best of both per feature): {total_best_signal:.4f}")
print(f"  Total signal lost by Z-scoring:         {total_lost:.4f}")
print(f"  Potential recovery (%):                 {(total_best_signal/total_z_signal - 1)*100:.1f}%")

# Top 20 features by BEST IC
print(f"\n{'=' * 70}")
print("TOP 20 FEATURES BY BEST AVAILABLE IC")
print(f"{'=' * 70}")
best_features = []
for col in feature_cols:
    raw_abs = abs(raw_ics[col]['mean_ic'])
    z_abs = abs(zscored_ics[col]['mean_ic'])
    best = max(raw_abs, z_abs)
    source = 'RAW' if raw_abs > z_abs else 'Z-SCORED'
    best_features.append({'Feature': col, 'Best_IC': best, 'Source': source, 
                         'Raw_IC': raw_ics[col]['mean_ic'], 'Z_IC': zscored_ics[col]['mean_ic']})

best_df = pd.DataFrame(best_features).sort_values('Best_IC', ascending=False)
print(f"{'Feature':<30} {'Best |IC|':>10} {'From':>10} {'Raw IC':>10} {'Z IC':>10}")
print("-" * 75)
for _, row in best_df.head(20).iterrows():
    print(f"{row['Feature']:<30} {row['Best_IC']:>10.4f} {row['Source']:>10} {row['Raw_IC']:>10.4f} {row['Z_IC']:>10.4f}")

# Count how many of the top 20 have RAW as best source
raw_best_count = len(best_df.head(20)[best_df.head(20)['Source'] == 'RAW'])
print(f"\nOf top 20 features: {raw_best_count} are better RAW, {20-raw_best_count} better Z-SCORED")

# Save results
results_df.to_csv('data/zscore_impact_analysis.csv', index=False)
best_df.to_csv('data/best_features_analysis.csv', index=False)
print(f"\nResults saved to data/zscore_impact_analysis.csv and data/best_features_analysis.csv")

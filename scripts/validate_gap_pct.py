"""
Validate Gap_Pct feature: Is the IC real across all hours, or concentrated in hour 1?

Tests:
1. IC breakdown by hour of day (9, 10, 11, 12, 13, 14, 15)
2. IC when excluding first hour of day
3. Distribution analysis of Gap_Pct values
4. Correlation with other features (redundancy check)
5. Stability across time (rolling IC)
"""

import os, sys
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm

sys.path.append(os.getcwd())
from scripts.tickers import TICKERS
from scripts.feature_utils import compute_features

RAW_CACHE_DIR = "data/raw_upstox_cache"
SAMPLE_TICKERS = TICKERS[:30]

print("=" * 70)
print("GAP_PCT VALIDATION ANALYSIS")
print("=" * 70)

# Load data
all_raw = []
for ticker in tqdm(SAMPLE_TICKERS, desc="Loading data"):
    cache_file = os.path.join(RAW_CACHE_DIR, f"{ticker.replace('.NS','')}.csv")
    if not os.path.exists(cache_file):
        continue
    try:
        df = pd.read_csv(cache_file, parse_dates=['timestamp'])
        if len(df) < 100:
            continue
        df = df.set_index('timestamp')
        df_1h = df.resample('1h', origin='start_day').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum'
        }).dropna(subset=['open', 'close'])
        if df_1h.index.tz is not None:
            df_1h.index = df_1h.index.tz_convert('Asia/Kolkata')
        df_1h = df_1h[(df_1h.index.hour >= 9) & (df_1h.index.hour < 16)]
        if len(df_1h) < 50:
            continue
        df_1h = df_1h.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'volume': 'Volume'
        })
        df_feat = compute_features(df_1h, legacy=False)
        df_feat['Next_Hour_Return'] = df_feat['Close'].shift(-1) / df_feat['Close'] - 1
        df_feat['Ticker'] = ticker
        df_feat['DateTime_Hour'] = df_feat.index.floor('h')
        df_feat['Hour'] = df_feat.index.hour
        
        # Compute Gap_Pct
        date_int = df_feat.index.year * 10000 + df_feat.index.month * 100 + df_feat.index.day
        day_open = df_feat.groupby(date_int)['Open'].transform('first')
        prev_day_close = df_feat.groupby(date_int)['Close'].transform('last')
        # Shift prev_day_close by 1 trading day  
        # For each day, get the previous day's closing price
        daily_close = df_feat.groupby(date_int)['Close'].last()
        daily_close_shifted = daily_close.shift(1)
        # Map back
        df_feat['Prev_Day_Close'] = pd.Series(date_int, index=df_feat.index).map(daily_close_shifted)
        df_feat['Day_Open'] = day_open
        df_feat['Gap_Pct'] = (day_open / (df_feat['Prev_Day_Close'] + 1e-8)) - 1
        
        # Mark first hour of day
        first_hour_mask = df_feat.groupby(date_int).cumcount() == 0
        df_feat['Is_First_Hour'] = first_hour_mask.astype(int)
        
        all_raw.append(df_feat)
    except Exception as e:
        continue

df_all = pd.concat(all_raw, ignore_index=False)
df_all = df_all.dropna(subset=['Next_Hour_Return', 'Gap_Pct'])
df_all['Query_ID'] = df_all.groupby('DateTime_Hour').ngroup()

# Filter queries with >= 5 tickers
query_sizes = df_all.groupby('Query_ID').size()
valid_queries = query_sizes[query_sizes >= 5].index
df_all = df_all[df_all['Query_ID'].isin(valid_queries)].copy()

print(f"\nDataset: {len(df_all):,} rows, {df_all['Query_ID'].nunique():,} queries")
print(f"Gap_Pct stats: mean={df_all['Gap_Pct'].mean():.5f}, std={df_all['Gap_Pct'].std():.5f}")
print(f"  min={df_all['Gap_Pct'].min():.4f}, max={df_all['Gap_Pct'].max():.4f}")
print(f"  median={df_all['Gap_Pct'].median():.5f}")

# ============================================================
# TEST 1: IC by Hour of Day
# ============================================================
print(f"\n{'=' * 70}")
print("TEST 1: GAP_PCT IC BY HOUR OF DAY")
print(f"{'=' * 70}")

def compute_ic_for_subset(df_subset, feature_col='Gap_Pct'):
    ics = []
    for qid, group in df_subset.groupby('Query_ID'):
        if len(group) < 5:
            continue
        feat = group[feature_col].values
        target = group['Next_Hour_Return'].values
        valid = np.isfinite(feat) & np.isfinite(target)
        if valid.sum() < 5 or np.std(feat[valid]) < 1e-10:
            continue
        corr, _ = spearmanr(feat[valid], target[valid])
        if not np.isnan(corr):
            ics.append(corr)
    if not ics:
        return 0.0, 0.0, 0, 0.0
    mean_ic = np.mean(ics)
    std_ic = np.std(ics)
    count = len(ics)
    t_stat = abs(mean_ic) / (std_ic / np.sqrt(count) + 1e-8) if count > 0 else 0
    return mean_ic, std_ic, count, t_stat

print(f"\n{'Hour':>6} {'Count':>8} {'Mean IC':>10} {'Std IC':>10} {'ICIR':>8} {'t-stat':>8} {'Sig':>6}")
print("-" * 60)

for hour in sorted(df_all['Hour'].unique()):
    subset = df_all[df_all['Hour'] == hour]
    mean_ic, std_ic, count, t_stat = compute_ic_for_subset(subset)
    icir = abs(mean_ic) / (std_ic + 1e-8)
    sig = "***" if t_stat > 3.0 else ("**" if t_stat > 2.0 else ("*" if t_stat > 1.5 else ""))
    print(f"{hour:>6} {count:>8} {mean_ic:>10.5f} {std_ic:>10.5f} {icir:>8.4f} {t_stat:>8.2f} {sig:>6}")

# ============================================================
# TEST 2: IC excluding first hour
# ============================================================
print(f"\n{'=' * 70}")
print("TEST 2: GAP_PCT IC — FIRST HOUR vs REST OF DAY")
print(f"{'=' * 70}")

first_hour = df_all[df_all['Is_First_Hour'] == 1]
rest_of_day = df_all[df_all['Is_First_Hour'] == 0]

mean_ic_first, std_ic_first, count_first, t_first = compute_ic_for_subset(first_hour)
mean_ic_rest, std_ic_rest, count_rest, t_rest = compute_ic_for_subset(rest_of_day)

print(f"\n  First hour only:  IC = {mean_ic_first:+.5f}  ICIR = {abs(mean_ic_first)/(std_ic_first+1e-8):.4f}  t = {t_first:.2f}  (n={count_first})")
print(f"  Rest of day:      IC = {mean_ic_rest:+.5f}  ICIR = {abs(mean_ic_rest)/(std_ic_rest+1e-8):.4f}  t = {t_rest:.2f}  (n={count_rest})")

# ============================================================
# TEST 3: Does Gap_Pct VARY within a day across stocks?
# ============================================================
print(f"\n{'=' * 70}")
print("TEST 3: CROSS-SECTIONAL VARIATION OF GAP_PCT")
print(f"{'=' * 70}")

# For each query, how much does Gap_Pct vary across stocks?
gap_variation = df_all.groupby('Query_ID')['Gap_Pct'].agg(['std', 'mean', 'count'])
print(f"\n  Avg cross-sectional std of Gap_Pct: {gap_variation['std'].mean():.5f}")
print(f"  Avg cross-sectional mean of Gap_Pct: {gap_variation['mean'].mean():.5f}")
print(f"  Ratio (std/|mean|): {gap_variation['std'].mean() / (abs(gap_variation['mean'].mean()) + 1e-8):.2f}")
print(f"  -> {'GOOD: Sufficient cross-sectional variation for ranking' if gap_variation['std'].mean() > 0.001 else 'BAD: Too little variation'}")

# Does the variation change by hour?
print(f"\n  Cross-sectional std by hour:")
df_all['Gap_Std_Group'] = df_all.groupby('Query_ID')['Gap_Pct'].transform('std')
for hour in sorted(df_all['Hour'].unique()):
    subset = df_all[df_all['Hour'] == hour]
    mean_std = subset['Gap_Std_Group'].mean()
    print(f"    Hour {hour}: avg std = {mean_std:.5f}")

# ============================================================
# TEST 4: Correlation with existing features
# ============================================================
print(f"\n{'=' * 70}")
print("TEST 4: CORRELATION WITH EXISTING FEATURES")
print(f"{'=' * 70}")

existing_features = ['Return', 'HL_Range', 'OC_Range', 'RSI_14', 'PercentB', 
                     'VWAP_Dist', 'Intraday_Return', 'Price_Zscore', 'ROC_12']

print(f"\n  {'Feature':<25} {'Corr with Gap_Pct':>20}")
print(f"  {'-'*50}")
for feat in existing_features:
    if feat in df_all.columns:
        valid = df_all[['Gap_Pct', feat]].dropna()
        if len(valid) > 100:
            corr = valid['Gap_Pct'].corr(valid[feat])
            redundant = "-> REDUNDANT" if abs(corr) > 0.7 else ""
            print(f"  {feat:<25} {corr:>20.4f} {redundant}")

# ============================================================
# TEST 5: Rolling IC stability (quarterly)
# ============================================================
print(f"\n{'=' * 70}")
print("TEST 5: IC STABILITY OVER TIME (QUARTERLY)")
print(f"{'=' * 70}")

df_all['Date'] = pd.to_datetime(df_all.index)
df_all['Quarter'] = df_all['Date'].dt.to_period('Q')

print(f"\n  {'Quarter':>10} {'IC':>10} {'t-stat':>8} {'Queries':>8}")
print(f"  {'-'*40}")
for q in sorted(df_all['Quarter'].unique()):
    subset = df_all[df_all['Quarter'] == q]
    mean_ic, std_ic, count, t_stat = compute_ic_for_subset(subset)
    sig = "***" if t_stat > 3.0 else ("**" if t_stat > 2.0 else ("*" if t_stat > 1.5 else ""))
    print(f"  {str(q):>10} {mean_ic:>10.5f} {t_stat:>8.2f} {count:>8} {sig}")

# ============================================================
# VERDICT
# ============================================================
print(f"\n{'=' * 70}")
print("VERDICT")
print(f"{'=' * 70}")

if t_rest > 3.0 and abs(mean_ic_rest) > 0.01:
    print("\n  [OK] Gap_Pct signal PERSISTS beyond first hour")
    print(f"     First hour IC: {mean_ic_first:+.5f}")
    print(f"     Rest of day IC: {mean_ic_rest:+.5f}")
    print(f"     -> SAFE TO ADD to feature set")
elif t_rest > 2.0:
    print("\n  [WARN] Gap_Pct signal WEAKENS after first hour but still significant")
    print(f"     First hour IC: {mean_ic_first:+.5f}")
    print(f"     Rest of day IC: {mean_ic_rest:+.5f}")
    print(f"     -> ADD but consider hour-interaction feature")
else:
    print("\n  [FAIL] Gap_Pct signal is CONCENTRATED in first hour only")
    print(f"     First hour IC: {mean_ic_first:+.5f}")
    print(f"     Rest of day IC: {mean_ic_rest:+.5f}")
    print(f"     -> Only useful for first hour predictions, or needs to be")
    print(f"       transformed (e.g., Gap_Pct * Is_First_Hour)")

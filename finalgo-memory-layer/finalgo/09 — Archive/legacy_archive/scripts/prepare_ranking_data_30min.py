"""
Prepare ranking data for the 30-minute standalone XGBoost model.
- Loads cached 15-minute candles from data/raw_upstox_cache_15min/
- Resamples to 30-minute intervals aligned on 15-minute offsets (09:15, 09:45...)
- Computes standard + 30-min specific features using compute_features_30min()
- Creates Next_30Min_Return labels and Query_IDs (grouped by 30-minute floor)
- Excludes temporal, raw volume, and target columns from cross-sectional Z-scoring
- Applies cross-sectional Z-scoring per Query_ID
- Saves to data/ranking_data_upstox_30min_1y.csv
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.append(os.getcwd())

from scripts.feature_utils import compute_features_30min

# ========================================
# CONFIG
# ========================================
INPUT_CACHE_DIR = "data/raw_upstox_cache_15min"
OUTPUT_CSV = "data/ranking_data_upstox_30min_1y.csv"

print("=" * 60)
print("30-MIN DATA PIPELINE — Feature Engineering & Preparation")
print("=" * 60)

if not os.path.exists(INPUT_CACHE_DIR) or not os.listdir(INPUT_CACHE_DIR):
    print(f"[FATAL] Cache directory {INPUT_CACHE_DIR} is empty or does not exist.")
    print("Please run scripts/collect_upstox_15min_1y.py first to fetch raw candles.")
    sys.exit(1)

# Get list of all cache CSVs
cache_files = glob.glob(os.path.join(INPUT_CACHE_DIR, "*.csv"))
print(f"Found cache files for {len(cache_files)} tickers.")
print()

# ========================================
# COMPUTE FEATURES PER TICKER
# ========================================
all_data = []

for file_path in tqdm(cache_files, desc="Processing Tickers"):
    ticker_name = os.path.basename(file_path).replace(".csv", "") + ".NS"
    try:
        df = pd.read_csv(file_path, parse_dates=['timestamp'])
        
        if df.empty:
            continue
            
        # Resample to 30min (Indian market starts at 09:15, so align using offset='15min')
        df = df.set_index('timestamp')
        df = df.resample('30min', origin='start_day', offset='15min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna(subset=['open'])
        
        # Standardize column names
        col_map = {
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'volume': 'Volume'
        }
        df = df.rename(columns=col_map)
        
        # Ensure correct column ordering and type conversion
        df.index.name = 'DateTime'
        df['DateTime'] = df.index
        
        # Drop rows with missing values in price/volume
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
        
        if len(df) < 50:
            continue
            
        df['Ticker'] = ticker_name
        df['DateTime'] = df.index  # Keep a column version of DateTime
        
        # Compute 30-minute features
        df = compute_features_30min(df, legacy=False)
        
        # Target: return of the next 30-min bar
        df['Next_30Min_Return'] = df['Close'].shift(-1) / df['Close'] - 1
        
        all_data.append(df)
        
    except Exception as e:
        print(f"[ERROR] Failed to process {ticker_name}: {e}")

if not all_data:
    print("[FATAL] No data was processed successfully.")
    sys.exit(1)

# ========================================
# COMBINE & CREATE QUERY IDS
# ========================================
print("\nCombining all tickers...")
df_all = pd.concat(all_data, ignore_index=True)

# Drop rows with NaN labels (the last row per ticker won't have next return)
df_all = df_all.dropna(subset=['Next_30Min_Return'])

print(f"Total rows after combining: {len(df_all):,}")

# Create query_id: group by 30-minute floor
df_all['DateTime'] = pd.to_datetime(df_all['DateTime'])
df_all['DateTime_30Min'] = df_all['DateTime'].dt.floor('30min')

# Sort chronologically to preserve temporal order
df_all = df_all.sort_values('DateTime_30Min')
df_all['Query_ID'] = df_all.groupby('DateTime_30Min').ngroup()

# Filter: only keep queries with >= 5 tickers for robust relative ranking
query_sizes = df_all.groupby('Query_ID').size()
valid_queries = query_sizes[query_sizes >= 5].index
df_all = df_all[df_all['Query_ID'].isin(valid_queries)].copy()

# Re-index Query_IDs sequentially after filtering
df_all = df_all.sort_values('DateTime_30Min')
df_all['Query_ID'] = df_all.groupby('DateTime_30Min').ngroup()

print(f"  Queries with >= 5 tickers: {df_all['Query_ID'].nunique():,}")
print(f"  Avg tickers per query:     {df_all.groupby('Query_ID').size().mean():.1f}")

# ========================================
# ADD MARKET CONTEXT FEATURES
# ========================================
print("\nAdding Market Context Features...")
df_all['Market_Mean_Return'] = df_all.groupby('Query_ID')['Return'].transform('mean')
df_all['Relative_Return'] = df_all['Return'] - df_all['Market_Mean_Return']
df_all['Market_Mean_Volatility'] = df_all.groupby('Query_ID')['HL_Range'].transform('mean')
df_all['Relative_Volatility'] = df_all['HL_Range'] / (df_all['Market_Mean_Volatility'] + 1e-8)

# ========================================
# CROSS-SECTIONAL Z-SCORING
# ========================================
print("\nApplying Cross-Sectional Z-Scoring...")

# Exclude list: targets, identifiers, raw price/vol, temporal features, and raw z-scores
exclude_cols = [
    'DateTime', 'DateTime_30Min', 'Query_ID', 'Ticker', 'Next_30Min_Return',
    'Open', 'High', 'Low', 'Close', 'Volume',
    'Market_Mean_Return', 'Relative_Return',
    'Market_Mean_Volatility', 'Relative_Volatility',
    'Hour', 'DayOfWeek', 'Is_Open_Hour', 'Is_Close_Hour', 'Time_To_Close',
    'Intra_Hour_Position', 'Raw_Volume_Zscore'
]
feature_cols = [col for col in df_all.columns if col not in exclude_cols]

for col in tqdm(feature_cols, desc="Z-Scoring"):
    mean = df_all.groupby('Query_ID')[col].transform('mean')
    std = df_all.groupby('Query_ID')[col].transform('std')
    df_all[col] = (df_all[col] - mean) / (std + 1e-8)

# Drop any remaining NaN values in features
df_all = df_all.dropna(subset=feature_cols)

# ========================================
# DATA QUALITY REPORT
# ========================================
print(f"\n{'=' * 60}")
print("DATA QUALITY REPORT")
print(f"{'=' * 60}")

nan_count = df_all[feature_cols].isna().sum().sum()
inf_count = np.isinf(df_all[feature_cols].select_dtypes(include=[np.number])).sum().sum()
print(f"  NaN values: {nan_count}")
print(f"  Inf values: {inf_count}")

returns = df_all['Next_30Min_Return']
print(f"  Label (Next_30Min_Return) stats:")
print(f"    Mean:   {returns.mean():.6f}")
print(f"    Median: {returns.median():.6f}")
print(f"    Std:    {returns.std():.6f}")
print(f"    Min:    {returns.min():.6f}")
print(f"    Max:    {returns.max():.6f}")

dates = df_all['DateTime'].sort_values()
print(f"  Date range:   {dates.iloc[0]} -> {dates.iloc[-1]}")
print(f"  Trading days: {df_all['DateTime'].dt.date.nunique()}")

# ========================================
# SAVE
# ========================================
print(f"\nSaving dataset to {OUTPUT_CSV}...")
df_all.to_csv(OUTPUT_CSV, index=False)

# Log info
print(f"\n{'=' * 60}")
print(f"[SUCCESS] SAVED: {OUTPUT_CSV}")
print(f"   Rows: {df_all.shape[0]:,} | Cols: {df_all.shape[1]}")
print(f"   Queries: {df_all['Query_ID'].nunique():,}")
print(f"   Features: {len(feature_cols)}")
print(f"{'=' * 60}")
print()

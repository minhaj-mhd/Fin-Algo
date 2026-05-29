#!/usr/bin/env python
"""
Upstox Training Data Compiler
Reads all daily Parquets from data/historical/,
runs feature engineering, calculates targets, assigns Query IDs,
performs Z-scoring, and saves to data/ranking_data_upstox.csv
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.append(os.getcwd())

from scripts.feature_utils import compute_features

OUTPUT_CSV = "data/ranking_data_upstox.csv"

def compile_dataset():
    print("=" * 60)
    print("COMPILING TRAINING DATASET FROM PARQUET CACHE")
    print("=" * 60)

    # Find all Parquet files
    parquet_files = sorted(glob.glob("data/historical/*.parquet"))
    if not parquet_files:
        print("[ERROR] No Parquet files found in data/historical/. Run scripts/data_collector.py first.")
        return

    print(f"Found {len(parquet_files)} daily Parquet files.")
    
    # Load all files
    dfs = []
    for file in tqdm(parquet_files, desc="Reading Parquet files"):
        try:
            df = pd.read_parquet(file)
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    if not dfs:
        print("[FATAL] All Parquet files were empty or failed to load.")
        return

    df_raw = pd.concat(dfs, ignore_index=True)
    df_raw['DateTime'] = pd.to_datetime(df_raw['DateTime'])
    df_raw = df_raw.sort_values(['Ticker', 'DateTime']).reset_index(drop=True)

    print(f"Loaded {len(df_raw):,} raw rows across {df_raw['Ticker'].nunique()} tickers.")
    
    # Run feature engineering per ticker
    processed_dfs = []
    grouped = df_raw.groupby('Ticker')
    
    for ticker, group in tqdm(grouped, desc="Computing features per ticker"):
        try:
            # Set index to DateTime as expected by compute_features
            group_idx = group.set_index('DateTime').sort_index()
            
            # Ensure no duplicates
            group_idx = group_idx[~group_idx.index.duplicated(keep='first')]
            
            if len(group_idx) < 25:
                # Need enough history for indicators like EMA/SMA/RSI
                continue
                
            # Compute features using standardized legacy=False
            df_feat = compute_features(group_idx, legacy=False)
            
            # Calculate Next_Hour_Return (label)
            df_feat['Next_Hour_Return'] = df_feat['Close'].shift(-1) / df_feat['Close'] - 1
            
            df_feat['Ticker'] = ticker
            df_feat['DateTime'] = df_feat.index
            
            processed_dfs.append(df_feat.reset_index(drop=True))
        except Exception as e:
            print(f"\nError processing ticker {ticker}: {e}")

    if not processed_dfs:
        print("[FATAL] No tickers successfully processed. Aborting.")
        return

    df_all = pd.concat(processed_dfs, ignore_index=True)
    df_all.dropna(subset=['Next_Hour_Return'], inplace=True) # Drop the last bar where label is NaN

    # Group by hourly timestamp to create Query_IDs
    df_all['DateTime_Hour'] = df_all['DateTime'].dt.floor('h')
    df_all['Query_ID'] = df_all.groupby('DateTime_Hour').ngroup()
    
    # Filter out query groups with too few tickers to avoid Z-scoring anomalies
    min_tickers_per_query = 5
    query_sizes = df_all.groupby('Query_ID').size()
    valid_queries = query_sizes[query_sizes >= min_tickers_per_query].index
    df_all = df_all[df_all['Query_ID'].isin(valid_queries)].copy()
    
    # Re-index Query_IDs to be contiguous starting from 0
    df_all['Query_ID'] = df_all.groupby('DateTime_Hour').ngroup()

    print(f"\nProcessed dataset statistics:")
    print(f"  Total rows: {df_all.shape[0]:,}")
    print(f"  Unique hours (queries): {df_all['Query_ID'].nunique():,}")
    print(f"  Avg tickers per query: {df_all.groupby('Query_ID').size().mean():.1f}")

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
    print("Applying Cross-Sectional Z-Scoring...")
    
    exclude_cols = [
        'DateTime', 'DateTime_Hour', 'Query_ID', 'Ticker', 'Next_Hour_Return',
        'Open', 'High', 'Low', 'Close', 'Volume',
        'Market_Mean_Return', 'Relative_Return',
        'Market_Mean_Volatility', 'Relative_Volatility',
        'Hour', 'DayOfWeek'
    ]
    feature_cols = [col for col in df_all.columns if col not in exclude_cols]

    for col in tqdm(feature_cols, desc="Z-Scoring"):
        mean = df_all.groupby('Query_ID')[col].transform('mean')
        std = df_all.groupby('Query_ID')[col].transform('std')
        df_all[col] = (df_all[col] - mean) / (std + 1e-8)

    # Drop any remaining NaNs in features
    df_all.dropna(inplace=True)

    # Save to CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_all.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n{'=' * 60}")
    print(f"[SUCCESS] SAVED: {OUTPUT_CSV}")
    print(f"   Rows: {df_all.shape[0]:,} | Cols: {df_all.shape[1]}")
    print(f"   Queries: {df_all['Query_ID'].nunique():,}")
    print(f"   Tickers/Query: {df_all.groupby('Query_ID').size().mean():.1f}")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    compile_dataset()

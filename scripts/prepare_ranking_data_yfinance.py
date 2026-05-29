"""
Prepare ranking data using Yahoo Finance Historical API (2 Years Lookback)
- Fetches 730 days of 1-hour candles from Yahoo Finance
- Converts timestamps to Indian Standard Time (Asia/Kolkata)
- Filters to market hours (9:15 AM - 3:30 PM IST)
- Computes all features using the shared compute_features(..., legacy=False) module
- Creates Next_Hour_Return labels, Query_IDs, and cross-sectional Z-scoring
- Saves to data/ranking_data_yfinance_2y.csv
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm

sys.path.append(os.getcwd())

from scripts.tickers import TICKERS
from scripts.feature_utils import compute_features

OUTPUT_CSV = "data/ranking_data_yfinance_2y.csv"
LOOKBACK_PERIOD = "730d"  # 2 years (max available for 1h interval)
RATE_LIMIT_PAUSE = 0.1  # seconds between API calls to avoid throttling

def main():
    print("=" * 70)
    print("YFINANCE DATA PIPELINE — 2-Year Hourly Ranking Data")
    print("=" * 70)

    tickers = TICKERS
    print(f"Tickers count: {len(tickers)}")
    print(f"Lookback: {LOOKBACK_PERIOD}")
    print(f"Interval: 1h")
    print()

    all_data = []
    failed_tickers = []

    for i, ticker in enumerate(tqdm(tickers, desc="Fetching yfinance Data")):
        try:
            # Download hourly data
            df = yf.download(ticker, period=LOOKBACK_PERIOD, interval="1h", progress=False, auto_adjust=True)
            
            if df is None or df.empty:
                raise Exception("Empty DataFrame returned")

            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]

            # Ensure we have the required columns
            required = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(c in df.columns for c in required):
                raise Exception(f"Missing columns: {set(required) - set(df.columns)}")

            df = df[required].copy()

            # Convert timezone to Asia/Kolkata (IST) to correctly slice trading hours
            if df.index.tz is not None:
                df.index = df.index.tz_convert('Asia/Kolkata')
            else:
                df.index = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')

            # Filter to NSE market hours only (9:15 AM - 3:30 PM IST)
            # DatetimeIndex.hour will be 9, 10, 11, 12, 13, 14, 15
            df = df[(df.index.hour >= 9) & (df.index.hour < 16)]

            if len(df) < 25:
                raise Exception(f"Only {len(df)} bars — need at least 25 for feature computation")

            df['Ticker'] = ticker
            df['DateTime'] = df.index  # Preserve for grouping

            # Compute features using the corrected math formulas
            df = compute_features(df, legacy=False)

            # Create label: Next hour return
            df['Next_Hour_Return'] = df['Close'].shift(-1) / df['Close'] - 1

            all_data.append(df)

        except Exception as e:
            failed_tickers.append((ticker, str(e)))

        # Respect API rate limits
        if i % 10 == 0:
            time.sleep(RATE_LIMIT_PAUSE)

    # Report download stats
    print(f"\n{'=' * 70}")
    print("DATA FETCH COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Successfully fetched: {len(all_data)} / {len(tickers)} tickers")
    print(f"  Failed:               {len(failed_tickers)}")
    
    if failed_tickers:
        print(f"\n  Failed tickers (first 10 shown):")
        for t, reason in failed_tickers[:10]:
            print(f"    {t}: {reason[:80]}")

    if not all_data:
        print("[FATAL] No data fetched. Aborting.")
        sys.exit(1)

    # Combine all DataFrames
    print(f"\n{'=' * 70}")
    print("CREATING RANKING DATASET...")
    print(f"{'=' * 70}")

    df_all = pd.concat(all_data, ignore_index=False)
    df_all.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)

    # Create Query_ID: group by hour
    df_all['DateTime'] = pd.to_datetime(df_all['DateTime'])
    df_all['DateTime_Hour'] = df_all['DateTime'].dt.floor('h')
    df_all['Query_ID'] = df_all.groupby('DateTime_Hour').ngroup()

    df_all = df_all.reset_index(drop=True)

    print(f"  Total raw rows: {df_all.shape[0]:,}")
    print(f"  Unique hours (queries): {df_all['Query_ID'].nunique():,}")
    print(f"  Avg tickers per query: {df_all.groupby('Query_ID').size().mean():.1f}")

    # Add Market Context Features
    print("\nAdding Market Context Features...")
    df_all['Market_Mean_Return'] = df_all.groupby('Query_ID')['Return'].transform('mean')
    df_all['Relative_Return'] = df_all['Return'] - df_all['Market_Mean_Return']
    df_all['Market_Mean_Volatility'] = df_all.groupby('Query_ID')['HL_Range'].transform('mean')
    df_all['Relative_Volatility'] = df_all['HL_Range'] / (df_all['Market_Mean_Volatility'] + 1e-8)

    # Cross-Sectional Z-Scoring
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

    # Drop rows with NaN label (last bar per ticker has no forward return)
    df_all.dropna(subset=['Next_Hour_Return'], inplace=True)
    
    # Fill remaining NaNs in features with 0 (neutral score)
    df_all[feature_cols] = df_all[feature_cols].fillna(0)

    # Data Quality Check
    nan_count = df_all[feature_cols].isna().sum().sum()
    inf_count = np.isinf(df_all[feature_cols].select_dtypes(include=[np.number])).sum().sum()
    print(f"\nData Quality check:")
    print(f"  NaN values: {nan_count}")
    print(f"  Inf values: {inf_count}")

    # Save output
    os.makedirs("data", exist_ok=True)
    df_all.to_csv(OUTPUT_CSV, index=False)

    print(f"\n{'=' * 70}")
    print(f"[SUCCESS] SAVED: {OUTPUT_CSV}")
    print(f"   Rows: {df_all.shape[0]:,} | Cols: {df_all.shape[1]}")
    print(f"   Queries: {df_all['Query_ID'].nunique():,}")
    print(f"   Feature columns count: {len(feature_cols)}")
    print(f"=" * 70 + "\n")

if __name__ == "__main__":
    main()

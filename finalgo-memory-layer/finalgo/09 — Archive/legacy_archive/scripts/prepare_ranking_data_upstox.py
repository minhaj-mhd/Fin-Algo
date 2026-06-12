"""
Prepare ranking data using UPSTOX V2 Historical API (30min → 1hr resampled)
- Fetches 90 days of 30-minute candles from Upstox (max available for intraday)
- Resamples to 1-hour bars (matching the original training pipeline)
- Computes all 54 features using the shared compute_features() module
- Creates Next_Hour_Return labels, Query_IDs, and cross-sectional Z-scoring
- Saves to data/ranking_data_upstox.csv
"""

import os
import sys
import time
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta

sys.path.append(os.getcwd())

from scripts.tickers import TICKERS
from scripts.feature_utils import compute_features
from scripts.upstox_broker import UpstoxSandboxBroker

# ========================================
# CONFIG
# ========================================
OUTPUT_CSV = "data/ranking_data_upstox.csv"
LOOKBACK_DAYS = 90       # Upstox max for 30min intraday
RATE_LIMIT_PAUSE = 0.15  # seconds between API calls (~7/sec, well under 10/sec limit)

# ========================================
# INITIALIZE BROKER
# ========================================
print("=" * 60)
print("UPSTOX DATA PIPELINE — 90-Day Hourly Ranking Data")
print("=" * 60)

broker = UpstoxSandboxBroker()
tickers = TICKERS

print(f"Tickers: {len(tickers)}")
print(f"Lookback: {LOOKBACK_DAYS} days")
print(f"Interval: 30min -> resampled to 1hr")
print()

# ========================================
# FETCH 30-MIN DATA & RESAMPLE TO 1HR
# ========================================
all_data = []
failed_tickers = []
yf_fallback_tickers = []

for i, ticker in enumerate(tqdm(tickers, desc="Fetching Upstox Data")):
    try:
        # Fetch 30-min data from Upstox (will auto-resample to 1hr internally)
        df = broker.get_historical_data(ticker, interval='60minute', days=LOOKBACK_DAYS)

        if df is None or df.empty:
            raise Exception("Empty DataFrame returned")

        # Standardize column names (Upstox returns lowercase)
        col_map = {
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'volume': 'Volume', 'timestamp': 'DateTime'
        }
        df = df.rename(columns=col_map)

        # Ensure we have the required columns
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(c in df.columns for c in required):
            raise Exception(f"Missing columns: {set(required) - set(df.columns)}")

        # Set DateTime as index (required by compute_features for Hour/DayOfWeek)
        if 'DateTime' in df.columns:
            df['DateTime'] = pd.to_datetime(df['DateTime'])
            df = df.set_index('DateTime')
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')

        # Drop OI column if present
        if 'oi' in df.columns:
            df = df.drop(columns=['oi'])

        # Filter to market hours only (9:15 AM - 3:30 PM IST)
        if hasattr(df.index, 'hour'):
            df = df[(df.index.hour >= 9) & (df.index.hour < 16)]

        if len(df) < 25:
            raise Exception(f"Only {len(df)} bars — need at least 25 for feature computation")

        df['Ticker'] = ticker
        df['DateTime'] = df.index  # Preserve for grouping

        # Compute features
        df = compute_features(df, legacy=False)

        # Create label: Next hour return
        df['Next_Hour_Return'] = df['Close'].shift(-1) / df['Close'] - 1

        all_data.append(df)

        # Verify source
        if hasattr(df, '_upstox_source'):
            pass  # Pure Upstox
        # The broker falls back to yfinance internally if Upstox fails
        # We track this via the tqdm output

    except Exception as e:
        failed_tickers.append((ticker, str(e)))
        # Try yfinance fallback for this ticker
        try:
            import yfinance as yf
            yf_df = yf.download(ticker, period="90d", interval="1h", progress=False, auto_adjust=True)
            if isinstance(yf_df.columns, pd.MultiIndex):
                yf_df.columns = [col[0] for col in yf_df.columns]
            yf_df = yf_df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            if len(yf_df) >= 25:
                yf_df['Ticker'] = ticker
                yf_df['DateTime'] = yf_df.index
                yf_df = compute_features(yf_df, legacy=False)
                yf_df['Next_Hour_Return'] = yf_df['Close'].shift(-1) / yf_df['Close'] - 1
                all_data.append(yf_df)
                yf_fallback_tickers.append(ticker)
        except Exception:
            pass

    # Rate limit
    if i % 5 == 0:
        time.sleep(RATE_LIMIT_PAUSE)

# ========================================
# REPORT FETCH RESULTS
# ========================================
print(f"\n{'=' * 60}")
print(f"DATA FETCH COMPLETE")
print(f"{'=' * 60}")
print(f"  Successfully fetched: {len(all_data)} / {len(tickers)} tickers")
print(f"  Failed (no data):     {len(failed_tickers) - len(yf_fallback_tickers)}")
print(f"  yfinance fallback:    {len(yf_fallback_tickers)}")

if failed_tickers:
    print(f"\n  Failed tickers:")
    for t, reason in failed_tickers[:10]:
        print(f"    {t}: {reason[:80]}")
    if len(failed_tickers) > 10:
        print(f"    ... and {len(failed_tickers) - 10} more")

if not all_data:
    print("[FATAL] No data fetched. Aborting.")
    sys.exit(1)

# ========================================
# COMBINE & CREATE QUERY IDS
# ========================================
print(f"\n{'=' * 60}")
print("CREATING RANKING DATASET...")
print(f"{'=' * 60}")

df_all = pd.concat(all_data, ignore_index=False)
df_all.dropna(inplace=True)

# Create query_id: group by hour (all stocks at the same hour share same query)
df_all['DateTime'] = pd.to_datetime(df_all['DateTime'])
df_all['DateTime_Hour'] = df_all['DateTime'].dt.floor('h')
df_all['Query_ID'] = df_all.groupby('DateTime_Hour').ngroup()

df_all = df_all.reset_index(drop=True)

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
    'Hour', 'DayOfWeek', 'Is_Open_Hour', 'Is_Close_Hour', 'Time_To_Close'
]
feature_cols = [col for col in df_all.columns if col not in exclude_cols]

for col in tqdm(feature_cols, desc="Z-Scoring"):
    mean = df_all.groupby('Query_ID')[col].transform('mean')
    std = df_all.groupby('Query_ID')[col].transform('std')
    df_all[col] = (df_all[col] - mean) / (std + 1e-8)

# ========================================
# DATA QUALITY CHECK
# ========================================
print(f"\n{'=' * 60}")
print("DATA QUALITY REPORT")
print(f"{'=' * 60}")

# Check for NaN/Inf
nan_count = df_all[feature_cols].isna().sum().sum()
inf_count = np.isinf(df_all[feature_cols].select_dtypes(include=[np.number])).sum().sum()
print(f"  NaN values: {nan_count}")
print(f"  Inf values: {inf_count}")

# Label distribution
returns = df_all['Next_Hour_Return'].dropna()
print(f"  Label (Next_Hour_Return) stats:")
print(f"    Mean:   {returns.mean():.6f}")
print(f"    Median: {returns.median():.6f}")
print(f"    Std:    {returns.std():.6f}")
print(f"    Min:    {returns.min():.6f}")
print(f"    Max:    {returns.max():.6f}")

# Date range
dates = df_all['DateTime'].sort_values()
print(f"  Date range: {dates.iloc[0]} -> {dates.iloc[-1]}")
print(f"  Trading days: {df_all['DateTime'].dt.date.nunique()}")

# ========================================
# SAVE
# ========================================
df_all.dropna(inplace=True)  # Drop rows with NaN labels (last row per ticker)

os.makedirs("data", exist_ok=True)
df_all.to_csv(OUTPUT_CSV, index=False)

# Also save feature list for reference
feature_cols_final = [col for col in df_all.columns if col not in exclude_cols]
print(f"\n  Feature columns: {len(feature_cols_final)}")

print(f"\n{'=' * 60}")
print(f"✅ SAVED: {OUTPUT_CSV}")
print(f"   Rows: {df_all.shape[0]:,} | Cols: {df_all.shape[1]}")
print(f"   Queries: {df_all['Query_ID'].nunique():,}")
print(f"   Tickers/Query: {df_all.groupby('Query_ID').size().mean():.1f}")
print(f"{'=' * 60}")

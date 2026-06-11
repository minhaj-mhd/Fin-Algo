"""
Fetch 3.5 years of Upstox 15-minute candle data for all tickers.

Strategy:
- Upstox allows fetching 1 quarter (90 days) per call for intraday intervals
- We loop from Jan 2023 to today in 85-day chunks to stay safely under the limit
- Fetches 15-min candles directly
- Saves raw OHLCV per ticker, then builds full ranking dataset

Estimated time: ~15-25 minutes for 47 tickers x 3 years.
"""

import os
import sys
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date, time as dtime
from tqdm import tqdm

sys.path.append(os.getcwd())

from scripts.tickers import TICKERS
from scripts.feature_utils import compute_features
from scripts.upstox_broker import UpstoxSandboxBroker

# ============================================================
# CONFIG
# ============================================================
OUTPUT_CSV      = "data/ranking_data_upstox_15min_3y.csv"
RAW_CACHE_DIR   = "data/raw_upstox_cache_15min_3y"      # Per-ticker CSVs to avoid re-fetching
START_DATE      = date(2023, 1, 1)                      # Start date for ~3.5 years
CHUNK_DAYS      = 30                                    # Upstox limit for 1minute data is 31 days
RATE_PAUSE      = 0.4                                   # seconds between API calls
MIN_BARS        = 50                                    # Minimum bars needed for feature computation

os.makedirs(RAW_CACHE_DIR, exist_ok=True)

broker = UpstoxSandboxBroker()

# ============================================================
# BUILD DATE CHUNKS: Jan 2023 → today
# ============================================================
def build_date_chunks(start: date, end: date, chunk_days: int):
    """Generate (from_date, to_date) pairs in chunk_days intervals."""
    chunks = []
    current = start
    while current < end:
        chunk_end = min(current + timedelta(days=chunk_days), end)
        chunks.append((current.strftime('%Y-%m-%d'), chunk_end.strftime('%Y-%m-%d')))
        current = chunk_end + timedelta(days=1)
    return chunks

TODAY      = date.today()
DATE_CHUNKS = build_date_chunks(START_DATE, TODAY, CHUNK_DAYS)

print("=" * 65)
print("UPSTOX 15-MIN 3-YEAR DATA COLLECTOR")
print("=" * 65)
print(f"  Tickers       : {len(TICKERS)}")
print(f"  Start date    : {START_DATE}")
print(f"  End date      : {TODAY}")
print(f"  Date chunks   : {len(DATE_CHUNKS)} x ~{CHUNK_DAYS}-day windows")
print(f"  Total API calls: ~{len(TICKERS) * len(DATE_CHUNKS)}")
print(f"  Cache dir     : {RAW_CACHE_DIR}")
print()

# ============================================================
# FETCH PER TICKER (with per-ticker CSV caching)
# ============================================================
def fetch_ticker_full_history(ticker: str, date_chunks: list) -> pd.DataFrame:
    """
    Fetch complete 15-min OHLCV history for a ticker by looping over date chunks.
    Uses a local CSV cache to avoid re-fetching already-collected windows.
    """
    cache_file = os.path.join(RAW_CACHE_DIR, f"{ticker.replace('.NS','')}.csv")

    # Load cached data if exists
    existing = None
    if os.path.exists(cache_file):
        try:
            existing = pd.read_csv(cache_file, parse_dates=['timestamp'])
            existing['timestamp'] = pd.to_datetime(existing['timestamp'], utc=True)
        except Exception:
            existing = None

    already_have_until = None
    if existing is not None and not existing.empty:
        already_have_until = existing['timestamp'].max().date()

    instrument_key = broker.get_instrument_key(ticker)
    api_instance = broker.data_api_client

    import upstox_client
    hist_api = upstox_client.HistoryApi(api_instance)

    new_chunks = []
    for (from_str, to_str) in date_chunks:
        # Skip chunks we already have fully cached
        chunk_end_date = datetime.strptime(to_str, '%Y-%m-%d').date()
        if already_have_until and chunk_end_date <= already_have_until:
            continue
        new_chunks.append((from_str, to_str))

    df_new = None
    if not new_chunks:
        df_new = existing
    else:
        all_candles = []
        for (from_str, to_str) in new_chunks:
            try:
                resp = hist_api.get_historical_candle_data1(
                    instrument_key,
                    '1minute',   # Native Upstox interval (15minute is not supported directly)
                    to_str,
                    from_str,
                    '2.0'
                )
                if resp.status == 'success' and resp.data and resp.data.candles:
                    all_candles.extend(resp.data.candles)
                time.sleep(RATE_PAUSE)
            except Exception as e:
                # If rate limited, wait and retry once
                if '429' in str(e) or 'Too Many' in str(e):
                    time.sleep(5)
                    try:
                        resp = hist_api.get_historical_candle_data1(
                            instrument_key, '1minute', to_str, from_str, '2.0'
                        )
                        if resp.status == 'success' and resp.data and resp.data.candles:
                            all_candles.extend(resp.data.candles)
                    except Exception:
                        pass
                time.sleep(RATE_PAUSE)

        if not all_candles:
            df_new = existing
        else:
            # Parse candles → DataFrame
            df_new = pd.DataFrame(
                all_candles,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi']
            )
            df_new['timestamp'] = pd.to_datetime(df_new['timestamp'], utc=True)
            df_new = df_new.drop_duplicates(subset=['timestamp']).sort_values('timestamp')

            # Resample new 1min candles to 15min
            df_new = df_new.set_index('timestamp')
            df_new = df_new.resample('15min', origin='start_day').agg({
                'open': 'first', 'high': 'max', 'low': 'min',
                'close': 'last', 'volume': 'sum', 'oi': 'last'
            }).dropna(subset=['open', 'close']).reset_index()

            # Merge with existing cache
            if existing is not None and not existing.empty:
                df_new = pd.concat([existing, df_new]).drop_duplicates(
                    subset=['timestamp']
                ).sort_values('timestamp')

    if df_new is None or df_new.empty:
        return None

    # Filter to market hours: 9:15 AM – 3:15 PM IST (start of last 15-min candle)
    # Upstox timestamps are in IST (+05:30)
    ist_ts = df_new['timestamp'].dt.tz_convert('Asia/Kolkata')
    
    # Extract time component for easier filtering
    time_series = ist_ts.dt.time
    market_open = dtime(9, 15)
    market_close = dtime(15, 15)
    
    df_new = df_new[(time_series >= market_open) & (time_series <= market_close)].copy()
    df_new['timestamp'] = df_new['timestamp'].dt.tz_convert('Asia/Kolkata')

    # Save raw 15-min data for caching
    df_new.to_csv(cache_file, index=False)

    return df_new if not df_new.empty else None


# ============================================================
# MAIN LOOP: Fetch all tickers
# ============================================================
print("Phase 1: Fetching raw OHLCV data from Upstox...")
print("-" * 65)

all_ticker_dfs = []
fetch_stats = {'ok': 0, 'partial': 0, 'failed': 0}

for ticker in tqdm(TICKERS, desc="Tickers"):
    try:
        df_15m = fetch_ticker_full_history(ticker, DATE_CHUNKS)

        if df_15m is None or df_15m.empty or len(df_15m) < MIN_BARS:
            fetch_stats['failed'] += 1
            tqdm.write(f"  [SKIP] {ticker}: only {len(df_15m) if df_15m is not None else 0} bars")
            continue

        # Standardize columns for feature computation
        df_15m = df_15m.rename(columns={
            'timestamp': 'DateTime',
            'open': 'Open', 'high': 'High',
            'low': 'Low', 'close': 'Close', 'volume': 'Volume'
        })
        if 'oi' in df_15m.columns:
            df_15m = df_15m.drop(columns=['oi'])
        df_15m['Ticker'] = ticker

        # Compute features
        df_feat = df_15m.copy()
        df_feat.index = pd.DatetimeIndex(df_feat['DateTime'])
        df_feat = compute_features(df_feat, legacy=False)
        df_feat['DateTime'] = df_feat.index
        df_feat['Ticker'] = ticker

        # Next 15-min return label
        df_feat['Next_15Min_Return'] = df_feat['Close'].shift(-1) / df_feat['Close'] - 1

        all_ticker_dfs.append(df_feat)
        fetch_stats['ok'] += 1
        tqdm.write(f"  [OK] {ticker}: {len(df_feat)} bars ({df_feat['DateTime'].min().date()} -> {df_feat['DateTime'].max().date()})")

    except Exception as e:
        fetch_stats['failed'] += 1
        tqdm.write(f"  [FAIL] {ticker}: {str(e)[:80]}")

print(f"\nFetch results: {fetch_stats['ok']} OK | {fetch_stats['failed']} failed")

if not all_ticker_dfs:
    print("[FATAL] No data fetched. Check your UPSTOX_ANALYTICS_ACCESS_TOKEN in .env")
    sys.exit(1)

# ============================================================
# BUILD RANKING DATASET
# ============================================================
print("\nPhase 2: Building ranking dataset...")
print("-" * 65)

df_all = pd.concat(all_ticker_dfs, ignore_index=True)
df_all['DateTime'] = pd.to_datetime(df_all['DateTime'])
df_all['DateTime_15Min'] = df_all['DateTime'].dt.floor('15min')

# Drop rows with NaN labels (last row per ticker)
df_all = df_all.dropna(subset=['Next_15Min_Return'])

# Create Query_ID: all tickers at the same 15min share one query
df_all = df_all.sort_values('DateTime_15Min')
df_all['Query_ID'] = df_all.groupby('DateTime_15Min').ngroup()

# Filter: only keep queries with >= 5 tickers (otherwise ranking is meaningless)
query_sizes = df_all.groupby('Query_ID').size()
valid_queries = query_sizes[query_sizes >= 5].index
df_all = df_all[df_all['Query_ID'].isin(valid_queries)].copy()
print(f"  Kept {df_all['Query_ID'].nunique()} queries with >= 5 tickers")

# Re-number Query_IDs sequentially after filtering
df_all = df_all.sort_values('DateTime_15Min')
df_all['Query_ID'] = df_all.groupby('DateTime_15Min').ngroup()

print(f"  Total rows    : {len(df_all):,}")
print(f"  Unique queries: {df_all['Query_ID'].nunique():,}")
print(f"  Avg tickers/Q : {df_all.groupby('Query_ID').size().mean():.1f}")
dates = df_all['DateTime'].sort_values()
print(f"  Date range    : {dates.iloc[0].date()} -> {dates.iloc[-1].date()}")
print(f"  Trading days  : {df_all['DateTime'].dt.date.nunique()}")

# ============================================================
# MARKET CONTEXT FEATURES
# ============================================================
print("\nAdding market context features...")
df_all['Market_Mean_Return']     = df_all.groupby('Query_ID')['Return'].transform('mean')
df_all['Relative_Return']        = df_all['Return'] - df_all['Market_Mean_Return']
df_all['Market_Mean_Volatility'] = df_all.groupby('Query_ID')['HL_Range'].transform('mean')
df_all['Relative_Volatility']    = df_all['HL_Range'] / (df_all['Market_Mean_Volatility'] + 1e-8)

# ============================================================
# CROSS-SECTIONAL Z-SCORING (per query)
# ============================================================
print("Applying cross-sectional Z-scoring...")

exclude_cols = {
    'DateTime', 'DateTime_15Min', 'Query_ID', 'Ticker', 'Next_15Min_Return',
    'Open', 'High', 'Low', 'Close', 'Volume',
    'Market_Mean_Return', 'Relative_Return',
    'Market_Mean_Volatility', 'Relative_Volatility',
    'Hour', 'DayOfWeek', 'Is_Open_Hour', 'Is_Close_Hour', 'Time_To_Close'
}
feature_cols = [c for c in df_all.columns if c not in exclude_cols]

for col in tqdm(feature_cols, desc="Z-Scoring"):
    grp_mean = df_all.groupby('Query_ID')[col].transform('mean')
    grp_std  = df_all.groupby('Query_ID')[col].transform('std')
    df_all[col] = (df_all[col] - grp_mean) / (grp_std + 1e-8)

# Final NaN drop
df_all = df_all.dropna(subset=feature_cols)

# ============================================================
# SAVE
# ============================================================
print(f"\nPhase 3: Saving dataset...")
df_all.to_csv(OUTPUT_CSV, index=False)

print("\n" + "=" * 65)
print(f"DONE")
print(f"  Saved to      : {OUTPUT_CSV}")
print(f"  Rows          : {len(df_all):,}")
print(f"  Queries       : {df_all['Query_ID'].nunique():,}")
print(f"  Features      : {len(feature_cols)}")
print(f"  Date range    : {df_all['DateTime'].min().date()} -> {df_all['DateTime'].max().date()}")
print("=" * 65)
print()
print("Next step: run scripts/training/train_ranking_15min_3y.py pointing to this new CSV.")

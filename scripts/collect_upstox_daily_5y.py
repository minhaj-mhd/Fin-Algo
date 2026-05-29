"""
Fetch 5 years of Upstox 1-day daily candle data for all tickers, compute features,
and compile a cross-sectionally Z-scored daily ranking dataset.

Strategy:
- Fetches daily candles ('day' interval) from Upstox.
- Loop from 5 years ago to today in 1-year chunks to safely handle any API limits.
- Local CSV caching per ticker is implemented under data/raw_upstox_daily_cache/ to avoid re-fetching.
- Standardizes columns and computes 54 technical features using features_utils.py.
- Recalculates 52-week high/low columns using a 250-day lookback window (instead of 1625 hourly bars).
- Creates Next_Day_Return labels and Query_IDs by daily date.
- Applies cross-sectional Z-scoring per trading day.
- Saves the clean rank dataset to data/ranking_data_upstox_daily_5y.csv.
"""

import os
import sys
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from tqdm import tqdm

sys.path.append(os.getcwd())

from scripts.tickers import TICKERS
from scripts.feature_utils import compute_features
from scripts.upstox_broker import UpstoxSandboxBroker

# ============================================================
# CONFIG
# ============================================================
OUTPUT_CSV      = "data/ranking_data_upstox_daily_5y.csv"
RAW_CACHE_DIR   = "data/raw_upstox_daily_cache"   # Cache folder
START_DATE      = date(2021, 5, 29)                # 5 years ago from May 2026
CHUNK_DAYS      = 350                             # Safe annual chunk size (under Upstox limits)
RATE_PAUSE      = 0.2                             # Seconds between API calls
MIN_BARS        = 100                             # Minimum daily bars required for feature computation

os.makedirs(RAW_CACHE_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)

broker = UpstoxSandboxBroker()

# ============================================================
# DATE CHUNKS
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

TODAY = date.today()
DATE_CHUNKS = build_date_chunks(START_DATE, TODAY, CHUNK_DAYS)

print("=" * 70)
print("UPSTOX 5-YEAR DAILY DATA COLLECTOR")
print("=" * 70)
print(f"  Universe Tickers : {len(TICKERS)}")
print(f"  Start Date       : {START_DATE}")
print(f"  End Date         : {TODAY}")
print(f"  Date Chunks      : {len(DATE_CHUNKS)} chunks")
print(f"  Cache Directory  : {RAW_CACHE_DIR}")
print("-" * 70)

# ============================================================
# FETCH PER TICKER (Upstox with yfinance fallback)
# ============================================================
def fetch_ticker_daily_history(ticker: str, date_chunks: list) -> pd.DataFrame:
    """
    Fetch complete 5-year 1-day OHLCV history for a ticker.
    Uses local cache if available. Falls back to yfinance if Upstox fails completely.
    """
    cache_file = os.path.join(RAW_CACHE_DIR, f"{ticker.replace('.NS','')}.csv")
    
    # Load cached data if exists
    existing = None
    if os.path.exists(cache_file):
        try:
            existing = pd.read_csv(cache_file, parse_dates=['timestamp'])
            existing['timestamp'] = pd.to_datetime(existing['timestamp'])
        except Exception:
            existing = None

    already_have_until = None
    if existing is not None and not existing.empty:
        already_have_until = existing['timestamp'].max().date()

    new_chunks = []
    for (from_str, to_str) in date_chunks:
        chunk_end_date = datetime.strptime(to_str, '%Y-%m-%d').date()
        if already_have_until and chunk_end_date <= already_have_until:
            continue
        new_chunks.append((from_str, to_str))

    df_new = None
    if not new_chunks:
        df_new = existing
    else:
        all_candles = []
        instrument_key = None
        
        try:
            instrument_key = broker.get_instrument_key(ticker)
            api_instance = broker.data_api_client
            import upstox_client
            hist_api = upstox_client.HistoryApi(api_instance)
        except Exception as e:
            print(f"  [WARN] Failed to init Upstox API client for {ticker}: {e}")

        if instrument_key is not None:
            for (from_str, to_str) in new_chunks:
                try:
                    resp = hist_api.get_historical_candle_data1(
                        instrument_key,
                        'day',
                        to_str,
                        from_str,
                        '2.0'
                    )
                    if resp.status == 'success' and resp.data and resp.data.candles:
                        all_candles.extend(resp.data.candles)
                    time.sleep(RATE_PAUSE)
                except Exception as e:
                    # Retry once if rate limited
                    if '429' in str(e) or 'Too Many' in str(e):
                        time.sleep(5)
                        try:
                            resp = hist_api.get_historical_candle_data1(
                                instrument_key, 'day', to_str, from_str, '2.0'
                            )
                            if resp.status == 'success' and resp.data and resp.data.candles:
                                all_candles.extend(resp.data.candles)
                        except Exception:
                            pass
                    else:
                        print(f"  [WARN] API error on chunk {from_str}->{to_str} for {ticker}: {e}")
                    time.sleep(RATE_PAUSE)

        if not all_candles:
            df_new = existing
        else:
            # Parse candles -> DataFrame
            parsed_df = pd.DataFrame(
                all_candles,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi']
            )
            parsed_df['timestamp'] = pd.to_datetime(parsed_df['timestamp'])
            parsed_df = parsed_df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')

            # Merge with existing cache
            if existing is not None and not existing.empty:
                df_new = pd.concat([existing, parsed_df]).drop_duplicates(
                    subset=['timestamp']
                ).sort_values('timestamp')
            else:
                df_new = parsed_df

    # If Upstox collection failed or yielded insufficient bars, fall back to yfinance
    if df_new is None or df_new.empty or len(df_new) < MIN_BARS:
        print(f"  [FALLBACK] Fetching {ticker} via yfinance...")
        try:
            import yfinance as yf
            # 5-year lookback is ~1825 days
            yf_df = yf.download(ticker, period="5y", interval="1d", progress=False, auto_adjust=True)
            if isinstance(yf_df.columns, pd.MultiIndex):
                yf_df.columns = [col[0] for col in yf_df.columns]
            
            if yf_df is not None and not yf_df.empty and len(yf_df) >= MIN_BARS:
                df_new = pd.DataFrame({
                    'timestamp': yf_df.index,
                    'open': yf_df['Open'],
                    'high': yf_df['High'],
                    'low': yf_df['Low'],
                    'close': yf_df['Close'],
                    'volume': yf_df['Volume'],
                    'oi': 0
                }).reset_index(drop=True)
                df_new._is_yfinance = True
        except Exception as e:
            print(f"  [FAIL] yfinance fallback failed for {ticker}: {e}")

    if df_new is not None and not df_new.empty:
        # Save cache (raw data)
        df_new.to_csv(cache_file, index=False)
        return df_new
        
    return None

# ============================================================
# MAIN COLLECTION & FEATURE ENGINEERING
# ============================================================
print("Phase 1: Collecting raw historical daily OHLCV...")
all_ticker_dfs = []
failed_tickers = []
yfinance_counts = 0

for ticker in tqdm(TICKERS, desc="Stocks"):
    try:
        df_raw = fetch_ticker_daily_history(ticker, DATE_CHUNKS)
        if df_raw is None or df_raw.empty or len(df_raw) < MIN_BARS:
            failed_tickers.append(ticker)
            continue

        if hasattr(df_raw, '_is_yfinance'):
            yfinance_counts += 1

        # Format columns for features
        df_feat = df_raw.rename(columns={
            'timestamp': 'DateTime',
            'open': 'Open', 'high': 'High',
            'low': 'Low', 'close': 'Close', 'volume': 'Volume'
        })
        if 'oi' in df_feat.columns:
            df_feat = df_feat.drop(columns=['oi'])
            
        df_feat['Ticker'] = ticker
        
        # Ensure DateTime is index and parsed correctly
        df_feat['DateTime'] = pd.to_datetime(df_feat['DateTime'])
        df_feat = df_feat.set_index('DateTime')

        # Compute technical indicators
        df_feat = compute_features(df_feat, legacy=False)
        df_feat['DateTime'] = df_feat.index
        df_feat['Ticker'] = ticker

        # Correct 52-week High/Low features for daily candles (250 trading days)
        high_52w = df_feat['High'].rolling(250, min_periods=50).max()
        low_52w  = df_feat['Low'].rolling(250, min_periods=50).min()
        df_feat['Dist_52W_High'] = (df_feat['Close'] - high_52w) / (high_52w + 1e-8)
        df_feat['Dist_52W_Low']  = (df_feat['Close'] - low_52w)  / (low_52w  + 1e-8)

        # Label: Next Day Return (Close-to-Close)
        df_feat['Next_Day_Return'] = df_feat['Close'].shift(-1) / df_feat['Close'] - 1

        all_ticker_dfs.append(df_feat)

    except Exception as e:
        failed_tickers.append(ticker)
        print(f"  [ERROR] {ticker}: {e}")

print(f"\nFetch completed: {len(all_ticker_dfs)} OK, {len(failed_tickers)} failed.")
print(f"yfinance fallbacks: {yfinance_counts}")
if failed_tickers:
    print(f"Failed tickers: {failed_tickers}")

if not all_ticker_dfs:
    print("[FATAL] No stock data collected. Verify network/API access tokens.")
    sys.exit(1)

# ============================================================
# COMPILE CROSS-SECTIONAL RANKING DATASET
# ============================================================
print("\nPhase 2: Building cross-sectional ranking dataset...")
df_all = pd.concat(all_ticker_dfs, ignore_index=True)
df_all['DateTime'] = pd.to_datetime(df_all['DateTime'])

# Drop rows where target label Next_Day_Return is NaN (typically the last day per stock)
df_all = df_all.dropna(subset=['Next_Day_Return'])

# Query_ID based on daily date (stocks traded on the same day belong to same query)
df_all = df_all.sort_values('DateTime')
df_all['Query_ID'] = df_all.groupby(df_all['DateTime'].dt.date).ngroup()

# Filter: Only keep dates with >= 5 tickers to maintain cross-sectional ranking integrity
query_sizes = df_all.groupby('Query_ID').size()
valid_queries = query_sizes[query_sizes >= 5].index
df_all = df_all[df_all['Query_ID'].isin(valid_queries)].copy()

# Re-index Query_IDs sequentially after filtering
df_all = df_all.sort_values('DateTime')
df_all['Query_ID'] = df_all.groupby(df_all['DateTime'].dt.date).ngroup()

print(f"  Total samples  : {len(df_all):,}")
print(f"  Trading days   : {df_all['Query_ID'].nunique():,}")
print(f"  Avg stocks/day : {df_all.groupby('Query_ID').size().mean():.1f}")
date_min, date_max = df_all['DateTime'].min(), df_all['DateTime'].max()
print(f"  Date span      : {date_min.date()} -> {date_max.date()}")

# ============================================================
# MARKET CONTEXT FEATURES
# ============================================================
print("\nCalculating market context features...")
df_all['Market_Mean_Return']     = df_all.groupby('Query_ID')['Return'].transform('mean')
df_all['Relative_Return']        = df_all['Return'] - df_all['Market_Mean_Return']
df_all['Market_Mean_Volatility'] = df_all.groupby('Query_ID')['HL_Range'].transform('mean')
df_all['Relative_Volatility']    = df_all['HL_Range'] / (df_all['Market_Mean_Volatility'] + 1e-8)

# ============================================================
# CROSS-SECTIONAL Z-SCORING
# ============================================================
print("Applying daily cross-sectional Z-scoring...")

exclude_cols = {
    'DateTime', 'Query_ID', 'Ticker', 'Next_Day_Return',
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

# Fill any remaining NaNs in features with 0 (neutral score)
df_all[feature_cols] = df_all[feature_cols].fillna(0)

# ============================================================
# SAVE FINAL DATASET
# ============================================================
print(f"\nPhase 3: Saving dataset...")
df_all.to_csv(OUTPUT_CSV, index=False)

print("\n" + "=" * 70)
print(f"DAILY 5-YEAR DATASET CREATION SUCCESSFUL")
print(f"  Output path    : {OUTPUT_CSV}")
print(f"  Total rows     : {len(df_all):,}")
print(f"  Total queries  : {df_all['Query_ID'].nunique():,}")
print(f"  Features count : {len(feature_cols)}")
print(f"  Date span      : {df_all['DateTime'].min().date()} -> {df_all['DateTime'].max().date()}")
print("=" * 70)
print()

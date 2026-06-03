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
from scripts.feature_utils import compute_features_daily_xgb, compute_features_daily_transformer_v2
from scripts.upstox_broker import UpstoxSandboxBroker

# ============================================================
# CONFIG
# ============================================================
OUTPUT_CSV_XGB         = "data/ranking_data_upstox_daily_5y.csv"             # XGBoost dataset
OUTPUT_CSV_TRANSFORMER = "data/ranking_data_upstox_daily_5y_transformer.csv" # Transformer dataset
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
print("Dual-Dataset Mode: XGBoost + Transformer")
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
# HELPER: Build ranking dataset from list of per-ticker DataFrames
# ============================================================
def build_ranking_dataset(all_dfs, dataset_name, output_csv):
    """Compile, add market context, Z-score, and save a ranking dataset."""
    print(f"\n{'='*70}")
    print(f"Building {dataset_name} ranking dataset...")
    print(f"{'='*70}")
    
    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all['DateTime'] = pd.to_datetime(df_all['DateTime'])
    
    # Drop rows where target label Next_Day_Return is NaN
    df_all = df_all.dropna(subset=['Next_Day_Return'])
    
    # Query_ID based on daily date
    df_all = df_all.sort_values('DateTime')
    df_all['Query_ID'] = df_all.groupby(df_all['DateTime'].dt.date).ngroup()
    
    # Filter: Only keep dates with >= 5 tickers
    query_sizes = df_all.groupby('Query_ID').size()
    valid_queries = query_sizes[query_sizes >= 5].index
    df_all = df_all[df_all['Query_ID'].isin(valid_queries)].copy()
    
    # Re-index Query_IDs sequentially
    df_all = df_all.sort_values('DateTime')
    df_all['Query_ID'] = df_all.groupby(df_all['DateTime'].dt.date).ngroup()
    
    print(f"  Total samples  : {len(df_all):,}")
    print(f"  Trading days   : {df_all['Query_ID'].nunique():,}")
    print(f"  Avg stocks/day : {df_all.groupby('Query_ID').size().mean():.1f}")
    date_min, date_max = df_all['DateTime'].min(), df_all['DateTime'].max()
    print(f"  Date span      : {date_min.date()} -> {date_max.date()}")
    
    # Market context features
    print("  Calculating market context features...")
    df_all['Market_Mean_Return']     = df_all.groupby('Query_ID')['Return'].transform('mean')
    df_all['Relative_Return']        = df_all['Return'] - df_all['Market_Mean_Return']
    df_all['Market_Mean_Volatility'] = df_all.groupby('Query_ID')['HL_Range'].transform('mean')
    df_all['Relative_Volatility']    = df_all['HL_Range'] / (df_all['Market_Mean_Volatility'] + 1e-8)
    
    # Cross-sectional Z-scoring
    print("  Applying daily cross-sectional Z-scoring...")
    exclude_cols = {
        'DateTime', 'Query_ID', 'Ticker', 'Next_Day_Return',
        'Open', 'High', 'Low', 'Close', 'Volume',
        'Market_Mean_Return', 'Relative_Return',
        'Market_Mean_Volatility', 'Relative_Volatility',
        'Hour', 'DayOfWeek', 'DayOfMonth', 'MonthOfYear',
        'Is_Open_Hour', 'Is_Close_Hour', 'Time_To_Close',
        'Is_Month_Start', 'Is_Month_End', 'WeekOfMonth'
    }
    feature_cols = [c for c in df_all.columns if c not in exclude_cols]
    
    for col in tqdm(feature_cols, desc=f"  Z-Scoring ({dataset_name})"):
        grp_mean = df_all.groupby('Query_ID')[col].transform('mean')
        grp_std  = df_all.groupby('Query_ID')[col].transform('std')
        df_all[col] = (df_all[col] - grp_mean) / (grp_std + 1e-8)
    
    # Fill remaining NaNs
    df_all[feature_cols] = df_all[feature_cols].fillna(0)
    
    # Save
    print(f"  Saving to {output_csv}...")
    df_all.to_csv(output_csv, index=False)
    
    print(f"  [{dataset_name}] Features: {len(feature_cols)}, Rows: {len(df_all):,}")
    return len(feature_cols), len(df_all)


# ============================================================
# PHASE 1: COLLECT RAW OHLCV DATA (SINGLE PASS)
# ============================================================
print("Phase 1: Collecting raw historical daily OHLCV...")
raw_ticker_data = {}  # ticker -> standardized DataFrame (Open/High/Low/Close/Volume indexed by DateTime)
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

        # Standardize columns
        df_std = df_raw.rename(columns={
            'timestamp': 'DateTime',
            'open': 'Open', 'high': 'High',
            'low': 'Low', 'close': 'Close', 'volume': 'Volume'
        })
        if 'oi' in df_std.columns:
            df_std = df_std.drop(columns=['oi'])
        
        df_std['DateTime'] = pd.to_datetime(df_std['DateTime'])
        df_std = df_std.set_index('DateTime')
        
        raw_ticker_data[ticker] = df_std

    except Exception as e:
        failed_tickers.append(ticker)
        print(f"  [ERROR] {ticker}: {e}")

print(f"\nFetch completed: {len(raw_ticker_data)} OK, {len(failed_tickers)} failed.")
print(f"yfinance fallbacks: {yfinance_counts}")
if failed_tickers:
    print(f"Failed tickers: {failed_tickers}")

if not raw_ticker_data:
    print("[FATAL] No stock data collected. Verify network/API access tokens.")
    sys.exit(1)


# ============================================================
# PHASE 2A: COMPUTE XGBOOST FEATURES
# ============================================================
print("\n" + "=" * 70)
print("Phase 2A: Computing XGBoost features (same as 1hr model + 52W fix)...")
print("=" * 70)
xgb_dfs = []
for ticker, df_std in tqdm(raw_ticker_data.items(), desc="XGB Features"):
    try:
        df_feat = compute_features_daily_xgb(df_std)
        df_feat['DateTime'] = df_feat.index
        df_feat['Ticker'] = ticker
        df_feat['Next_Day_Return'] = df_feat['Close'].shift(-1) / df_feat['Close'] - 1
        xgb_dfs.append(df_feat)
    except Exception as e:
        print(f"  [ERROR] XGB features for {ticker}: {e}")

xgb_n_features, xgb_n_rows = build_ranking_dataset(xgb_dfs, "XGBoost", OUTPUT_CSV_XGB)


# ============================================================
# PHASE 2B: COMPUTE TRANSFORMER FEATURES
# ============================================================
print("\n" + "=" * 70)
print("Phase 2B: Computing Transformer-optimized features...")
print("=" * 70)
tf_dfs = []
for ticker, df_std in tqdm(raw_ticker_data.items(), desc="Transformer Features"):
    try:
        df_feat = compute_features_daily_transformer_v2(df_std)
        df_feat['DateTime'] = df_feat.index
        df_feat['Ticker'] = ticker
        df_feat['Next_Day_Return'] = df_feat['Close'].shift(-1) / df_feat['Close'] - 1
        tf_dfs.append(df_feat)
    except Exception as e:
        print(f"  [ERROR] Transformer features for {ticker}: {e}")

tf_n_features, tf_n_rows = build_ranking_dataset(tf_dfs, "Transformer", OUTPUT_CSV_TRANSFORMER)


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("DUAL-DATASET DAILY 5-YEAR COLLECTION COMPLETE")
print("=" * 70)
print(f"  XGBoost Dataset:")
print(f"    Path     : {OUTPUT_CSV_XGB}")
print(f"    Features : {xgb_n_features}")
print(f"    Rows     : {xgb_n_rows:,}")
print(f"  Transformer Dataset:")
print(f"    Path     : {OUTPUT_CSV_TRANSFORMER}")
print(f"    Features : {tf_n_features}")
print(f"    Rows     : {tf_n_rows:,}")
print("=" * 70)
print()


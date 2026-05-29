"""
Fetch 1 year of Upstox 15-minute candle data for all tickers and cache locally.

Strategy:
- Upstox V3 API supports custom unit and interval parameters.
- We fetch 15-minute candles directly using: https://api.upstox.com/v3/historical-candle/{instrument_key}/minutes/15/{to_date}/{from_date}
- We loop from 1 year ago to today in 85-day chunks to stay safely under limits.
- Saves raw OHLCV per ticker in data/raw_upstox_cache_15min/ to avoid re-fetching.
"""

import os
import sys
import time
import requests
import pandas as pd
from datetime import datetime, timedelta, date
from tqdm import tqdm

sys.path.append(os.getcwd())

from scripts.tickers import TICKERS
from scripts.upstox_broker import UpstoxSandboxBroker

# ============================================================
# CONFIG
# ============================================================
RAW_CACHE_DIR   = "data/raw_upstox_cache_15min"      # Per-ticker CSVs to avoid re-fetching
START_DATE      = date.today() - timedelta(days=365) # 1 year lookback
CHUNK_DAYS      = 25                                 # Stay under 30-day limit for sub-15min intervals
RATE_PAUSE      = 0.4                                # seconds between API calls
MIN_BARS        = 50                                 # Minimum bars needed for feature computation

os.makedirs(RAW_CACHE_DIR, exist_ok=True)

broker = UpstoxSandboxBroker()

# ============================================================
# BUILD DATE CHUNKS
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
print("UPSTOX 15-MINUTE 1-YEAR DATA COLLECTOR (V3 REST)")
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
    Fetch complete 15-minute OHLCV history for a ticker by looping over date chunks.
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
    already_have_from = None
    if existing is not None and not existing.empty:
        already_have_until = existing['timestamp'].max().date()
        already_have_from = existing['timestamp'].min().date()

    instrument_key = broker.get_instrument_key(ticker)

    new_chunks = []
    for (from_str, to_str) in date_chunks:
        chunk_start_date = datetime.strptime(from_str, '%Y-%m-%d').date()
        chunk_end_date = datetime.strptime(to_str, '%Y-%m-%d').date()
        
        # Skip chunks we already have fully cached (completely covered by cached range)
        if already_have_from and already_have_until:
            if chunk_start_date >= already_have_from and chunk_end_date <= already_have_until:
                continue
        new_chunks.append((from_str, to_str))

    df_new = None
    if not new_chunks:
        df_new = existing
    else:
        all_candles = []
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {broker.analytics_token}'
        }
        for (from_str, to_str) in new_chunks:
            url = f"https://api.upstox.com/v3/historical-candle/{instrument_key}/minutes/15/{to_str}/{from_str}"
            try:
                response = requests.get(url, headers=headers)
                
                # Check for rate limiting
                if response.status_code == 429:
                    time.sleep(5)
                    response = requests.get(url, headers=headers)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('status') == 'success' and 'data' in data and data['data'].get('candles'):
                        all_candles.extend(data['data']['candles'])
                    else:
                        tqdm.write(f"  [WARN] Failed to parse candles for {ticker} ({from_str} -> {to_str}): {data}")
                else:
                    tqdm.write(f"  [WARN] HTTP {response.status_code} for {ticker} ({from_str} -> {to_str}): {response.text[:150]}")
                time.sleep(RATE_PAUSE)
            except Exception as e:
                # If error, wait and retry once
                time.sleep(2)
                try:
                    response = requests.get(url, headers=headers)
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('status') == 'success' and 'data' in data and data['data'].get('candles'):
                            all_candles.extend(data['data']['candles'])
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

            # Merge with existing cache
            if existing is not None and not existing.empty:
                df_new = pd.concat([existing, df_new]).drop_duplicates(
                    subset=['timestamp']
                ).sort_values('timestamp')

    if df_new is None or df_new.empty:
        return None

    # Filter to market hours: 9:15 AM – 3:30 PM IST (3:45 – 10:00 UTC)
    ist_ts = df_new['timestamp'].dt.tz_convert('Asia/Kolkata')
    df_15m = df_new[(ist_ts.dt.hour >= 9) & (ist_ts.dt.hour < 16)].copy()
    df_15m['timestamp'] = df_15m['timestamp'].dt.tz_convert('Asia/Kolkata')

    # Save raw 15-min data for caching
    df_15m.to_csv(cache_file, index=False)

    return df_15m if not df_15m.empty else None


# ============================================================
# MAIN LOOP: Fetch all tickers
# ============================================================
print("Phase 1: Fetching raw 15-min candles from Upstox...")
print("-" * 65)

fetch_stats = {'ok': 0, 'failed': 0}

for ticker in tqdm(TICKERS, desc="Tickers"):
    try:
        df_15m = fetch_ticker_full_history(ticker, DATE_CHUNKS)

        if df_15m is None or df_15m.empty or len(df_15m) < MIN_BARS:
            fetch_stats['failed'] += 1
            tqdm.write(f"  [SKIP] {ticker}: only {len(df_15m) if df_15m is not None else 0} bars")
            continue

        fetch_stats['ok'] += 1
        tqdm.write(f"  [OK] {ticker}: {len(df_15m)} bars ({df_15m['timestamp'].min().date()} -> {df_15m['timestamp'].max().date()})")

    except Exception as e:
        fetch_stats['failed'] += 1
        tqdm.write(f"  [FAIL] {ticker}: {str(e)[:80]}")

print(f"\nFetch results: {fetch_stats['ok']} OK | {fetch_stats['failed']} failed")
print("=" * 65)
print("Data collection and caching complete.")
print("Next step: run scripts/prepare_ranking_data_15min.py to generate features.")
print("=" * 65)

"""
Fetch 5-minute candle data for May 2026 for the 72 traded tickers and cache locally.
"""

import os
import sys
import time
import requests
import pandas as pd
from datetime import datetime, timedelta, date
from tqdm import tqdm

sys.path.append(os.getcwd())

from scripts.upstox_broker import UpstoxSandboxBroker

# Target only the 72 traded tickers identified in task-483
TRADED_TICKERS = [
    'TATACOMM.NS', 'TIMKEN.NS', 'GNFC.NS', 'ALKEM.NS', 'DEVYANI.NS', 'MPHASIS.NS', 
    'LALPATHLAB.NS', 'ZEEL.NS', 'SRF.NS', 'POWERGRID.NS', 'SKFINDIA.NS', 'IGL.NS', 
    'PVRINOX.NS', 'HAVELLS.NS', 'WHIRLPOOL.NS', 'BATAINDIA.NS', 'AWL.NS', 'BALKRISIND.NS', 
    'COROMANDEL.NS', 'BHEL.NS', 'MARUTI.NS', 'COALINDIA.NS', 'SBICARD.NS', 'STARHEALTH.NS', 
    'MRF.NS', 'EXIDEIND.NS', 'PRESTIGE.NS', 'KPIL.NS', 'CROMPTON.NS', 'SYNGENE.NS', 
    'M&MFIN.NS', 'RAMCOCEM.NS', 'DABUR.NS', 'SUNDARMFIN.NS', 'JSWSTEEL.NS', 'ATUL.NS', 
    'SBILIFE.NS', 'THERMAX.NS', 'PAGEIND.NS', 'GRANULES.NS', 'RELIANCE.NS', 'ARE&M.NS', 
    'JKCEMENT.NS', 'KEC.NS', 'MANAPPURAM.NS', 'PIDILITIND.NS', 'SUNTV.NS', 'TATACONSUM.NS', 
    'METROPOLIS.NS', 'NYKAA.NS', 'RELAXO.NS', 'BANKBARODA.NS', 'ICICIGI.NS', 'DALBHARAT.NS', 
    'TATAPOWER.NS', 'LTTS.NS', 'BRIGADE.NS', 'CIPLA.NS', 'ATGL.NS', 'BAJFINANCE.NS', 
    'SCHAEFFLER.NS', 'AIAENG.NS', 'APOLLOTYRE.NS', 'MUTHOOTFIN.NS', 'TECHM.NS', 'SHREECEM.NS', 
    'BHARATFORG.NS', 'ICICIPRULI.NS', 'BERGEPAINT.NS', 'ESCORTS.NS', 'AARTIIND.NS', 'PETRONET.NS'
]

# ============================================================
# CONFIG
# ============================================================
RAW_CACHE_DIR   = "data/raw_upstox_cache_5min"
START_DATE      = date(2026, 5, 1)
END_DATE        = date(2026, 5, 30)
CHUNK_DAYS      = 15
RATE_PAUSE      = 0.4

os.makedirs(RAW_CACHE_DIR, exist_ok=True)

broker = UpstoxSandboxBroker()

def build_date_chunks(start: date, end: date, chunk_days: int):
    chunks = []
    current = start
    while current < end:
        chunk_end = min(current + timedelta(days=chunk_days), end)
        chunks.append((current.strftime('%Y-%m-%d'), chunk_end.strftime('%Y-%m-%d')))
        current = chunk_end + timedelta(days=1)
    return chunks

DATE_CHUNKS = build_date_chunks(START_DATE, END_DATE, CHUNK_DAYS)

print("=" * 65)
print("OPTIMIZED UPSTOX 5-MINUTE DATA COLLECTOR (MAY 2026)")
print("=" * 65)
print(f"  Traded Tickers: {len(TRADED_TICKERS)}")
print(f"  Start date    : {START_DATE}")
print(f"  End date      : {END_DATE}")
print(f"  Cache dir     : {RAW_CACHE_DIR}")
print()

def fetch_ticker_5min(ticker: str, date_chunks: list) -> pd.DataFrame:
    cache_file = os.path.join(RAW_CACHE_DIR, f"{ticker.replace('.NS','')}.csv")
    
    # Check cache to avoid re-fetching
    if os.path.exists(cache_file):
        try:
            df_exist = pd.read_csv(cache_file)
            if not df_exist.empty and len(df_exist) > 100:
                tqdm.write(f"  [CACHE-HIT] {ticker}: already has {len(df_exist)} cached rows.")
                return df_exist
        except Exception:
            pass

    instrument_key = broker.get_instrument_key(ticker)
    all_candles = []
    
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {broker.analytics_token}'
    }
    
    for (from_str, to_str) in date_chunks:
        url = f"https://api.upstox.com/v3/historical-candle/{instrument_key}/minutes/5/{to_str}/{from_str}"
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 429:
                time.sleep(5)
                response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success' and 'data' in data and data['data'].get('candles'):
                    all_candles.extend(data['data']['candles'])
            else:
                tqdm.write(f"  [WARN] HTTP {response.status_code} for {ticker} ({from_str} -> {to_str})")
            time.sleep(RATE_PAUSE)
        except Exception as e:
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
        return None

    df = pd.DataFrame(
        all_candles,
        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi']
    )
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')

    ist_ts = df['timestamp'].dt.tz_convert('Asia/Kolkata')
    df_5m = df[(ist_ts.dt.hour >= 9) & (ist_ts.dt.hour < 16)].copy()
    df_5m['timestamp'] = df_5m['timestamp'].dt.tz_convert('Asia/Kolkata')

    df_5m['time_int'] = df_5m['timestamp'].dt.hour * 60 + df_5m['timestamp'].dt.minute
    df_5m = df_5m[(df_5m['time_int'] >= 9 * 60 + 15) & (df_5m['time_int'] <= 15 * 60 + 30)]
    df_5m = df_5m.drop(columns=['time_int'])

    if not df_5m.empty:
        df_5m.to_csv(cache_file, index=False)
        return df_5m
    return None

print("Fetching raw 5-minute candles from Upstox...")
print("-" * 65)

fetch_stats = {'ok': 0, 'failed': 0}

for ticker in tqdm(TRADED_TICKERS, desc="Tickers"):
    try:
        df_5m = fetch_ticker_5min(ticker, DATE_CHUNKS)
        if df_5m is None or df_5m.empty:
            fetch_stats['failed'] += 1
            tqdm.write(f"  [SKIP] {ticker}: no data")
            continue

        fetch_stats['ok'] += 1
    except Exception as e:
        fetch_stats['failed'] += 1
        tqdm.write(f"  [FAIL] {ticker}: {str(e)}")

print(f"\nFetch results: {fetch_stats['ok']} OK | {fetch_stats['failed']} failed")
print("=" * 65)
print("Data collection complete.")

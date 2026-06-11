"""
Fetch 10 years of Upstox 1-day daily candle data for all tickers and Indian indices,
and store them as per-instrument parquet files under data/raw_daily_10y/.

Features:
- Incremental updates: checks existing parquet files and fetches only missing data.
- Handles rate limits and retries on 429.
- Falls back to yfinance if Upstox fails.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from tqdm import tqdm

sys.path.append(os.getcwd())

from scripts.tickers import TICKERS
from scripts.upstox_broker import UpstoxSandboxBroker

# CONFIG
OUTPUT_DIR = "data/raw_daily_10y"
START_DATE = date(2016, 6, 10)  # 10 years ago from June 2026
CHUNK_DAYS = 350
RATE_PAUSE = 0.2
MIN_BARS = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)

# NSE INDICES MAPPING
INDICES = {
    "NIFTY_50": {"upstox": "NSE_INDEX|Nifty 50", "yf": "^NSEI"},
    "NIFTY_500": {"upstox": "NSE_INDEX|Nifty 500", "yf": "NIFTY500.NS"},
    "NIFTY_BANK": {"upstox": "NSE_INDEX|Nifty Bank", "yf": "^NSEBANK"},
    "NIFTY_IT": {"upstox": "NSE_INDEX|Nifty IT", "yf": "^CNXIT"},
    "NIFTY_AUTO": {"upstox": "NSE_INDEX|Nifty Auto", "yf": "^CNXAUTO"},
    "NIFTY_PHARMA": {"upstox": "NSE_INDEX|Nifty Pharma", "yf": "^CNXPHARMA"},
    "NIFTY_METAL": {"upstox": "NSE_INDEX|Nifty Metal", "yf": "^CNXMETAL"},
    "NIFTY_FMCG": {"upstox": "NSE_INDEX|Nifty FMCG", "yf": "^CNXFMCG"},
    "NIFTY_ENERGY": {"upstox": "NSE_INDEX|Nifty Energy", "yf": "^CNXENERGY"},
    "NIFTY_REALTY": {"upstox": "NSE_INDEX|Nifty Realty", "yf": "^CNXREALTY"},
    "INDIA_VIX": {"upstox": "NSE_INDEX|India Vix", "yf": "^INDIAVIX"}
}

def build_date_chunks(start: date, end: date, chunk_days: int):
    chunks = []
    current = start
    while current < end:
        chunk_end = min(current + timedelta(days=chunk_days), end)
        chunks.append((current.strftime('%Y-%m-%d'), chunk_end.strftime('%Y-%m-%d')))
        current = chunk_end + timedelta(days=1)
    return chunks

def fetch_instrument_daily_history(symbol: str, upstox_key: str, yf_symbol: str, is_index: bool = False) -> pd.DataFrame:
    """
    Fetch complete 10-year OHLCV daily history.
    Uses local parquet cache if available. Falls back to yfinance if Upstox fails.
    """
    filename = symbol.replace('.NS', '')
    cache_file = os.path.join(OUTPUT_DIR, f"{filename}.parquet")
    
    existing = None
    if os.path.exists(cache_file):
        try:
            existing = pd.read_parquet(cache_file)
            existing['timestamp'] = pd.to_datetime(existing['timestamp'])
        except Exception:
            existing = None

    already_have_until = None
    if existing is not None and not existing.empty:
        already_have_until = existing['timestamp'].max().date()

    today = date.today()
    
    # Determine what chunks are needed
    if already_have_until and already_have_until >= today - timedelta(days=1):
        # Already fully up to date
        return existing
        
    start_point = START_DATE
    if already_have_until:
        start_point = already_have_until + timedelta(days=1)
        
    date_chunks = build_date_chunks(start_point, today, CHUNK_DAYS)
    
    if not date_chunks:
        return existing

    all_candles = []
    
    # Try Upstox fetch if we have access
    broker = UpstoxSandboxBroker()
    api_instance = broker.data_api_client
    
    try:
        import upstox_client
        hist_api = upstox_client.HistoryApi(api_instance)
        
        # Get instrument key
        if is_index:
            instrument_key = upstox_key
        else:
            instrument_key = broker.get_instrument_key(symbol)
            
        if instrument_key:
            for (from_str, to_str) in date_chunks:
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
                        print(f"  [WARN] Upstox API error for {symbol} ({from_str} -> {to_str}): {e}")
                    time.sleep(RATE_PAUSE)
    except Exception as e:
        print(f"  [WARN] Upstox connection failed for {symbol}: {e}")

    df_new = None
    if all_candles:
        parsed_df = pd.DataFrame(
            all_candles,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi']
        )
        parsed_df['timestamp'] = pd.to_datetime(parsed_df['timestamp'])
        parsed_df = parsed_df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
        
        if existing is not None and not existing.empty:
            df_new = pd.concat([existing, parsed_df]).drop_duplicates(
                subset=['timestamp']
            ).sort_values('timestamp')
        else:
            df_new = parsed_df

    # Fallback to yfinance if no data fetched or missing historical chunks
    if df_new is None or df_new.empty or (not already_have_until and len(df_new) < MIN_BARS):
        print(f"  [FALLBACK] Fetching {symbol} via yfinance...")
        try:
            import yfinance as yf
            yf_df = yf.download(yf_symbol, start=START_DATE.strftime('%Y-%m-%d'), end=today.strftime('%Y-%m-%d'), interval="1d", progress=False, auto_adjust=True, timeout=15)
            if isinstance(yf_df.columns, pd.MultiIndex):
                yf_df.columns = [col[0] for col in yf_df.columns]
                
            if yf_df is not None and not yf_df.empty:
                df_new = pd.DataFrame({
                    'timestamp': yf_df.index,
                    'open': yf_df['Open'],
                    'high': yf_df['High'],
                    'low': yf_df['Low'],
                    'close': yf_df['Close'],
                    'volume': yf_df['Volume'] if 'Volume' in yf_df.columns else 0,
                    'oi': 0
                }).reset_index(drop=True)
                df_new['timestamp'] = pd.to_datetime(df_new['timestamp'])
        except Exception as e:
            print(f"  [FAIL] yfinance fallback failed for {symbol}: {e}")

    if df_new is not None and not df_new.empty:
        # Save to parquet cache
        df_new.to_parquet(cache_file, index=False)
        return df_new
        
    return existing

def main():
    print("=" * 70)
    print("UPSTOX 10-YEAR DAILY HISTORICAL DATA COLLECTOR")
    print("=" * 70)
    
    # Collect indices first
    print("\nFetching Indian Indices & VIX...")
    for idx_name, mapping in tqdm(INDICES.items(), desc="Indices"):
        try:
            fetch_instrument_daily_history(
                symbol=idx_name,
                upstox_key=mapping["upstox"],
                yf_symbol=mapping["yf"],
                is_index=True
            )
        except Exception as e:
            print(f"  [ERROR] Failed to fetch index {idx_name}: {e}")
            
    # Collect universe tickers
    print("\nFetching Universe Tickers...")
    for ticker in tqdm(TICKERS, desc="Tickers"):
        try:
            fetch_instrument_daily_history(
                symbol=ticker,
                upstox_key=None,
                yf_symbol=ticker,
                is_index=False
            )
        except Exception as e:
            print(f"  [ERROR] Failed to fetch ticker {ticker}: {e}")
            
    print("\nCollection finished. Files saved under data/raw_daily_10y/")

if __name__ == "__main__":
    main()

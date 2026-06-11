"""
Fetch 10 years of global daily macro data from yfinance and save them
as per-instrument parquet files under data/raw_global_daily/.

Assets collected:
- ^GSPC (S&P 500)
- ^IXIC (Nasdaq)
- ^N225 (Nikkei 225)
- ^HSI (Hang Seng Index)
- USDINR=X (USD/INR)
- BZ=F (Brent Crude)
- GC=F (Gold)
- DX-Y.NYB (DXY)
- ^TNX (10Y Treasury Yield)
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from tqdm import tqdm

sys.path.append(os.getcwd())

# CONFIG
OUTPUT_DIR = "data/raw_global_daily"
START_DATE = date(2016, 6, 10) # 10 years ago
TIMEOUT = 15

os.makedirs(OUTPUT_DIR, exist_ok=True)

GLOBAL_ASSETS = {
    "SP500": "^GSPC",
    "NASDAQ": "^IXIC",
    "NIKKEI": "^N225",
    "HSI": "^HSI",
    "USDINR": "USDINR=X",
    "BRENT": "BZ=F",
    "GOLD": "GC=F",
    "DXY": "DX-Y.NYB",
    "US10Y": "^TNX"
}

def fetch_global_asset_history(name: str, yf_symbol: str) -> pd.DataFrame:
    """
    Fetch daily history of global asset from yfinance.
    Saves and updates local parquet file.
    """
    cache_file = os.path.join(OUTPUT_DIR, f"{name}.parquet")
    
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
    
    if already_have_until and already_have_until >= today - timedelta(days=1):
        return existing
        
    start_point = START_DATE
    if already_have_until:
        start_point = already_have_until + timedelta(days=1)
        
    df_new = None
    try:
        import yfinance as yf
        print(f"Fetching {name} ({yf_symbol}) from {start_point} to {today}...")
        
        # Download data
        yf_df = yf.download(
            yf_symbol, 
            start=start_point.strftime('%Y-%m-%d'), 
            end=today.strftime('%Y-%m-%d'), 
            interval="1d", 
            progress=False, 
            auto_adjust=True,
            timeout=TIMEOUT
        )
        
        if yf_df is not None and not yf_df.empty:
            # Flatten multiindex columns if any
            if isinstance(yf_df.columns, pd.MultiIndex):
                yf_df.columns = [col[0] for col in yf_df.columns]
                
            parsed_df = pd.DataFrame({
                'timestamp': yf_df.index,
                'open': yf_df['Open'],
                'high': yf_df['High'],
                'low': yf_df['Low'],
                'close': yf_df['Close'],
                'volume': yf_df['Volume'] if 'Volume' in yf_df.columns else 0,
                'oi': 0
            }).reset_index(drop=True)
            parsed_df['timestamp'] = pd.to_datetime(parsed_df['timestamp'])
            
            if existing is not None and not existing.empty:
                df_new = pd.concat([existing, parsed_df]).drop_duplicates(
                    subset=['timestamp']
                ).sort_values('timestamp')
            else:
                df_new = parsed_df
    except Exception as e:
        print(f"  [WARN] Failed to fetch {name}: {e}")
        
    if df_new is not None and not df_new.empty:
        df_new.to_parquet(cache_file, index=False)
        return df_new
        
    return existing

def main():
    print("=" * 70)
    print("GLOBAL DAILY MACRO DATA COLLECTOR (10-YEAR)")
    print("=" * 70)
    
    for name, sym in tqdm(GLOBAL_ASSETS.items(), desc="Global Assets"):
        try:
            fetch_global_asset_history(name, sym)
            time.sleep(0.5) # Avoid aggressive polling
        except Exception as e:
            print(f"  [ERROR] Failed to collect {name}: {e}")
            
    print("\nGlobal daily macro collection completed.")

if __name__ == "__main__":
    main()

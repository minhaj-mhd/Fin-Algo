import os
import sys
import glob
import json
import pandas as pd
from datetime import date

sys.path.append(os.getcwd())
from scripts.tickers import TICKERS
from scripts.upstox_broker import UpstoxSandboxBroker

def main():
    start_date = "2026-02-13"
    end_date = "2026-03-31"
    
    # 1. Load canonical trading days from NIFTY 50
    nifty_path = "data/raw_index_cache/nifty50_15m.csv"
    if not os.path.exists(nifty_path):
        print(f"Error: {nifty_path} not found.")
        sys.exit(1)
        
    nifty_df = pd.read_csv(nifty_path)
    # the existing cache might just be naive UTC or timezone aware
    try:
        nifty_df['timestamp'] = pd.to_datetime(nifty_df['ts'], utc=True).dt.tz_convert('Asia/Kolkata')
    except Exception:
        # fallback
        nifty_df['timestamp'] = pd.to_datetime(nifty_df['ts'])
        
    nifty_dates = sorted(nifty_df['timestamp'].dt.date.unique())
    
    start_d = pd.to_datetime(start_date).date()
    end_d = pd.to_datetime(end_date).date()
    
    canonical_dates = [d for d in nifty_dates if start_d <= d <= end_d]
    canonical_dates_str = [d.strftime('%Y-%m-%d') for d in canonical_dates]
    
    print(f"Found {len(canonical_dates)} canonical trading days between {start_date} and {end_date}.")
    
    # Gap window: 2026-02-21 to 2026-03-23
    gap_start = pd.to_datetime("2026-02-21").date()
    gap_end = pd.to_datetime("2026-03-23").date()
    canonical_gap_dates = [d for d in canonical_dates if gap_start <= d <= gap_end]
    canonical_gap_dates_str = [d.strftime('%Y-%m-%d') for d in canonical_gap_dates]
    
    overlap_dates = [d for d in canonical_dates if d not in canonical_gap_dates]
    overlap_dates_str = [d.strftime('%Y-%m-%d') for d in overlap_dates]

    # 2. Check each ticker
    cache_dir = "data/raw_upstox_cache_15min_3y"
    missing_data = {}
    
    for ticker in TICKERS:
        ticker_base = ticker.replace('.NS', '')
        csv_path = os.path.join(cache_dir, f"{ticker_base}.csv")
        
        if not os.path.exists(csv_path):
            missing_data[ticker] = canonical_gap_dates_str # entirely missing
            continue
            
        df = pd.read_csv(csv_path)
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert('Asia/Kolkata')
        except:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        ticker_dates = set(df['timestamp'].dt.date.unique())
        
        missing_days = []
        for d in canonical_gap_dates:
            if d not in ticker_dates:
                missing_days.append(d.strftime('%Y-%m-%d'))
                
        if missing_days:
            missing_data[ticker] = missing_days

    print(f"Found {len(missing_data)} tickers with missing days in the gap window.")
    
    # 3. Get instrument keys
    broker = UpstoxSandboxBroker()
    instrument_keys = {}
    for ticker in TICKERS:
        try:
            instrument_keys[ticker] = broker.get_instrument_key(ticker)
        except Exception as e:
            print(f"Failed to get instrument key for {ticker}: {e}")
            instrument_keys[ticker] = None
            
    # 4. Save manifest.json
    manifest = {
        "start_date": start_date,
        "end_date": end_date,
        "gap_start": "2026-02-21",
        "gap_end": "2026-03-23",
        "canonical_dates": canonical_dates_str,
        "canonical_gap_dates": canonical_gap_dates_str,
        "overlap_dates": overlap_dates_str,
        "missing_data": missing_data,
        "instrument_keys": instrument_keys,
        "tickers": TICKERS
    }
    
    with open("manifest.json", "w") as f:
        json.dump(manifest, f, indent=4)
        
    print("Saved manifest.json.")

if __name__ == '__main__':
    main()

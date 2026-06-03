#!/usr/bin/env python
"""
Upstox Daily Data Collector
Fetches hourly (resampled from 30min) data for the 172 TICKERS from Upstox
and saves the raw candles into daily Parquet files: data/historical/YYYY-MM-DD.parquet
"""

import os
import sys
import time
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta

sys.path.append(os.getcwd())

from scripts.tickers import TICKERS
from scripts.upstox_broker import UpstoxSandboxBroker

# Rate limit configuration
RATE_LIMIT_PAUSE = 0.15  # seconds between API calls

def collect_data(days=2, force=False, target_ticker=None):
    print("=" * 60)
    print("STARTING UPSTOX DAILY DATA COLLECTOR")
    print(f"Lookback Days: {days}")
    print(f"Force Overwrite: {force}")
    print("=" * 60)

    # Initialize broker
    broker = UpstoxSandboxBroker()
    
    tickers = [target_ticker] if target_ticker else TICKERS
    print(f"Processing {len(tickers)} tickers...")

    all_dfs = []
    failed_tickers = []
    yf_fallbacks = []

    for i, ticker in enumerate(tqdm(tickers, desc="Fetching data")):
        try:
            # Fetch hourly data (resampled from 30min internally in broker)
            df = broker.get_historical_data(ticker, interval='60minute', days=days)
            
            if df is None or df.empty:
                raise ValueError("No data returned from Upstox")

            # Standardize column names
            col_map = {
                'open': 'Open', 'high': 'High', 'low': 'Low',
                'close': 'Close', 'volume': 'Volume', 'timestamp': 'DateTime'
            }
            df = df.rename(columns=col_map)
            
            required = ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']
            if not all(c in df.columns for c in required):
                raise ValueError(f"Missing required columns: {set(required) - set(df.columns)}")
            
            # Select and order standard raw columns
            df = df[required].copy()
            df['Ticker'] = ticker
            
            all_dfs.append(df)

        except Exception as e:
            failed_tickers.append(ticker)
            # yfinance fallback to guarantee no missing data gaps
            try:
                import yfinance as yf
                yf_df = yf.download(ticker, period=f"{days}d", interval="1h", progress=False, auto_adjust=True)
                if not yf_df.empty:
                    if isinstance(yf_df.columns, pd.MultiIndex):
                        yf_df.columns = [col[0] for col in yf_df.columns]
                    yf_df = yf_df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                    yf_df['DateTime'] = yf_df.index
                    yf_df['Ticker'] = ticker
                    yf_df = yf_df[required + ['Ticker']].copy()
                    all_dfs.append(yf_df)
                    yf_fallbacks.append(ticker)
            except Exception as yf_err:
                print(f"\n[ERROR] Fallback failed for {ticker}: {yf_err}")

        # Rate limiting
        if i % 5 == 0:
            time.sleep(RATE_LIMIT_PAUSE)

    print("\n" + "=" * 60)
    print("FETCH STATISTICS")
    print(f"Successfully fetched (Upstox): {len(all_dfs) - len(yf_fallbacks)} / {len(tickers)}")
    print(f"Successfully fetched (yfinance): {len(yf_fallbacks)} / {len(tickers)}")
    print(f"Failed completely: {len(tickers) - len(all_dfs)}")
    if failed_tickers:
        print(f"Failed Upstox Tickers (first 10): {failed_tickers[:10]}")
    print("=" * 60)

    if not all_dfs:
        print("[FATAL] No data collected at all. Aborting.")
        return

    # Combine all data
    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all['DateTime'] = pd.to_datetime(df_all['DateTime'])
    
    # Ensure localized timezone parsing or string conversion to extract the local date
    # In India, trading is timezone +05:30. Extract the YYYY-MM-DD date part.
    df_all['DateStr'] = df_all['DateTime'].dt.strftime('%Y-%m-%d')
    
    # Filter to market hours (9:00 AM to 4:00 PM to capture standard trading hours 9:15-15:30)
    df_all = df_all[(df_all['DateTime'].dt.hour >= 9) & (df_all['DateTime'].dt.hour < 16)]

    # Group by date and save to individual Parquet files
    os.makedirs("data/historical", exist_ok=True)
    grouped = df_all.groupby('DateStr')
    
    saved_files = []
    for date_str, group in grouped:
        # Check if weekend (no trading)
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        if date_obj.weekday() >= 5: # 5 = Saturday, 6 = Sunday
            # Skip unless there's actually a special trading session
            if len(group) < 50: # arbitrary low threshold for weekend noise
                continue

        filename = f"data/historical/{date_str}.parquet"
        
        if os.path.exists(filename) and not force:
            print(f"File {filename} already exists. Skipping (use --force to overwrite).")
            continue
            
        # Drop temporary helper column
        group_to_save = group.drop(columns=['DateStr']).copy()
        
        # Sort values for clean parquet storage
        group_to_save = group_to_save.sort_values(['Ticker', 'DateTime']).reset_index(drop=True)
        
        # Save to parquet
        group_to_save.to_parquet(filename, index=False, engine='pyarrow')
        print(f"Saved: {filename} ({len(group_to_save)} rows, {group_to_save['Ticker'].nunique()} tickers)")
        saved_files.append(filename)
        
    print("\nCollection and storage finished.")
    print(f"Total daily files created/updated: {len(saved_files)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upstox Daily Parquet Data Collector")
    parser.add_argument("--days", type=int, default=2, help="Number of historical days to fetch")
    parser.add_argument("--force", action="store_true", help="Overwrite existing parquet files")
    parser.add_argument("--ticker", type=str, default=None, help="Fetch data only for this specific ticker (debug)")
    
    args = parser.parse_args()
    collect_data(days=args.days, force=args.force, target_ticker=args.ticker)

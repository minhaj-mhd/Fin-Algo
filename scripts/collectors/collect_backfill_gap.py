import os
import sys
import json
import time
import pandas as pd
from datetime import datetime, time as dtime
from tqdm import tqdm

sys.path.append(os.getcwd())
from scripts.upstox_broker import UpstoxSandboxBroker
import upstox_client

OUTPUT_DIR = "data/raw_upstox_cache_15min_3y_backfill"
CACHE_DIR = "data/raw_upstox_cache_15min_3y"
RATE_PAUSE = 0.4
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    with open("manifest.json", "r") as f:
        manifest = json.load(f)
        
    start_date = manifest["start_date"]
    end_date = manifest["end_date"]
    tickers = manifest["tickers"]
    instrument_keys = manifest["instrument_keys"]
    overlap_dates = manifest["overlap_dates"]
    
    broker = UpstoxSandboxBroker()
    api_instance = broker.data_api_client
    hist_api = upstox_client.HistoryApi(api_instance)
    
    coverage_md = "# Backfill Coverage Report\n\n"
    coverage_md += f"**Window:** {start_date} to {end_date}\n\n"
    
    mismatches = []
    missing_tickers = []
    
    # We will fetch in 1 chunk since it's < 90 days.
    # But wait, 1minute data limit is 31 days. So we need to split it.
    # Feb 13 to Mar 31 is 46 days. So we split into two chunks.
    chunk1_start = "2026-02-13"
    chunk1_end = "2026-03-05"
    chunk2_start = "2026-03-06"
    chunk2_end = "2026-03-31"
    
    chunks = [(chunk1_start, chunk1_end), (chunk2_start, chunk2_end)]
    
    for ticker in tqdm(tickers):
        ikey = instrument_keys.get(ticker)
        if not ikey:
            missing_tickers.append(ticker)
            continue
            
        all_candles = []
        for (f_str, t_str) in chunks:
            try:
                resp = hist_api.get_historical_candle_data1(
                    ikey, '1minute', t_str, f_str, '2.0'
                )
                if resp.status == 'success' and resp.data and resp.data.candles:
                    all_candles.extend(resp.data.candles)
                time.sleep(RATE_PAUSE)
            except Exception as e:
                if '429' in str(e) or 'Too Many' in str(e):
                    time.sleep(5)
                    try:
                        resp = hist_api.get_historical_candle_data1(
                            ikey, '1minute', t_str, f_str, '2.0'
                        )
                        if resp.status == 'success' and resp.data and resp.data.candles:
                            all_candles.extend(resp.data.candles)
                    except:
                        pass
                time.sleep(RATE_PAUSE)
                
        if not all_candles:
            missing_tickers.append(ticker)
            continue
            
        # Parse candles
        df_new = pd.DataFrame(
            all_candles,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi']
        )
        df_new['timestamp'] = pd.to_datetime(df_new['timestamp'], utc=True)
        df_new = df_new.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
        
        # Resample to 15min
        df_new = df_new.set_index('timestamp')
        df_new = df_new.resample('15min', origin='start_day').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum', 'oi': 'last'
        }).dropna(subset=['open', 'close']).reset_index()
        
        # Filter to market hours 9:15 to 15:15 IST
        ist_ts = df_new['timestamp'].dt.tz_convert('Asia/Kolkata')
        time_series = ist_ts.dt.time
        market_open = datetime.strptime("09:15", "%H:%M").time()
        market_close = datetime.strptime("15:15", "%H:%M").time()
        df_new = df_new[(time_series >= market_open) & (time_series <= market_close)].copy()
        df_new['timestamp'] = df_new['timestamp'].dt.tz_convert('Asia/Kolkata')
        
        df_new.to_csv(os.path.join(OUTPUT_DIR, f"{ticker.replace('.NS','')}.csv"), index=False)
        
        # Adjustment Check vs Existing Cache
        cache_path = os.path.join(CACHE_DIR, f"{ticker.replace('.NS','')}.csv")
        if os.path.exists(cache_path):
            existing = pd.read_csv(cache_path)
            try:
                existing['timestamp'] = pd.to_datetime(existing['timestamp'], utc=True).dt.tz_convert('Asia/Kolkata')
            except:
                existing['timestamp'] = pd.to_datetime(existing['timestamp'])
                
            # Filter overlap
            exist_overlap = existing[existing['timestamp'].dt.strftime('%Y-%m-%d').isin(overlap_dates)].copy()
            new_overlap = df_new[df_new['timestamp'].dt.strftime('%Y-%m-%d').isin(overlap_dates)].copy()
            
            merged = pd.merge(exist_overlap, new_overlap, on='timestamp', suffixes=('_old', '_new'))
            if len(merged) > 0:
                diffs = (merged['close_old'] - merged['close_new']).abs().max()
                if diffs > 0.01:
                    mismatches.append(f"{ticker}: Max diff {diffs:.4f}")
                    
    # Write report
    coverage_md += "## Missing Tickers\n"
    if missing_tickers:
        for t in missing_tickers:
            coverage_md += f"- {t}\n"
    else:
        coverage_md += "None. All tickers fetched successfully.\n\n"
        
    coverage_md += "## Adjustment Mismatches (Overlap)\n"
    if mismatches:
        for m in mismatches:
            coverage_md += f"- {m}\n"
    else:
        coverage_md += "None. Overlap days match exactly to the paisa for all tickers.\n"
        
    with open("coverage_report.md", "w") as f:
        f.write(coverage_md)
        
    print("Done. coverage_report.md generated.")

if __name__ == '__main__':
    main()

"""
Download True OOS Data (June 4 - July 10, 2026) from Upstox
==================================================
Downloads the 15-minute historical data for all 172 tickers and APPENDS it to the master cache.
"""

import sys, os, json, time, warnings
sys.path.insert(0, os.getcwd())
warnings.filterwarnings("ignore")

import pandas as pd
from datetime import datetime, date
from pathlib import Path

import upstox_client
from dotenv import load_dotenv
load_dotenv()

from scripts.tickers import TICKERS

OOS_FROM = date(2026, 6, 4)
OOS_TO   = date(2026, 7, 10)

CACHE_15M = Path("data/raw_upstox_cache_15min_3y")
CACHE_15M.mkdir(parents=True, exist_ok=True)

RATE_PAUSE = 0.25

# API Setup
tok = os.getenv("UPSTOX_ANALYTICS_ACCESS_TOKEN")
if not tok:
    print("[ERROR] UPSTOX_ANALYTICS_ACCESS_TOKEN not found in environment!")
    sys.exit(1)

cfg = upstox_client.Configuration(sandbox=False)
cfg.sandbox = False
cfg.host = "https://api.upstox.com"
cfg.access_token = tok
api_client = upstox_client.ApiClient(cfg)
v3api = upstox_client.HistoryV3Api(api_client)

# instrument key cache
with open("scripts/instrument_cache.json") as f:
    INST_MAP = json.load(f)

def get_ik(ticker):
    sym = ticker.replace(".NS", "")
    return INST_MAP.get(ticker) or INST_MAP.get(sym)

print("=" * 70)
print(f"  Downloading 15-min data for {len(TICKERS)} tickers")
print(f"  Period : {OOS_FROM} to {OOS_TO}")
print("=" * 70)

ok_count = 0
fail_tickers = []

for idx, ticker in enumerate(TICKERS, 1):
    sym = ticker.replace(".NS", "")
    cache_path = CACHE_15M / f"{sym}.csv"
    
    df_existing = None
    if cache_path.exists():
        try:
            df_existing = pd.read_csv(cache_path)
            if not df_existing.empty and "timestamp" in df_existing.columns:
                df_existing["timestamp"] = pd.to_datetime(df_existing["timestamp"], utc=True)
                # If we already have data past July 10, skip
                if df_existing["timestamp"].max().date() >= OOS_TO:
                    print(f"[{idx:3d}/{len(TICKERS)}] {ticker:<15} -> Already Cached past {OOS_TO}")
                    ok_count += 1
                    continue
        except Exception:
            pass

    ik = get_ik(ticker)
    if not ik:
        print(f"[{idx:3d}/{len(TICKERS)}] {ticker:<15} -> NO INSTRUMENT KEY")
        fail_tickers.append(ticker)
        continue

    frm = OOS_FROM.strftime("%Y-%m-%d")
    to  = OOS_TO.strftime("%Y-%m-%d")
    
    success = False
    chunks = [
        (date(2026, 6, 4).strftime("%Y-%m-%d"), date(2026, 6, 30).strftime("%Y-%m-%d")),
        (date(2026, 7, 1).strftime("%Y-%m-%d"), date(2026, 7, 10).strftime("%Y-%m-%d"))
    ]
    
    for attempt in range(3):
        try:
            chunk_success = True
            all_rows = []
            for frm, to in chunks:
                resp = v3api.get_historical_candle_data1(ik, "minutes", "15", to, frm)
                if resp.status == "success" and resp.data and resp.data.candles:
                    all_rows.extend(resp.data.candles)
                else:
                    chunk_success = False
                    time.sleep(1)
                    break
            
            if chunk_success and all_rows:
                df_new = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
                df_new["timestamp"] = pd.to_datetime(df_new["timestamp"], utc=True)
                
                # Append securely
                if df_existing is not None and not df_existing.empty:
                    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                else:
                    df_combined = df_new
                    
                df_combined = df_combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
                
                # Convert back to standard isoformat for CSV
                df_combined["timestamp"] = df_combined["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
                df_combined.to_csv(cache_path, index=False)
                
                print(f"[{idx:3d}/{len(TICKERS)}] {ticker:<15} -> Downloaded and Appended ({len(df_new)} new rows)")
                ok_count += 1
                success = True
                break
            
        except Exception as e:
            if "429" in str(e) or "Too Many" in str(e):
                time.sleep(3)
                continue
            print(f"[{idx:3d}/{len(TICKERS)}] {ticker:<15} -> Error: {str(e)}")
            break
        time.sleep(RATE_PAUSE)
        
    if not success:
        fail_tickers.append(ticker)
    
    time.sleep(RATE_PAUSE)

print("=" * 70)
print(f"Download complete: {ok_count}/{len(TICKERS)} tickers OK.")
if fail_tickers:
    print(f"Failed tickers: {fail_tickers}")
print("=" * 70)

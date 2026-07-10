"""
Download True OOS Data (June 5-18, 2026) from Upstox
==================================================
Downloads the 15-minute historical data for all 172 tickers and caches it.
Run this script first to fetch the dataset.

Run: python -m scripts.backtests.download_true_oos_data
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

OOS_FROM = date(2026, 6, 5)
OOS_TO   = date(2026, 6, 18)

CACHE_15M = Path("data/oos_cache_15min_jun2026")
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
    
    # Check if we already have it fully downloaded
    if cache_path.exists():
        try:
            df_existing = pd.read_csv(cache_path)
            if not df_existing.empty and "timestamp" in df_existing.columns:
                df_existing["timestamp"] = pd.to_datetime(df_existing["timestamp"], utc=True)
                if df_existing["timestamp"].max().date() >= OOS_TO:
                    print(f"[{idx:3d}/{len(TICKERS)}] {ticker:<15} -> Already Cached")
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
    for attempt in range(3):
        try:
            resp = v3api.get_historical_candle_data1(ik, "minutes", "15", to, frm)
            if resp.status == "success" and resp.data and resp.data.candles:
                rows = resp.data.candles
                df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
                df.to_csv(cache_path, index=False)
                print(f"[{idx:3d}/{len(TICKERS)}] {ticker:<15} -> Downloaded ({len(df)} rows)")
                ok_count += 1
                success = True
                break
            else:
                time.sleep(1)
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

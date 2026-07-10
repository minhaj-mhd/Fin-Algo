"""One-off incremental refresh of data/raw_daily_10y/*.parquet via yfinance.

Upstox broker has no live access token right now, so this fetches ONLY the
missing recent window (last cached date + 1 -> today) per ticker via yfinance
and merges into the existing parquet cache (dedup by timestamp). Does not
touch the pre-existing 10y history.

Exploratory data-refresh utility, not a production collector.
"""
import os
import sys
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

sys.path.append(os.getcwd())
from scripts.tickers import TICKERS

OUTPUT_DIR = "data/raw_daily_10y"
TODAY = date.today()


def refresh_one(symbol):
    filename = symbol.replace(".NS", "")
    cache_file = os.path.join(OUTPUT_DIR, f"{filename}.parquet")
    if not os.path.exists(cache_file):
        return filename, "no_cache", 0
    existing = pd.read_parquet(cache_file)
    existing["timestamp"] = pd.to_datetime(existing["timestamp"])
    last = existing["timestamp"].max().date()
    if last >= TODAY - timedelta(days=1):
        return filename, "already_current", 0
    start = last + timedelta(days=1)
    try:
        yf_df = yf.download(symbol, start=start.strftime("%Y-%m-%d"),
                             end=(TODAY + timedelta(days=1)).strftime("%Y-%m-%d"),
                             interval="1d", progress=False, auto_adjust=True, timeout=15)
    except Exception as e:
        return filename, f"error:{e}", 0
    if yf_df is None or yf_df.empty:
        return filename, "no_new_data", 0
    if isinstance(yf_df.columns, pd.MultiIndex):
        yf_df.columns = [c[0] for c in yf_df.columns]
    new_df = pd.DataFrame({
        "timestamp": yf_df.index,
        "open": yf_df["Open"], "high": yf_df["High"], "low": yf_df["Low"],
        "close": yf_df["Close"], "volume": yf_df.get("Volume", 0), "oi": 0,
    }).reset_index(drop=True)
    new_df["timestamp"] = pd.to_datetime(new_df["timestamp"]).dt.tz_localize(None)
    existing["timestamp"] = existing["timestamp"].dt.tz_localize(None)
    merged = pd.concat([existing, new_df]).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    merged.to_parquet(cache_file, index=False)
    return filename, "updated", len(new_df)


def main():
    results = {"updated": 0, "already_current": 0, "no_cache": 0, "error": 0, "no_new_data": 0}
    errors = []
    for i, ticker in enumerate(TICKERS):
        filename, status, n_new = refresh_one(ticker)
        if status.startswith("error"):
            results["error"] += 1
            errors.append((filename, status))
        else:
            results[status] = results.get(status, 0) + 1
        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{len(TICKERS)} processed...")
    print("DONE:", results)
    if errors:
        print("errors:", errors[:10])


if __name__ == "__main__":
    main()

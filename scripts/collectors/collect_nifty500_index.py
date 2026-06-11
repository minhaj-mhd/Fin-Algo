"""
Fetch NIFTY 500 index history (daily + hourly) from Upstox V3 HistoryV3Api.

Outputs:
  data/raw_index_cache/nifty500_1d.csv   — daily OHLCV, 2022-01-01 → today
  data/raw_index_cache/nifty500_1h.csv   — native hourly OHLCV, IST tz-naive, 15:15 stub dropped

Usage:
    python scripts/collectors/collect_nifty500_index.py
    python scripts/collectors/collect_nifty500_index.py --proxy   # generate proxy from universe mean
"""
import os, sys, time, argparse, warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.append(os.getcwd())

import upstox_client
from scripts.upstox_broker import UpstoxSandboxBroker

OUT_DIR   = 'data/raw_index_cache'
OUT_DAILY = os.path.join(OUT_DIR, 'nifty500_1d.csv')
OUT_HOURLY = os.path.join(OUT_DIR, 'nifty500_1h.csv')

INDEX_KEY   = 'NSE_INDEX|Nifty 500'
START_DATE  = date(2022, 1, 1)
CHUNK_DAYS  = 90
RATE_PAUSE  = 0.3
VALID_TODS  = {'09:15', '10:15', '11:15', '12:15', '13:15', '14:15'}

os.makedirs(OUT_DIR, exist_ok=True)


def date_chunks(start, end, n):
    out, cur = [], start
    while cur < end:
        ce = min(cur + timedelta(days=n), end)
        out.append((cur.strftime('%Y-%m-%d'), ce.strftime('%Y-%m-%d')))
        cur = ce + timedelta(days=1)
    return out


def fetch_series(v3, unit, interval, label):
    chunks = date_chunks(START_DATE, date.today(), CHUNK_DAYS)
    rows = []
    for frm, to in chunks:
        for attempt in range(3):
            try:
                resp = v3.get_historical_candle_data1(INDEX_KEY, unit, interval, to, frm)
                if resp.status == 'success' and resp.data and resp.data.candles:
                    rows.extend(resp.data.candles)
                break
            except Exception as e:
                if '429' in str(e) or 'Too Many' in str(e):
                    print(f"  rate-limited ({frm}->{to}), sleeping 5s …")
                    time.sleep(5)
                    continue
                print(f"  [warn] {label} {frm}->{to}: {str(e)[:80]}")
                break
        time.sleep(RATE_PAUSE)
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
    df = df.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
    return df


def normalize_hourly(df):
    dt = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
    df = df.copy()
    df['timestamp'] = dt
    df = df[dt.dt.strftime('%H:%M').isin(VALID_TODS)].reset_index(drop=True)
    return df


def normalize_daily(df):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.normalize()
    return df


def build_proxy_from_universe(out_daily, out_hourly):
    src = 'data/ranking_data_upstox_1h_v3_3y.csv'
    print(f"  Building proxy from {src} …")
    df = pd.read_csv(src, usecols=['DateTime', 'Query_ID', 'Market_Mean_Return', 'Next_Hour_Return'])
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df['date'] = df['DateTime'].dt.normalize()

    # Hourly proxy: mean return per query -> cumulative synthetic index
    qret = df.groupby('DateTime')['Next_Hour_Return'].mean().reset_index()
    qret = qret.sort_values('DateTime').rename(columns={'DateTime': 'timestamp', 'Next_Hour_Return': 'close'})
    qret['close'] = (1 + qret['close']).cumprod() * 1000
    qret['open'] = qret['close']
    qret['high'] = qret['close']
    qret['low'] = qret['close']
    qret['volume'] = 0
    qret['oi'] = 0
    qret[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi']].to_csv(out_hourly, index=False)
    print(f"  Saved hourly proxy → {out_hourly}  ({len(qret)} bars)")

    # Daily proxy: re-sample to daily
    dret = qret.copy()
    dret['date'] = pd.to_datetime(dret['timestamp']).dt.normalize()
    daily = dret.groupby('date').agg(
        open=('close', 'first'), high=('close', 'max'), low=('close', 'min'),
        close=('close', 'last'), volume=('volume', 'sum'), oi=('oi', 'last')
    ).reset_index().rename(columns={'date': 'timestamp'})
    daily.to_csv(out_daily, index=False)
    print(f"  Saved daily proxy  → {out_daily}   ({len(daily)} days)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--proxy', action='store_true',
                    help='Use universe equal-weight proxy instead of Upstox index fetch')
    args = ap.parse_args()

    if args.proxy:
        print("=== PROXY MODE (no Upstox fetch) ===")
        build_proxy_from_universe(OUT_DAILY, OUT_HOURLY)
        return

    print(f"Connecting to Upstox …")
    broker = UpstoxSandboxBroker()
    v3 = upstox_client.HistoryV3Api(broker.data_api_client)

    print(f"Fetching NIFTY 500 daily series …")
    d1 = fetch_series(v3, 'days', '1', 'daily')
    if d1 is None:
        print("  [ERROR] Daily fetch returned no data. Run with --proxy as fallback.")
        sys.exit(1)
    d1 = normalize_daily(d1)
    d1.to_csv(OUT_DAILY, index=False)
    print(f"  Saved {OUT_DAILY}  rows={len(d1)}  span={d1['timestamp'].min()} -> {d1['timestamp'].max()}")

    print(f"Fetching NIFTY 500 hourly series …")
    h1 = fetch_series(v3, 'hours', '1', 'hourly')
    if h1 is None:
        print("  [ERROR] Hourly fetch returned no data.")
        sys.exit(1)
    h1 = normalize_hourly(h1)
    h1.to_csv(OUT_HOURLY, index=False)
    print(f"  Saved {OUT_HOURLY}  rows={len(h1)}  span={h1['timestamp'].min()} -> {h1['timestamp'].max()}")

    print("\nSanity checks:")
    d1c = pd.read_csv(OUT_DAILY); h1c = pd.read_csv(OUT_HOURLY)
    print(f"  Daily bars/year ≈ {len(d1c) / 4.5:.0f}  (expect ~250)")
    print(f"  Hourly unique ToDs: {sorted(pd.to_datetime(h1c['timestamp']).dt.strftime('%H:%M').unique())}")
    print("Done.")


if __name__ == '__main__':
    main()

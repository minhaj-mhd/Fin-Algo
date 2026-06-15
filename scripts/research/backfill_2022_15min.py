"""
Backfill 2022 native 15-min candles (Upstox V3 minutes/15) into the existing
data/raw_upstox_cache_15min_3y/ cache. ADDITIVE: only fetches 2022-01-01..2022-12-31
and merges in front of the already-cached 2023+ rows (deduped on timestamp). 2023+ rows
are never refetched or altered.

Purpose: extend the v20 rolling-1h panel back to v10's 2022 start so the comparison is
on equal footing. Probe (scripts/research/probe_2022_15min.py) confirmed 2022 is available
(2021 is not).
"""
import os, sys, time, glob
import pandas as pd
from datetime import date, timedelta
from tqdm import tqdm
sys.path.append(os.getcwd())

import upstox_client
from scripts.upstox_broker import UpstoxSandboxBroker

CACHE_DIR  = 'data/raw_upstox_cache_15min_3y'
START      = date(2022, 1, 1)
END        = date(2022, 12, 31)
CHUNK_DAYS = 30
RATE_PAUSE = 0.25
COLS       = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi']

broker = UpstoxSandboxBroker()
v3 = upstox_client.HistoryV3Api(broker.data_api_client)


def chunks(start, end, n):
    out, cur = [], start
    while cur < end:
        ce = min(cur + timedelta(days=n), end)
        out.append((cur.strftime('%Y-%m-%d'), ce.strftime('%Y-%m-%d')))
        cur = ce + timedelta(days=1)
    return out


CHUNKS = chunks(START, END, CHUNK_DAYS)


def fetch_2022(ik):
    rows = []
    for (frm, to) in CHUNKS:
        for attempt in range(2):
            try:
                r = v3.get_historical_candle_data1(ik, 'minutes', '15', to, frm)
                if r.status == 'success' and r.data and r.data.candles:
                    rows.extend(r.data.candles)
                break
            except Exception as e:
                if '429' in str(e) or 'Too Many' in str(e):
                    time.sleep(3); continue
                break
        time.sleep(RATE_PAUSE)
    return rows


def main():
    files = sorted(glob.glob(os.path.join(CACHE_DIR, '*.csv')))
    # map cache filename stem -> a ticker the broker understands (.NS suffix)
    print(f"Backfilling 2022 into {CACHE_DIR}  ({len(files)} tickers, {len(CHUNKS)} chunks each)")
    added, skipped, empty = 0, 0, 0
    for fp in tqdm(files, desc='Tickers'):
        stem = os.path.splitext(os.path.basename(fp))[0]
        try:
            existing = pd.read_csv(fp)
            existing['timestamp'] = pd.to_datetime(existing['timestamp'], utc=True)
            if existing['timestamp'].min().year <= 2022:
                skipped += 1; continue                       # already has 2022
            ik = broker.get_instrument_key(stem + '.NS')
            raw = fetch_2022(ik)
            if not raw:
                empty += 1; continue
            new = pd.DataFrame(raw, columns=COLS)
            new['timestamp'] = pd.to_datetime(new['timestamp'], utc=True)
            # session bars only (09:15..15:15 IST start), matches existing cache
            ist = new['timestamp'].dt.tz_convert('Asia/Kolkata')
            new = new[(ist.dt.time >= pd.Timestamp('09:15').time()) &
                      (ist.dt.time <= pd.Timestamp('15:15').time())]
            merged = (pd.concat([new, existing], ignore_index=True)
                        .drop_duplicates('timestamp').sort_values('timestamp'))
            # write back in IST, same format as the existing cache
            merged['timestamp'] = merged['timestamp'].dt.tz_convert('Asia/Kolkata')
            merged.to_csv(fp, index=False)
            added += 1
        except Exception as e:
            tqdm.write(f"  [skip] {stem}: {str(e)[:80]}")
            empty += 1
    print(f"\nDone. backfilled={added}  already_had_2022={skipped}  no_2022_or_error={empty}")


if __name__ == '__main__':
    main()

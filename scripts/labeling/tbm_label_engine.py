"""
TBM Label Engine — Phase 1.

For every 1h signal bar across all tickers in the 15m cache, walks the 4
constituent sub-bars of the HOLD period to assign a triple-barrier label:
  0 = SL hit first  (or ambiguous → stop-first per D6)
  1 = TP hit first
  2 = Timeout (neither barrier hit within the 1h hold)

Outputs: data/tbm_labels_1h.parquet

Schema:
  DateTime      : 1h signal bar start time (IST naive, matches ranking CSV)
  Ticker        : e.g. RELIANCE (no .NS suffix)
  label         : 0/1/2
  realized_gross: actual hold return (from entry_price to exit price)
  realized_net  : realized_gross - COST
  entry_price   : close of last 15m bar of signal period
  atr           : ATR14 on 1h bars at signal time
  R             : barrier size = M * atr (in price units)
  weight        : label-uniqueness weight (1 / avg_concurrency at timestamp)

Usage:
    python scripts/labeling/tbm_label_engine.py [--m 1.0] [--cost_floor 3]
"""

import os, sys, glob, argparse
import numpy as np
import pandas as pd
from datetime import date as date_type

sys.path.append(os.getcwd())

# ── config ────────────────────────────────────────────────────────────────────
CACHE_DIR  = 'data/raw_upstox_cache_15min_3y'
OUT_PATH   = 'data/tbm_labels_1h.parquet'
COST       = 0.0006   # 6 bps
ATR_PERIOD = 14

# ── session map ───────────────────────────────────────────────────────────────
# For each 1h signal bar start (HH:MM), map to:
#   - entry_hm : start time of the LAST constituent 15m bar of the signal period
#                (its close = entry price)
#   - sub_hms  : start times of the 4 15m bars in the HOLD period
SIGNAL_CONFIG = {
    '09:15': ('10:00', ['10:15', '10:30', '10:45', '11:00']),
    '10:15': ('11:00', ['11:15', '11:30', '11:45', '12:00']),
    '11:15': ('12:00', ['12:15', '12:30', '12:45', '13:00']),
    '12:15': ('13:00', ['13:15', '13:30', '13:45', '14:00']),
    '13:15': ('14:00', ['14:15', '14:30', '14:45', '15:00']),
}

SIGNAL_ORDER = ['09:15', '10:15', '11:15', '12:15', '13:15']

# ── helpers ───────────────────────────────────────────────────────────────────

def _hm_to_offset(hm: str) -> pd.Timedelta:
    h, m = int(hm[:2]), int(hm[3:])
    return pd.Timedelta(hours=h, minutes=m)


def load_15m(ticker: str) -> pd.DataFrame | None:
    path = os.path.join(CACHE_DIR, f'{ticker}.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df['timestamp'] = (pd.to_datetime(df['timestamp'], utc=True)
                       .dt.tz_convert('Asia/Kolkata')
                       .dt.tz_localize(None))
    df = df.rename(columns={'open':'Open','high':'High','low':'Low',
                             'close':'Close','volume':'Volume'})
    df = df.set_index('timestamp').sort_index()
    df = df[['Open','High','Low','Close','Volume']]
    return df


def build_signal_bars(df15: pd.DataFrame) -> pd.DataFrame:
    """Build 1h OHLCV signal bars from 15m constituent bars.
    Returns one row per (date, signal_time) with OHLCV + entry_price.
    """
    records = []
    day_groups = df15.groupby(df15.index.date)

    for day_date, day_df in day_groups:
        day_index_set = set(day_df.index)
        base = pd.Timestamp(year=day_date.year, month=day_date.month, day=day_date.day)

        for sig_hm, (entry_hm, sub_hms) in SIGNAL_CONFIG.items():
            # The 4 constituent bars of the signal period
            # signal bar at sig_hm covers sig_hm to sig_hm+60; constituents:
            # sig_hm, sig_hm+15, sig_hm+30, sig_hm+45
            h, m = int(sig_hm[:2]), int(sig_hm[3:])
            constituent_ts = [
                base + pd.Timedelta(hours=h, minutes=m + 15*i)
                for i in range(4)
            ]
            if not all(ts in day_index_set for ts in constituent_ts):
                continue

            rows = [day_df.loc[ts] for ts in constituent_ts]
            entry_ts = constituent_ts[-1]  # last constituent = entry bar
            entry_price = day_df.loc[entry_ts, 'Close']

            records.append({
                'date':        day_date,
                'signal_time': sig_hm,
                'sig_Open':    rows[0]['Open'],
                'sig_High':    max(r['High']   for r in rows),
                'sig_Low':     min(r['Low']    for r in rows),
                'sig_Close':   rows[-1]['Close'],
                'sig_Volume':  sum(r['Volume'] for r in rows),
                'entry_price': entry_price,
            })

    return pd.DataFrame(records)


def add_atr(df1h: pd.DataFrame, period: int = ATR_PERIOD) -> pd.DataFrame:
    """Add ATR{period} computed on the 1h signal bars (per ticker, time-sorted)."""
    df = df1h.copy().reset_index(drop=True)
    prev_close = df['sig_Close'].shift(1)
    tr = pd.concat([
        df['sig_High'] - df['sig_Low'],
        (df['sig_High'] - prev_close).abs(),
        (df['sig_Low']  - prev_close).abs(),
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(period, min_periods=period).mean()
    return df


def label_one_bar(df15_day: pd.DataFrame, day_date,
                  sig_hm: str, entry_price: float,
                  atr: float, m: float, cost_floor_mult: int) -> dict | None:
    """
    Returns a label dict for one (ticker, date, signal_time) bar, or None if
    the bar fails cost-floor or lacks sub-bar data.
    """
    if np.isnan(atr) or atr <= 0:
        return None
    R = m * atr
    # Cost floor: barrier must be ≥ cost_floor_mult × COST (in absolute return terms)
    if R / entry_price < cost_floor_mult * COST:
        return None

    TP = entry_price + R
    SL = entry_price - R

    _, sub_hms = SIGNAL_CONFIG[sig_hm]
    base = pd.Timestamp(year=day_date.year, month=day_date.month, day=day_date.day)

    label = 2       # default: timeout
    realized = np.nan
    ambiguous = False
    touch_bar = None

    for i, hm in enumerate(sub_hms):
        h, mn = int(hm[:2]), int(hm[3:])
        ts = base + pd.Timedelta(hours=h, minutes=mn)
        if ts not in df15_day.index:
            label = 2
            break
        bar = df15_day.loc[ts]
        hi, lo, cl = float(bar['High']), float(bar['Low']), float(bar['Close'])

        hit_tp = hi >= TP
        hit_sl = lo <= SL

        if hit_tp and hit_sl:
            # Ambiguous: both barriers inside same 15m bar → stop-first (D6)
            ambiguous = True
            label = 0
            realized = (SL - entry_price) / entry_price
            touch_bar = hm
            break
        elif hit_tp:
            label = 1
            realized = (TP - entry_price) / entry_price
            touch_bar = hm
            break
        elif hit_sl:
            label = 0
            realized = (SL - entry_price) / entry_price
            touch_bar = hm
            break

    if label == 2:
        # Timeout: realized = close of last sub-bar
        last_hm = sub_hms[-1]
        h, mn = int(last_hm[:2]), int(last_hm[3:])
        ts = base + pd.Timedelta(hours=h, minutes=mn)
        if ts in df15_day.index:
            realized = (float(df15_day.loc[ts, 'Close']) - entry_price) / entry_price
        touch_bar = last_hm

    if np.isnan(realized):
        return None

    return {
        'label':      label,
        'realized_gross': realized,
        'realized_net':   realized - COST,
        'entry_price':    entry_price,
        'atr':            atr,
        'R':              R,
        'ambiguous':      ambiguous,
        'touch_bar':      touch_bar,
    }


def compute_uniqueness_weights(labels_df: pd.DataFrame) -> pd.DataFrame:
    """
    Uniqueness weight = 1 / (number of tickers at same DateTime).
    Since all samples at the same timestamp have identical hold windows [T, T+1h],
    their concurrency = N_tickers_at_T. Weight = 1/N.
    """
    n_per_dt = labels_df.groupby('DateTime')['Ticker'].transform('count')
    labels_df['weight'] = 1.0 / n_per_dt
    return labels_df


# ── main ──────────────────────────────────────────────────────────────────────

def main(m: float = 1.0, cost_floor_mult: int = 3):
    print("=" * 64)
    print(f"TBM Label Engine  m={m}  cost_floor={cost_floor_mult}×  COST={COST*10000:.0f}bps")
    print("=" * 64)

    csvs = sorted(glob.glob(os.path.join(CACHE_DIR, '*.csv')))
    tickers = [os.path.basename(p).replace('.csv', '') for p in csvs]
    print(f"  Tickers found: {len(tickers)}")

    os.makedirs('data', exist_ok=True)
    all_records = []
    skipped = 0

    for i, tkr in enumerate(tickers, 1):
        if i % 20 == 0:
            print(f"  [{i}/{len(tickers)}] {tkr} ...")

        df15 = load_15m(tkr)
        if df15 is None or df15.empty:
            skipped += 1
            continue

        df1h = build_signal_bars(df15)
        if df1h.empty:
            skipped += 1
            continue

        df1h = add_atr(df1h)

        day_groups = {d: grp.set_index('timestamp') if 'timestamp' in grp.columns
                      else grp
                      for d, grp in df15.groupby(df15.index.date)}

        for _, row in df1h.iterrows():
            d = row['date']
            sig_hm = row['signal_time']
            entry_price = row['entry_price']
            atr_val = row['atr']

            day_df = df15[df15.index.date == d]
            if day_df.empty:
                continue

            result = label_one_bar(
                day_df, d, sig_hm, entry_price, atr_val, m, cost_floor_mult
            )
            if result is None:
                continue

            # Build DateTime matching the 1h feature CSV convention
            h, mn = int(sig_hm[:2]), int(sig_hm[3:])
            dt = pd.Timestamp(year=d.year, month=d.month, day=d.day, hour=h, minute=mn)

            all_records.append({
                'DateTime':       dt,
                'Ticker':         tkr,
                **result,
            })

    print(f"\n  Labeling complete.")
    print(f"  Total records: {len(all_records):,}  |  Tickers skipped: {skipped}")

    if not all_records:
        print("  ❌ No records generated — check CACHE_DIR and config.")
        return

    df_out = pd.DataFrame(all_records)
    df_out = compute_uniqueness_weights(df_out)

    # Summary stats
    vc = df_out['label'].value_counts(normalize=True).sort_index()
    print(f"\n  Label distribution:")
    print(f"    SL (0): {vc.get(0,0):.1%}")
    print(f"    TP (1): {vc.get(1,0):.1%}")
    print(f"    TO (2): {vc.get(2,0):.1%}")
    print(f"    Ambiguous (stop-first): {df_out['ambiguous'].sum():,} ({df_out['ambiguous'].mean():.1%})")
    print(f"\n  Mean realized gross (all):  {df_out['realized_gross'].mean()*10000:+.2f} bps")
    print(f"  Mean realized net   (all):  {df_out['realized_net'].mean()*10000:+.2f} bps")
    print(f"  Mean weight:                {df_out['weight'].mean():.4f}")

    df_out.to_parquet(OUT_PATH, index=False)
    print(f"\n  ✅ Saved → {OUT_PATH}  ({os.path.getsize(OUT_PATH)/1e6:.1f} MB)")
    print(f"  Shape: {df_out.shape}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--m',           type=float, default=1.0, help='ATR multiplier')
    ap.add_argument('--cost_floor',  type=int,   default=3,   help='Min barrier as N×cost')
    args = ap.parse_args()
    main(m=args.m, cost_floor_mult=args.cost_floor)

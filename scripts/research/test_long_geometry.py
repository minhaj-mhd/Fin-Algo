"""
RESEARCH: Does a smaller barrier / longer horizon make LONGS predictable?

User hypothesis: down-moves are violent (hit -1xATR within 1h -> short TP
resolves), but up-moves grind slowly (need 3-4 candles for +1%). So the 1h/1ATR
label throws away genuine slow climbs as TIMEOUTS, training the long model to
predict only rare news-spikes (unpredictable). Reducing the long TP target
and/or extending the horizon should let the predictable grind become the label.

This tests SEVERAL (m, horizon) configs, STRICTLY INTRADAY (no overnight), and
for each measures:
  - timeout rate (lower = more bars resolve)
  - unconditional long net WR
  - the SEPARATION SPREAD: long WR across daily-trend / momentum / IBS strata.
    At 1h/1ATR every stratum was ~42% (spread ~1pp => no signal). If a new
    geometry widens the spread (e.g. uptrend 50% vs downtrend 43%), a separable
    long signal EXISTS at that geometry and a full retrain is justified.

Reads 15m cache directly; reuses tbm_label_engine helpers.
Output: data/model_analysis/tbm_1h/long_geometry_scan.txt (+ console)
"""
import os, sys, glob
import numpy as np
import pandas as pd

sys.path.append(os.getcwd())
from scripts.labeling.tbm_label_engine import (
    load_15m, build_signal_bars, add_atr, SIGNAL_CONFIG, SIGNAL_ORDER
)

CACHE = 'data/raw_upstox_cache_15min_3y'
COST = 0.0006

# (label, m_barrier, horizon_hours)
CONFIGS = [
    ('m1.0_H1 (baseline)', 1.0, 1),
    ('m0.5_H1 (smaller)',  0.5, 1),
    ('m1.0_H2 (longer)',   1.0, 2),
    ('m0.5_H2 (small+long)', 0.5, 2),
    ('m0.75_H3 (mod+3h)',  0.75, 3),
    ('m1.0_H3 (1pct+3h)',  1.0, 3),
]

def wr(x): return float((x > 0).mean()) if len(x) else np.nan


def daily_trend_for_ticker(path):
    """Daily trend indicators as of the PRIOR complete day (no lookahead)."""
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df['timestamp'] = (pd.to_datetime(df['timestamp'], utc=True)
                       .dt.tz_convert('Asia/Kolkata').dt.tz_localize(None))
    df['date'] = df['timestamp'].dt.date
    daily = df.groupby('date').agg(d_close=('close', 'last')).reset_index()
    daily = daily.sort_values('date').reset_index(drop=True)
    if len(daily) < 60:
        return None
    c = daily['d_close']
    sma20 = c.rolling(20, min_periods=20).mean()
    daily['roc10'] = (c / c.shift(10) - 1.0).shift(1)
    daily['above_sma20'] = (c > sma20).astype(float).shift(1)
    daily['Ticker'] = os.path.basename(path).replace('.csv', '')
    return daily[['date', 'Ticker', 'roc10', 'above_sma20']]


def label_long_path(intraday_15m_day, entry_ts, entry_price, atr, m, horizon_h):
    """Walk forward up to horizon_h*4 intraday 15m bars after entry_ts.
    Returns (label, long_realized_gross) or None if insufficient forward bars.
    label: 1=TP(up) first, 0=SL(down) first, 2=timeout.
    Exit at first touch; timeout -> exit at last available forward bar (intraday).
    """
    if np.isnan(atr) or atr <= 0:
        return None
    R = m * atr
    TP = entry_price + R
    SL = entry_price - R
    n_need = horizon_h * 4
    fwd = intraday_15m_day[intraday_15m_day.index > entry_ts]
    if len(fwd) < n_need:
        return None                      # require FULL intraday horizon (no overnight, unbiased)
    fwd = fwd.iloc[:n_need]
    for ts, bar in fwd.iterrows():
        hi, lo = float(bar['High']), float(bar['Low'])
        hit_tp = hi >= TP
        hit_sl = lo <= SL
        if hit_tp and hit_sl:
            return 0, (SL - entry_price) / entry_price     # stop-first (conservative)
        if hit_tp:
            return 1, (TP - entry_price) / entry_price
        if hit_sl:
            return 0, (SL - entry_price) / entry_price
    # timeout: exit at last forward bar close
    last_close = float(fwd.iloc[-1]['Close'])
    return 2, (last_close - entry_price) / entry_price


def main():
    paths = sorted(glob.glob(os.path.join(CACHE, '*.csv')))
    print(f"Tickers: {len(paths)}  Configs: {len(CONFIGS)}")

    # accumulate per-config records: list of dicts {DateTime,Ticker,label,gross}
    recs = {c[0]: [] for c in CONFIGS}

    for i, p in enumerate(paths, 1):
        tkr = os.path.basename(p).replace('.csv', '')
        df15 = load_15m(tkr)
        if df15 is None or df15.empty:
            continue
        df1h = build_signal_bars(df15)
        if df1h.empty:
            continue
        df1h = add_atr(df1h)

        # index 15m by day for fast slicing
        by_day = {d: g for d, g in df15.groupby(df15.index.date)}

        for _, row in df1h.iterrows():
            d = row['date']; sig_hm = row['signal_time']
            entry_price = row['entry_price']; atr_val = row['atr']
            if d not in by_day:
                continue
            day15 = by_day[d]
            entry_hm = SIGNAL_CONFIG[sig_hm][0]
            eh, em = int(entry_hm[:2]), int(entry_hm[3:])
            entry_ts = pd.Timestamp(d.year, d.month, d.day, eh, em)
            if entry_ts not in day15.index:
                continue
            h, mn = int(sig_hm[:2]), int(sig_hm[3:])
            dt = pd.Timestamp(d.year, d.month, d.day, h, mn)
            for name, m, H in CONFIGS:
                res = label_long_path(day15, entry_ts, entry_price, atr_val, m, H)
                if res is None:
                    continue
                lab, gross = res
                recs[name].append((dt, tkr, lab, gross))

        if i % 40 == 0:
            print(f"  [{i}/{len(paths)}] {tkr}")

    # ── load daily trend + IBS for stratification ────────────────────────────
    print("\nLoading daily-trend + IBS for stratification ...")
    dt_parts = []
    for p in paths:
        d = daily_trend_for_ticker(p)
        if d is not None:
            dt_parts.append(d)
    daily_all = pd.concat(dt_parts, ignore_index=True)
    va = pd.read_parquet('data/tbm_feature_views/A_meanrev.parquet')
    va['DateTime'] = pd.to_datetime(va['DateTime'])
    va = va[['DateTime', 'Ticker', 'IBS']]

    out_lines = []
    def emit(s):
        print(s); out_lines.append(s)

    emit("=" * 84)
    emit(f"{'config':22s} {'n':>8s} {'TO%':>6s} {'uncond':>7s} | "
         f"{'aSMA20':>7s} {'bSMA20':>7s} {'ROC>0':>7s} {'ROC<0':>7s} {'ovrsld':>7s} {'SPREAD':>7s}")
    emit("=" * 84)

    for name, m, H in CONFIGS:
        df = pd.DataFrame(recs[name], columns=['DateTime', 'Ticker', 'label', 'gross'])
        if df.empty:
            emit(f"{name:22s}  (no records)")
            continue
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df['date'] = df['DateTime'].dt.date
        df['long_net'] = df['gross'] - COST
        to_rate = (df['label'] == 2).mean()
        uncond = wr(df['long_net'].values)

        mm = df.merge(daily_all, on=['date', 'Ticker'], how='left')
        mm = mm.merge(va, on=['DateTime', 'Ticker'], how='left')
        mm = mm.dropna(subset=['above_sma20', 'roc10'])

        ib_lo = mm['IBS'] <= mm['IBS'].quantile(0.33)
        strata = {
            'aSMA20': mm[mm['above_sma20'] == 1]['long_net'].values,
            'bSMA20': mm[mm['above_sma20'] == 0]['long_net'].values,
            'ROC>0':  mm[mm['roc10'] > 0]['long_net'].values,
            'ROC<0':  mm[mm['roc10'] < 0]['long_net'].values,
            'ovrsld': mm[ib_lo]['long_net'].values,
        }
        wrs = {k: wr(v) for k, v in strata.items()}
        # separation spread among the trend strata (the signal indicator)
        trend_wrs = [wrs['aSMA20'], wrs['bSMA20'], wrs['ROC>0'], wrs['ROC<0']]
        trend_wrs = [w for w in trend_wrs if not np.isnan(w)]
        spread = (max(trend_wrs) - min(trend_wrs)) * 100 if trend_wrs else np.nan

        emit(f"{name:22s} {len(df):8,d} {to_rate*100:5.1f}% {uncond:6.1%} | "
             f"{wrs['aSMA20']:6.1%} {wrs['bSMA20']:6.1%} {wrs['ROC>0']:6.1%} "
             f"{wrs['ROC<0']:6.1%} {wrs['ovrsld']:6.1%} {spread:5.1f}pp")

    emit("=" * 84)
    emit("READ: SPREAD ~1pp => no separable long signal (same as 1h/1ATR).")
    emit("      SPREAD >5pp with sensible direction (uptrend>downtrend) => signal")
    emit("      EXISTS at that geometry -> full purged-WF retrain justified.")

    os.makedirs('data/model_analysis/tbm_1h', exist_ok=True)
    with open('data/model_analysis/tbm_1h/long_geometry_scan.txt', 'w', encoding='utf-8') as f:
        f.write("\n".join(out_lines))
    print("\nSaved -> data/model_analysis/tbm_1h/long_geometry_scan.txt")


if __name__ == '__main__':
    main()

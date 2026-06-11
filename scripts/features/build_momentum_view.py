"""
Momentum/Breakout Feature View D — long-side specialist.

Computes direction-specific long features absent from the existing 86-feature matrix.
These are momentum/continuation signals, not mean-reversion oscillators.

Features (26 total):
  From 1h OHLCV (per-ticker rolling, no lookahead):
    ADX_14, Plus_DI_14, Minus_DI_14, DI_Diff_14
    Breakout_3H, Breakout_5H, Breakout_10H
    High_Rank_10H, High_Rank_20H
    Vol_Trend_3H, Vol_Trend_5H
    Consec_Up
    MOM_Accel_3H
    RS_1H, RS_Cumul_3H, RS_Cumul_5H

  From 15m cache (gap and opening range):
    Gap_Pct, Gap_Up, Gap_Down, Gap_Abs
    ORB_High, ORB_Low, ORB_Width
    ORB_Breakout, Dist_ORB_High, Dist_ORB_Low

Reads:
  data/ranking_data_upstox_1h_v3_3y.csv
  data/raw_upstox_cache_15min_3y/{TICKER}.csv
  data/tbm_labels_1h.parquet

Writes:
  data/tbm_feature_views/D_momentum.parquet

Usage:
    python scripts/features/build_momentum_view.py
"""

import os, sys, glob
import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.append(os.getcwd())

# ── config ────────────────────────────────────────────────────────────────────
FEAT_FILE   = 'data/ranking_data_upstox_1h_v3_3y.csv'
CACHE_DIR   = 'data/raw_upstox_cache_15min_3y'
LABELS_FILE = 'data/tbm_labels_1h.parquet'
OUT_PATH    = 'data/tbm_feature_views/D_momentum.parquet'
OUT_DIR     = 'data/tbm_feature_views'

# Signal bar start → entry bar (last 15m bar of signal period)
# entry_price = close of this 15m bar
SIGNAL_TO_ENTRY = {
    '09:15': '10:00',
    '10:15': '11:00',
    '11:15': '12:00',
    '12:15': '13:00',
    '13:15': '14:00',
}

META_COLS = ['DateTime', 'Ticker', 'label', 'realized_gross', 'realized_net',
             'entry_price', 'atr', 'R', 'weight', 'YearMonth']

# ── ADX computation ───────────────────────────────────────────────────────────

def _wilder_smooth(arr: np.ndarray, n: int) -> np.ndarray:
    """Wilder's smoothing (used in ADX/DI calculation)."""
    out = np.full(len(arr), np.nan)
    # Find first window of n non-nan values
    finite = np.where(np.isfinite(arr))[0]
    if len(finite) < n:
        return out
    start = finite[0]
    if start + n - 1 >= len(arr):
        return out
    out[start + n - 1] = np.nanmean(arr[start:start + n])
    for i in range(start + n, len(arr)):
        if np.isfinite(arr[i]):
            out[i] = out[i-1] - out[i-1] / n + arr[i]
        else:
            out[i] = out[i-1]
    return out


def compute_adx(hi: pd.Series, lo: pd.Series, cl: pd.Series, period: int = 14):
    """Returns (ADX, +DI, -DI) as Series aligned to input index."""
    hi, lo, cl = hi.values, lo.values, cl.values
    n = len(hi)

    prev_hi = np.empty(n); prev_hi[:] = np.nan; prev_hi[1:] = hi[:-1]
    prev_lo = np.empty(n); prev_lo[:] = np.nan; prev_lo[1:] = lo[:-1]
    prev_cl = np.empty(n); prev_cl[:] = np.nan; prev_cl[1:] = cl[:-1]

    # True Range
    tr = np.maximum.reduce([
        hi - lo,
        np.abs(hi - prev_cl),
        np.abs(lo - prev_cl),
    ])

    # Directional Movement
    up   = hi - prev_hi
    down = prev_lo - lo
    plus_dm  = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    plus_dm[0] = minus_dm[0] = np.nan
    tr[0] = np.nan

    # Wilder smooth
    tr_s   = _wilder_smooth(tr,       period)
    pdm_s  = _wilder_smooth(plus_dm,  period)
    ndm_s  = _wilder_smooth(minus_dm, period)

    with np.errstate(divide='ignore', invalid='ignore'):
        pdi = 100.0 * pdm_s / np.where(tr_s > 0, tr_s, np.nan)
        ndi = 100.0 * ndm_s / np.where(tr_s > 0, tr_s, np.nan)
        dx  = 100.0 * np.abs(pdi - ndi) / np.where((pdi + ndi) > 0, pdi + ndi, np.nan)

    adx = _wilder_smooth(np.where(np.isfinite(dx), dx, np.nan), period)

    idx = pd.RangeIndex(n)
    return pd.Series(adx, index=idx), pd.Series(pdi, index=idx), pd.Series(ndi, index=idx)


# ── per-ticker 1h features ────────────────────────────────────────────────────

def build_1h_momentum(df_tkr: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 1h-derived momentum features for one ticker.
    df_tkr is sorted by DateTime, with Open/High/Low/Close/Volume/Return/Market_Mean_Return.
    Returns df_tkr with new columns added.
    """
    df = df_tkr.copy().reset_index(drop=True)

    # 1. ADX
    adx, pdi, ndi = compute_adx(df['High'], df['Low'], df['Close'])
    df['ADX_14']      = adx.values
    df['Plus_DI_14']  = pdi.values
    df['Minus_DI_14'] = ndi.values
    df['DI_Diff_14']  = pdi.values - ndi.values

    # 2. N-bar channel breakout
    # Current close vs previous N bars' high (shift(1) avoids current bar)
    for n in [3, 5, 10]:
        prev_high_max = df['High'].shift(1).rolling(n, min_periods=n).max()
        df[f'Breakout_{n}H'] = (df['Close'] > prev_high_max).astype(float)

    # 3. Percentile rank of close in N-bar rolling window
    for n in [10, 20]:
        df[f'High_Rank_{n}H'] = df['Close'].rolling(n, min_periods=n).rank(pct=True)

    # 4. Volume directional bias: sum(Volume * sign(Return)) over N bars
    ret_sign = np.sign(df['Return'].fillna(0))
    vol_signed = df['Volume'] * ret_sign
    for n in [3, 5]:
        df[f'Vol_Trend_{n}H'] = vol_signed.rolling(n, min_periods=n).sum()
    # Normalise by mean volume so it's scale-invariant across tickers
    mean_vol = df['Volume'].rolling(20, min_periods=5).mean()
    mean_vol = mean_vol.replace(0, np.nan)
    for n in [3, 5]:
        df[f'Vol_Trend_{n}H'] = df[f'Vol_Trend_{n}H'] / mean_vol

    # 5. Consecutive up-bars (capped at 5)
    consec = np.zeros(len(df), dtype=float)
    for i in range(1, len(df)):
        r = df['Return'].iloc[i]
        if np.isfinite(r) and r > 0:
            consec[i] = min(consec[i-1] + 1, 5)
        else:
            consec[i] = 0
    df['Consec_Up'] = consec

    # 6. Momentum acceleration: ROC_3 minus its own 3-bar lag
    roc3 = df['Close'].pct_change(3)
    df['MOM_Accel_3H'] = roc3 - roc3.shift(3)

    # 7. Relative strength vs cross-sectional market mean
    df['RS_1H'] = df['Return'] - df['Market_Mean_Return']
    log_excess = np.log1p(df['Return'].clip(-0.5, 1.0)) - \
                 np.log1p(df['Market_Mean_Return'].clip(-0.5, 1.0))
    for n in [3, 5]:
        df[f'RS_Cumul_{n}H'] = log_excess.rolling(n, min_periods=n).sum()

    return df


# ── 15m gap and ORB features ──────────────────────────────────────────────────

def load_15m_ticker(ticker: str) -> pd.DataFrame | None:
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
    return df


def build_gap_orb_features(ticker: str) -> pd.DataFrame:
    """
    Build one row per (signal_datetime, ticker) with gap and ORB features.
    Returns a DataFrame with DateTime, Gap_Pct, Gap_Up, Gap_Down, Gap_Abs,
    ORB_High, ORB_Low, ORB_Width, ORB_Breakout, Dist_ORB_High, Dist_ORB_Low.
    """
    df15 = load_15m_ticker(ticker)
    if df15 is None or df15.empty:
        return pd.DataFrame()

    signal_times_hm = ['09:15', '10:15', '11:15', '12:15', '13:15']
    records = []

    dates = sorted(set(df15.index.date))
    for i, d in enumerate(dates):
        day_df = df15[df15.index.date == d]
        base = pd.Timestamp(year=d.year, month=d.month, day=d.day)

        # ── Opening Range (09:15 and 09:30 15m bars) ──
        orb_ts = [base + pd.Timedelta(hours=9, minutes=15),
                  base + pd.Timedelta(hours=9, minutes=30)]
        orb_bars = [day_df.loc[ts] for ts in orb_ts if ts in day_df.index]

        if len(orb_bars) < 1:
            continue  # no data for this day
        orb_high  = max(b['High'] for b in orb_bars)
        orb_low   = min(b['Low']  for b in orb_bars)
        orb_mid   = (orb_high + orb_low) / 2
        orb_width = (orb_high - orb_low) / orb_mid if orb_mid > 0 else np.nan

        # ── Overnight gap ──
        # Gap = first bar open / previous day's last close
        open_915_ts = base + pd.Timedelta(hours=9, minutes=15)
        if open_915_ts not in day_df.index:
            continue
        open_price = day_df.loc[open_915_ts, 'Open']
        gap_pct = np.nan
        if i > 0:
            prev_d   = dates[i - 1]
            prev_day = df15[df15.index.date == prev_d]
            if not prev_day.empty:
                prev_close = prev_day['Close'].iloc[-1]
                if prev_close > 0:
                    gap_pct = open_price / prev_close - 1

        # ── Per-signal features ──
        for sig_hm in signal_times_hm:
            h, m = int(sig_hm[:2]), int(sig_hm[3:])
            entry_hm = SIGNAL_TO_ENTRY[sig_hm]
            eh, em = int(entry_hm[:2]), int(entry_hm[3:])
            entry_ts = base + pd.Timedelta(hours=eh, minutes=em)

            if entry_ts not in day_df.index:
                continue
            entry_price = day_df.loc[entry_ts, 'Close']

            sig_dt = base + pd.Timedelta(hours=h, minutes=m)
            orb_breakout  = float(entry_price > orb_high)
            dist_orb_high = (entry_price - orb_high) / entry_price if entry_price > 0 else np.nan
            dist_orb_low  = (entry_price - orb_low)  / entry_price if entry_price > 0 else np.nan

            records.append({
                'DateTime':     sig_dt,
                'Ticker':       ticker,
                'Gap_Pct':      gap_pct,
                'Gap_Up':       float(gap_pct > 0.002) if np.isfinite(gap_pct) else np.nan,
                'Gap_Down':     float(gap_pct < -0.002) if np.isfinite(gap_pct) else np.nan,
                'Gap_Abs':      abs(gap_pct) if np.isfinite(gap_pct) else np.nan,
                'ORB_High':     orb_high,
                'ORB_Low':      orb_low,
                'ORB_Width':    orb_width,
                'ORB_Breakout': orb_breakout,
                'Dist_ORB_High': dist_orb_high,
                'Dist_ORB_Low':  dist_orb_low,
            })

    return pd.DataFrame(records)


# ── main ──────────────────────────────────────────────────────────────────────

VIEW_D_COLS = [
    'ADX_14', 'Plus_DI_14', 'Minus_DI_14', 'DI_Diff_14',
    'Breakout_3H', 'Breakout_5H', 'Breakout_10H',
    'High_Rank_10H', 'High_Rank_20H',
    'Vol_Trend_3H', 'Vol_Trend_5H',
    'Consec_Up', 'MOM_Accel_3H',
    'RS_1H', 'RS_Cumul_3H', 'RS_Cumul_5H',
    'Gap_Pct', 'Gap_Up', 'Gap_Down', 'Gap_Abs',
    'ORB_High', 'ORB_Low', 'ORB_Width',
    'ORB_Breakout', 'Dist_ORB_High', 'Dist_ORB_Low',
]


def main():
    print("=" * 64)
    print("Momentum/Breakout Feature View D Builder")
    print("=" * 64)

    # ── Load 1h features ──
    print(f"\nLoading {FEAT_FILE} ...")
    df_feat = pd.read_csv(FEAT_FILE)
    df_feat['Ticker']   = df_feat['Ticker'].str.replace(r'\.NS$', '', regex=True)
    df_feat['DateTime'] = pd.to_datetime(df_feat['DateTime'])
    df_feat['YearMonth'] = df_feat['DateTime'].dt.to_period('M').astype(str)
    print(f"  {df_feat.shape[0]:,} rows | {df_feat['Ticker'].nunique()} tickers")

    # Required columns from 1h CSV
    required = ['Open', 'High', 'Low', 'Close', 'Volume',
                'Return', 'Market_Mean_Return']
    missing = [c for c in required if c not in df_feat.columns]
    if missing:
        raise ValueError(f"Missing columns in 1h CSV: {missing}")

    # ── Step 1: Compute per-ticker 1h momentum features ──
    print("\nComputing 1h momentum features per ticker ...")
    tickers = sorted(df_feat['Ticker'].unique())
    parts_1h = []
    for i, tkr in enumerate(tickers, 1):
        if i % 30 == 0:
            print(f"  [{i}/{len(tickers)}] {tkr}")
        sub = df_feat[df_feat['Ticker'] == tkr].sort_values('DateTime').copy()
        sub = build_1h_momentum(sub)
        parts_1h.append(sub)

    df_1h_momentum = pd.concat(parts_1h, ignore_index=True)
    print(f"  Done. {df_1h_momentum.shape[0]:,} rows")

    # ── Step 2: Compute per-ticker gap/ORB features from 15m cache ──
    print(f"\nComputing gap/ORB features from {CACHE_DIR} ...")
    csvs    = sorted(glob.glob(os.path.join(CACHE_DIR, '*.csv')))
    cache_tickers = [os.path.basename(p).replace('.csv', '') for p in csvs]
    in_both = [t for t in cache_tickers if t in set(tickers)]
    print(f"  {len(cache_tickers)} cache tickers | {len(in_both)} overlap with 1h CSV")

    parts_orb = []
    for i, tkr in enumerate(in_both, 1):
        if i % 30 == 0:
            print(f"  [{i}/{len(in_both)}] {tkr}")
        df_orb = build_gap_orb_features(tkr)
        if not df_orb.empty:
            parts_orb.append(df_orb)

    df_gap_orb = pd.concat(parts_orb, ignore_index=True)
    df_gap_orb['DateTime'] = pd.to_datetime(df_gap_orb['DateTime'])
    print(f"  Done. {df_gap_orb.shape[0]:,} gap/ORB rows")

    # ── Step 3: Merge 1h momentum + gap/ORB ──
    print("\nMerging 1h momentum with gap/ORB ...")
    df_merged = df_1h_momentum.merge(
        df_gap_orb, on=['DateTime', 'Ticker'], how='left'
    )
    gap_coverage = df_merged['Gap_Pct'].notna().mean()
    orb_coverage = df_merged['ORB_Breakout'].notna().mean()
    print(f"  Gap coverage: {gap_coverage:.1%}  |  ORB coverage: {orb_coverage:.1%}")

    # ── Step 4: Join with TBM labels ──
    print(f"\nJoining with {LABELS_FILE} ...")
    df_labels = pd.read_parquet(LABELS_FILE)
    df_labels['DateTime'] = pd.to_datetime(df_labels['DateTime'])

    df_out = df_merged.merge(
        df_labels[['DateTime', 'Ticker', 'label', 'realized_gross', 'realized_net',
                   'entry_price', 'atr', 'R', 'weight']],
        on=['DateTime', 'Ticker'],
        how='inner',
    )
    print(f"  Merged: {df_out.shape[0]:,} rows")

    # ── Step 5: Build final view ──
    available_d = [c for c in VIEW_D_COLS if c in df_out.columns]
    missing_d   = [c for c in VIEW_D_COLS if c not in df_out.columns]
    if missing_d:
        print(f"  Missing features: {missing_d}")

    key_cols = ['DateTime', 'Ticker', 'label', 'realized_gross', 'realized_net',
                'entry_price', 'atr', 'R', 'weight', 'YearMonth']
    df_view = df_out[key_cols + available_d].copy()

    # Coverage report
    print(f"\n  View D feature summary:")
    print(f"    Total features: {len(available_d)}")
    for col in available_d:
        pct_valid = df_view[col].notna().mean()
        print(f"    {col:<22}: {pct_valid:.1%} valid")

    label_vc = df_view['label'].value_counts(normalize=True).sort_index()
    print(f"\n  Label balance: SL:{label_vc.get(0,0):.1%}  TP:{label_vc.get(1,0):.1%}  TO:{label_vc.get(2,0):.1%}")

    os.makedirs(OUT_DIR, exist_ok=True)
    df_view.to_parquet(OUT_PATH, index=False)
    size_mb = os.path.getsize(OUT_PATH) / 1e6
    print(f"\n  Saved: {OUT_PATH}  ({df_view.shape[0]:,} rows x {df_view.shape[1]} cols, {size_mb:.1f} MB)")
    print("\nNext: python scripts/validation/purged_wf_tbm.py --side long")


if __name__ == '__main__':
    main()

"""
Build the dense aligned tensor panel for the dual-resolution cross-sectional
transformer (Conv-2026-06-12-Sophisticated-Transformer).

Rebuilds BOTH timeframes from ONE source (raw 15m cache), reproducing the vetted
pipeline in scripts/collectors/rebuild_aligned_datasets.py, with two additions the
user asked for:
  * KEEP the 14:15 (2:15-3:15) 1h context candle  -> present in the input sequence,
    label = NaN (never a decision point; it is already the 13:15 decision's label).
  * Emit CLOCK-TIME slot ids per bar (1h: 6 slots 09:15..14:15 ; 15m: 25 slots
    09:15..15:15) for a learned time-of-day embedding in the model.

Production CSVs are NOT touched (we build panels in-memory from the raw cache).

Outputs (data/transformer_panel/):
  X_1h.npy   (T1, N, F)  X_15m.npy (T15, N, F)  Y_ret.npy (T1, N)
  slot_1h.npy (T1,)  slot_15m.npy (T15,)  end15.npy (T1,)
  ts_1h.npy (T1,) ts_15m.npy (T15,)  date_idx.npy (T1,)
  macro.npy (D, M)  macro_dates.npy (D,)  sector_ids.npy (N,)  meta.json
"""
import os, sys, glob, json, warnings
from datetime import time as dtime
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')
sys.path.append(os.getcwd())
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from scripts.feature_utils import compute_features
from scripts.features.build_feature_views import VIEW_A, VIEW_B, VIEW_C
from scripts.sector_map import SECTOR_MAP

CACHE_DIR  = 'data/raw_upstox_cache_15min_3y'
MACRO_FILE = 'data/ranking_data_daily_macro_v3.csv'
OUT_DIR    = 'data/transformer_panel'

MKT_OPEN  = dtime(9, 15)
M15_CLOSE = dtime(15, 15)
VALID_1H_TODS = {'09:15', '10:15', '11:15', '12:15', '13:15', '14:15'}  # 6 full hours incl 14:15
MIN_BARS  = 60

FEATURES = list(dict.fromkeys(VIEW_A + VIEW_B + VIEW_C))   # 81 vetted, lookahead-free
MARKET_COLS = ['Market_Mean_Return', 'Relative_Return', 'Market_Mean_Volatility', 'Relative_Volatility']

MACRO_COLS = [
    'Breadth_AD_Ratio', 'Breadth_Pct_Above_SMA_50', 'Breadth_Pct_Above_SMA_200',
    'Breadth_Pct_Near_52W_High', 'Breadth_Return_Dispersion',
    'Nifty50_Dist_SMA_20', 'Nifty50_Dist_SMA_50', 'Nifty50_Dist_SMA_200',
    'Nifty50_Return_5D', 'Nifty50_Return_20D', 'Nifty500_Return_5D', 'Nifty500_Return_20D',
    'VIX_Level', 'VIX_Change_5D', 'VIX_Percentile_1Y',
    'SP500_Return_1D', 'SP500_Change_5D', 'NASDAQ_Return_1D', 'NASDAQ_Change_5D',
    'NIKKEI_Return_1D', 'NIKKEI_Change_5D', 'HSI_Return_1D', 'HSI_Change_5D',
    'USDINR_Change_5D', 'BRENT_Change_5D', 'GOLD_Change_5D', 'DXY_Change_5D', 'US10Y_Change_5D',
]


# ── per-ticker builders (mirror rebuild_aligned_datasets.py) ────────────────────
def load_ticker_ohlcv(path):
    raw = pd.read_csv(path)
    if not {'timestamp', 'open', 'high', 'low', 'close', 'volume'}.issubset(raw.columns):
        return None
    dt = pd.to_datetime(raw['timestamp'], utc=True).dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
    df = pd.DataFrame({
        'DateTime': dt,
        'Open': raw['open'].astype(float), 'High': raw['high'].astype(float),
        'Low': raw['low'].astype(float), 'Close': raw['close'].astype(float),
        'Volume': raw['volume'].astype(float),
    }).dropna(subset=['DateTime', 'Open', 'Close'])
    df = df.drop_duplicates(subset='DateTime').sort_values('DateTime').set_index('DateTime')
    t = df.index.time
    df = df[(t >= MKT_OPEN) & (t <= M15_CLOSE)]
    return df if len(df) >= MIN_BARS else None


def session_masked_fwd(close):
    by_day = close.groupby(close.index.normalize())
    return by_day.shift(-1) / close - 1.0


def build_15m(ohlc, ticker):
    feat = compute_features(ohlc[['Open', 'High', 'Low', 'Close', 'Volume']].copy(), legacy=False)
    feat['Next_Ret'] = session_masked_fwd(feat['Close'])   # context-only label (kept all bars)
    feat['DateTime'] = feat.index
    feat['Ticker'] = ticker
    return feat


def build_1h(ohlc, ticker):
    h1 = ohlc[['Open', 'High', 'Low', 'Close', 'Volume']].resample(
        '1h', origin='start_day', offset='15min'
    ).agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})
    h1 = h1.dropna(subset=['Open', 'Close'])
    h1 = h1[pd.Index(h1.index).strftime('%H:%M').isin(VALID_1H_TODS)]   # 6 full hours incl 14:15
    if len(h1) < MIN_BARS:
        return None
    feat = compute_features(h1, legacy=False)
    feat['Next_Ret'] = session_masked_fwd(feat['Close'])   # NaN at 14:15 (last bar) -> context only
    feat['DateTime'] = feat.index
    feat['Ticker'] = ticker
    return feat


def cross_sectional(df_all):
    """Market-context features + per-query cross-sectional z-scoring. KEEPS context rows
    (NaN forward return retained); only drops rows missing feature values / tiny queries."""
    df = df_all.copy()
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df = df.sort_values('DateTime')
    df['Query_ID'] = df.groupby('DateTime').ngroup()
    sizes = df.groupby('Query_ID').size()
    df = df[df['Query_ID'].isin(sizes[sizes >= 5].index)].copy()
    df['Query_ID'] = df.groupby('DateTime').ngroup()
    df['Market_Mean_Return']     = df.groupby('Query_ID')['Return'].transform('mean')
    df['Relative_Return']        = df['Return'] - df['Market_Mean_Return']
    df['Market_Mean_Volatility'] = df.groupby('Query_ID')['HL_Range'].transform('mean')
    df['Relative_Volatility']    = df['HL_Range'] / (df['Market_Mean_Volatility'] + 1e-8)
    df = df.replace([np.inf, -np.inf], np.nan)
    zcols = [c for c in FEATURES if c not in MARKET_COLS]
    for col in zcols:
        g = df.groupby('Query_ID')[col]
        df[col] = (df[col] - g.transform('mean')) / (g.transform('std') + 1e-8)
    df = df.dropna(subset=zcols)              # drop early-history bars lacking features
    return df


def tod_slot(ts_ns, step_min):
    """clock slot index = minutes since 09:15 // step."""
    t = pd.to_datetime(ts_ns)
    mins = t.hour * 60 + t.minute - (9 * 60 + 15)
    return (mins // step_min).astype(np.int32)


def pivot_panel(df, tickers, with_label):
    tmap = {t: i for i, t in enumerate(tickers)}
    df = df[df['Ticker'].isin(tmap)].copy()
    ts = np.sort(df['DateTime'].unique())
    tsmap = {int(pd.Timestamp(t).value): i for i, t in enumerate(ts)}
    ti = df['DateTime'].astype('datetime64[ns]').astype('int64').map(tsmap).to_numpy()
    ni = df['Ticker'].map(tmap).to_numpy()
    T, N, F = len(ts), len(tickers), len(FEATURES)
    X = np.full((T, N, F), np.nan, dtype=np.float32)
    X[ti, ni, :] = df[FEATURES].to_numpy(dtype=np.float32)
    Y = None
    if with_label:
        Y = np.full((T, N), np.nan, dtype=np.float32)
        Y[ti, ni] = df['Next_Ret'].to_numpy(dtype=np.float32)
    ts_ns = ts.astype('datetime64[ns]').astype('int64')
    return X, Y, ts_ns


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 70)
    print("DUAL-RES PANEL BUILDER v2 (single 15m source; 14:15 context candle; time slots)")
    print("=" * 70)
    print(f"Features: {len(FEATURES)}")

    tickers = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(f'{CACHE_DIR}/*.csv'))
    print(f"Tickers: {len(tickers)}")

    frames_15m, frames_1h = [], []
    for tk in tqdm(tickers, desc='features'):
        ohlc = load_ticker_ohlcv(f'{CACHE_DIR}/{tk}.csv')
        if ohlc is None:
            continue
        try:
            frames_15m.append(build_15m(ohlc, tk))
            f1 = build_1h(ohlc, tk)
            if f1 is not None:
                frames_1h.append(f1)
        except Exception as e:
            tqdm.write(f"  [skip] {tk}: {str(e)[:60]}")

    # ── 15m context panel (keep ALL bars) ──────────────────────────────────────
    print("\n[15m] cross-sectional z-score + pivot")
    d15 = cross_sectional(pd.concat(frames_15m, ignore_index=True))
    del frames_15m
    X15, _, ts15 = pivot_panel(d15, tickers, with_label=False)
    del d15
    slot15 = tod_slot(ts15, 15)
    print(f"   X_15m {X15.shape}  slots {slot15.min()}..{slot15.max()}")

    # ── 1h panel (keep 14:15 context; label NaN there) ─────────────────────────
    print("\n[1h] cross-sectional z-score + pivot")
    d1 = cross_sectional(pd.concat(frames_1h, ignore_index=True))
    del frames_1h
    X1, Y, ts1 = pivot_panel(d1, tickers, with_label=True)
    del d1
    slot1 = tod_slot(ts1, 60)
    print(f"   X_1h {X1.shape}  slots {slot1.min()}..{slot1.max()}  "
          f"decision rows (finite label): {np.isfinite(Y).sum():,}")

    # ── align 15m to 1h: the 15m bar closing WITH 1h bar t is ts1[t]+45m ────────
    print("\n[align] 1h@T <-> 15m@T+45m")
    ts15_map = {int(t): i for i, t in enumerate(ts15)}
    off45 = np.int64(45 * 60 * 1_000_000_000)
    end15 = np.array([ts15_map.get(int(t + off45), -1) for t in ts1], dtype=np.int32)
    aligned = end15 >= 0
    chk = aligned.nonzero()[0]
    lhs = ts15[end15[chk]] + np.int64(15 * 60 * 1_000_000_000)
    rhs = ts1[chk] + np.int64(60 * 60 * 1_000_000_000)
    assert np.all(lhs == rhs), "ALIGNMENT VIOLATION"
    print(f"   aligned {aligned.mean()*100:.1f}%  [OK] close-time assertion passed")

    # ── macro (daily, market-level, PIT) ───────────────────────────────────────
    print("\n[macro] daily VIX/breadth/global")
    mdf = pd.read_csv(MACRO_FILE, usecols=['DateTime'] + MACRO_COLS)
    mdf['DateTime'] = pd.to_datetime(mdf['DateTime']).dt.normalize()
    mdf = mdf.drop_duplicates('DateTime').sort_values('DateTime').reset_index(drop=True)
    macro = mdf[MACRO_COLS].to_numpy(dtype=np.float32)
    macro_dates = mdf['DateTime'].to_numpy().astype('datetime64[ns]').astype('int64')
    mdate_map = {int(d): i for i, d in enumerate(macro_dates)}
    date_idx = np.array([mdate_map.get(int(pd.Timestamp(t).normalize().value), -1) for t in ts1],
                        dtype=np.int32)
    print(f"   macro {macro.shape}  1h bars with macro: {(date_idx>=0).mean()*100:.1f}%")

    # sectors
    sec_norm = {k.replace('.NS', ''): v for k, v in SECTOR_MAP.items()}
    sectors = sorted(set(sec_norm.values()))
    secmap = {s: i for i, s in enumerate(sectors)}
    sector_ids = np.array([secmap.get(sec_norm.get(t, 'MISC'), secmap['MISC']) for t in tickers],
                          dtype=np.int32)

    # ── save ───────────────────────────────────────────────────────────────────
    for name, arr in [('X_1h', X1), ('X_15m', X15), ('Y_ret', Y), ('slot_1h', slot1),
                      ('slot_15m', slot15), ('end15', end15), ('ts_1h', ts1), ('ts_15m', ts15),
                      ('date_idx', date_idx), ('macro', macro), ('macro_dates', macro_dates),
                      ('sector_ids', sector_ids)]:
        np.save(f'{OUT_DIR}/{name}.npy', arr)
    meta = {
        'features': FEATURES, 'n_features': len(FEATURES), 'tickers': tickers, 'n_tickers': len(tickers),
        'sectors': sectors, 'macro_cols': MACRO_COLS,
        'n_slots_1h': 6, 'n_slots_15m': 25,
        'shapes': {'X_1h': list(X1.shape), 'X_15m': list(X15.shape), 'macro': list(macro.shape)},
        'note': '14:15 context candle kept (label NaN); per-query z-scored; single 15m source; time slots emitted',
    }
    with open(f'{OUT_DIR}/meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 70)
    print(f"SAVED -> {OUT_DIR}/   X_1h {X1.shape}  X_15m {X15.shape}")
    print(f"  up-rate {np.nanmean(Y > 0)*100:.2f}%   finite labels {np.isfinite(Y).sum():,}")
    print("  1h time-of-day slots:", sorted(set(slot1.tolist())), "(5=14:15 context)")
    print("=" * 70)


if __name__ == '__main__':
    main()

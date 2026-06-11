"""
Rebuild PERFECTLY ALIGNED 15-min and 1-hour ranking datasets from ONE source.

Why: the original 1h dataset was resampled from 30-min Upstox candles in UTC with
origin='start_day', which (under the +5:30 IST offset) distorted the hourly grid —
a bar labeled 09:30 actually spanned 09:45-10:45 and the opening 15 min was dropped.
The 15m and 1h grids therefore did NOT correspond. Both also had an overnight leak in
the forward-return target (plain shift(-1) with no session mask).

Fix (this script):
  * Single source  : the cached raw 15-min OHLCV (data/raw_upstox_cache_15min_3y/),
                     itself built from native 1-minute Upstox candles.
  * IST throughout : timestamps are IST wall-clock (tz-naive), resampling done in IST.
  * 15m grid       : native quarter-hours 09:15 ... 15:15 (left-labeled).
  * 1h grid        : resample('1h', origin='start_day', offset='15min') -> bins anchored
                     to the 09:15 open: 09:15,10:15,11:15,12:15,13:15,14:15 (6 full bars).
                     GUARANTEE: each 1h bar == exactly the four 15m bars [T,T+15,T+30,T+45].
  * Forward returns: SESSION-MASKED (NaN at each day's last bar) -> no overnight leak.

Outputs (non-destructive new files):
  data/ranking_data_upstox_15min_3y_clean.csv
  data/ranking_data_upstox_1h_3y_clean.csv

Same feature pipeline as the originals: compute_features(legacy=False) + market-context
features + per-query cross-sectional z-scoring. Feature columns match the trained models'
expected set (so the clean data is eval-ready; retrain recommended for time-based features).
"""
import os, sys, glob, warnings
import numpy as np
import pandas as pd
from datetime import time as dtime
from tqdm import tqdm
warnings.filterwarnings('ignore')
sys.path.append(os.getcwd())

from scripts.feature_utils import compute_features

CACHE_DIR = 'data/raw_upstox_cache_15min_3y'
OUT_15M   = 'data/ranking_data_upstox_15min_3y_clean.csv'
OUT_1H    = 'data/ranking_data_upstox_1h_3y_clean.csv'
MIN_BARS  = 60
MKT_OPEN  = dtime(9, 15)
M15_CLOSE = dtime(15, 15)   # last 15m bar label
VALID_1H_TODS = {'09:15', '10:15', '11:15', '12:15', '13:15', '14:15'}  # 6 full hours

def load_ticker_ohlcv(path):
    raw = pd.read_csv(path)
    if not {'timestamp', 'open', 'high', 'low', 'close', 'volume'}.issubset(raw.columns):
        return None
    # IST wall-clock, tz-naive (consistent with how backtests load DateTime)
    dt = pd.to_datetime(raw['timestamp'], utc=True).dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
    df = pd.DataFrame({
        'DateTime': dt,
        'Open': raw['open'].astype(float), 'High': raw['high'].astype(float),
        'Low': raw['low'].astype(float),  'Close': raw['close'].astype(float),
        'Volume': raw['volume'].astype(float),
    }).dropna(subset=['DateTime', 'Open', 'Close'])
    df = df.drop_duplicates(subset='DateTime').sort_values('DateTime').set_index('DateTime')
    # market hours only (15m labels 09:15 .. 15:15)
    t = df.index.time
    df = df[(t >= MKT_OPEN) & (t <= M15_CLOSE)]
    return df if len(df) >= MIN_BARS else None

def session_masked_fwd(close: pd.Series) -> pd.Series:
    """Next-bar return, NaN'd across day boundaries (no overnight leak)."""
    by_day = close.groupby(close.index.normalize())
    nxt = by_day.shift(-1)
    return nxt / close - 1.0

def build_15m(ohlc, ticker):
    feat = compute_features(ohlc[['Open', 'High', 'Low', 'Close', 'Volume']].copy(), legacy=False)
    feat['Next_15Min_Return'] = session_masked_fwd(feat['Close'])
    feat['DateTime'] = feat.index
    feat['Ticker'] = ticker
    return feat

def build_1h(ohlc, ticker):
    h1 = ohlc[['Open', 'High', 'Low', 'Close', 'Volume']].resample(
        '1h', origin='start_day', offset='15min'
    ).agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})
    h1 = h1.dropna(subset=['Open', 'Close'])
    h1 = h1[pd.Index(h1.index).strftime('%H:%M').isin(VALID_1H_TODS)]   # 6 full hours only
    if len(h1) < MIN_BARS:
        return None
    feat = compute_features(h1, legacy=False)
    feat['Next_Hour_Return'] = session_masked_fwd(feat['Close'])
    feat['DateTime'] = feat.index
    feat['Ticker'] = ticker
    return feat

def build_ranking(df_all, ret_col):
    df_all = df_all.copy()
    df_all['DateTime'] = pd.to_datetime(df_all['DateTime'])
    df_all = df_all.dropna(subset=[ret_col])
    # Query_ID = all tickers sharing a bar timestamp
    df_all = df_all.sort_values('DateTime')
    df_all['Query_ID'] = df_all.groupby('DateTime').ngroup()
    sizes = df_all.groupby('Query_ID').size()
    df_all = df_all[df_all['Query_ID'].isin(sizes[sizes >= 5].index)].copy()
    df_all = df_all.sort_values('DateTime')
    df_all['Query_ID'] = df_all.groupby('DateTime').ngroup()
    # market context
    df_all['Market_Mean_Return']     = df_all.groupby('Query_ID')['Return'].transform('mean')
    df_all['Relative_Return']        = df_all['Return'] - df_all['Market_Mean_Return']
    df_all['Market_Mean_Volatility'] = df_all.groupby('Query_ID')['HL_Range'].transform('mean')
    df_all['Relative_Volatility']    = df_all['HL_Range'] / (df_all['Market_Mean_Volatility'] + 1e-8)
    # cross-sectional z-scoring (per query)
    exclude = {
        'DateTime', 'Query_ID', 'Ticker', ret_col,
        'Open', 'High', 'Low', 'Close', 'Volume',
        'Market_Mean_Return', 'Relative_Return',
        'Market_Mean_Volatility', 'Relative_Volatility',
        'Hour', 'DayOfWeek', 'Is_Open_Hour', 'Is_Close_Hour', 'Time_To_Close',
    }
    feat_cols = [c for c in df_all.columns if c not in exclude]
    df_all = df_all.replace([np.inf, -np.inf], np.nan)
    for col in tqdm(feat_cols, desc='  z-score', leave=False):
        g = df_all.groupby('Query_ID')[col]
        df_all[col] = (df_all[col] - g.transform('mean')) / (g.transform('std') + 1e-8)
    df_all = df_all.dropna(subset=feat_cols)
    return df_all, feat_cols

# ── main ────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("REBUILD ALIGNED DATASETS (single 15-min source, IST, session-masked)")
print("=" * 70)
files = sorted(glob.glob(os.path.join(CACHE_DIR, '*.csv')))
print(f"  Cache tickers: {len(files)}")

frames_15m, frames_1h = [], []
stats = {'ok': 0, 'skip': 0}
for path in tqdm(files, desc='Tickers'):
    ticker = os.path.splitext(os.path.basename(path))[0] + '.NS'
    ohlc = load_ticker_ohlcv(path)
    if ohlc is None:
        stats['skip'] += 1; continue
    try:
        f15 = build_15m(ohlc, ticker)
        f1h = build_1h(ohlc, ticker)
        if f15 is not None: frames_15m.append(f15)
        if f1h is not None: frames_1h.append(f1h)
        stats['ok'] += 1
    except Exception as e:
        stats['skip'] += 1
        tqdm.write(f"  [skip] {ticker}: {str(e)[:70]}")

print(f"  OK={stats['ok']}  skip={stats['skip']}")

print("\nBuilding 15m ranking dataset...")
df15_all = pd.concat(frames_15m, ignore_index=True)
df15_final, fc15 = build_ranking(df15_all, 'Next_15Min_Return')
df15_final.to_csv(OUT_15M, index=False)
print(f"  Saved {OUT_15M}  rows={len(df15_final):,}  queries={df15_final['Query_ID'].nunique():,}  feats={len(fc15)}")

print("\nBuilding 1h ranking dataset...")
df1h_all = pd.concat(frames_1h, ignore_index=True)
df1h_final, fc1h = build_ranking(df1h_all, 'Next_Hour_Return')
df1h_final.to_csv(OUT_1H, index=False)
print(f"  Saved {OUT_1H}  rows={len(df1h_final):,}  queries={df1h_final['Query_ID'].nunique():,}  feats={len(fc1h)}")

# ── alignment + leak verification (on RAW prices, pre-zscore not available here, so
#    re-derive close-to-close from the saved DateTime grid is not possible after zscore;
#    instead verify time-of-day grids and overnight masking) ──────────────────────
print("\n" + "=" * 70)
print("VERIFICATION")
print("=" * 70)
tod15 = sorted(pd.to_datetime(df15_final['DateTime']).dt.strftime('%H:%M').unique())
tod1h = sorted(pd.to_datetime(df1h_final['DateTime']).dt.strftime('%H:%M').unique())
print(f"  15m times-of-day: {tod15}")
print(f"  1h  times-of-day: {tod1h}")
# overnight leak check: last bar of each day should have NaN forward return -> dropped.
# Confirm no 1h row exists whose next-day join would be overnight: check last tod not retained as a tradeable signal with a same-day next bar.
d15 = pd.to_datetime(df15_all['DateTime'])
last15 = df15_all.assign(dt=d15, day=d15.dt.normalize())
last_mask = last15.groupby(['Ticker', 'day'])['dt'].transform('max') == last15['dt']
leak15 = last15[last_mask]['Next_15Min_Return'].notna().sum()
print(f"  15m last-bar-of-day rows with non-NaN fwd return (should be 0): {leak15}")
d1h = pd.to_datetime(df1h_all['DateTime'])
last1h = df1h_all.assign(dt=d1h, day=d1h.dt.normalize())
last_mask1 = last1h.groupby(['Ticker', 'day'])['dt'].transform('max') == last1h['dt']
leak1h = last1h[last_mask1]['Next_Hour_Return'].notna().sum()
print(f"  1h  last-bar-of-day rows with non-NaN fwd return (should be 0): {leak1h}")
print("\nDone.")

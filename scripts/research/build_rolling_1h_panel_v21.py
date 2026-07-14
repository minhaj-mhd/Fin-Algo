"""
Build the v20 ROLLING-1h RESEARCH panel.

Idea (user spec): instead of v10's 6 NON-overlapping exchange hourly bars per day
(09:15, 10:15, ... 14:15), emit OVERLAPPING 1-hour windows stepped every 15 minutes:
    [09:15-10:15], [09:30-10:30], [09:45-10:45], ... [14:15-15:15], ...
A rolling 1h window == 4 consecutive native 15-min bars, so we aggregate locally from
data/raw_upstox_cache_15min_3y/ -- NO Upstox API calls.

This is a byte-faithful clone of scripts/collectors/collect_upstox_1h_v3.py's feature /
z-score / ranking pipeline (compute_features(legacy=False) + per-query cross-sectional
z-score + Relative_* columns) so the resulting feature SCHEMA is identical to v10's 86
features. The ONLY difference is the candle grid.

Each window is keyed by its ENTRY time T (the moment the hour completes, = window close).
Label: Next_Hour_Return = close(T+1h) / close(T) - 1, session-masked by EXACT-timestamp
match (NaN if T+1h is past the session / next day) -> no overnight leak, mirrors v10.

============================  RESEARCH ONLY  ============================
No Gauntlet, no registry stamp, no verdict authority (AGENTS.md Model Metric Discipline).
WARNING - overlapping windows share 45/60 min, so CONSECUTIVE ROWS are ~75% identical in
both features and label. Point estimates (avg rho, WR@k) from the monthly purged WF are
comparable to v10, but effective N ~= 1/4 of row count: do NOT t-test these numbers or
read significance into them. The purpose is anchor-agnostic 1h ranking that can be served
every 15 min, NOT a claim of a better per-trade edge (the 1h price/volume edge is
information-limited per prior research).

Output: data/research/v20_rolling_1h/panel.parquet
"""
import os, sys, glob, warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
warnings.filterwarnings('ignore')
sys.path.append(os.getcwd())

from scripts.feature_utils import compute_features

SRC_DIR     = 'data/raw_upstox_cache_15min_3y'
OUT_DIR     = 'data/research/v21_rolling_1h_dynamic'
OUT_PARQUET = os.path.join(OUT_DIR, 'panel.parquet')
STEP        = pd.Timedelta('15min')
HOUR        = pd.Timedelta('60min')
MIN_BARS    = 300          # need enough 15-min bars to form a usable rolling series
MIN_PER_Q   = 5            # min tickers per cross-section (matches v10 build_ranking)
os.makedirs(OUT_DIR, exist_ok=True)

from datetime import date
import yfinance as yf

# Load Macro: Nifty 50
nifty = pd.read_csv('data/raw_index_cache/nifty50_15m.csv')
ist_times = []
for t_str in nifty['ts']:
    t = pd.to_datetime(t_str)
    if t.date() < date(2026, 6, 1):
        ist_times.append(t.tz_localize(None))
    else:
        ist_times.append(t.tz_convert('Asia/Kolkata').tz_localize(None))
nifty['ts'] = ist_times
nifty = nifty.sort_values('ts').reset_index(drop=True)
nifty['nifty_ret_2h'] = nifty['close'] / nifty['close'].shift(8) - 1
nifty_map = dict(zip(nifty['ts'], nifty['nifty_ret_2h']))

# Load Macro: S&P 500
sp500 = yf.download('^GSPC', start='2022-01-01', end='2026-07-30', progress=False)
if isinstance(sp500.columns, pd.MultiIndex):
    sp500.columns = sp500.columns.get_level_values(0)
sp500 = sp500.reset_index()
sp500['Date'] = pd.to_datetime(sp500['Date']).dt.date
sp500['sp500_ret'] = sp500['Close'].pct_change()
sp500_ret_dict = {r['Date']: r['sp500_ret'] for _, r in sp500.iterrows()}

prev_sp500_cache = {}
def get_prev_sp500_ret(curr_date):
    if curr_date in prev_sp500_cache: return prev_sp500_cache[curr_date]
    prev_dates = [d for d in sp500_ret_dict.keys() if d < curr_date]
    ret = sp500_ret_dict[max(prev_dates)] if prev_dates else 0
    prev_sp500_cache[curr_date] = ret
    return ret


def build_ticker(ticker, raw):
    """15-min OHLCV -> overlapping 1h windows (15-min step) -> features + next-1h label."""
    dt = pd.to_datetime(raw['timestamp'], utc=True).dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
    df = pd.DataFrame({
        'DateTime': dt,
        'Open': raw['open'].astype(float), 'High': raw['high'].astype(float),
        'Low': raw['low'].astype(float), 'Close': raw['close'].astype(float),
        'Volume': raw['volume'].astype(float),
    }).dropna(subset=['DateTime', 'Open', 'Close'])
    df = df.drop_duplicates('DateTime').sort_values('DateTime').reset_index(drop=True)
    if len(df) < MIN_BARS:
        return None

    t = df['DateTime']
    # Rolling 1h window = the 4 consecutive 15-min bars [k-3 .. k].
    #   Open = first bar's open, Close = last bar's close, High/Low/Volume aggregated.
    #   Window is valid only if those 4 bars are exactly 45 min end-to-end apart, which is
    #   true iff they are 4 consecutive intraday bars (false across the overnight gap).
    win = pd.DataFrame({
        'DateTime': t + STEP,                       # entry time T = hour-close (last bar start + 15m)
        'Open':   df['Open'].shift(3),
        'High':   df['High'].rolling(4).max(),
        'Low':    df['Low'].rolling(4).min(),
        'Close':  df['Close'],
        'Volume': df['Volume'].rolling(4).sum(),
    })
    contiguous = (t - t.shift(3)) == (3 * STEP)
    win = win[contiguous.values].dropna(subset=['Open', 'High', 'Low', 'Close']).copy()
    win = win.drop_duplicates('DateTime').sort_values('DateTime').set_index('DateTime')
    if len(win) < MIN_BARS:
        return None

    feat = compute_features(win[['Open', 'High', 'Low', 'Close', 'Volume']].copy(), legacy=False)

    # next-1h forward return on the rolling grid: close at EXACTLY T+1h (exact-match reindex
    # -> NaN whenever T+1h is not a same-session window close == session mask, no overnight leak)
    close_at = feat['Close']
    fwd = close_at.reindex(close_at.index + HOUR)
    feat['Next_Hour_Return'] = fwd.values / close_at.values - 1.0

    feat['DateTime'] = feat.index
    feat['Ticker'] = ticker
    
    # --- Inject Dynamic Macro Features ---
    # 1. Time of Day (Continuous float: hour + minute/60)
    feat['Macro_TimeOfDay'] = feat['DateTime'].dt.hour + feat['DateTime'].dt.minute / 60.0
    
    # 2. SP500 Previous Day Return
    # Map each unique date to its previous S&P 500 return
    date_map = {d: get_prev_sp500_ret(d) for d in feat['DateTime'].dt.date.unique()}
    feat['Macro_SP500_Ret'] = feat['DateTime'].dt.date.map(date_map)
    
    # 3. Nifty Trailing 2H Return
    feat['Macro_Nifty_2H'] = feat['DateTime'].map(nifty_map)
    
    # Forward fill or backfill any minor Nifty gaps (e.g. if Upstox index data missed a specific 15m timestamp)
    feat['Macro_Nifty_2H'] = feat['Macro_Nifty_2H'].ffill().bfill()
    
    return feat


def build_ranking(df_all):
    """VERBATIM from collect_upstox_1h_v3.py: per-query cross-sectional z-score + Relative_* cols.
    Keeps the output schema identical to the v10 dataset so the same 86 features emerge."""
    df_all = df_all.copy()
    df_all['DateTime'] = pd.to_datetime(df_all['DateTime'])
    df_all = df_all.dropna(subset=['Next_Hour_Return']).sort_values('DateTime')
    df_all['Query_ID'] = df_all.groupby('DateTime').ngroup()
    sizes = df_all.groupby('Query_ID').size()
    df_all = df_all[df_all['Query_ID'].isin(sizes[sizes >= MIN_PER_Q].index)].copy()
    df_all = df_all.sort_values('DateTime')
    df_all['Query_ID'] = df_all.groupby('DateTime').ngroup()
    df_all['Market_Mean_Return']     = df_all.groupby('Query_ID')['Return'].transform('mean')
    df_all['Relative_Return']        = df_all['Return'] - df_all['Market_Mean_Return']
    df_all['Market_Mean_Volatility'] = df_all.groupby('Query_ID')['HL_Range'].transform('mean')
    df_all['Relative_Volatility']    = df_all['HL_Range'] / (df_all['Market_Mean_Volatility'] + 1e-8)
    exclude = {'DateTime', 'Query_ID', 'Ticker', 'Next_Hour_Return', 'Open', 'High', 'Low', 'Close', 'Volume',
               'Market_Mean_Return', 'Relative_Return', 'Market_Mean_Volatility', 'Relative_Volatility',
               'Hour', 'DayOfWeek', 'Is_Open_Hour', 'Is_Close_Hour', 'Time_To_Close'}
    feat_cols = [c for c in df_all.columns if c not in exclude and not c.startswith('Macro_')]
    df_all = df_all.replace([np.inf, -np.inf], np.nan)
    for col in tqdm(feat_cols, desc='  z-score', leave=False):
        g = df_all.groupby('Query_ID')[col]
        df_all[col] = (df_all[col] - g.transform('mean')) / (g.transform('std') + 1e-8)
        
    # Re-add Macro_ columns to the final list of features so they are saved
    final_features = feat_cols + [c for c in df_all.columns if c.startswith('Macro_')]
    return df_all.dropna(subset=final_features), final_features


def main():
    files = sorted(glob.glob(os.path.join(SRC_DIR, '*.csv')))
    print(f"Source: {SRC_DIR}  ({len(files)} tickers)")
    frames, ok, skip = [], 0, 0
    for fp in tqdm(files, desc='Tickers'):
        ticker = os.path.splitext(os.path.basename(fp))[0]
        try:
            raw = pd.read_csv(fp)
            f = build_ticker(ticker, raw)
            if f is not None and len(f):
                frames.append(f); ok += 1
            else:
                skip += 1
        except Exception as e:
            skip += 1; tqdm.write(f"  [skip] {ticker}: {str(e)[:70]}")
    print(f"  tickers OK={ok} skip={skip}")

    print("\nBuilding ranking dataset (cross-sectional z-score per 15-min entry time)...")
    df_all = pd.concat(frames, ignore_index=True)
    final, fc = build_ranking(df_all)

    # downcast feature/return cols to float32 to keep the parquet small
    for c in fc + ['Next_Hour_Return', 'Market_Mean_Return', 'Relative_Return',
                   'Market_Mean_Volatility', 'Relative_Volatility',
                   'Open', 'High', 'Low', 'Close', 'Volume']:
        if c in final.columns:
            final[c] = final[c].astype('float32')

    final.to_parquet(OUT_PARQUET, index=False)

    months = sorted(pd.to_datetime(final['DateTime']).dt.strftime('%Y-%m').unique())
    tods   = sorted(pd.to_datetime(final['DateTime']).dt.strftime('%H:%M').unique())
    print(f"\nSaved {OUT_PARQUET}")
    print(f"  rows={len(final):,}  queries={final['Query_ID'].nunique():,}  feats={len(fc)}")
    print(f"  span: {months[0]} -> {months[-1]} ({len(months)} months)")
    print(f"  entry times/day: {len(tods)}  -> {tods[0]} .. {tods[-1]}")
    print(f"  avg tickers/query: {final.groupby('Query_ID').size().mean():.1f}")
    qpd = final.groupby(pd.to_datetime(final['DateTime']).dt.date)['Query_ID'].nunique()
    print(f"  avg labeled entry-times/day: {qpd.mean():.1f}  (v10 has 5)")


if __name__ == '__main__':
    main()

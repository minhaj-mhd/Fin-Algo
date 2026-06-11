"""
Build enriched 15m panel with all detector inputs precomputed.

Output: data/structural_panel_15m.parquet

Usage:
    python scripts/structural/build_panel.py
    python scripts/structural/build_panel.py --rebuild   # force rebuild even if cached
"""
import os, sys, glob, argparse, warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')
sys.path.append(os.getcwd())

from scripts.sector_map import SECTOR_MAP
from scripts.feature_utils import ATR

CACHE_DIR  = 'data/raw_upstox_cache_15min_3y'
INDEX_FILE = 'data/raw_index_cache/nifty500_1h.csv'
OUT_FILE   = 'data/structural_panel_15m.parquet'

EPS = 1e-8
VALID_TODS = {  # drop 15:15 stub if present; keep full session
    '09:15','09:30','09:45','10:00','10:15','10:30','10:45',
    '11:00','11:15','11:30','11:45','12:00','12:15','12:30','12:45',
    '13:00','13:15','13:30','13:45','14:00','14:15','14:30','14:45',
    '15:00',
}


def load_raw(ticker):
    path = os.path.join(CACHE_DIR, f"{ticker.replace('.NS','')}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df['ts'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
    df = df.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'})
    df = df[['ts','Open','High','Low','Close','Volume']].drop_duplicates('ts').sort_values('ts').reset_index(drop=True)
    df['tod'] = df['ts'].dt.strftime('%H:%M')
    df = df[df['tod'].isin(VALID_TODS)].reset_index(drop=True)
    return df


def session_vwap(df):
    """Cumulative VWAP reset each calendar day (tz-naive)."""
    df = df.copy()
    df['date'] = df['ts'].dt.normalize()
    df['pv'] = (df['High'] + df['Low'] + df['Close']) / 3 * df['Volume']
    df['cum_pv'] = df.groupby('date')['pv'].cumsum()
    df['cum_vol'] = df.groupby('date')['Volume'].cumsum()
    df['session_vwap'] = df['cum_pv'] / (df['cum_vol'] + EPS)
    return df.drop(columns=['pv','cum_pv','cum_vol','date'])


def opening_range(df):
    """Opening range high/low from 09:15 and 09:30 bars (the first two 15m bars)."""
    df = df.copy()
    df['date'] = df['ts'].dt.normalize()
    # Opening range = first two bars of the day (09:15 + 09:30)
    or_bars = df[df['tod'].isin({'09:15','09:30'})].groupby('date').agg(
        or_high=('High','max'), or_low=('Low','min')
    ).reset_index()
    df = df.merge(or_bars, on='date', how='left')
    df['or_high'] = df.groupby('date')['or_high'].ffill()
    df['or_low']  = df.groupby('date')['or_low'].ffill()
    return df.drop(columns=['date'])


def build_ticker_features(df):
    """Compute all per-ticker detector inputs."""
    d = session_vwap(df)
    d = opening_range(d)

    # ATR
    ohlc = d[['Open','High','Low','Close','Volume']].rename(columns=str.title)
    ohlc.index = d['ts']
    d['atr_15m']      = ATR(ohlc, 14).values
    d['atr_pct']      = d['atr_15m'] / (d['Close'] + EPS)
    d['atr_pct_4bar']  = d['atr_pct'].rolling(4).mean()
    d['atr_pct_20bar'] = d['atr_pct'].rolling(20).mean()

    # RVOL and volume averages
    d['avg_vol_20']   = d['Volume'].rolling(20).mean()
    d['rvol_20']      = d['Volume'] / (d['avg_vol_20'] + EPS)
    d['avg_range_pct_20'] = ((d['High'] - d['Low']) / (d['Close'] + EPS)).rolling(20).mean()

    # Returns (pct change)
    d['ret_15m'] = d['Close'].pct_change(1)
    d['ret_30m'] = d['Close'].pct_change(2)
    d['ret_60m'] = d['Close'].pct_change(4)

    # Range stats
    d['range_pct']      = (d['High'] - d['Low']) / (d['Close'] + EPS)
    d['close_location'] = (d['Close'] - d['Low']) / (d['High'] - d['Low'] + EPS)
    d['upper_wick_pct'] = (d['High'] - d[['Open','Close']].max(axis=1)) / (d['High'] - d['Low'] + EPS)
    d['lower_wick_pct'] = (d[['Open','Close']].min(axis=1) - d['Low'])  / (d['High'] - d['Low'] + EPS)

    # Rolling highs/lows EXCLUDING current bar (shifted)
    d['roll_high_4']  = d['High'].shift(1).rolling(4).max()
    d['roll_low_4']   = d['Low'].shift(1).rolling(4).min()
    d['roll_high_6']  = d['High'].shift(1).rolling(6).max()
    d['roll_low_6']   = d['Low'].shift(1).rolling(6).min()

    # Previous bar close (for VWAP reclaim crossover check)
    d['prev_close']  = d['Close'].shift(1)
    d['prev_vwap']   = d['session_vwap'].shift(1)

    return d


def build_panel():
    files = sorted(glob.glob(os.path.join(CACHE_DIR, '*.csv')))
    tickers = [os.path.basename(f).replace('.csv','') + '.NS' for f in files]
    tickers = [t for t in tickers if t in SECTOR_MAP]
    print(f"Building panel: {len(tickers)} tickers x 15m bars ...")

    frames = []
    for ticker in tqdm(tickers, desc='Tickers'):
        raw = load_raw(ticker)
        if raw is None or len(raw) < 100:
            continue
        feat = build_ticker_features(raw)
        feat['Ticker'] = ticker
        feat['Sector'] = SECTOR_MAP[ticker]
        frames.append(feat[['ts','Ticker','Sector','Open','High','Low','Close','Volume',
                             'session_vwap','or_high','or_low',
                             'atr_15m','atr_pct','atr_pct_4bar','atr_pct_20bar',
                             'avg_vol_20','rvol_20','avg_range_pct_20',
                             'ret_15m','ret_30m','ret_60m',
                             'range_pct','close_location','upper_wick_pct','lower_wick_pct',
                             'roll_high_4','roll_low_4','roll_high_6','roll_low_6',
                             'prev_close','prev_vwap','tod']])

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(['ts','Ticker']).reset_index(drop=True)

    print("Adding cross-sectional features ...")
    # Sector returns
    sector_ret = panel.groupby(['ts','Sector'])['ret_60m'].transform('mean')
    panel['sector_ret_60m'] = sector_ret
    panel['relative_ret_60m'] = panel['ret_60m'] - panel['sector_ret_60m']

    sector_ret_30m = panel.groupby(['ts','Sector'])['ret_30m'].transform('mean')
    panel['sector_ret_30m'] = sector_ret_30m
    panel['relative_ret_30m'] = panel['ret_30m'] - panel['sector_ret_30m']

    # NIFTY 500 return
    nifty = pd.read_csv(INDEX_FILE)
    nifty['ts']     = pd.to_datetime(nifty['timestamp'])
    nifty['nifty_ret_60m'] = nifty['close'].pct_change(4)
    nifty = nifty[['ts','nifty_ret_60m']].sort_values('ts')
    panel = pd.merge_asof(panel.sort_values('ts'), nifty, on='ts', direction='backward')

    panel = panel.sort_values(['ts','Ticker']).reset_index(drop=True)
    panel.to_parquet(OUT_FILE, index=False)
    print(f"Saved {OUT_FILE}  rows={len(panel):,}  tickers={panel['Ticker'].nunique()}")
    print(f"Span: {panel['ts'].min()} -> {panel['ts'].max()}")
    return panel


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--rebuild', action='store_true')
    args = ap.parse_args()

    if not args.rebuild and os.path.exists(OUT_FILE):
        print(f"Panel already exists at {OUT_FILE}. Use --rebuild to regenerate.")
    else:
        build_panel()

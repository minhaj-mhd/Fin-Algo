"""
Build the v22 ROLLING-1h RESEARCH panel.

Idea (user spec): instead of v10's 6 NON-overlapping exchange hourly bars per day
(09:15, 10:15, ... 14:15), emit OVERLAPPING 1-hour windows stepped every 15 minutes:
    [09:15-10:15], [09:30-10:30], [09:45-10:45], ... [14:15-15:15], ...
A rolling 1h window == 4 consecutive native 15-min bars, so we aggregate locally from
data/raw_upstox_cache_15min_3y/ -- NO Upstox API calls.

This version adds new feature groups:
- Group T: Time_Sin, Time_Cos
- Group V: VWAP_Slope, VWAP_Zscore, Price_VWAP_Ratio
- Group R: RelStr_Nifty_1H, RelStr_Nifty_2H, Beta_Nifty, Resid_Return_Nifty
- Helper: Is_Intraday_High
- Group N: Macro_Nifty_1H, Macro_Nifty_RealVol, Macro_Nifty_ATR_Pct, Macro_Nifty_Gap, Macro_India_VIX
- Group M: Breadth_Pct_Positive, Breadth_Pct_Above_VWAP, Breadth_AD_Ratio, Breadth_Pct_NewHigh, Breadth_Median_Return, Disp_CrossSec_Std, Disp_CrossSec_MAD

Output: data/research/v22_rolling_1h_dynamic/panel.parquet
"""
import os, sys, glob, warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
warnings.filterwarnings('ignore')
sys.path.append(os.getcwd())

from scripts.feature_utils import compute_features

SRC_DIR     = 'data/raw_upstox_cache_15min_3y'
OUT_DIR     = 'data/research/v22_rolling_1h_dynamic'
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

# Compute Nifty Features (Group N & R helpers)
nifty['nifty_ret_1h'] = nifty['close'] / nifty['close'].shift(4) - 1
nifty['nifty_ret_2h'] = nifty['close'] / nifty['close'].shift(8) - 1
nifty['nifty_realvol'] = nifty['close'].pct_change().rolling(14).std()
nifty['nifty_gap'] = nifty['open'] / nifty['close'].shift(1) - 1

nifty_atr = pd.concat([
    nifty['high'] - nifty['low'],
    (nifty['high'] - nifty['close'].shift(1)).abs(),
    (nifty['low'] - nifty['close'].shift(1)).abs()
], axis=1).max(axis=1)
nifty['nifty_atr_pct'] = (nifty_atr.rolling(14).mean() / nifty['close'])

nifty_1h_map = dict(zip(nifty['ts'], nifty['nifty_ret_1h']))
nifty_2h_map = dict(zip(nifty['ts'], nifty['nifty_ret_2h']))
nifty_realvol_map = dict(zip(nifty['ts'], nifty['nifty_realvol']))
nifty_gap_map = dict(zip(nifty['ts'], nifty['nifty_gap']))
nifty_atr_map = dict(zip(nifty['ts'], nifty['nifty_atr_pct']))

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

# Load Macro: India VIX
vix = yf.download('^INDIAVIX', start='2022-01-01', end='2026-07-30', progress=False)
if isinstance(vix.columns, pd.MultiIndex):
    vix.columns = vix.columns.get_level_values(0)
vix = vix.reset_index()
vix['Date'] = pd.to_datetime(vix['Date']).dt.date
vix_dict = {r['Date']: r['Close'] for _, r in vix.iterrows()}

prev_vix_cache = {}
def get_prev_vix(curr_date):
    if curr_date in prev_vix_cache: return prev_vix_cache[curr_date]
    prev_dates = [d for d in vix_dict.keys() if d < curr_date]
    ret = vix_dict[max(prev_dates)] if prev_dates else np.nan
    prev_vix_cache[curr_date] = ret
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

    # Helper: Is_Intraday_High
    df['Date'] = df['DateTime'].dt.date
    df['daily_high'] = df.groupby('Date')['High'].cummax()
    daily_high_map = dict(zip(df['DateTime'], df['daily_high']))
    
    # Calculate daily VWAP
    df['typ'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['vol_typ'] = df['typ'] * df['Volume']
    df['cum_vol_typ'] = df.groupby('Date')['vol_typ'].cumsum()
    df['cum_vol'] = df.groupby('Date')['Volume'].cumsum()
    df['vwap'] = df['cum_vol_typ'] / df['cum_vol']
    vwap_map = dict(zip(df['DateTime'], df['vwap']))

    t = df['DateTime']
    # Rolling 1h window
    win = pd.DataFrame({
        'DateTime': t + STEP,
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

    # Label
    close_at = feat['Close']
    fwd = close_at.reindex(close_at.index + HOUR)
    feat['Next_Hour_Return'] = fwd.values / close_at.values - 1.0

    feat['DateTime'] = feat.index
    feat['Ticker'] = ticker
    
    # --- Inject Dynamic Macro Features ---
    # Group T: Time_Sin, Time_Cos
    feat['Macro_TimeOfDay'] = feat['DateTime'].dt.hour + feat['DateTime'].dt.minute / 60.0
    feat['Time_Sin'] = np.sin(2 * np.pi * feat['Macro_TimeOfDay'] / 24.0)
    feat['Time_Cos'] = np.cos(2 * np.pi * feat['Macro_TimeOfDay'] / 24.0)
    
    # SP500 Previous Day Return
    date_map = {d: get_prev_sp500_ret(d) for d in feat['DateTime'].dt.date.unique()}
    feat['Macro_SP500_Ret'] = feat['DateTime'].dt.date.map(date_map)
    
    # Group N: Macro_Nifty_1H, Macro_Nifty_2H (Legacy from V21), etc.
    feat['Macro_Nifty_1H'] = feat['DateTime'].map(nifty_1h_map).ffill().bfill()
    feat['Macro_Nifty_2H'] = feat['DateTime'].map(nifty_2h_map).ffill().bfill()
    feat['Macro_Nifty_RealVol'] = feat['DateTime'].map(nifty_realvol_map).ffill().bfill()
    feat['Macro_Nifty_ATR_Pct'] = feat['DateTime'].map(nifty_atr_map).ffill().bfill()
    feat['Macro_Nifty_Gap'] = feat['DateTime'].map(nifty_gap_map).ffill().bfill()
    
    date_map_vix = {d: get_prev_vix(d) for d in feat['DateTime'].dt.date.unique()}
    feat['Macro_India_VIX'] = feat['DateTime'].dt.date.map(date_map_vix)
    
    # Group V: VWAP_Slope, VWAP_Zscore, Price_VWAP_Ratio
    feat['VWAP'] = feat['DateTime'].map(vwap_map)
    feat['Price_VWAP_Ratio'] = feat['Close'] / feat['VWAP']
    feat['VWAP_Slope'] = (feat['VWAP'] / feat['VWAP'].shift(4)) - 1
    feat['VWAP_Zscore'] = (feat['Close'] - feat['VWAP']) / (feat['Close'].rolling(14).std() + 1e-8)
    
    # Group R: RelStr_Nifty_1H, RelStr_Nifty_2H, Beta_Nifty, Resid_Return_Nifty
    stock_ret_1h = feat['Close'] / feat['Close'].shift(4) - 1
    stock_ret_2h = feat['Close'] / feat['Close'].shift(8) - 1
    feat['RelStr_Nifty_1H'] = stock_ret_1h - feat['Macro_Nifty_1H']
    feat['RelStr_Nifty_2H'] = stock_ret_2h - feat['Macro_Nifty_2H']
    
    cov = stock_ret_1h.rolling(14).cov(feat['Macro_Nifty_1H'])
    var = feat['Macro_Nifty_1H'].rolling(14).var()
    feat['Beta_Nifty'] = cov / (var + 1e-8)
    feat['Resid_Return_Nifty'] = stock_ret_1h - feat['Beta_Nifty'] * feat['Macro_Nifty_1H']
    
    # Helper: Is_Intraday_High
    feat['Is_Intraday_High'] = (feat['Close'] >= feat['DateTime'].map(daily_high_map)).astype(float)
    
    return feat


def build_ranking(df_all):
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
    
    # Group M: Breadth features (cross-sectional)
    df_all['Breadth_Pct_Positive'] = df_all.groupby('Query_ID')['Return'].transform(lambda x: (x > 0).mean())
    df_all['Breadth_Pct_Above_VWAP'] = df_all.groupby('Query_ID')['Price_VWAP_Ratio'].transform(lambda x: (x > 1).mean())
    def ad_ratio(x):
        adv = (x > 0).sum()
        dec = (x < 0).sum()
        return adv / dec if dec > 0 else np.nan
    df_all['Breadth_AD_Ratio'] = df_all.groupby('Query_ID')['Return'].transform(ad_ratio).fillna(1.0)
    df_all['Breadth_Pct_NewHigh'] = df_all.groupby('Query_ID')['Is_Intraday_High'].transform('mean')
    df_all['Breadth_Median_Return'] = df_all.groupby('Query_ID')['Return'].transform('median')
    df_all['Disp_CrossSec_Std'] = df_all.groupby('Query_ID')['Return'].transform('std')
    df_all['Disp_CrossSec_MAD'] = df_all.groupby('Query_ID')['Return'].transform(lambda x: (x - x.mean()).abs().mean())

    exclude = {'DateTime', 'Query_ID', 'Ticker', 'Next_Hour_Return', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP',
               'Market_Mean_Return', 'Relative_Return', 'Market_Mean_Volatility', 'Relative_Volatility',
               'Hour', 'DayOfWeek', 'Is_Open_Hour', 'Is_Close_Hour', 'Time_To_Close'}
    
    feat_cols_to_zscore = []
    feat_cols_raw = []
    for c in df_all.columns:
        if c in exclude:
            continue
        if c.startswith('Macro_') or c.startswith('Breadth_') or c.startswith('Disp_') or c.startswith('Time_'):
            feat_cols_raw.append(c)
        else:
            feat_cols_to_zscore.append(c)

    df_all = df_all.replace([np.inf, -np.inf], np.nan)
    for col in tqdm(feat_cols_to_zscore, desc='  z-score', leave=False):
        g = df_all.groupby('Query_ID')[col]
        df_all[col] = (df_all[col] - g.transform('mean')) / (g.transform('std') + 1e-8)
        
    final_features = feat_cols_to_zscore + feat_cols_raw
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
                   'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']:
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

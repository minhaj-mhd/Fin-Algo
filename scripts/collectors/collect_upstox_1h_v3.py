"""
Fetch NATIVE 1-hour candles directly from the Upstox V3 historical API and build a clean
1-hour ranking dataset. No resampling/anchoring choices by us — these are the exchange's
canonical hourly bars (09:15-anchored, 7 bars/day incl. the 15:15 stub).

Why: removes all resampling ambiguity. Live inference can fetch the IDENTICAL V3 candles
(HistoryV3Api, unit='hours', interval='1') -> guaranteed train/serve consistency.

Construction:
  * native 1h via HistoryV3Api.get_historical_candle_data1(ik,'hours','1',to,from)
  * IST wall-clock (tz-naive), drop the 15:15 partial stub -> 6 full-hour bars 09:15..14:15
  * session-masked Next_Hour_Return (NaN at each day's last bar -> no overnight leak)
  * same feature pipeline as production: compute_features(legacy=False) + market context + z-score

Output: data/ranking_data_upstox_1h_v3_3y.csv  (non-destructive new file)
Cache : data/raw_upstox_cache_1h_v3/{TICKER}.csv  (native 1h OHLCV per ticker)
"""
import os, sys, time, glob, warnings
import numpy as np
import pandas as pd
from datetime import date, timedelta, datetime, time as dtime
from tqdm import tqdm
warnings.filterwarnings('ignore')
sys.path.append(os.getcwd())

import upstox_client
from scripts.upstox_broker import UpstoxSandboxBroker
from scripts.feature_utils import compute_features

OUTPUT_CSV    = 'data/ranking_data_upstox_1h_v3_3y.csv'
RAW_CACHE_DIR = 'data/raw_upstox_cache_1h_v3'
START_DATE    = date(2022, 1, 1)
CHUNK_DAYS    = 90
RATE_PAUSE    = 0.25
MIN_BARS      = 60
VALID_TODS    = {'09:15', '10:15', '11:15', '12:15', '13:15', '14:15'}  # drop 15:15 stub
os.makedirs(RAW_CACHE_DIR, exist_ok=True)

broker = UpstoxSandboxBroker()
v3 = upstox_client.HistoryV3Api(broker.data_api_client)

from scripts.tickers import TICKERS

def date_chunks(start, end, n):
    out, cur = [], start
    while cur < end:
        ce = min(cur + timedelta(days=n), end)
        out.append((cur.strftime('%Y-%m-%d'), ce.strftime('%Y-%m-%d')))
        cur = ce + timedelta(days=1)
    return out

CHUNKS = date_chunks(START_DATE, date.today(), CHUNK_DAYS)
print(f"Tickers={len(TICKERS)}  chunks/ticker={len(CHUNKS)}  ~{len(TICKERS)*len(CHUNKS)} calls")

def fetch_native_1h(ticker):
    cache = os.path.join(RAW_CACHE_DIR, f"{ticker.replace('.NS','')}.csv")
    if os.path.exists(cache):
        try:
            ex = pd.read_csv(cache)
            if len(ex) >= MIN_BARS:
                return ex
        except Exception:
            pass
    ik = broker.get_instrument_key(ticker)
    rows = []
    for (frm, to) in CHUNKS:
        for attempt in range(2):
            try:
                resp = v3.get_historical_candle_data1(ik, 'hours', '1', to, frm)
                if resp.status == 'success' and resp.data and resp.data.candles:
                    rows.extend(resp.data.candles)
                break
            except Exception as e:
                if '429' in str(e) or 'Too Many' in str(e):
                    time.sleep(3); continue
                break
        time.sleep(RATE_PAUSE)
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
    df = df.drop_duplicates(subset='timestamp')
    df.to_csv(cache, index=False)
    return df

def session_masked_fwd(close):
    by_day = close.groupby(close.index.normalize())
    return by_day.shift(-1) / close - 1.0

def build_ticker(ticker, raw):
    dt = pd.to_datetime(raw['timestamp'], utc=True).dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
    df = pd.DataFrame({
        'DateTime': dt,
        'Open': raw['open'].astype(float), 'High': raw['high'].astype(float),
        'Low': raw['low'].astype(float), 'Close': raw['close'].astype(float),
        'Volume': raw['volume'].astype(float),
    }).dropna(subset=['DateTime', 'Open', 'Close'])
    df = df.drop_duplicates('DateTime').sort_values('DateTime').set_index('DateTime')
    # full-hour bars only (drop 15:15 stub + any non-standard)
    df = df[pd.Index(df.index).strftime('%H:%M').isin(VALID_TODS)]
    if len(df) < MIN_BARS:
        return None
    feat = compute_features(df[['Open', 'High', 'Low', 'Close', 'Volume']].copy(), legacy=False)
    feat['Next_Hour_Return'] = session_masked_fwd(feat['Close'])
    feat['DateTime'] = feat.index
    feat['Ticker'] = ticker
    return feat

def build_ranking(df_all):
    df_all = df_all.copy()
    df_all['DateTime'] = pd.to_datetime(df_all['DateTime'])
    df_all = df_all.dropna(subset=['Next_Hour_Return']).sort_values('DateTime')
    df_all['Query_ID'] = df_all.groupby('DateTime').ngroup()
    sizes = df_all.groupby('Query_ID').size()
    df_all = df_all[df_all['Query_ID'].isin(sizes[sizes >= 5].index)].copy()
    df_all = df_all.sort_values('DateTime')
    df_all['Query_ID'] = df_all.groupby('DateTime').ngroup()
    df_all['Market_Mean_Return']     = df_all.groupby('Query_ID')['Return'].transform('mean')
    df_all['Relative_Return']        = df_all['Return'] - df_all['Market_Mean_Return']
    df_all['Market_Mean_Volatility'] = df_all.groupby('Query_ID')['HL_Range'].transform('mean')
    df_all['Relative_Volatility']    = df_all['HL_Range'] / (df_all['Market_Mean_Volatility'] + 1e-8)
    exclude = {'DateTime', 'Query_ID', 'Ticker', 'Next_Hour_Return', 'Open', 'High', 'Low', 'Close', 'Volume',
               'Market_Mean_Return', 'Relative_Return', 'Market_Mean_Volatility', 'Relative_Volatility',
               'Hour', 'DayOfWeek', 'Is_Open_Hour', 'Is_Close_Hour', 'Time_To_Close'}
    feat_cols = [c for c in df_all.columns if c not in exclude]
    df_all = df_all.replace([np.inf, -np.inf], np.nan)
    for col in tqdm(feat_cols, desc='  z-score', leave=False):
        g = df_all.groupby('Query_ID')[col]
        df_all[col] = (df_all[col] - g.transform('mean')) / (g.transform('std') + 1e-8)
    return df_all.dropna(subset=feat_cols), feat_cols

print("\nPhase 1: fetch native 1h per ticker...")
frames, ok, skip = [], 0, 0
for ticker in tqdm(TICKERS, desc='Tickers'):
    try:
        raw = fetch_native_1h(ticker)
        if raw is None: skip += 1; continue
        f = build_ticker(ticker, raw)
        if f is not None: frames.append(f); ok += 1
        else: skip += 1
    except Exception as e:
        skip += 1; tqdm.write(f"  [skip] {ticker}: {str(e)[:70]}")
print(f"  OK={ok} skip={skip}")

print("\nPhase 2: build ranking dataset...")
df_all = pd.concat(frames, ignore_index=True)
final, fc = build_ranking(df_all)
final.to_csv(OUTPUT_CSV, index=False)
months = sorted(pd.to_datetime(final['DateTime']).dt.strftime('%Y-%m').unique())
print(f"\nSaved {OUTPUT_CSV}")
print(f"  rows={len(final):,}  queries={final['Query_ID'].nunique():,}  feats={len(fc)}")
print(f"  span: {months[0]} -> {months[-1]} ({len(months)} months)")
print(f"  tods: {sorted(pd.to_datetime(final['DateTime']).dt.strftime('%H:%M').unique())}")
print(f"  avg tickers/query: {final.groupby('Query_ID').size().mean():.1f}")

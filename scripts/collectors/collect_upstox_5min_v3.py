"""
Fetch NATIVE 5-minute candles from the Upstox V3 API and build a clean 5-minute
ranking dataset — to get ACTUAL numbers for the resolution×horizon matrix at 5m.

Mirrors collect_upstox_30min_v3.py exactly (same feature pipeline, session-masked
forward return, cross-sectional z-score) so results are comparable. Differences:
  * V3 API: HistoryV3Api.get_historical_candle_data1(ik,'minutes','5',to,from)
  * 5m needs SMALL date windows per request (~25 days; large ranges -> HTTP 400)
  * regular-session bars only (09:15..15:25 IST); session_masked Next_5Min_Return
Output: data/ranking_data_upstox_5min_v3_3y.csv   Cache: data/raw_upstox_cache_5min_v3/{T}.csv
"""
import os, sys, time, warnings
import numpy as np, pandas as pd
from datetime import date, timedelta, time as dtime
from tqdm import tqdm
warnings.filterwarnings('ignore'); sys.path.append(os.getcwd())

import upstox_client
from scripts.upstox_broker import UpstoxSandboxBroker
from scripts.feature_utils import compute_features
from scripts.tickers import TICKERS

OUTPUT_CSV    = 'data/ranking_data_upstox_5min_v3_3y.csv'
RAW_CACHE_DIR = 'data/raw_upstox_cache_5min_v3'
START_DATE    = date(2023, 1, 1)
CHUNK_DAYS    = 25
RATE_PAUSE    = 0.22
MIN_BARS      = 300
SESS_START, SESS_END = dtime(9, 15), dtime(15, 25)
os.makedirs(RAW_CACHE_DIR, exist_ok=True)

broker = UpstoxSandboxBroker()
v3 = upstox_client.HistoryV3Api(broker.data_api_client)

def date_chunks(start, end, n):
    out, cur = [], start
    while cur < end:
        ce = min(cur + timedelta(days=n), end)
        out.append((cur.strftime('%Y-%m-%d'), ce.strftime('%Y-%m-%d'))); cur = ce + timedelta(days=1)
    return out
CHUNKS = date_chunks(START_DATE, date.today(), CHUNK_DAYS)
print(f"Tickers={len(TICKERS)}  chunks/ticker={len(CHUNKS)}  ~{len(TICKERS)*len(CHUNKS)} calls", flush=True)

def fetch_5m(ticker):
    cache = os.path.join(RAW_CACHE_DIR, f"{ticker.replace('.NS','')}.csv")
    if os.path.exists(cache):
        try:
            ex = pd.read_csv(cache)
            if len(ex) >= MIN_BARS: return ex
        except Exception: pass
    ik = broker.get_instrument_key(ticker); rows = []
    for (frm, to) in CHUNKS:
        for attempt in range(2):
            try:
                resp = v3.get_historical_candle_data1(ik, 'minutes', '5', to, frm)
                if resp.status == 'success' and resp.data and resp.data.candles:
                    rows.extend(resp.data.candles)
                break
            except Exception as e:
                if '429' in str(e) or 'Too Many' in str(e): time.sleep(3); continue
                break
        time.sleep(RATE_PAUSE)
    if not rows: return None
    df = pd.DataFrame(rows, columns=['timestamp','open','high','low','close','volume','oi']).drop_duplicates('timestamp')
    df.to_csv(cache, index=False)
    return df

def session_masked_fwd(close):
    return close.groupby(close.index.normalize()).shift(-1) / close - 1.0

def build_ticker(ticker, raw):
    dt = pd.to_datetime(raw['timestamp'], utc=True).dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
    df = pd.DataFrame({'DateTime': dt, 'Open': raw['open'].astype(float), 'High': raw['high'].astype(float),
                       'Low': raw['low'].astype(float), 'Close': raw['close'].astype(float),
                       'Volume': raw['volume'].astype(float)}).dropna(subset=['DateTime','Open','Close'])
    df = df.drop_duplicates('DateTime').sort_values('DateTime').set_index('DateTime')
    t = pd.Index(df.index).time
    df = df[(t >= SESS_START) & (t <= SESS_END)]
    if len(df) < MIN_BARS: return None
    feat = compute_features(df[['Open','High','Low','Close','Volume']].copy(), legacy=False)
    feat['Next_5Min_Return'] = session_masked_fwd(feat['Close'])
    feat['DateTime'] = feat.index; feat['Ticker'] = ticker
    return feat

def build_ranking(df_all):
    df_all = df_all.copy(); df_all['DateTime'] = pd.to_datetime(df_all['DateTime'])
    df_all = df_all.dropna(subset=['Next_5Min_Return']).sort_values('DateTime')
    df_all['Query_ID'] = df_all.groupby('DateTime').ngroup()
    sizes = df_all.groupby('Query_ID').size()
    df_all = df_all[df_all['Query_ID'].isin(sizes[sizes >= 5].index)].copy()
    df_all['Query_ID'] = df_all.groupby('DateTime').ngroup()
    df_all['Market_Mean_Return']     = df_all.groupby('Query_ID')['Return'].transform('mean')
    df_all['Relative_Return']        = df_all['Return'] - df_all['Market_Mean_Return']
    df_all['Market_Mean_Volatility'] = df_all.groupby('Query_ID')['HL_Range'].transform('mean')
    df_all['Relative_Volatility']    = df_all['HL_Range'] / (df_all['Market_Mean_Volatility'] + 1e-8)
    exclude = {'DateTime','Query_ID','Ticker','Next_5Min_Return','Open','High','Low','Close','Volume',
               'Market_Mean_Return','Relative_Return','Market_Mean_Volatility','Relative_Volatility',
               'Hour','DayOfWeek','Is_Open_Hour','Is_Close_Hour','Time_To_Close'}
    feat_cols = [c for c in df_all.columns if c not in exclude]
    df_all = df_all.replace([np.inf, -np.inf], np.nan)
    for col in feat_cols:
        g = df_all.groupby('Query_ID')[col]
        df_all[col] = (df_all[col] - g.transform('mean')) / (g.transform('std') + 1e-8)
    return df_all.dropna(subset=feat_cols), feat_cols

print("\nPhase 1: fetch native 5m per ticker...", flush=True)
frames, ok, skip = [], 0, 0
for ticker in tqdm(TICKERS, desc='Tickers'):
    try:
        raw = fetch_5m(ticker)
        if raw is None: skip += 1; continue
        f = build_ticker(ticker, raw)
        if f is not None: frames.append(f); ok += 1
        else: skip += 1
    except Exception as e:
        skip += 1; tqdm.write(f"  [skip] {ticker}: {str(e)[:70]}")
print(f"  OK={ok} skip={skip}", flush=True)
if not frames:
    print("[FATAL] No data collected."); sys.exit(1)

print("\nPhase 2: build ranking dataset (this is the heavy step)...", flush=True)
df_all = pd.concat(frames, ignore_index=True)
final, fc = build_ranking(df_all)
final.to_csv(OUTPUT_CSV, index=False)
months = sorted(pd.to_datetime(final['DateTime']).dt.strftime('%Y-%m').unique())
print(f"\nSaved {OUTPUT_CSV}", flush=True)
print(f"  rows={len(final):,}  queries={final['Query_ID'].nunique():,}  feats={len(fc)}")
print(f"  span: {months[0]} -> {months[-1]} ({len(months)} months)")
print(f"  avg tickers/query: {final.groupby('Query_ID').size().mean():.1f}")

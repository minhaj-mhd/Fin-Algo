import os, sys, json, urllib.parse, datetime as dt, warnings
import concurrent.futures as cf
sys.path.insert(0, os.getcwd())
warnings.filterwarnings("ignore")

import requests
import numpy as np
import pandas as pd
import yfinance as yf

import xgboost as xgb
from scripts.tickers import TICKERS
from scripts.feature_utils import build_rolling_1h_ohlcv, compute_features

# ── CONFIG ──────────────────────────────────────────────────────────────────
TODAY       = dt.date.today()
N_SESSIONS  = 5   # this week (5 trading days)
ENTRY_FROM  = "10:15"
ENTRY_TO    = "14:15"
HOLD        = pd.Timedelta(hours=1)

V20_LONG  = "models/research/v20_rolling_1h/xgb_long_model.json"
V20_SHORT = "models/research/v20_rolling_1h/xgb_short_model.json"
V20_META  = "models/research/v20_rolling_1h/metadata.json"

COST_BPS_PRIMARY = 6.0

# uniform live per-trade notional
CAPITAL, MAX_SLOTS, MARGIN = 99517.68, 5, 5.0
NOTIONAL     = (CAPITAL / MAX_SLOTS) * MARGIN
BROKERAGE_RT = 20.0
STT_RATE     = 0.00025
SLIP_PCT     = 0.0003

CACHE = os.path.join(os.environ.get("TEMP", "/tmp"), "claude",
                     "c--Users-loq-Desktop-Trading-finalgo", "today_top1_15m_cache")
os.makedirs(CACHE, exist_ok=True)
HDR = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Accept": "application/json"}
INSTR = json.load(open("scripts/instrument_cache.json"))

def fetch_15m(ticker):
    sym = ticker.replace(".NS", "")
    ik = INSTR.get(sym)
    if not ik: return ticker, None
    cache_f = os.path.join(CACHE, f"{sym}_{TODAY}.csv")
    if os.path.exists(cache_f):
        df = pd.read_csv(cache_f, parse_dates=["timestamp"])
    else:
        enc = urllib.parse.quote(ik, safe="")
        frm = TODAY - dt.timedelta(days=25)
        candles = []
        try:
            r = requests.get(f"https://api.upstox.com/v3/historical-candle/{enc}/minutes/15/{TODAY}/{frm}", headers=HDR, timeout=25).json()
            candles += r.get("data", {}).get("candles", [])
            r2 = requests.get(f"https://api.upstox.com/v3/historical-candle/intraday/{enc}/minutes/15", headers=HDR, timeout=25).json()
            candles += r2.get("data", {}).get("candles", [])
        except Exception:
            return ticker, None
        if not candles: return ticker, None
        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        df = df.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
        df.to_csv(cache_f, index=False)
    df = df.set_index("timestamp").rename(columns={
        "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
    return ticker, df[["Open", "High", "Low", "Close", "Volume"]].dropna()


print("=" * 82)
print(f"  NEW GATES SIMULATION (FRESH SCORES)  |  last {N_SESSIONS} sessions")
print("=" * 82)

# NIFTY LOGIC
nifty = yf.download('^NSEI', period='15d', interval='15m', progress=False)
if isinstance(nifty.columns, pd.MultiIndex):
    nifty.columns = nifty.columns.get_level_values(0)
nifty = nifty.reset_index()
nifty.rename(columns={'Datetime': 'ts', 'Open': 'open', 'Close': 'close'}, inplace=True)
if nifty['ts'].dt.tz is not None: nifty['ts'] = nifty['ts'].dt.tz_localize(None)

nifty['ts'] = nifty['ts'] + pd.Timedelta(hours=5, minutes=30)
nifty['nifty_ret_2h'] = nifty['close'] / nifty['close'].shift(8) - 1
nifty['date'] = nifty['ts'].dt.date
daily_open = nifty.groupby('date')['open'].first().reset_index()
daily_open.rename(columns={'open': 'daily_open'}, inplace=True)
nifty = pd.merge(nifty, daily_open, on='date', how='left')
nifty['nifty_intraday'] = nifty['close'] / nifty['daily_open'] - 1
nifty_map = dict(zip(nifty['ts'], nifty['nifty_ret_2h']))
nifty_intra_map = dict(zip(nifty['ts'], nifty['nifty_intraday']))

def get_nifty_context(ts):
    floored_ts = ts.replace(minute=(ts.minute // 15) * 15, second=0, microsecond=0)
    ret_2h = nifty_map.get(floored_ts, None)
    intra = nifty_intra_map.get(floored_ts, None)
    return ret_2h, intra

v20_feats = json.load(open(V20_META))["features"]
bst_l = xgb.Booster(); bst_l.load_model(V20_LONG)
bst_s = xgb.Booster(); bst_s.load_model(V20_SHORT)

print(f"[1/4] Fetching 15-min candles for {len(TICKERS)} tickers (cache reuse)...")
raw15 = {}
with cf.ThreadPoolExecutor(max_workers=8) as ex:
    for tk, df in ex.map(fetch_15m, TICKERS):
        if df is not None and len(df) >= 40:
            raw15[tk] = df

print("[2/4] Building rolling-1h features...")
feat, close_1h = {}, {}
for tk, df15 in raw15.items():
    try:
        h1 = build_rolling_1h_ohlcv(df15)
        if len(h1) < 20: continue
        feat[tk] = compute_features(h1[["Open", "High", "Low", "Close", "Volume"]].copy(), legacy=False)
        close_1h[tk] = h1["Close"]
    except Exception:
        continue

all_dates = set()
for f in feat.values():
    for tsx in f.index:
        if ENTRY_FROM <= tsx.strftime("%H:%M") <= ENTRY_TO:
            all_dates.add(tsx.date())
sessions = sorted(all_dates)[-N_SESSIONS:]

def score_anchor(ts):
    rows, tickers = [], []
    for tk, f in feat.items():
        if ts in f.index:
            rows.append(f.loc[ts]); tickers.append(tk)
    if len(tickers) < 10: return None
    X = pd.DataFrame(rows, index=tickers)
    X["Market_Mean_Return"] = X["Return"].mean()
    X["Relative_Return"] = X["Return"] - X["Market_Mean_Return"]
    X["Market_Mean_Volatility"] = X["HL_Range"].mean()
    X["Relative_Volatility"] = X["HL_Range"] / (X["Market_Mean_Volatility"] + 1e-8)
    Xm = np.nan_to_num(X[v20_feats].values.astype(np.float32))
    dm = xgb.DMatrix(Xm, feature_names=v20_feats)
    ls = bst_l.predict(dm); ss = bst_s.predict(dm)
    l_c = ls - ls.mean(); s_c = ss - ss.mean()
    return pd.DataFrame({"ticker": tickers, "ls": ls, "ss": ss,
                         "long_conv": l_c - s_c, "short_conv": s_c - l_c})

def exit_price(tk, ts):
    s = close_1h[tk]; ex_ts = ts + HOLD
    if ex_ts in s.index: return float(s.loc[ex_ts]), ex_ts
    after = s[s.index >= ex_ts]
    if not after.empty: return float(after.iloc[0]), after.index[0]
    same = s[s.index.date == ts.date()]
    return float(same.iloc[-1]), same.index[-1]

print("[3/4] Scoring anchors + generating gated trades...\n")
long_trades = []
short_trades = []

for d in sessions:
    anchors = sorted({tsx for f in feat.values() for tsx in f.index
                      if tsx.date() == d and ENTRY_FROM <= tsx.strftime("%H:%M") <= ENTRY_TO})
    for ts in anchors:
        sc = score_anchor(ts)
        if sc is None: continue
        
        n_2h, n_intra = get_nifty_context(ts)
        if pd.isna(n_2h) or pd.isna(n_intra): continue
        
        # Long Gate
        if n_2h > 0.0025 and n_intra > 0.0020:
            best_l = sc.sort_values('long_conv', ascending=False).iloc[0]
            tk = best_l["ticker"]
            ep = float(close_1h[tk].loc[ts])
            if ep > 0:
                xp, xts = exit_price(tk, ts)
                gross_bps = ((xp - ep) / ep) * 1e4
                long_trades.append({'ts': ts, 'ticker': tk, 'side': 'LONG', 
                                    'gross_bps': gross_bps, 'net_bps': gross_bps - COST_BPS_PRIMARY})
                
        # Short Gate
        t_time = ts.time()
        if (n_2h <= 0.0025 or n_intra > 0.0036):
            if (t_time < dt.time(11, 30) or t_time > dt.time(13, 0)):
                cands = sc[sc['ss'] > 0.082].sort_values('short_conv', ascending=False)
                if len(cands) > 0:
                    best_s = cands.iloc[0]
                    tk = best_s["ticker"]
                    ep = float(close_1h[tk].loc[ts])
                    if ep > 0:
                        xp, xts = exit_price(tk, ts)
                        gross_bps = ((ep - xp) / ep) * 1e4
                        short_trades.append({'ts': ts, 'ticker': tk, 'side': 'SHORT', 
                                             'gross_bps': gross_bps, 'net_bps': gross_bps - COST_BPS_PRIMARY})

print("[4/4] Results:")
ldf = pd.DataFrame(long_trades)
sdf = pd.DataFrame(short_trades)
cdf = pd.concat([ldf, sdf])

if len(cdf) == 0:
    print("NO TRADES TAKEN THIS WEEK.")
else:
    print(f"Total Trades: {len(cdf)}")
    print(f"Combined Avg Net BPS: {cdf.net_bps.mean():.2f}")
    
    if len(sdf) > 0:
        print(f"\n--- SHORTS ---")
        print(f"Trades: {len(sdf)} | Win Rate: {(sdf.net_bps > 0).mean():.1%} | Avg Net BPS: {sdf.net_bps.mean():.2f}")
    if len(ldf) > 0:
        print(f"\n--- LONGS ---")
        print(f"Trades: {len(ldf)} | Win Rate: {(ldf.net_bps > 0).mean():.1%} | Avg Net BPS: {ldf.net_bps.mean():.2f}")

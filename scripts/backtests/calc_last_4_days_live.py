import sys, os, json, time, warnings
sys.path.insert(0, os.getcwd())
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime, date, timedelta
from pathlib import Path

import upstox_client
from dotenv import load_dotenv
load_dotenv()

from scripts.tickers import TICKERS
from scripts.feature_utils import compute_features

# 1. DOWNLOAD JULY DATA
CACHE_JULY = Path("data/oos_cache_15min_jul2026")
CACHE_JULY.mkdir(parents=True, exist_ok=True)
OOS_FROM = date(2026, 6, 19)
OOS_TO   = date(2026, 7, 10)

tok = os.getenv("UPSTOX_ANALYTICS_ACCESS_TOKEN")
cfg = upstox_client.Configuration(sandbox=False)
cfg.host = "https://api.upstox.com"
cfg.access_token = tok
api_client = upstox_client.ApiClient(cfg)
v3api = upstox_client.HistoryV3Api(api_client)

with open("scripts/instrument_cache.json") as f:
    INST_MAP = json.load(f)

def get_ik(ticker):
    sym = ticker.replace(".NS", "")
    return INST_MAP.get(ticker) or INST_MAP.get(sym)

print("Downloading July Data...")
for idx, ticker in enumerate(TICKERS, 1):
    sym = ticker.replace(".NS", "")
    cache_path = CACHE_JULY / f"{sym}.csv"
    if not cache_path.exists():
        ik = get_ik(ticker)
        if ik:
            try:
                resp = v3api.get_historical_candle_data1(ik, "minutes", "15", OOS_TO.strftime("%Y-%m-%d"), OOS_FROM.strftime("%Y-%m-%d"))
                if resp.status == "success" and resp.data and resp.data.candles:
                    df = pd.DataFrame(resp.data.candles, columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
                    df.to_csv(cache_path, index=False)
            except:
                pass
        time.sleep(0.1)

print("Processing Features and Running Model...")

def load_all_data(ticker):
    sym = ticker.replace(".NS", "")
    dfs = []
    
    # 1. warmup
    p1 = Path(f"data/raw_upstox_cache_15min_3y/{sym}.csv")
    if p1.exists():
        d = pd.read_csv(p1)
        if len(d) > 500: d = d.iloc[-500:]
        dfs.append(d)
        
    # 2. june oos
    p2 = Path(f"data/oos_cache_15min_jun2026/{sym}.csv")
    if p2.exists(): dfs.append(pd.read_csv(p2))
        
    # 3. july oos
    p3 = Path(f"data/oos_cache_15min_jul2026/{sym}.csv")
    if p3.exists(): dfs.append(pd.read_csv(p3))
        
    if not dfs: return None
    df = pd.concat(dfs, ignore_index=True)
    dt_col = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()][0]
    df[dt_col] = pd.to_datetime(df[dt_col], format='mixed', utc=True)
    df["DateTime"] = df[dt_col].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
    df = df.drop_duplicates("DateTime").sort_values("DateTime").set_index("DateTime")
    return df[["Open","High","Low","Close","Volume"]].dropna()

def build_rolling_1h(df_15m):
    if len(df_15m) < 4: return pd.DataFrame()
    out = pd.DataFrame(index=df_15m.index)
    out["Open"] = df_15m["Open"].shift(3)
    out["High"] = df_15m["High"].rolling(4).max()
    out["Low"] = df_15m["Low"].rolling(4).min()
    out["Close"] = df_15m["Close"]
    out["Volume"] = df_15m["Volume"].rolling(4).sum()
    out["Next_Hour_Return"] = out["Close"].shift(-4) / out["Close"] - 1
    out = out.dropna()
    tods = out.index.strftime("%H:%M")
    out = out[pd.Series(tods).isin({"09:15", "10:15", "11:15", "12:15", "13:15", "14:15"}).values]
    return out

v20_feats = json.load(open("models/research/v20_rolling_1h/metadata.json"))["features"]
bs = xgb.Booster()
bs.load_model("models/research/v20_rolling_1h/xgb_short_model.json")

target_dates = [date(2026, 7, 6), date(2026, 7, 7), date(2026, 7, 8), date(2026, 7, 9)]
anchors = []

all_features = []
for ticker in TICKERS:
    df_15 = load_all_data(ticker)
    if df_15 is None: continue
    df_1h = build_rolling_1h(df_15)
    if len(df_1h) < 10: continue
    
    try:
        f20 = compute_features(df_1h[["Open","High","Low","Close","Volume"]].copy(), legacy=False)
        f20 = f20[f20.index.strftime("%H:%M").isin({"10:15", "11:15", "12:15", "13:15", "14:15"})]
        f20 = f20[f20.index.date >= date(2026, 7, 6)]
        f20 = f20[f20.index.date <= date(2026, 7, 9)]
        
        # Merge next hour return
        f20["Next_Hour_Return"] = df_1h["Next_Hour_Return"]
        f20["Ticker"] = ticker
        all_features.append(f20.reset_index())
    except:
        pass

if not all_features:
    print("No features generated.")
    sys.exit()

master = pd.concat(all_features, ignore_index=True)
master = master.dropna(subset=["Next_Hour_Return"])

# compute cross-sectional features per timestamp
def _cs(df):
    if "Return" in df.columns:
        df["Market_Mean_Return"] = df["Return"].mean()
        df["Relative_Return"] = df["Return"] - df["Market_Mean_Return"]
    if "HL_Range" in df.columns:
        df["Market_Mean_Volatility"] = df["HL_Range"].mean()
        df["Relative_Volatility"] = df["HL_Range"] / (df["Market_Mean_Volatility"] + 1e-8)
    for c in v20_feats:
        if c not in df.columns: df[c] = 0.0
    return df

master = master.groupby("DateTime", group_keys=False).apply(_cs)
if "DateTime" not in master.columns:
    master = master.reset_index()
    if "index" in master.columns:
        master = master.rename(columns={"index": "DateTime"})
master = master.dropna(subset=v20_feats)
master["retbps"] = master["Next_Hour_Return"] * 10000

bl = xgb.Booster()
bl.load_model("models/research/v20_rolling_1h/xgb_long_model.json")
X = xgb.DMatrix(np.nan_to_num(master[v20_feats].values.astype(np.float32)), feature_names=v20_feats)
master["ss"] = bs.predict(X)
master["ls"] = bl.predict(X)

def _calc_mixed(g):
    l_c = g["ls"] - g["ls"].mean()
    s_c = g["ss"] - g["ss"].mean()
    g["short_conviction"] = s_c - l_c
    return g

if "DateTime" in master.columns:
    group_col = "DateTime"
else:
    group_col = master.index.names[0] if master.index.names[0] else master.index

master = master.groupby(group_col, group_keys=False).apply(_calc_mixed)

trades = []
last = {}
COST = 6
NOTIONAL = 99517.68

# print(master.head()) # debug
    
# Check if DateTime is in columns or index
if "DateTime" in master.columns:
    group_col = "DateTime"
else:
    group_col = master.index.names[0] if master.index.names[0] else master.index
        
    for ts, g in master.groupby(group_col):
        # Rank purely by the shadow (mixed) conviction rate, no hard threshold
        cands = g.sort_values("short_conviction", ascending=False)
        
        picks = []
        for _, row in cands.iterrows():
            t = row["Ticker"]
            curr_ts = row["DateTime"] if "DateTime" in row else ts
            
            skip = False
            if t in last:
                try:
                    delta = (pd.to_datetime(curr_ts) - pd.to_datetime(last[t])).total_seconds()
                except:
                    delta = (curr_ts - last[t]) / 1e9 if isinstance(curr_ts, (int, float)) else 0
                if delta < 7200:
                    skip = True
            
            if skip: continue
            
            picks.append((row, curr_ts))
            if len(picks) == 1:
                break
                
        for row, curr_ts in picks:
            last[row["Ticker"]] = curr_ts
            trades.append((curr_ts, row["Ticker"], -row["retbps"], row["short_conviction"]))

print(f"Total trades selected: {len(trades)}")
if trades:
    td = pd.DataFrame(trades, columns=["ts", "tk", "pnl", "score"])
    td["ts"] = pd.to_datetime(td["ts"])
    td["net6"] = td.pnl - COST
    td["bookRs"] = td.net6 / 10000 * NOTIONAL
    td["date"] = td.ts.dt.date
    print(td.date.value_counts())
    
    print("="*60)
    print("  SHORT MODEL (MIXED CONVICTION) - LIVE JULY 6-9, 2026")
    print("="*60)
    for d in target_dates:
        d_trades = td[td.date == d]
        n = len(d_trades)
        if n == 0:
            print(f"  {d}: n=0 | No trades taken")
        else:
            net = d_trades.net6.mean()
            rs = d_trades.bookRs.sum()
            tickers = ", ".join(d_trades.tk.tolist())
            print(f"  {d}: n={n} | net@6={net:+.2f} | Rs {rs:+,.0f} | {tickers}")
else:
    print("No trades found >= 0.0826 threshold for July 6-9.")

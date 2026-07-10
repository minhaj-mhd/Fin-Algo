"""Dated top-1 long panel with RVOL + own-gap, to join with the idx2h gate."""
import os, sys, json, datetime as dt, warnings
import concurrent.futures as cf
sys.path.insert(0, os.getcwd()); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, xgboost as xgb
from scripts.tickers import TICKERS
from scripts.feature_utils import build_rolling_1h_ohlcv, compute_features

TODAY = dt.date.today(); OOS_FROM = dt.date(2026, 6, 16)
HOLD = pd.Timedelta(hours=1)
CACHE = os.path.join(os.environ.get("TEMP", "/tmp"), "claude", "c--Users-loq-Desktop-Trading-finalgo", "today_top1_15m_cache")
INSTR = json.load(open("scripts/instrument_cache.json"))
v20 = json.load(open("models/research/v20_rolling_1h/metadata.json"))["features"]


def load(t):
    sym = t.replace(".NS", ""); p = os.path.join(CACHE, f"{sym}_{TODAY}.csv")
    if not os.path.exists(p) or sym not in INSTR: return t, None
    df = pd.read_csv(p, parse_dates=["timestamp"]).set_index("timestamp").rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"})
    return t, df[["Open","High","Low","Close","Volume"]].dropna()


bl = xgb.Booster(); bl.load_model("models/research/v20_rolling_1h/xgb_long_model.json")
bs = xgb.Booster(); bs.load_model("models/research/v20_rolling_1h/xgb_short_model.json")
raw, feat, c1h = {}, {}, {}
with cf.ThreadPoolExecutor(max_workers=8) as ex:
    for t, df in ex.map(load, TICKERS):
        if df is None or len(df) < 40: continue
        raw[t] = df
        try:
            h = build_rolling_1h_ohlcv(df)
            if len(h) >= 20: feat[t] = compute_features(h[["Open","High","Low","Close","Volume"]].copy(), legacy=False); c1h[t] = h["Close"]
        except Exception: pass
anchors = sorted({ts for f in feat.values() for ts in f.index if ts.date() >= OOS_FROM and "10:15" <= ts.strftime("%H:%M") <= "14:15"})
rows = []
for ts in anchors:
    fr, tks = [], []
    for t, f in feat.items():
        if ts in f.index and (ts+HOLD) in c1h[t].index: fr.append(f.loc[ts]); tks.append(t)
    if len(tks) < 10: continue
    X = pd.DataFrame(fr, index=tks)
    X["Market_Mean_Return"]=X["Return"].mean(); X["Relative_Return"]=X["Return"]-X["Market_Mean_Return"]
    X["Market_Mean_Volatility"]=X["HL_Range"].mean(); X["Relative_Volatility"]=X["HL_Range"]/(X["Market_Mean_Volatility"]+1e-8)
    dm = xgb.DMatrix(np.nan_to_num(X[v20].values.astype(np.float32)), feature_names=v20)
    ls = bl.predict(dm); i = int(np.argmax(ls)); t = tks[i]
    ep = float(c1h[t].loc[ts]); xp = float(c1h[t].loc[ts+HOLD])
    rows.append({"date": ts.date().isoformat(), "anchor": ts.strftime("%H:%M"), "ticker": t.replace(".NS",""),
                 "rvol": float(X["RVOL"].iloc[i]), "Lg": (xp-ep)/ep*1e4})
pd.DataFrame(rows).to_csv(f"data/backtests/long_top1_rvol_{TODAY}.csv", index=False)

# join idx2h and analyze the 47
d = pd.read_csv(f"data/backtests/long_top1_rvol_{TODAY}.csv")
g = pd.read_csv("data/backtests/nifty_2h_long_gate_2026-07-10.csv")[["date","anchor","idx2h"]]
m = d.merge(g, on=["date","anchor"], how="left")
q = m[m.idx2h >= 0.3].copy()
def blk(x,l):
    print(f"  {l:30s} n={len(x):2d} WR={100*(x.Lg>0).mean():3.0f}% avg Lg={x.Lg.mean():+7.1f} net@6={x.Lg.mean()-6:+7.1f} bps  medRVOL={x.rvol.median():.2f}")
print("\n=== the 47 qualifying longs (idx2h>=0.3) split by RVOL ===")
blk(q, "ALL 47")
blk(q[q.rvol >= 1.0], "RVOL >= 1.0 (capitulation)")
blk(q[q.rvol < 1.0], "RVOL <  1.0 (low-vol drift)")
print("\n=== per-day avg RVOL of the qualifying longs (loser days vs rest) ===")
for day, x in q.groupby("date"):
    flag = "  <-- LOSER" if x.Lg.mean()-6 < 0 else ""
    print(f"  {day}  n={len(x)}  medRVOL={x.rvol.median():.2f}  net@6={x.Lg.mean()-6:+7.1f}{flag}")

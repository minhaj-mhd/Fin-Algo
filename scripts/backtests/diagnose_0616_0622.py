"""
Diagnose why the long gate flipped negative on 2026-06-16 & 06-22
=================================================================
Re-scores the loser days (06-16, 06-22) and two winning up-gap days (06-25, 07-03)
and, for each 15-min anchor, dumps the top-1 LONG pick with characteristics:
ticker, sector, market-cap rank, its OWN overnight gap, last-bar return, RVOL,
long_score, conviction, and the realized 1h fwd return. Goal: find what was
different about the picks/regime on the two loser days.
"""
import os, sys, json, datetime as dt, warnings
import concurrent.futures as cf
sys.path.insert(0, os.getcwd()); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, xgboost as xgb
from scripts.tickers import TICKERS
from scripts.feature_utils import build_rolling_1h_ohlcv, compute_features
from scripts.sector_map import SECTOR_MAP

TODAY = dt.date.today()
DAYS = ["2026-06-16", "2026-06-22", "2026-06-25", "2026-07-03"]
ENTRY_FROM, ENTRY_TO = "10:15", "14:15"
HOLD = pd.Timedelta(hours=1)
CACHE = os.path.join(os.environ.get("TEMP", "/tmp"), "claude",
                     "c--Users-loq-Desktop-Trading-finalgo", "today_top1_15m_cache")
INSTR = json.load(open("scripts/instrument_cache.json"))
v20_feats = json.load(open("models/research/v20_rolling_1h/metadata.json"))["features"]
try:
    MC = {t.replace(".NS", ""): v.get("rank") for t, v in
          json.load(open("data/marketcap_ranks.json", encoding="utf-8"))["tickers"].items()}
except Exception:
    MC = {}


def load(ticker):
    sym = ticker.replace(".NS", "")
    p = os.path.join(CACHE, f"{sym}_{TODAY}.csv")
    if not os.path.exists(p) or sym not in INSTR:
        return ticker, None
    df = pd.read_csv(p, parse_dates=["timestamp"]).set_index("timestamp").rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
    return ticker, df[["Open", "High", "Low", "Close", "Volume"]].dropna()


bst_l = xgb.Booster(); bst_l.load_model("models/research/v20_rolling_1h/xgb_long_model.json")
bst_s = xgb.Booster(); bst_s.load_model("models/research/v20_rolling_1h/xgb_short_model.json")
raw15, feat, close_1h = {}, {}, {}
with cf.ThreadPoolExecutor(max_workers=8) as ex:
    for tk, df in ex.map(load, TICKERS):
        if df is None or len(df) < 40:
            continue
        raw15[tk] = df
        try:
            h1 = build_rolling_1h_ohlcv(df)
            if len(h1) >= 20:
                feat[tk] = compute_features(h1[["Open", "High", "Low", "Close", "Volume"]].copy(), legacy=False)
                close_1h[tk] = h1["Close"]
        except Exception:
            pass


def stock_gap(tk, day):
    """today's first 15m open / prev session last close - 1, in %."""
    df = raw15[tk]
    dd = pd.Timestamp(day).date()
    today = df[df.index.date == dd]
    prev = df[df.index.date < dd]
    if today.empty or prev.empty:
        return np.nan
    return (today["Open"].iloc[0] / prev["Close"].iloc[-1] - 1) * 100


for day in DAYS:
    dd = pd.Timestamp(day).date()
    anchors = sorted({ts for f in feat.values() for ts in f.index
                      if ts.date() == dd and ENTRY_FROM <= ts.strftime("%H:%M") <= ENTRY_TO})
    print("\n" + "=" * 100)
    print(f"  {day}   (top-1 LONG per anchor)")
    print("=" * 100)
    print(f"  {'anchor':6s} {'ticker':12s} {'sector':8s} {'mcap':>4s} {'gap%':>6s} "
          f"{'lastRet%':>8s} {'RVOL':>5s} {'lscore':>7s} {'conv':>7s} {'fwd1h_bps':>9s}")
    for ts in anchors:
        rows, tks = [], []
        for tk, f in feat.items():
            if ts in f.index and (ts + HOLD) in close_1h[tk].index:
                rows.append(f.loc[ts]); tks.append(tk)
        if len(tks) < 10:
            continue
        X = pd.DataFrame(rows, index=tks)
        X["Market_Mean_Return"] = X["Return"].mean(); X["Relative_Return"] = X["Return"] - X["Market_Mean_Return"]
        X["Market_Mean_Volatility"] = X["HL_Range"].mean(); X["Relative_Volatility"] = X["HL_Range"] / (X["Market_Mean_Volatility"] + 1e-8)
        dm = xgb.DMatrix(np.nan_to_num(X[v20_feats].values.astype(np.float32)), feature_names=v20_feats)
        ls = bst_l.predict(dm); ss = bst_s.predict(dm)
        lconv = (ls - ls.mean()) - (ss - ss.mean())
        i = int(np.argmax(ls)); tk = tks[i]; sym = tk.replace(".NS", "")
        ep = float(close_1h[tk].loc[ts]); xp = float(close_1h[tk].loc[ts + HOLD])
        fwd = (xp - ep) / ep * 1e4
        rv = float(X["RVOL"].iloc[i]) if "RVOL" in X else np.nan
        ret = float(X["Return"].iloc[i]) * 100
        print(f"  {ts.strftime('%H:%M'):6s} {sym:12s} {SECTOR_MAP.get(tk,'?'):8s} "
              f"{str(MC.get(sym,'-')):>4s} {stock_gap(tk,day):+6.2f} {ret:+8.2f} {rv:5.2f} "
              f"{ls[i]:+7.3f} {lconv[i]:+7.4f} {fwd:+9.1f}")

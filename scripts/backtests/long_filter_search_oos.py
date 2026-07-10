"""
LONG-side filter search (OOS)  —  cut the bad longs, keep the good ones
======================================================================
The short leg works; the long leg is net-negative at every conviction level.
Goal: find entry-time discriminators that separate winning top-1 longs from
losing ones, so we can gate the long book down to only the good signals.

OOS = 2026-06-16..07-09 (v20 trained 06-15). For every 15-min anchor [10:15..
14:15] we score the universe, take the top-1 long (argmax long_score = live
pick), and record a vector of causal entry-time features + the realized 1h fwd
return. We also keep the FULL long candidate pool for large-n monotonicity.

Analyses:
  [1] top-1 long by anchor time
  [2] top-1 long by feature tercile (net@6 low/mid/high)  -> where longs die
  [3] full-pool Spearman(feature, fwd return) (large n, robust direction)
  [4] best single/'combined' filter applied to the top-1 long stream

Run: python -m scripts.backtests.long_filter_search_oos
"""
import os, sys, json, datetime as dt, warnings
import concurrent.futures as cf
sys.path.insert(0, os.getcwd()); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, xgboost as xgb
from scipy.stats import spearmanr
from scripts.tickers import TICKERS
from scripts.feature_utils import build_rolling_1h_ohlcv, compute_features

TODAY = dt.date.today()
OOS_FROM = dt.date(2026, 6, 16)
ENTRY_FROM, ENTRY_TO = "10:15", "14:15"
HOLD = pd.Timedelta(hours=1)
COST6 = 6
V20_META = "models/research/v20_rolling_1h/metadata.json"
CACHE = os.path.join(os.environ.get("TEMP", "/tmp"), "claude",
                     "c--Users-loq-Desktop-Trading-finalgo", "today_top1_15m_cache")
INSTR = json.load(open("scripts/instrument_cache.json"))

# candidate entry-time discriminators (must exist in compute_features output)
FEATS = ["Return", "Relative_Return", "RVOL", "RSI_14", "Dist_52W_High", "Dist_52W_Low",
         "Dist_SMA_50", "VWAP_Dist", "IBS", "Stoch_K", "Alpha_3H", "HL_Range"]

# market-cap rank
try:
    _mc = json.load(open("data/marketcap_ranks.json", encoding="utf-8"))["tickers"]
    MCAP = {t.replace(".NS", ""): v.get("rank") for t, v in _mc.items()}
except Exception:
    MCAP = {}


def load(ticker):
    sym = ticker.replace(".NS", "")
    cf_ = os.path.join(CACHE, f"{sym}_{TODAY}.csv")
    if not os.path.exists(cf_) or sym not in INSTR: return ticker, None
    df = pd.read_csv(cf_, parse_dates=["timestamp"]).set_index("timestamp").rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
    return ticker, df[["Open", "High", "Low", "Close", "Volume"]].dropna()


v20_feats = json.load(open(V20_META))["features"]
bst_l = xgb.Booster(); bst_l.load_model("models/research/v20_rolling_1h/xgb_long_model.json")
bst_s = xgb.Booster(); bst_s.load_model("models/research/v20_rolling_1h/xgb_short_model.json")
print("Building features from cached OOS candles...")
feat, close_1h = {}, {}
with cf.ThreadPoolExecutor(max_workers=8) as ex:
    for tk, df in ex.map(load, TICKERS):
        if df is None or len(df) < 40: continue
        try:
            h1 = build_rolling_1h_ohlcv(df)
            if len(h1) < 20: continue
            feat[tk] = compute_features(h1[["Open", "High", "Low", "Close", "Volume"]].copy(), legacy=False)
            close_1h[tk] = h1["Close"]
        except Exception: pass
anchors = sorted({ts for f in feat.values() for ts in f.index
                  if ts.date() >= OOS_FROM and ENTRY_FROM <= ts.strftime("%H:%M") <= ENTRY_TO})
FEATS = [c for c in FEATS if all(c in f.columns for f in list(feat.values())[:1])]
print(f"  {len(feat)} tickers, {len(anchors)} anchors, feats={FEATS}\n")

top_rows, pool_rows = [], []
for ts in anchors:
    fr, tks = [], []
    for tk, f in feat.items():
        if ts in f.index and (ts + HOLD) in close_1h[tk].index:
            fr.append(f.loc[ts]); tks.append(tk)
    if len(tks) < 10: continue
    X = pd.DataFrame(fr, index=tks)
    X["Market_Mean_Return"] = X["Return"].mean(); X["Relative_Return"] = X["Return"] - X["Market_Mean_Return"]
    X["Market_Mean_Volatility"] = X["HL_Range"].mean(); X["Relative_Volatility"] = X["HL_Range"] / (X["Market_Mean_Volatility"] + 1e-8)
    dm = xgb.DMatrix(np.nan_to_num(X[v20_feats].values.astype(np.float32)), feature_names=v20_feats)
    ls = bst_l.predict(dm); ss = bst_s.predict(dm)
    lconv = (ls - ls.mean()) - (ss - ss.mean())
    t1 = int(np.argmax(ls))
    for i, tk in enumerate(tks):
        ep = float(close_1h[tk].loc[ts]); xp = float(close_1h[tk].loc[ts + HOLD])
        if ep <= 0: continue
        lg = (xp - ep) / ep * 1e4
        sym = tk.replace(".NS", "")
        row = {"anchor": ts.strftime("%H:%M"), "long_score": float(ls[i]), "Lconv": float(lconv[i]),
               "Market_Mean_Return": float(X["Market_Mean_Return"].iloc[0]),
               "mcap_rank": MCAP.get(sym), "fwd": lg, "is_top1": (i == t1)}
        for c in FEATS:
            row[c] = float(X[c].iloc[i]) if c in X.columns else np.nan
        pool_rows.append(row)
        if i == t1:
            top_rows.append(row)

top = pd.DataFrame(top_rows); pool = pd.DataFrame(pool_rows)
top["net6"] = top.fwd - COST6
print(f"  {len(top)} top-1 long trades | baseline net@6 = {top.net6.mean():+.2f} bps  "
      f"(WR {100*(top.fwd>0).mean():.0f}%)\n")

CAND = FEATS + ["Lconv", "long_score", "Market_Mean_Return", "mcap_rank"]


def net(x): return x.net6.mean() if len(x) else float("nan")
def wr(x):  return 100 * (x.fwd > 0).mean() if len(x) else float("nan")

print("=" * 92); print("  [1] TOP-1 LONG by anchor time"); print("=" * 92)
for a in sorted(top.anchor.unique()):
    x = top[top.anchor == a]
    print(f"   {a}   n={len(x):3d}  net@6={net(x):+7.2f}  WR={wr(x):3.0f}%")

print("\n" + "=" * 92)
print("  [2] TOP-1 LONG net@6 by feature TERCILE (bottom / mid / top)  — where longs die")
print("=" * 92)
print(f"   {'feature':20s} {'net@6 LOW':>10s} {'net@6 MID':>10s} {'net@6 HIGH':>11s}   spread(H-L)")
disc = []
for c in CAND:
    s = top.dropna(subset=[c])
    if s[c].nunique() < 3 or len(s) < 30: continue
    try:
        q = pd.qcut(s[c], 3, labels=["L", "M", "H"], duplicates="drop")
    except Exception:
        continue
    if q.nunique() < 3: continue
    nl, nm, nh = net(s[q == "L"]), net(s[q == "M"]), net(s[q == "H"])
    disc.append((c, nl, nm, nh, nh - nl))
    print(f"   {c:20s} {nl:+10.2f} {nm:+10.2f} {nh:+11.2f}   {nh-nl:+8.2f}")

print("\n" + "=" * 92)
print("  [3] FULL long POOL: Spearman(feature, 1h fwd return)  (large n, robust direction)")
print("=" * 92)
for c in CAND:
    s = pool.dropna(subset=[c, "fwd"])
    if len(s) < 500 or s[c].nunique() < 5: continue
    rho, _ = spearmanr(s[c], s.fwd)
    teff = rho * np.sqrt(len(s) / 4)   # overlap-deflated ~ /sqrt(4)
    print(f"   {c:20s} rho={rho:+.4f}  n={len(s):5d}  t/2~{teff/1:+.1f}")

top.to_csv(f"data/backtests/long_filter_top1_{TODAY}.csv", index=False)
pool.to_csv(f"data/backtests/long_filter_pool_{TODAY}.csv", index=False)
print(f"\n[SAVED] data/backtests/long_filter_top1_{TODAY}.csv  ({len(top)} rows)")
print("Exploratory, no Gauntlet. net@6 = 1h fwd gross - 6bps.")

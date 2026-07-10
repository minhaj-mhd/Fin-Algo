"""
OOS conviction CAPS + FLOORS, both sides  (v20_rolling_1h)
==========================================================
Extends short_conv_gt04_oos.py to BOTH sides and fine-grained buckets, to locate:
  * an UPPER cap (where the side's edge inverts — short's is >0.04)
  * a LOWER floor (does very-low conviction, ~0.012 and below, also fail?)

For LONG trades bucket by Long_Conviction; for SHORT trades bucket by
Short_Conviction (= -Long_Conviction per ticker). Top-1 = argmax of that side's
raw score (the live pick). 1h-forward return per side. OOS = 2026-06-16..07-09
(v20 trained 06-15). Overlap: rolling 1h @15-min => ~4x; use t/2.

Run: python -m scripts.backtests.conviction_caps_oos
"""
import os, sys, json, datetime as dt, warnings
import concurrent.futures as cf
sys.path.insert(0, os.getcwd()); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, xgboost as xgb
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


def load(ticker):
    sym = ticker.replace(".NS", "")
    cache_f = os.path.join(CACHE, f"{sym}_{TODAY}.csv")
    if not os.path.exists(cache_f) or sym not in INSTR: return ticker, None
    df = pd.read_csv(cache_f, parse_dates=["timestamp"]).set_index("timestamp").rename(
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
print(f"  {len(feat)} tickers, {len(anchors)} OOS anchors\n")

rows = []
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
    ls, ss = bst_l.predict(dm), bst_s.predict(dm)
    long_conv = (ls - ls.mean()) - (ss - ss.mean())
    t1l, t1s = int(np.argmax(ls)), int(np.argmax(ss))
    for i, tk in enumerate(tks):
        ep = float(close_1h[tk].loc[ts]); xp = float(close_1h[tk].loc[ts + HOLD])
        if ep <= 0: continue
        lg = (xp - ep) / ep * 1e4
        rows.append((ts.date().isoformat(), ts.strftime("%H:%M"),
                     float(long_conv[i]), (i == t1l), lg,          # LONG
                     float(-long_conv[i]), (i == t1s), -lg))       # SHORT
cols = ["date", "anchor", "Lconv", "is_t1L", "Lg", "Sconv", "is_t1S", "Sg"]
sc = pd.DataFrame(rows, columns=cols)
print(f"  {len(sc)} ticker-rows  ({sc.is_t1L.sum()} top-1 longs, {sc.is_t1S.sum()} top-1 shorts)\n")

BUCKETS = [(-9, 0.0, "conv < 0"), (0.0, 0.006, "[0,     0.006)"), (0.006, 0.009, "[0.006, 0.009)"),
           (0.009, 0.011, "[0.009, 0.011)"), (0.011, 0.013, "[0.011, 0.013)"), (0.013, 0.016, "[0.013, 0.016)"),
           (0.016, 0.020, "[0.016, 0.020)"), (0.020, 0.025, "[0.020, 0.025)"), (0.025, 0.030, "[0.025, 0.030)"),
           (0.030, 0.040, "[0.030, 0.040)"), (0.040, 0.060, "[0.040, 0.060)"), (0.060, 9, ">= 0.060")]


def rep(x, g, label):
    if len(x) == 0:
        print(f"    {label:16s} n=    0"); return
    gg = x[g]; t = gg.mean() / (gg.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 and gg.std() > 0 else float("nan")
    print(f"    {label:16s} n={len(x):5d}  WR={100*(gg>0).mean():3.0f}%  gross={gg.mean():+7.2f}  "
          f"net@6={gg.mean()-COST6:+7.2f} bps  t={t:+4.1f} (t/2~{t/2:+.1f})")


def table(df, convcol, gcol, title):
    print(title)
    for lo, hi, lbl in BUCKETS:
        rep(df[(df[convcol] >= lo) & (df[convcol] < hi)], gcol, lbl)
    print()


print("=" * 96)
print("  LONG side — bucket by Long_Conviction")
print("=" * 96)
table(sc[sc.is_t1L], "Lconv", "Lg", "  [A] TOP-1 LONG stream (live picks):")
table(sc,            "Lconv", "Lg", "  [B] FULL long candidate pool:")

print("=" * 96)
print("  SHORT side — bucket by Short_Conviction")
print("=" * 96)
table(sc[sc.is_t1S], "Sconv", "Sg", "  [C] TOP-1 SHORT stream (live picks):")
table(sc,            "Sconv", "Sg", "  [D] FULL short candidate pool:")

sc.to_csv(f"data/backtests/conviction_caps_oos_{TODAY}.csv", index=False)
print(f"[SAVED] data/backtests/conviction_caps_oos_{TODAY}.csv")
print("Exploratory, no Gauntlet. net@6 = gross - 6bps round-trip. Use t/2 for overlap.")

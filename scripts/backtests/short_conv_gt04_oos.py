"""
OOS test: do SHORT picks with conviction > 0.04 fail?  (v20_rolling_1h)
======================================================================
Hypothesis (from the live/reconstruction diagnostic): top-1 SHORT picks whose
conviction exceeds 0.04 are net-losing — over-extended shorts mean-revert.

Tested OOS (v20 trained_at 2026-06-15  ->  window = 2026-06-16 .. 2026-07-09,
17 sessions the model never saw). At every 15-min anchor [10:15..14:15] the whole
universe is scored with live parity; for EVERY ticker we record its
Short_Conviction and its realized 1h-forward SHORT return, and whether it is the
top-1 short (argmax short_score = the live pick).

Two arms, per the user:
  ARM A  conv>0.04 AND it IS the top-1 short pick   (what we'd actually trade)
  ARM B  conv>0.04 AND it is NOT the top-1 short    (population of high-conv shorts)
plus conviction-bucket tables for the top-1 stream and the full candidate pool.

Overlap caveat: rolling 1h @15-min step => ~4x overlap; naive t inflated ~2x.

Run: python -m scripts.backtests.short_conv_gt04_oos
"""
import os, sys, json, urllib.parse, datetime as dt, warnings
import concurrent.futures as cf
sys.path.insert(0, os.getcwd()); warnings.filterwarnings("ignore")
import requests, numpy as np, pandas as pd, xgboost as xgb
from scripts.tickers import TICKERS
from scripts.feature_utils import build_rolling_1h_ohlcv, compute_features

TODAY = dt.date.today()
OOS_FROM = dt.date(2026, 6, 16)      # first session AFTER v20 training cutoff (06-15)
ENTRY_FROM, ENTRY_TO = "10:15", "14:15"
HOLD = pd.Timedelta(hours=1)
CONV_HI = 0.04
COST6, COST10 = 6, 10
V20_META = "models/research/v20_rolling_1h/metadata.json"
CACHE = os.path.join(os.environ.get("TEMP", "/tmp"), "claude",
                     "c--Users-loq-Desktop-Trading-finalgo", "today_top1_15m_cache")
HDR = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Accept": "application/json"}
INSTR = json.load(open("scripts/instrument_cache.json"))


def fetch_15m(ticker):
    sym = ticker.replace(".NS", ""); ik = INSTR.get(sym)
    if not ik: return ticker, None
    cache_f = os.path.join(CACHE, f"{sym}_{TODAY}.csv")
    if not os.path.exists(cache_f): return ticker, None
    df = pd.read_csv(cache_f, parse_dates=["timestamp"]).set_index("timestamp")
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
    return ticker, df[["Open", "High", "Low", "Close", "Volume"]].dropna()


v20_feats = json.load(open(V20_META))["features"]
bst_l = xgb.Booster(); bst_l.load_model("models/research/v20_rolling_1h/xgb_long_model.json")
bst_s = xgb.Booster(); bst_s.load_model("models/research/v20_rolling_1h/xgb_short_model.json")
print("Building features from cached OOS candles...")
feat, close_1h = {}, {}
with cf.ThreadPoolExecutor(max_workers=8) as ex:
    for tk, df in ex.map(fetch_15m, TICKERS):
        if df is None or len(df) < 40: continue
        try:
            h1 = build_rolling_1h_ohlcv(df)
            if len(h1) < 20: continue
            feat[tk] = compute_features(h1[["Open", "High", "Low", "Close", "Volume"]].copy(), legacy=False)
            close_1h[tk] = h1["Close"]
        except Exception: pass
print(f"  {len(feat)} tickers")

# OOS anchors
anchors = sorted({ts for f in feat.values() for ts in f.index
                  if ts.date() >= OOS_FROM and ENTRY_FROM <= ts.strftime("%H:%M") <= ENTRY_TO})
print(f"  {len(anchors)} OOS anchors ({OOS_FROM} .. {anchors[-1].date()})\n")

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
    short_conv = (ss - ss.mean()) - (ls - ls.mean())
    top1 = int(np.argmax(ss))
    for i, tk in enumerate(tks):
        ep = float(close_1h[tk].loc[ts]); xp = float(close_1h[tk].loc[ts + HOLD])
        if ep <= 0: continue
        rows.append((ts.date().isoformat(), ts.strftime("%H:%M"), tk.replace(".NS", ""),
                     float(short_conv[i]), (i == top1), (ep - xp) / ep * 1e4))
sc = pd.DataFrame(rows, columns=["date", "anchor", "ticker", "conv", "is_top1", "gross_bps"])
sc["net6"] = sc.gross_bps - COST6; sc["net10"] = sc.gross_bps - COST10
print(f"  {len(sc)} short candidate-rows;  {sc.is_top1.sum()} top-1 short picks\n")


def rep(x, label):
    if len(x) == 0:
        print(f"  {label:34s} n=  0"); return
    g = x.gross_bps; t = g.mean() / (g.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 and g.std() > 0 else float("nan")
    print(f"  {label:34s} n={len(x):4d}  WR={100*(g>0).mean():3.0f}%  gross={g.mean():+7.2f}  "
          f"net@6={x.net6.mean():+7.2f}  net@10={x.net10.mean():+7.2f} bps  t={t:+4.1f} (t/2~{t/2:+.1f})")

SEP = "=" * 100
print(SEP); print("  OOS SHORT — conviction > 0.04 hypothesis  (v20, 2026-06-16..07-09)"); print(SEP)

print("\n[1] TOP-1 SHORT PICKS (the live-traded stream), bucketed by conviction")
t1 = sc[sc.is_top1]
rep(t1[t1.conv < 0.011],            "top1  conv < 0.011")
rep(t1[(t1.conv >= 0.011) & (t1.conv <= 0.04)], "top1  conv in [0.011, 0.04]")
rep(t1[t1.conv > 0.04],             "top1  conv > 0.04   (ARM A)")
rep(t1,                             "top1  ALL")

print("\n[2] conv > 0.04 shorts — ARM A (top-1) vs ARM B (NOT top-1)")
hi = sc[sc.conv > 0.04]
rep(hi[hi.is_top1],  "conv>0.04 & IS top-1   (ARM A)")
rep(hi[~hi.is_top1], "conv>0.04 & NOT top-1  (ARM B)")
rep(hi,              "conv>0.04 ALL candidates")

print("\n[3] FULL candidate pool by conviction bucket (cross-sectional monotonicity)")
for lo, hival, lbl in [(-9, 0.0, "conv < 0"), (0.0, 0.011, "[0, 0.011)"), (0.011, 0.02, "[0.011, 0.02)"),
                       (0.02, 0.03, "[0.02, 0.03)"), (0.03, 0.04, "[0.03, 0.04)"),
                       (0.04, 0.06, "[0.04, 0.06)"), (0.06, 9, ">= 0.06")]:
    rep(sc[(sc.conv >= lo) & (sc.conv < hival)], lbl)

print("\n[4] Per-day net@6 of top-1 conv>0.04 shorts (ARM A)")
armA = t1[t1.conv > 0.04]
for d in sorted(sc.date.unique()):
    x = armA[armA.date == d]
    if len(x): print(f"  {d}  n={len(x):2d}  net@6={x.net6.mean():+7.2f}  WR={100*(x.gross_bps>0).mean():3.0f}%")

out = f"data/backtests/short_conv_gt04_oos_{TODAY}.csv"
sc.to_csv(out, index=False)
print(f"\n[SAVED] {out}  ({len(sc)} rows)")
print("Overlap caveat: rolling 1h @15-min => ~4x overlap; use t/2 as the honest guide. Exploratory, no Gauntlet.")

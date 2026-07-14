"""
true_oos_dyn_floor.py -- TRUE OOS (3rd tier) for the Dynamic Probability Floor.

The dev/holdout framework has three tiers:
  DEV        : develop gates on the frozen feed.parquet          (2025-08-01 -> 2026-06-11)
  PROXY OOS  : sealed dev_holdout HOLDOUT, feed-based, run.py     (2026-06-11 -> 2026-07-10)
  TRUE OOS   : INDEPENDENT fresh-from-Upstox rebuild (this file). Re-loads the raw Nifty cache
               with the timezone-bug fix (oos_jul10_backtest.py style), re-fetches S&P live,
               re-scores the panel, and applies ONLY the isolated short Dynamic Probability
               Floor -- so it replicates the proxy result from a separate data path.

Isolated short-only gate (identical to configs/dyn_prob_floor_short.json):
  base floor = 99.92pct of ss (0.0788); +0.028 tighten when SP500prev>+0.5% AND Nifty2h>=-0.10%;
  no lunch veto, no long engine; 1-slot lock, shorts-priority, 10:15-14:15.

Genuine-unseen window is 2026-06-11 -> 2026-07-10 (Jun4-Jun10 overlaps DEV; reported separately
and labelled ~in-sample). Primary metric = net bps/trade + t-stat.
"""
import json
from datetime import time, date

import numpy as np
import pandas as pd
import xgboost as xgb

COST_BPS = 6.0
SHORT_THRESH = 0.0788        # 99.92pct of ss (DEV-estimated), == user's "base floor"
SHORT_DYN = 0.0788 + 0.028   # +0.028 penalty -> 0.1068
SP500_GT = 0.005
NIFTY2H_GTE = -0.0010
GENUINE_START = date(2026, 6, 11)   # everything >= this is truly unseen (Jun4-10 is DEV)


def load_nifty_2h():
    """Fresh raw Nifty cache with the double-convention tz fix (oos_jul10_backtest style)."""
    nifty = pd.read_csv("data/raw_index_cache/nifty50_15m.csv")
    ist = []
    for t_str in nifty["ts"]:
        t = pd.to_datetime(t_str)
        if t.date() < date(2026, 6, 1):
            ist.append(t.tz_localize(None))               # old rows: already IST, mislabelled +0000
        else:
            ist.append(t.tz_convert("Asia/Kolkata").tz_localize(None))  # new rows: true UTC -> IST
    nifty["ts"] = ist
    nifty = nifty.sort_values("ts").reset_index(drop=True)
    nifty["nifty_ret_2h"] = nifty["close"] / nifty["close"].shift(8) - 1
    return dict(zip(nifty["ts"], nifty["nifty_ret_2h"]))


def load_sp500_prev():
    import yfinance as yf
    sp = yf.download("^GSPC", start="2025-07-01", end="2026-07-30", progress=False)
    if isinstance(sp.columns, pd.MultiIndex):
        sp.columns = sp.columns.get_level_values(0)
    sp = sp.reset_index()
    sp["Date"] = pd.to_datetime(sp["Date"]).dt.date
    sp["ret"] = sp["Close"].pct_change()
    d = {r["Date"]: r["ret"] for _, r in sp.iterrows()}
    days = sorted(d.keys())

    def prev(curr):
        p = [x for x in days if x < curr]
        return d[max(p)] if p else 0.0
    return prev


def _tstat(x):
    x = np.asarray(x, float)
    if len(x) < 2 or x.std(ddof=1) == 0:
        return 0.0
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))


def main():
    nifty_map = load_nifty_2h()
    prev_sp = load_sp500_prev()

    df = pd.read_parquet("data/research/v20_rolling_1h/panel.parquet")
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df = df[df["DateTime"].dt.date >= date(2026, 6, 4)]
    tmask = (df["DateTime"].dt.time >= time(10, 15)) & (df["DateTime"].dt.time <= time(14, 15))
    df = df[tmask].copy()
    df["nifty_ret_2h"] = df["DateTime"].map(nifty_map)
    df = df.dropna(subset=["nifty_ret_2h"])

    feats = json.load(open("models/research/v20_rolling_1h/metadata.json"))["features"]
    df = df.dropna(subset=feats + ["Next_Hour_Return"])
    bs = xgb.Booster(); bs.load_model("models/research/v20_rolling_1h/xgb_short_model.json")
    bl = xgb.Booster(); bl.load_model("models/research/v20_rolling_1h/xgb_long_model.json")
    X = xgb.DMatrix(np.nan_to_num(df[feats].values.astype(np.float32)), feature_names=feats)
    df["ss"] = bs.predict(X)
    df["ls"] = bl.predict(X)
    ss_mean = df.groupby("DateTime")["ss"].transform("mean")
    ls_mean = df.groupby("DateTime")["ls"].transform("mean")
    df["short_conviction"] = (df["ss"] - ss_mean) - (df["ls"] - ls_mean)

    groups = {ts: g for ts, g in df.groupby("DateTime")}
    trades = []
    locked_until = None
    for ts in sorted(groups.keys()):
        if locked_until is not None and ts < locked_until:
            continue
        g = groups[ts]
        sp = prev_sp(ts.date())
        n2h = g["nifty_ret_2h"].iloc[0]
        thr = SHORT_DYN if (sp > SP500_GT and n2h >= NIFTY2H_GTE) else SHORT_THRESH
        sc = g[g["ss"] > thr]
        if len(sc) == 0:
            continue
        p = sc.sort_values("short_conviction", ascending=False).iloc[0]
        trades.append((ts, "SHORT", p["Ticker"], -p["Next_Hour_Return"] * 10000.0))
        locked_until = ts + pd.Timedelta(hours=1)

    td = pd.DataFrame(trades, columns=["ts", "side", "tk", "gross_bps"])
    td["net_bps"] = td["gross_bps"] - COST_BPS
    td["date"] = td["ts"].dt.date
    td["month"] = td["ts"].dt.to_period("M")

    def rpt(sub, label):
        if len(sub) == 0:
            print(f"  {label:32s} no trades"); return
        print(f"  {label:32s} n={len(sub):3d}  win {(sub.net_bps>0).mean():.0%}  "
              f"NET {sub.net_bps.mean():+6.2f} bps (t={_tstat(sub.net_bps):+.2f})  "
              f"sum {sub.net_bps.sum():+.0f}")

    print("=" * 74)
    print(" TRUE OOS (fresh Upstox rebuild) -- isolated short Dynamic Probability Floor")
    print(f"   base={SHORT_THRESH} dyn=+0.028->{SHORT_DYN:.4f}  SP500>{SP500_GT} & Nifty2h>={NIFTY2H_GTE}")
    print("=" * 74)
    rpt(td[td.date >= GENUINE_START], f"GENUINE OOS (>= {GENUINE_START})")
    rpt(td[td.date < GENUINE_START], "Jun4-10 (overlaps DEV, ~IS)")
    rpt(td, "FULL Jun4-Jul10 (project window)")
    print("\n  monthly:")
    for mo in sorted(td["month"].unique()):
        rpt(td[td["month"] == mo], f"  {mo}")


if __name__ == "__main__":
    main()

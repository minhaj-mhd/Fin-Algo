"""
build_feed.py -- Freeze the v20 model signal feed for the dev/holdout gate framework.

WHY THIS EXISTS
---------------
The v20 model is a FIXED, already-held-out input. The trainer
(scripts/training/train_ranking_clean.py) does a strict 80/20 temporal split:
  train  = 2022-01 .. 2025-06   (first ~80% of months)
  val    = 2025-07
  UNTOUCHED test = 2025-08 .. onward   <-- the deployed model never saw this
So everything from 2025-08 on is genuinely out-of-sample *for the model*.
Empirically its rank-IC is stable there (rho ~0.02 short & long) and does NOT
degrade on the Jun-Jul 2026 window -- the model generalises. What overfit in the
"+26 bps / 13x" backtest was the GATE LAYER (thresholds tuned to that window),
not the model.

This script scores the panel ONCE with the deployed models and caches a compact
per-(timestamp, ticker) feed so gate/strategy research iterates in seconds without
reloading the 1.1 GB panel or re-hitting the network for S&P.

OUTPUT : data/research/dev_holdout/feed.parquet
RUN    : python scripts/research/dev_holdout/build_feed.py [--rebuild]
"""
import os
import argparse
from datetime import time, date

import numpy as np
import pandas as pd
import xgboost as xgb
import json

PANEL       = "data/research/v20_rolling_1h/panel.parquet"
MODEL_DIR   = "models/research/v20_rolling_1h"
NIFTY_CSV   = "data/raw_index_cache/nifty50_15m.csv"
OUT_DIR     = "data/research/dev_holdout"
FEED_PATH   = os.path.join(OUT_DIR, "feed.parquet")
SP500_CACHE = os.path.join(OUT_DIR, "sp500_prev.csv")

# Start of the model's untouched 20% test block. dev + holdout both live inside this.
OOS_START   = date(2025, 8, 1)
TIME_START  = time(10, 15)
TIME_END    = time(14, 15)


def load_sp500_prev(dates, rebuild=False):
    """Return {trade_date: prev-session S&P500 daily return}. Cached to CSV so
    iteration is offline + deterministic."""
    if os.path.exists(SP500_CACHE) and not rebuild:
        s = pd.read_csv(SP500_CACHE, parse_dates=["Date"])
        ret = {r.Date.date(): r.sp500_ret for r in s.itertuples()}
    else:
        import yfinance as yf
        sp = yf.download("^GSPC", start="2025-06-01", end="2026-07-30", progress=False)
        if isinstance(sp.columns, pd.MultiIndex):
            sp.columns = sp.columns.get_level_values(0)
        sp = sp.reset_index()
        sp["Date"] = pd.to_datetime(sp["Date"])
        sp["sp500_ret"] = sp["Close"].pct_change()
        sp[["Date", "sp500_ret"]].to_csv(SP500_CACHE, index=False)
        ret = {r.Date.date(): r.sp500_ret for r in sp.itertuples()}

    sorted_days = sorted(ret.keys())
    out = {}
    for d in dates:
        prev = [x for x in sorted_days if x < d]
        out[d] = ret[max(prev)] if prev else 0.0
    return out


def main(rebuild=False):
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- 1. Nifty trailing-2h return, aligned to signal timestamps ----
    # NOTE: the nifty cache is DOUBLE-CONVENTION (a parallel-rebuild artifact). Every row
    # is labelled +0000, but rows up to ~2026-06-09 are IST wall-clock (09:15-15:30)
    # mislabelled UTC, while rows from ~2026-06-10 are true UTC (03:45-10:00). Normalise
    # both to naive IST so they align with the panel's naive-IST DateTime, then keep the
    # NSE session and de-duplicate to a clean 15-min grid. (Cache should be re-collected;
    # see README.) Jun 8-9 hold both formats -> keep='last' picks the corrected grid.
    nifty = pd.read_csv(NIFTY_CSV)
    nifty["ts"] = pd.to_datetime(nifty["ts"], utc=True).dt.tz_localize(None)
    utc_rows = nifty["ts"].dt.date >= date(2026, 6, 10)
    nifty.loc[utc_rows, "ts"] = nifty.loc[utc_rows, "ts"] + pd.Timedelta(hours=5, minutes=30)
    nifty = nifty[(nifty["ts"].dt.time >= time(9, 15)) & (nifty["ts"].dt.time <= time(15, 30))]
    nifty = nifty.drop_duplicates("ts", keep="last").sort_values("ts").reset_index(drop=True)
    nifty["nifty_ret_2h"] = nifty["close"] / nifty["close"].shift(8) - 1
    nifty_map = dict(zip(nifty["ts"], nifty["nifty_ret_2h"]))

    # ---- 2. Panel -> OOS block, intraday window ----
    feats = json.load(open(os.path.join(MODEL_DIR, "metadata.json")))["features"]
    keep = list(dict.fromkeys(feats + [
        "DateTime", "Ticker", "Next_Hour_Return", "VWAP_Dist", "Market_Mean_Return"]))
    df = pd.read_parquet(PANEL, columns=keep)
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df = df[df["DateTime"].dt.date >= OOS_START]
    tmask = (df["DateTime"].dt.time >= TIME_START) & (df["DateTime"].dt.time <= TIME_END)
    df = df[tmask].copy()

    df["nifty_ret_2h"] = df["DateTime"].map(nifty_map)
    df = df.dropna(subset=["nifty_ret_2h"])
    df = df.dropna(subset=feats + ["Next_Hour_Return"])

    # ---- 3. Score with the FROZEN deployed models ----
    bs = xgb.Booster(); bs.load_model(os.path.join(MODEL_DIR, "xgb_short_model.json"))
    bl = xgb.Booster(); bl.load_model(os.path.join(MODEL_DIR, "xgb_long_model.json"))
    X = xgb.DMatrix(np.nan_to_num(df[feats].values.astype(np.float32)), feature_names=feats)
    df["ss"] = bs.predict(X)
    df["ls"] = bl.predict(X)

    # cross-sectional (per-timestamp) demeaned conviction -- identical to the audited script
    ss_mean = df.groupby("DateTime")["ss"].transform("mean")
    ls_mean = df.groupby("DateTime")["ls"].transform("mean")
    df["short_conviction"] = (df["ss"] - ss_mean) - (df["ls"] - ls_mean)
    df["long_conviction"]  = (df["ls"] - ls_mean) - (df["ss"] - ss_mean)

    # ---- 4. S&P500 prev-session return (macro gate input) ----
    dates = sorted(set(df["DateTime"].dt.date))
    sp_map = load_sp500_prev(dates, rebuild=rebuild)
    df["sp500_prev_ret"] = df["DateTime"].dt.date.map(sp_map)

    out = df[[
        "DateTime", "Ticker", "ss", "ls", "short_conviction", "long_conviction",
        "Next_Hour_Return", "VWAP_Dist", "Market_Mean_Return",
        "nifty_ret_2h", "sp500_prev_ret",
    ]].reset_index(drop=True)
    out.to_parquet(FEED_PATH, index=False)

    d = out["DateTime"].dt.date
    print(f"[feed] rows={len(out):,}  timestamps={out['DateTime'].nunique():,}")
    print(f"[feed] date range {d.min()} -> {d.max()}")
    print(f"[feed] saved -> {FEED_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="re-download S&P and rescore")
    a = ap.parse_args()
    main(rebuild=a.rebuild)

"""
three_tier.py -- Correct 3-tier split for the gate framework.

The 11-MONTH window (2025-08-01 .. 2026-06-30) is the model's held-out test block.
BOTH dev and proxy-oos live INSIDE it. TRUE OOS is the fresh-pull month BEYOND it (July).

  DEV        2025-08-01 .. 2026-05-31   (inside 11mo)  develop
  PROXY OOS  2026-06-01 .. 2026-06-30   (inside 11mo)  sealed confirm
  TRUE OOS   2026-07-01 .. 2026-07-10   (beyond 11mo)  fresh broker pull

Runs any dev_holdout config; primary metric = net bps/trade + t-stat.
"""
import os
import sys
import json
import argparse
from datetime import date

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from strategy import run_strategy, metrics, DEFAULT_CONFIG  # noqa: E402

FEED = "data/research/dev_holdout/feed.parquet"
HERE = os.path.dirname(__file__)

WINDOWS = [
    ("DEV        (inside 11mo)", date(2025, 8, 1),  date(2026, 5, 31)),
    ("PROXY OOS  (inside 11mo)", date(2026, 6, 1),  date(2026, 6, 30)),
    ("TRUE OOS   (beyond 11mo)", date(2026, 7, 1),  date(2026, 7, 10)),
]


def load_cfg(path):
    if not os.path.isabs(path) and not os.path.exists(path):
        path = os.path.join(HERE, path)
    with open(path) as f:
        return {**DEFAULT_CONFIG, **json.load(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dyn_prob_floor_short.json")
    a = ap.parse_args()
    cfg = load_cfg(a.config)

    feed = pd.read_parquet(FEED)
    feed["DateTime"] = pd.to_datetime(feed["DateTime"])
    d = feed["DateTime"].dt.date

    print("=" * 78)
    print(f" 3-TIER GATE FRAMEWORK   config={cfg['name']}")
    print(" 11-month frame = Aug 2025 -> Jun 2026 (model test block); July = fresh true OOS")
    print("=" * 78)
    for label, lo, hi in WINDOWS:
        fw = feed[(d >= lo) & (d <= hi)]
        td = run_strategy(fw, cfg)
        m = metrics(td, cfg)
        if m["n"] == 0:
            print(f"  {label}  {lo}..{hi}   no trades"); continue
        print(f"  {label}  {lo}..{hi}")
        print(f"      n={m['n']:3d}  win {m['win_rate']:.0%}  "
              f"NET {m['net_bps_mean']:+6.2f} bps  (t={m['net_bps_t']:+.2f})  "
              f"sum {td.net_bps.sum():+.0f}")


if __name__ == "__main__":
    main()

"""
run.py -- Develop gates on DEV, confirm on a pre-registered single-use HOLDOUT.

SPLIT (both windows are model-OOS; the split is for STRATEGY development):
  DEV     : 2025-08-01 .. HOLDOUT_START   <- iterate gates here as much as you like
  HOLDOUT : HOLDOUT_START .. end           <- truly unseen by the gate tuning

DISCIPLINE
  * DEV runs are free. Iterate, sweep, A/B configs on DEV.
  * A HOLDOUT run requires --confirm AND --hypothesis "<one-line pre-registration>".
    Every holdout look is appended to HOLDOUT_LEDGER.md. It is SINGLE-USE per
    hypothesis: if you tune after peeking, the holdout is burned (you have just
    started overfitting one level up -- exactly what killed the +26 bps gates).

USAGE
  # develop
  python scripts/research/dev_holdout/run.py --config configs/baseline.json
  python scripts/research/dev_holdout/run.py --config configs/baseline.json --neg-control 25
  # confirm (once)
  python scripts/research/dev_holdout/run.py --config configs/baseline.json \
        --confirm --hypothesis "baseline 213-gate config, expect OOS collapse"
"""
import os
import sys
import json
import argparse
from datetime import date, datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from strategy import run_strategy, metrics, monthly_table, DEFAULT_CONFIG  # noqa: E402

HERE = os.path.dirname(__file__)
FEED_PATH = "data/research/dev_holdout/feed.parquet"
LEDGER = os.path.join(HERE, "HOLDOUT_LEDGER.md")

DEV_START = date(2025, 8, 1)
# The single most important pre-registration decision. Everything >= this date is
# the holdout and must not inform gate design. ~1 month / ~22 trading days by default.
HOLDOUT_START = date(2026, 6, 11)


def load_config(path):
    if path in (None, "default"):
        return dict(DEFAULT_CONFIG)
    if not os.path.isabs(path) and not os.path.exists(path):
        path = os.path.join(HERE, path)
    with open(path) as f:
        return {**DEFAULT_CONFIG, **json.load(f)}


def slice_window(feed, which, holdout_start):
    d = feed["DateTime"].dt.date
    if which == "dev":
        return feed[(d >= DEV_START) & (d < holdout_start)]
    if which == "holdout":
        return feed[d >= holdout_start]
    return feed[d >= DEV_START]


def fmt(m):
    if m["n"] == 0:
        return "  no trades"
    c = m["compound"]
    return (
        f"  trades {m['n']:3d} (S {m['n_short']}/L {m['n_long']})  win {m['win_rate']:.1%}\n"
        f"  NET bps/trade {m['net_bps_mean']:+6.2f}  (t={m['net_bps_t']:+.2f})   <-- PRIMARY, leverage-independent\n"
        f"    short {m['short_net_bps']:+6.2f} bps (t={m['short_t']:+.2f}, win {m['short_win']:.0%})"
        f"   long {m['long_net_bps']:+6.2f} bps (t={m['long_t']:+.2f}, win {m['long_win']:.0%})\n"
        f"  [leverage-dependent] 5x compound: {c['return_x']:.2f}x  profit Rs.{c['total_profit']:+,.0f}"
        f"  maxDD Rs.{c['max_dd_rs']:+,.0f}  R/MDD {c['return_mdd']:.2f}"
    )


def neg_control(feed_win, cfg, k):
    """Random-pick within the same gated pools. Isolates whether conviction RANKING
    adds value on top of the gates + model membership."""
    vals = []
    for seed in range(k):
        c = {**cfg, "selection": "random_pool", "random_seed": seed}
        td = run_strategy(feed_win, c)
        if len(td):
            vals.append(td.net_bps.mean())
    if not vals:
        return None
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="default")
    ap.add_argument("--window", choices=["dev", "holdout", "all"], default="dev")
    ap.add_argument("--confirm", action="store_true", help="run on HOLDOUT (single-use)")
    ap.add_argument("--hypothesis", default=None, help="pre-registration line (required for holdout)")
    ap.add_argument("--neg-control", type=int, default=0, metavar="K", help="K random-pool controls")
    ap.add_argument("--holdout-start", default=None, help="override YYYY-MM-DD")
    ap.add_argument("--monthly", action="store_true")
    a = ap.parse_args()

    holdout_start = (datetime.strptime(a.holdout_start, "%Y-%m-%d").date()
                     if a.holdout_start else HOLDOUT_START)
    cfg = load_config(a.config)

    if not os.path.exists(FEED_PATH):
        sys.exit("feed.parquet missing -- run build_feed.py first.")
    feed = pd.read_parquet(FEED_PATH)
    feed["DateTime"] = pd.to_datetime(feed["DateTime"])

    window = "holdout" if a.confirm else a.window
    if window == "holdout" and not a.hypothesis:
        sys.exit("HOLDOUT is single-use: pass --hypothesis \"<pre-registration>\".")

    fw = slice_window(feed, window, holdout_start)
    d = fw["DateTime"].dt.date
    print("=" * 72)
    print(f" config   : {cfg['name']}   ({os.path.basename(a.config)})")
    print(f" window   : {window.upper()}   {d.min()} -> {d.max()}   "
          f"(dev/holdout boundary {holdout_start})")
    print("=" * 72)

    td = run_strategy(fw, cfg)
    m = metrics(td, cfg)
    print(fmt(m))

    if a.neg_control and m.get("n", 0):
        nc = neg_control(fw, cfg, a.neg_control)
        if nc:
            mean, std, k = nc
            edge = m["net_bps_mean"] - mean
            print(f"\n  neg-control (random pick in same gated pool, {k} seeds):"
                  f" {mean:+.2f} +/- {std:.2f} bps")
            print(f"  ranking edge over neg-control: {edge:+.2f} bps"
                  f"  ({'ADDS' if edge > 0 else 'NO'} value from conviction ranking)")

    if a.monthly and m.get("n", 0):
        print("\n" + monthly_table(td).to_string(index=False))

    if window == "holdout":
        line = (f"- {datetime.now().isoformat(timespec='seconds')} | config={cfg['name']} "
                f"| holdout {d.min()}..{d.max()} | n={m['n']} "
                f"net={m.get('net_bps_mean', float('nan')):+.2f}bps t={m.get('net_bps_t', 0):+.2f} "
                f"| HYPOTHESIS: {a.hypothesis}\n")
        with open(LEDGER, "a", encoding="utf-8") as f:
            f.write(line)
        print(f"\n  [ledger] holdout look recorded -> {os.path.relpath(LEDGER)}")
        print("  This hypothesis's holdout is now spent. Do not tune and re-peek.")


if __name__ == "__main__":
    main()

"""
ablation.py -- Screen every gate/constraint on DEV only (the holdout stays sealed).

Three free, high-power screens per constraint:

  1. LEAVE-ONE-OUT : full baseline MINUS this gate. dNet<0 => removing hurts => gate helps
                     in the presence of the others. Isolates marginal value in-context.
  2. ADD-ONE       : structural minimum PLUS just this gate. dNet>0 => helps standalone.
                     A gate that helps in BOTH is robust; only-one => interaction-dependent.
  3. NEG-CONTROL   : random pick inside each config's gated pool. net - negctrl = value the
                     model's RANKING adds; if ~0, the config is pure pool-carving (the part
                     that overfit).

Then STABILITY across consecutive DEV months -- the real OOS proxy. A gate net-positive in
8/10 months is a candidate; one whose edge is all-February is a mirage.

Run: python scripts/research/dev_holdout/ablation.py [--neg-seeds 10] [--holdout-start YYYY-MM-DD]
Output is advisory: it produces a shortlist to pre-register for ONE holdout look. It never
reads the holdout.
"""
import os
import sys
import json
import argparse
from datetime import date, datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from strategy import run_strategy, metrics, DEFAULT_CONFIG  # noqa: E402

HERE = os.path.dirname(__file__)
FEED_PATH = "data/research/dev_holdout/feed.parquet"
DEV_START = date(2025, 8, 1)
HOLDOUT_START = date(2026, 6, 11)

# Each constraint as an ON payload (baseline value) and an OFF payload (disabled).
GATES = {
    "short_dyn(0.110)":      (dict(short_dyn_thresh=0.110),      dict(short_dyn_thresh=None)),
    "lunch_veto":            (dict(lunch_veto=["11:30", "13:00"]), dict(lunch_veto=None)),
    "long_sp500_veto":       (dict(long_sp500_veto_lt=-0.005),   dict(long_sp500_veto_lt=None)),
    "long_nifty2h(>0.0025)": (dict(long_nifty2h_gt=0.0025),      dict(long_nifty2h_gt=None)),
    "long_vwap_gate":        (dict(long_vwap_gate=True),         dict(long_vwap_gate=False)),
    "long_market_gate":      (dict(long_market_gate=True),       dict(long_market_gate=False)),
    "long_conv_cap(0.030)":  (dict(long_conv_cap=0.030),         dict(long_conv_cap=None)),
    "short_conv_cap(0.04)":  (dict(short_conv_cap=0.04),         dict(short_conv_cap=None)),
}
# Gates that are ON in the shipped baseline (so leave-one-out applies).
BASELINE_ON = [g for g in GATES if g != "short_conv_cap(0.04)"]


def load_baseline():
    with open(os.path.join(HERE, "configs", "baseline.json")) as f:
        return {**DEFAULT_CONFIG, **json.load(f)}


def structural(baseline):
    """Baseline with every ablatable gate turned OFF -> the pure engine
    (time window + 1-slot + shorts-priority + short base thresh + longs, no extra gates)."""
    cfg = dict(baseline)
    for _, off in GATES.values():
        cfg.update(off)
    cfg["name"] = "structural_min"
    return cfg


def evaluate(feed, cfg, neg_seeds=0):
    td = run_strategy(feed, cfg)
    m = metrics(td, cfg)
    nc = np.nan
    if neg_seeds and m.get("n", 0):
        vals = []
        for s in range(neg_seeds):
            t2 = run_strategy(feed, {**cfg, "selection": "random_pool", "random_seed": s})
            if len(t2):
                vals.append(t2.net_bps.mean())
        nc = float(np.mean(vals)) if vals else np.nan
    return td, m, nc


def stability_row(td):
    if len(td) == 0:
        return dict(months=0, pos=0, worst=np.nan, mean=np.nan, by_month={})
    g = td.groupby("month")["net_bps"].mean()
    return dict(months=len(g), pos=int((g > 0).sum()), worst=float(g.min()),
                mean=float(g.mean()), by_month={str(k): float(v) for k, v in g.items()})


def verdict(name, loo_d, add_d, stab):
    stable = stab["months"] and stab["pos"] / stab["months"] >= 0.6
    helps_standalone = (not np.isnan(add_d)) and add_d > 0.5
    helps_incontext = (not np.isnan(loo_d)) and loo_d < -0.5
    if name == "lunch_veto":
        return "STRUCTURAL-PRIOR"
    if helps_standalone and stable:
        return "KEEP-CANDIDATE"
    if helps_incontext and not helps_standalone:
        return "CONTEXT/REGIME-ONLY"
    if (not np.isnan(loo_d) and loo_d > 0.2) or (not np.isnan(add_d) and add_d < -0.5):
        return "HURTS"
    return "NO-VALUE"


def line(name, m, nc=np.nan, ref=None):
    if m.get("n", 0) == 0:
        return f"  {name:24s}  no trades"
    d = "" if ref is None else f"  dNet {m['net_bps_mean'] - ref:+6.2f}"
    edge = "" if np.isnan(nc) else f"  rankEdge {m['net_bps_mean'] - nc:+5.2f}"
    return (f"  {name:24s} n{m['n']:>4d}  net {m['net_bps_mean']:+6.2f} (t{m['net_bps_t']:+4.1f})"
            f"  S {m['short_net_bps']:+6.1f} / L {m['long_net_bps']:+6.1f}{d}{edge}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neg-seeds", type=int, default=10)
    ap.add_argument("--holdout-start", default=None)
    a = ap.parse_args()
    hstart = (datetime.strptime(a.holdout_start, "%Y-%m-%d").date()
              if a.holdout_start else HOLDOUT_START)

    feed = pd.read_parquet(FEED_PATH)
    feed["DateTime"] = pd.to_datetime(feed["DateTime"])
    d = feed["DateTime"].dt.date
    dev = feed[(d >= DEV_START) & (d < hstart)]
    print("=" * 84)
    print(f" DEV ABLATION  window {dev['DateTime'].dt.date.min()} -> {dev['DateTime'].dt.date.max()}"
          f"   (holdout >= {hstart} stays SEALED)")
    print("=" * 84)

    base = load_baseline()
    struct = structural(base)
    td_base, m_base, nc_base = evaluate(dev, base, a.neg_seeds)
    td_str, m_str, nc_str = evaluate(dev, struct, a.neg_seeds)

    print("\nREFERENCE")
    print(line("FULL baseline", m_base, nc_base))
    print(line("structural min", m_str, nc_str))

    print("\nLEAVE-ONE-OUT  (baseline minus gate; dNet<0 => gate HELPS in-context)")
    loo_delta = {}
    for g in BASELINE_ON:
        cfg = {**base, **GATES[g][1], "name": f"base-{g}"}
        _, m, _ = evaluate(dev, cfg)
        print(line(f"-{g}", m, ref=m_base["net_bps_mean"]))
        loo_delta[g] = m["net_bps_mean"] - m_base["net_bps_mean"]

    print("\nADD-ONE  (structural plus one gate; dNet>0 => gate HELPS standalone)")
    add = {}
    for g in GATES:
        cfg = {**struct, **GATES[g][0], "name": f"struct+{g}"}
        td, m, nc = evaluate(dev, cfg, a.neg_seeds)
        print(line(f"+{g}", m, nc, ref=m_str["net_bps_mean"]))
        add[g] = (td, m, nc)

    # ---- mode toggle: shorts-priority (baseline) vs longs-priority. Not an additive gate,
    #      so it's tested as a straight A/B; LOO-style delta = flip minus baseline. ----
    print("\nMODE TOGGLE  (baseline is shorts-priority; flip => longs-priority)")
    td_lp, m_lp, _ = evaluate(dev, {**base, "shorts_priority": False, "name": "longs-priority"})
    print(line("flip: longs-priority", m_lp, ref=m_base["net_bps_mean"]))
    loo_delta["shorts_priority"] = m_lp["net_bps_mean"] - m_base["net_bps_mean"]

    # ---- stability across DEV months ----
    print("\nSTABILITY across DEV months  (net bps/trade per month; pos = #months>0)")
    rows = {"FULL baseline": stability_row(td_base), "structural min": stability_row(td_str),
            "flip:longs-priority": stability_row(td_lp)}
    for g in GATES:
        rows[f"+{g}"] = stability_row(add[g][0])
    months = sorted({mm for r in rows.values() for mm in r["by_month"]})
    tbl = pd.DataFrame(
        {name: {mm: r["by_month"].get(mm, np.nan) for mm in months} for name, r in rows.items()}
    ).T[months].round(0)
    tbl["pos/tot"] = [f"{rows[n]['pos']}/{rows[n]['months']}" for n in tbl.index]
    tbl["worst"] = [round(rows[n]["worst"], 1) if not np.isnan(rows[n]["worst"]) else np.nan
                    for n in tbl.index]
    with pd.option_context("display.width", 220, "display.max_columns", 40):
        print(tbl.to_string())

    # ---- Stage 1+2 scorecard: every constraint, one row, one verdict ----
    print("\n" + "=" * 84)
    print(" STAGE 1+2 SCORECARD  (all constraints; DEV only, holdout SEALED)")
    print("=" * 84)
    sc = []
    for g in GATES:
        am = add[g][1]
        stab = rows[f"+{g}"]
        add_d = (am["net_bps_mean"] - m_str["net_bps_mean"]) if am.get("n", 0) else np.nan
        edge = (am["net_bps_mean"] - add[g][2]) if am.get("n", 0) and not np.isnan(add[g][2]) else np.nan
        loo_d = loo_delta.get(g, np.nan)
        sc.append(dict(
            constraint=g,
            loo_dNet=("—" if np.isnan(loo_d) else round(loo_d, 1)),
            add_dNet=("—" if np.isnan(add_d) else round(add_d, 1)),
            stab=f"{stab['pos']}/{stab['months']}",
            worst=("—" if np.isnan(stab["worst"]) else round(stab["worst"], 1)),
            rankEdge=("—" if np.isnan(edge) else round(edge, 1)),
            verdict=verdict(g, loo_d, add_d, stab)))
    lp = rows["flip:longs-priority"]
    ld = loo_delta["shorts_priority"]
    sc.append(dict(constraint="shorts_priority(mode)", loo_dNet=round(ld, 1), add_dNet="—",
                   stab=f"{lp['pos']}/{lp['months']}", worst=round(lp["worst"], 1), rankEdge="—",
                   verdict=("CONTEXT/REGIME-ONLY" if ld < -0.5 else "NO-VALUE")))
    print(pd.DataFrame(sc).to_string(index=False))
    keepers = [r["constraint"] for r in sc if r["verdict"] == "KEEP-CANDIDATE"]
    print(f"\n  KEEP-CANDIDATEs (helps standalone + stable >=60% months): {keepers or '(none)'}")
    print("  Collapse keepers into ONE config, pre-register direction, then run.py --confirm ONCE.")
    print("  (Reminder: DEV-stable != OOS-safe -- baseline was 10/11 DEV months and still failed holdout.)")


if __name__ == "__main__":
    main()

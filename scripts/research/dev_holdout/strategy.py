"""
strategy.py -- Config-driven 1-slot gate engine + honest metrics.

The engine is a faithful generalisation of the audited script
(scripts/backtests/temp_11m_combined.py): same 1-slot lock, shorts-priority,
dynamic short threshold, long macro/nifty/VWAP/market/conviction gates, lunch
veto and 5x geometric compounding -- but every threshold/toggle is a config key
so a "gate hypothesis" is just a dict diff.

PRIMARY metric is per-trade net bps + t-stat (leverage-independent, honest).
The 5x compounding P&L is reported too but clearly labelled LEVERAGE-DEPENDENT:
it is the same net-bps stream times a 5x-reinvested capital base, NOT extra edge.
"""
from datetime import time as dtime

import numpy as np
import pandas as pd


DEFAULT_CONFIG = {
    "name": "baseline_213",
    "cost_bps": 6.0,
    # temporal
    "time_start": "10:15",
    "time_end": "14:15",
    "lunch_veto": ["11:30", "13:00"],   # inclusive block; null to disable
    "slot_lock_hours": 1.0,
    "shorts_priority": True,
    # short engine
    "short_enabled": True,
    "short_thresh": 0.082,
    "short_dyn_thresh": 0.110,          # risk-on tightened threshold; null to disable
    "short_dyn_sp500_gt": 0.005,
    "short_dyn_nifty2h_gte": -0.0010,
    "short_conv_cap": None,             # e.g. 0.04 (short inverted-U finding); null=off
    # long engine
    "long_enabled": True,
    "long_sp500_veto_lt": -0.005,       # block longs if prev S&P < this; null to disable
    "long_nifty2h_gt": 0.0025,          # require rising index; null to disable
    "long_vwap_gate": True,             # VWAP_Dist >= 0
    "long_market_gate": True,           # Market_Mean_Return >= 0
    "long_conv_cap": 0.030,             # long_conviction <= this; null=off
    # selection
    "selection": "rank",                # "rank" (top conviction) or "random_pool" (neg-control)
    "random_seed": 0,
    # capital (secondary, leverage-dependent metric only)
    "base_capital": 100000.0,
    "leverage": 5.0,
}


def _t(s):
    h, m = map(int, s.split(":"))
    return dtime(h, m)


def run_strategy(feed, config):
    """feed: DataFrame from build_feed. Returns trades DataFrame (one row per taken trade)."""
    cfg = {**DEFAULT_CONFIG, **config}
    t_start, t_end = _t(cfg["time_start"]), _t(cfg["time_end"])
    lunch = cfg["lunch_veto"]
    lunch_a, lunch_b = (_t(lunch[0]), _t(lunch[1])) if lunch else (None, None)
    lock = pd.Timedelta(hours=cfg["slot_lock_hours"])
    rng = np.random.default_rng(cfg["random_seed"])

    f = feed[(feed["DateTime"].dt.time >= t_start) & (feed["DateTime"].dt.time <= t_end)]
    groups = {ts: g for ts, g in f.groupby("DateTime")}

    trades = []
    locked_until = None
    for ts in sorted(groups.keys()):
        if locked_until is not None and ts < locked_until:
            continue
        tt = ts.time()
        if lunch_a is not None and lunch_a <= tt <= lunch_b:
            continue

        g = groups[ts]
        sp = g["sp500_prev_ret"].iloc[0]
        n2h = g["nifty_ret_2h"].iloc[0]

        # ---- short candidates ----
        short_cands = g.iloc[0:0]
        if cfg["short_enabled"]:
            thr = cfg["short_thresh"]
            if (cfg["short_dyn_thresh"] is not None
                    and sp > cfg["short_dyn_sp500_gt"] and n2h >= cfg["short_dyn_nifty2h_gte"]):
                thr = cfg["short_dyn_thresh"]
            sc = g[g["ss"] > thr]
            if cfg["short_conv_cap"] is not None:
                sc = sc[sc["short_conviction"] <= cfg["short_conv_cap"]]
            short_cands = sc

        # ---- long candidates ----
        long_cands = g.iloc[0:0]
        if cfg["long_enabled"]:
            ok = True
            if cfg["long_sp500_veto_lt"] is not None and sp < cfg["long_sp500_veto_lt"]:
                ok = False
            if cfg["long_nifty2h_gt"] is not None and not (n2h > cfg["long_nifty2h_gt"]):
                ok = False
            if ok:
                lc = g
                if cfg["long_vwap_gate"]:
                    lc = lc[lc["VWAP_Dist"] >= 0]
                if cfg["long_market_gate"]:
                    lc = lc[lc["Market_Mean_Return"] >= 0]
                if cfg["long_conv_cap"] is not None:
                    lc = lc[lc["long_conviction"] <= cfg["long_conv_cap"]]
                long_cands = lc

        def pick(cands, rank_col):
            if len(cands) == 0:
                return None
            if cfg["selection"] == "random_pool":
                return cands.iloc[rng.integers(len(cands))]
            return cands.sort_values(rank_col, ascending=False).iloc[0]

        if cfg.get("hour_side_gate") is not None:
            ts_str = ts.strftime("%H:%M")
            allowed_sides = cfg["hour_side_gate"].get(ts_str, [])
            if "SHORT" not in allowed_sides:
                short_cands = g.iloc[0:0]
            if "LONG" not in allowed_sides:
                long_cands = g.iloc[0:0]

        order = [("SHORT", short_cands, "short_conviction"),
                 ("LONG", long_cands, "long_conviction")]
        if not cfg["shorts_priority"]:
            order = order[::-1]

        taken = None
        for side, cands, col in order:
            p = pick(cands, col)
            if p is not None:
                sign = -1.0 if side == "SHORT" else 1.0
                taken = (ts, side, p["Ticker"], sign * p["Next_Hour_Return"] * 10000.0)
                break

        if taken is not None:
            trades.append(taken)
            locked_until = ts + lock

    td = pd.DataFrame(trades, columns=["ts", "side", "tk", "gross_bps"])
    if len(td):
        td["net_bps"] = td["gross_bps"] - cfg["cost_bps"]
        td["date"] = td["ts"].dt.date
        td["month"] = td["ts"].dt.to_period("M")
    return td


def _tstat(x):
    x = np.asarray(x, float)
    if len(x) < 2 or x.std(ddof=1) == 0:
        return 0.0
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))


def _compound(td, base_capital, leverage):
    """5x geometric compounding -- LEVERAGE-DEPENDENT, reported for context only."""
    cap = base_capital
    book = []
    for nb in td["net_bps"].values:
        pnl = (nb / 10000.0) * cap * leverage
        cap += pnl
        book.append(pnl)
    td = td.copy()
    td["bookRs"] = book
    td["capital"] = base_capital + np.cumsum(book)
    daily = td.groupby("date")["capital"].last()
    peak = daily.cummax()
    dd = (daily - peak)
    mdd = dd.min() if len(dd) else 0.0
    total = float(td["bookRs"].sum())
    return {
        "final_capital": base_capital + total,
        "total_profit": total,
        "return_x": (base_capital + total) / base_capital,
        "max_dd_rs": float(mdd),
        "return_mdd": abs(total / mdd) if mdd else float("nan"),
    }


def metrics(td, config):
    cfg = {**DEFAULT_CONFIG, **config}
    m = {"n": len(td)}
    if len(td) == 0:
        return m
    m["n_short"] = int((td.side == "SHORT").sum())
    m["n_long"] = int((td.side == "LONG").sum())
    m["win_rate"] = float((td.net_bps > 0).mean())
    m["net_bps_mean"] = float(td.net_bps.mean())
    m["net_bps_t"] = float(_tstat(td.net_bps))
    for side in ("SHORT", "LONG"):
        s = td[td.side == side]
        m[f"{side.lower()}_net_bps"] = float(s.net_bps.mean()) if len(s) else float("nan")
        m[f"{side.lower()}_t"] = float(_tstat(s.net_bps)) if len(s) else 0.0
        m[f"{side.lower()}_win"] = float((s.net_bps > 0).mean()) if len(s) else float("nan")
    m["compound"] = _compound(td, cfg["base_capital"], cfg["leverage"])
    return m


def monthly_table(td):
    rows = []
    for mo in sorted(td["month"].unique()):
        mt = td[td["month"] == mo]
        rows.append({
            "month": str(mo),
            "n": len(mt),
            "s": int((mt.side == "SHORT").sum()),
            "l": int((mt.side == "LONG").sum()),
            "win": float((mt.net_bps > 0).mean()),
            "net_bps": float(mt.net_bps.mean()),
        })
    return pd.DataFrame(rows)

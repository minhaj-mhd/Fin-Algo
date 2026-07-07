"""Rigorous deep-dive on the grid-search standout: TOP-1 SHORT, entry 13:15
(decision dt1=12:15, hold 13:15->14:15).

This cell was picked from a 2-side x 5-hour x 2-model grid, so it is POST-HOC. We
separate two questions and stress each:

  A. Is the RAW edge real (host-ranker top-1 short at 13:15, no Kronos)? -- panel only,
     so we can also check the PRE-cutoff window as an independent period.
  B. Does the Kronos LoRA veto genuinely ADD on top (vs base, vs a shuffled neg-control)?
     -- post-cutoff only (that's where scores exist).

Day-clustered bootstrap t (top-1@one hour = ~1 trade/day, so ~independent), outlier
sensitivity, sub-period split. Exploratory - NO Gauntlet authority.
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PANEL = os.path.join(ROOT, "data", "research", "entry_exit", "dualtf_trade_panel.csv")
SDIR = os.path.join(ROOT, "data", "research", "kronos_veto")
CUTOFF = pd.Timestamp("2025-09-09")
COSTS = {"6bps": 0.0006, "10bps": 0.0010}
THR_SHORT = 0.70
RNG = np.random.default_rng(7)


def boot(x, n=5000):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (np.nan, np.nan, np.nan, np.nan)
    means = x[RNG.integers(0, len(x), size=(n, len(x)))].mean(1)
    return x.mean(), x.mean() / means.std(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def bps(x, c):
    x = np.asarray(x, float)
    return (x - c).mean() * 1e4 if len(x) else np.nan


def short_top1(panel):
    s = panel[panel.dir == "short"].dropna(subset=["rkS_0"])
    t1 = s.loc[s.groupby("dt1")["rkS_0"].idxmin()].copy()
    t1["gross"] = -t1["nhr"]                      # short
    t1["tod"] = (t1.dt1 + pd.Timedelta(minutes=60)).dt.strftime("%H:%M")
    return t1


def all_shorts(panel):
    s = panel[panel.dir == "short"].copy()
    s["gross"] = -s["nhr"]
    s["tod"] = (s.dt1 + pd.Timedelta(minutes=60)).dt.strftime("%H:%M")
    return s


def main():
    panel = pd.read_csv(PANEL, usecols=["dt1", "ticker", "dir", "nhr", "rkS_0"], parse_dates=["dt1"])
    t1 = short_top1(panel)
    cell = t1[t1.tod == "13:15"].copy()
    post = cell[cell.dt1 >= CUTOFF]
    pre = cell[cell.dt1 < CUTOFF]

    print("=" * 78)
    print("A. RAW top-1 SHORT @ entry 13:15 (no Kronos)  [gross = short return]")
    print("=" * 78)
    for name, d in [("POST-cutoff", post), ("PRE-cutoff", pre), ("FULL", cell)]:
        g = d["gross"].values
        days = d["dt1"].dt.date.nunique()
        m, t, lo, hi = boot(g)
        for cn, c in COSTS.items():
            mm, tt, l2, h2 = boot(g - c)
            print(f"  {name:11s} @{cn:5s}: n={len(g):4d} days={days:4d} "
                  f"net={mm*1e4:+6.2f}bps t={tt:+5.2f} CI[{l2*1e4:+.2f},{h2*1e4:+.2f}] "
                  f"WR={(g>0).mean()*100:.1f}%")
    # outlier sensitivity (post)
    g = np.sort(post["gross"].values)
    print(f"\n  POST outlier check: mean gross {g.mean()*1e4:+.2f}bps | "
          f"drop top-3 wins -> {g[:-3].mean()*1e4:+.2f} | drop bot-3 -> {g[3:].mean()*1e4:+.2f} | "
          f"median {np.median(g)*1e4:+.2f} | winsor5% {np.clip(g,np.percentile(g,5),np.percentile(g,95)).mean()*1e4:+.2f}")
    # sub-period stability (post)
    post_sorted = post.sort_values("dt1")
    half = len(post_sorted) // 2
    for lbl, d in [("post-H1", post_sorted.iloc[:half]), ("post-H2", post_sorted.iloc[half:])]:
        print(f"  {lbl}: n={len(d)} net@6={bps(d.gross,0.0006):+.2f} net@10={bps(d.gross,0.0010):+.2f} "
              f"({d.dt1.min():%Y-%m-%d}..{d.dt1.max():%Y-%m-%d})")

    # rank vs hour: is the edge in rank-1 or the whole 13:15 short book?
    sall = all_shorts(panel)
    c13 = sall[(sall.tod == "13:15") & (sall.dt1 >= CUTOFF)]
    print(f"\n  hour-vs-rank (POST): top-1 short net@6={bps(post.gross,0.0006):+.2f} | "
          f"ALL 13:15 shorts (top-3) net@6={bps(c13.gross,0.0006):+.2f} (n={len(c13)})")
    print("  other-hours top-1 short net@6 (POST):", {
        tod: round(bps(t1[(t1.tod == tod) & (t1.dt1 >= CUTOFF)].gross, 0.0006), 2)
        for tod in ["10:15", "11:15", "12:15", "13:15", "14:15"]})

    # ---- B. Kronos veto on the cell (post only) ----
    base = pd.read_csv(os.path.join(SDIR, "scores_1h_base.csv"), parse_dates=["dt1"])[["ticker", "dt1", "p_up"]]
    lora = pd.read_csv(os.path.join(SDIR, "scores_1h_lora.csv"), parse_dates=["dt1"])[["ticker", "dt1", "p_up"]]
    m = post.merge(base.rename(columns={"p_up": "p_base"}), on=["ticker", "dt1"]).merge(
        lora.rename(columns={"p_up": "p_lora"}), on=["ticker", "dt1"])
    print("\n" + "=" * 78)
    print(f"B. Kronos veto ON the cell (post, n={len(m)}); short keep iff 1-p_up>={THR_SHORT}")
    print("=" * 78)
    for cn, c in COSTS.items():
        book = bps(m.gross, c)
        print(f"  book (no veto) @{cn}: {book:+.2f}bps")
        for tag, pcol in [("base", "p_base"), ("lora", "p_lora")]:
            keep = (1 - m[pcol].values) >= THR_SHORT
            kept = m.gross.values[keep]
            mm, tt, lo, hi = boot(kept - c)
            # neg-control: shuffle p_up, same rule, distribution of kept-net
            null = []
            p = m[pcol].values
            for _ in range(3000):
                k = (1 - RNG.permutation(p)) >= THR_SHORT
                null.append((m.gross.values[k] - c).mean() * 1e4 if k.sum() else np.nan)
            null = np.array(null)
            pval = float(np.mean(null >= mm * 1e4))
            print(f"    {tag}-veto: keep {keep.sum():3d}/{len(m)} net={mm*1e4:+6.2f} t={tt:+5.2f} "
                  f"CI[{lo*1e4:+.2f},{hi*1e4:+.2f}] | negctrl null {np.nanmean(null):+.2f} "
                  f"[{np.nanpercentile(null,2.5):+.2f},{np.nanpercentile(null,97.5):+.2f}] p={pval:.3f}")


if __name__ == "__main__":
    main()

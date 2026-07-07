"""Top-1 (per side, per decision) of the 1h host book, vetoed by Kronos base vs LoRA,
broken down by entry time-of-day.

Top-1 = the single best long (min rkL_0) and best short (min rkS_0) at each dt1 -- the
highest-conviction pick actually traded. We then apply the DEPLOYED Kronos veto rule
(keep LONG iff p_up>=0.50; keep SHORT iff 1-p_up>=0.70) with base vs LoRA p_up, and
report net bps/trade of all vs kept (survived) vs vetoed (dropped), overall and per
entry hour (dt1+60 = 10:15..14:15). Post-cutoff (>=2025-09-09) honest window.

Exploratory only - NO Gauntlet authority.
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PANEL = os.path.join(ROOT, "data", "research", "entry_exit", "dualtf_trade_panel.csv")
SDIR = os.path.join(ROOT, "data", "research", "kronos_veto")
CUTOFF = pd.Timestamp("2025-09-09")
THR_LONG, THR_SHORT = 0.50, 0.70          # deployed Kronos veto thresholds
COST = {"6bps": 0.0006, "10bps": 0.0010}


def top1_book(panel):
    longs = panel[panel.dir == "long"].dropna(subset=["rkL_0"])
    shorts = panel[panel.dir == "short"].dropna(subset=["rkS_0"])
    tl = longs.loc[longs.groupby("dt1")["rkL_0"].idxmin()]
    ts = shorts.loc[shorts.groupby("dt1")["rkS_0"].idxmin()]
    return pd.concat([tl, ts], ignore_index=True)


def load():
    panel = pd.read_csv(PANEL, usecols=["dt1", "ticker", "dir", "nhr", "rkL_0", "rkS_0"],
                        parse_dates=["dt1"])
    t1 = top1_book(panel)
    base = pd.read_csv(os.path.join(SDIR, "scores_1h_base.csv"), parse_dates=["dt1"])[["ticker", "dt1", "p_up"]]
    lora = pd.read_csv(os.path.join(SDIR, "scores_1h_lora.csv"), parse_dates=["dt1"])[["ticker", "dt1", "p_up"]]
    df = t1.merge(base.rename(columns={"p_up": "p_base"}), on=["ticker", "dt1"], how="inner")
    df = df.merge(lora.rename(columns={"p_up": "p_lora"}), on=["ticker", "dt1"], how="inner")
    df = df[df.dt1 >= CUTOFF].copy()
    df["sgn"] = np.where(df.dir == "long", 1.0, -1.0)
    df["gross"] = df.nhr * df.sgn
    df["tod"] = (df.dt1 + pd.Timedelta(minutes=60)).dt.strftime("%H:%M")
    return df


def keep_mask(df, pcol):
    p = df[pcol].values
    is_long = (df.dir == "long").values
    return np.where(is_long, p >= THR_LONG, (1.0 - p) >= THR_SHORT)


def netbps(gross, cost):
    return (gross - cost).mean() * 1e4 if len(gross) else float("nan")


def tstat(gross, cost):
    x = (gross - cost) * 1e4
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 and x.std() > 0 else 0.0


def report(df, cost_name, lines):
    c = COST[cost_name]
    for side in ["long", "short"]:
        s = df[df.dir == side]
        lines.append(f"\n===== TOP-1 {side.upper()}  (n={len(s)}, post-cutoff)  @{cost_name} =====")
        wr = (s.gross > 0).mean() * 100
        lines.append(f"  no-veto:            net={netbps(s.gross.values, c):+6.2f}bps  t={tstat(s.gross.values, c):+.2f}  rawWR={wr:.1f}%")
        for tag, pcol in [("base", "p_base"), ("lora", "p_lora")]:
            k = keep_mask(s, pcol)
            kept, drop = s[k], s[~k]
            lines.append(f"  {tag}-veto keep({k.sum()}/{len(s)}): "
                         f"KEPT net={netbps(kept.gross.values, c):+6.2f} t={tstat(kept.gross.values, c):+.2f} "
                         f"| VETOED net={netbps(drop.gross.values, c):+6.2f} (n={len(drop)}) "
                         f"| dNET={netbps(kept.gross.values, c) - netbps(s.gross.values, c):+6.2f}")
        # by time-of-day
        lines.append(f"  by entry hour (no-veto | lora-KEPT net@{cost_name}):")
        for tod in sorted(s.tod.unique()):
            st = s[s.tod == tod]
            k = keep_mask(st, "p_lora")
            kept = st[k]
            lines.append(f"    {tod}: n={len(st):3d}  no-veto={netbps(st.gross.values, c):+7.2f}  "
                         f"lora-KEPT={netbps(kept.gross.values, c):+7.2f} (kept {k.sum():3d})  "
                         f"base-KEPT={netbps(st[keep_mask(st, 'p_base')].gross.values, c):+7.2f}")


def main():
    df = load()
    lines = [f"TOP-1 of 1h-host book (min rank/side/dt1) vetoed by Kronos base vs LoRA - {pd.Timestamp.now():%Y-%m-%d %H:%M}",
             f"post-cutoff n={len(df)} (long {int((df.dir=='long').sum())}, short {int((df.dir=='short').sum())}); "
             f"veto thr: LONG keep p_up>={THR_LONG}, SHORT keep 1-p_up>={THR_SHORT}",
             f"p_up mean base {df.p_base.mean():.3f} / lora {df.p_lora.mean():.3f}"]
    for cn in ["6bps", "10bps"]:
        report(df, cn, lines)
    print("\n".join(lines))


if __name__ == "__main__":
    main()

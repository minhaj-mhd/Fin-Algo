"""Honest 2xATR (and sweep) stop-loss sim on the 13:15-short top-1 cell.

Stop MUST trigger on the intra-hour HIGH (max adverse excursion for a short), not the
close -- else it's the resolution artifact prior stop research already debunked. We take
the 1h bar labelled 13:15 (covers the 13:15->14:15 hold); its HIGH is the intra-hour peak.
If (high-entry)/entry >= k*ATR_frac the short is stopped at -k*ATR (even if it would have
closed a winner). ATR = 14-bar ATR on the 1h series ending at the entry bar.

Baseline no-stop return uses the same cache (close_12:15 -> close_13:15) and is checked
against the panel's nhr. Post-cutoff; full cell + LoRA-kept subset. Exploratory.
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PANEL = os.path.join(ROOT, "data", "research", "entry_exit", "dualtf_trade_panel.csv")
CACHE = os.path.join(ROOT, "data", "raw_upstox_cache_1h_v3")
SDIR = os.path.join(ROOT, "data", "research", "kronos_veto")
CUTOFF = pd.Timestamp("2025-09-09")
WIDTHS = [1.0, 1.5, 2.0, 2.5, 3.0]


def load_1h(sym):
    fp = os.path.join(CACHE, f"{sym}.csv")
    if not os.path.exists(fp):
        return None
    df = pd.read_csv(fp)
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    df = df.assign(ts=ts)
    df = df[(ts.dt.minute == 15) & (ts.dt.hour.between(9, 14))].sort_values("ts").reset_index(drop=True)
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(1)
    df["atr14"] = tr.rolling(14).mean()
    return df.set_index("ts")


def build_cell():
    p = pd.read_csv(PANEL, usecols=["dt1", "ticker", "dir", "nhr", "rkS_0"], parse_dates=["dt1"])
    s = p[p.dir == "short"].dropna(subset=["rkS_0"])
    t1 = s.loc[s.groupby("dt1")["rkS_0"].idxmin()].copy()
    t1["tod"] = (t1.dt1 + pd.Timedelta(minutes=60)).dt.strftime("%H:%M")
    cell = t1[(t1.tod == "13:15") & (t1.dt1 >= CUTOFF)].copy()
    lora = pd.read_csv(os.path.join(SDIR, "scores_1h_lora.csv"), parse_dates=["dt1"])[["ticker", "dt1", "p_up"]]
    return cell.merge(lora.rename(columns={"p_up": "p_lora"}), on=["ticker", "dt1"])


def simulate(m):
    rows = []
    caches = {}
    for r in m.itertuples(index=False):
        sym = r.ticker.replace(".NS", "")
        if sym not in caches:
            caches[sym] = load_1h(sym)
        df = caches[sym]
        if df is None:
            continue
        entry_bar = pd.Timestamp(r.dt1)          # 12:15 bar close = entry (price at 13:15)
        hold_bar = entry_bar + pd.Timedelta(minutes=60)   # 13:15 bar = the hold hour
        if entry_bar not in df.index or hold_bar not in df.index:
            continue
        entry = df.at[entry_bar, "close"]
        atr = df.at[entry_bar, "atr14"]
        high = df.at[hold_bar, "high"]
        exit_ = df.at[hold_bar, "close"]
        if not np.isfinite(entry) or not np.isfinite(atr) or entry <= 0 or atr <= 0:
            continue
        gross_short = -(exit_ / entry - 1.0)          # cache-based short return
        adverse_up = (high - entry) / entry           # short's max adverse excursion
        atr_frac = atr / entry
        rows.append(dict(gross=gross_short, adverse=adverse_up, atr_frac=atr_frac,
                         p_lora=r.p_lora, nhr=r.nhr))
    return pd.DataFrame(rows)


def summ(g, c):
    g = np.asarray(g, float) * 1e4
    l = g[g < 0]
    return (f"net={g.mean() - c:+6.2f} WR={(g > 0).mean() * 100:4.1f}% "
            f"avgLoss={l.mean() if len(l) else 0:+7.2f} worst={g.min():+8.2f}")


def run(sim, label):
    g0 = sim["gross"].values
    print(f"\n----- {label} (n={len(sim)}) -----")
    print(f"  panel-nhr short mean bps = {(-sim['nhr']).mean() * 1e4:+.2f} | cache short mean bps = {g0.mean() * 1e4:+.2f}  (sanity)")
    print(f"  NO STOP        @6bps: {summ(g0, 6)}   @10bps: {summ(g0, 10)}")
    for k in WIDTHS:
        stop = k * sim["atr_frac"].values
        stopped = sim["adverse"].values >= stop
        g = np.where(stopped, -stop, g0)
        print(f"  {k:.1f}xATR stop @6bps: {summ(g, 6)}   @10bps: {summ(g, 10)}   (stopped {stopped.sum():3d}/{len(sim)})")


def main():
    m = build_cell()
    sim = simulate(m)
    run(sim, "FULL 13:15-short cell")
    kl = (1 - sim["p_lora"].values) >= 0.70
    run(sim[kl].reset_index(drop=True), "LoRA-kept (veto survivors)")


if __name__ == "__main__":
    main()

"""
Index (Nifty 50) trailing-2h momentum as a LONG cut  —  OOS test
================================================================
Hypothesis: if Nifty 50 has already SOARED over the trailing ~2h (index
over-extension), new longs chase an exhausted move and revert -> CUT longs.

Rule tested: skip a top-1 long if  index_2h_ret(T) > cutoff  (sweep cutoffs).
Early-morning handling: for anchors where a full 2h of intraday isn't available
(before 11:15, since the session opens 09:15), the trailing window reaches into
pre-open, so we "count the overnight GAP as ~1 hour" by measuring the index
return from the PREVIOUS CLOSE through T (gap-inclusive). At/after 11:15 it's a
pure intraday 2h return.

Reuses the top-1 long returns from data/backtests/conviction_caps_oos_2026-07-10.csv
(date, anchor, is_t1L, Lg=1h fwd bps) and joins Nifty ^NSEI 15m (yfinance).
OOS = 2026-06-16..07-09 (v20 trained 06-15). net@6 = fwd - 6bps.

Run: python -m scripts.backtests.nifty_2h_long_gate_oos
"""
import os, sys, datetime as dt, warnings
sys.path.insert(0, os.getcwd()); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf

PANEL = "data/backtests/conviction_caps_oos_2026-07-10.csv"
COST6 = 6
SESSION_OPEN = dt.time(9, 15)
LOOKBACK = pd.Timedelta(hours=2)

# ── Nifty 50 15m (UTC -> IST naive) ──────────────────────────────────────────
nf = yf.download("^NSEI", start="2026-06-15", end="2026-07-11", interval="15m", progress=False)
if isinstance(nf.columns, pd.MultiIndex):
    nf.columns = nf.columns.get_level_values(0)
nf = nf[["Close"]].copy()
nf.index = pd.to_datetime(nf.index).tz_convert("Asia/Kolkata").tz_localize(None)
nifty = nf["Close"].sort_index()
# previous-session close per date (last bar of each prior trading day)
sess_close = nifty.groupby(nifty.index.date).last()
sess_dates = sorted(sess_close.index)
prev_close = {d: sess_close[sess_dates[i - 1]] for i, d in enumerate(sess_dates) if i > 0}
print(f"Nifty bars {len(nifty)} | {nifty.index.min()} -> {nifty.index.max()}")


def nifty_at(ts):
    """Nifty close at/just-before ts."""
    s = nifty[nifty.index <= ts]
    return float(s.iloc[-1]) if len(s) else np.nan


def index_2h_ret(ts):
    """Trailing ~2h Nifty return; gap-as-1h for early-session anchors."""
    now = nifty_at(ts)
    if np.isnan(now):
        return np.nan
    ref_time = ts - LOOKBACK
    d = ts.date()
    if ref_time.time() >= SESSION_OPEN and ref_time.date() == d:
        ref = nifty_at(ref_time)                    # full intraday 2h
    else:
        ref = prev_close.get(d, np.nan)             # early: gap-inclusive (prev close)
    if not ref or np.isnan(ref):
        return np.nan
    return (now / ref - 1) * 100                     # percent


# ── join to top-1 long trades ────────────────────────────────────────────────
p = pd.read_csv(PANEL)
lo = p[p.is_t1L].copy()
lo["ts"] = pd.to_datetime(lo.date + " " + lo.anchor)
lo["idx2h"] = lo.ts.map(index_2h_ret)
lo["net6"] = lo.Lg - COST6
lo = lo.dropna(subset=["idx2h"])
print(f"top-1 longs with index momentum: {len(lo)}  | baseline net@6 = {lo.net6.mean():+.2f} bps "
      f"(WR {100*(lo.Lg>0).mean():.0f}%)\n")
print(f"index_2h distribution: min {lo.idx2h.min():+.2f}%  med {lo.idx2h.median():+.2f}%  "
      f"max {lo.idx2h.max():+.2f}%  ( >0 in {100*(lo.idx2h>0).mean():.0f}% of anchors )\n")


def blk(x, label):
    if len(x) == 0:
        print(f"   {label:34s} n=  0"); return
    g = x.Lg; net = g.mean() - COST6
    t = g.mean() / (g.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 and g.std() > 0 else 0
    print(f"   {label:34s} n={len(x):3d}  WR={100*(g>0).mean():3.0f}%  net@6={net:+7.2f} bps  t/2~{t/2:+.1f}")


print("=" * 84)
print("  CUT LONGS when Nifty trailing-2h return > cutoff  (keep the rest)")
print("=" * 84)
blk(lo, "BASELINE (all top-1 longs)")
print("  " + "-" * 80)
for cut in [0.3, 0.5, 0.7, 1.0]:
    kept = lo[lo.idx2h <= cut]; cutset = lo[lo.idx2h > cut]
    print(f"  cutoff {cut:+.1f}% :")
    blk(kept, f"   KEPT  (idx2h <= {cut:+.1f}%)")
    blk(cutset, f"   CUT   (idx2h >  {cut:+.1f}%)  <- removed")

print("\n" + "=" * 84)
print("  Longs by index-2h-momentum bucket (is over-extension the bad zone?)")
print("=" * 84)
for lo_b, hi_b, lbl in [(-99, -0.5, "idx2h < -0.5% (index fell)"), (-0.5, 0.0, "[-0.5%, 0%)"),
                        (0.0, 0.5, "[0%, +0.5%)"), (0.5, 1.0, "[+0.5%, +1.0%)"),
                        (1.0, 99, ">= +1.0% (index soared)")]:
    blk(lo[(lo.idx2h >= lo_b) & (lo.idx2h < hi_b)], lbl)

lo[["date", "anchor", "idx2h", "Lg", "net6"]].to_csv(
    f"data/backtests/nifty_2h_long_gate_{dt.date.today()}.csv", index=False)
print(f"\n[SAVED] data/backtests/nifty_2h_long_gate_{dt.date.today()}.csv")
print("Exploratory, no Gauntlet. Early-morning (<11:15) uses gap-inclusive (prev-close) ref.")

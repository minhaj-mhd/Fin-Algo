"""
Where does the SHORT model underperform?  —  OOS regime probe
=============================================================
User thesis: the short leg is a mean-reversion engine that "thrives whether Nifty
is up or down". This probes for the regimes where that breaks — mirroring the
long-side Nifty trailing-2h gate (scripts/backtests/nifty_2h_long_gate_oos.py)
but on the top-1 SHORT stream (is_t1S, Sg = -fwd_bps).

Cuts tested (top-1 short, 1h hold, net@6 = Sg - 6bps):
  A. Trailing-2h index direction buckets     (does side care about regime?)
  B. Index gate sweeps  (cut shorts when idx2h below / above a cutoff)
  C. |idx2h| trend-STRENGTH buckets           (mean-reversion dies on trend days)
  D. Index move DURING the 1h hold            (diagnostic mechanism, not gate-able)
  E. Short conviction bands                   (re-confirm the >0.04 over-extension fail)
  F. Time-of-day

OOS = 2026-06-16..07-09 (v20 trained 06-15). Early anchors (<11:15) count the
overnight gap as ~1h (prev-close ref), same convention as the long gate.

Run: python -m scripts.backtests.short_regime_probe_oos
"""
import os, sys, datetime as dt, warnings
sys.path.insert(0, os.getcwd()); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf

PANEL = "data/backtests/conviction_caps_oos_2026-07-10.csv"
COST6 = 6
SESSION_OPEN = dt.time(9, 15)
LOOKBACK = pd.Timedelta(hours=2)
HOLD = pd.Timedelta(hours=1)

# ── Nifty 50 15m (UTC -> IST naive) ──────────────────────────────────────────
nf = yf.download("^NSEI", start="2026-06-15", end="2026-07-11", interval="15m", progress=False)
if isinstance(nf.columns, pd.MultiIndex):
    nf.columns = nf.columns.get_level_values(0)
nf = nf[["Close"]].copy()
nf.index = pd.to_datetime(nf.index).tz_convert("Asia/Kolkata").tz_localize(None)
nifty = nf["Close"].sort_index()
sess_close = nifty.groupby(nifty.index.date).last()
sess_dates = sorted(sess_close.index)
prev_close = {d: sess_close[sess_dates[i - 1]] for i, d in enumerate(sess_dates) if i > 0}
print(f"Nifty bars {len(nifty)} | {nifty.index.min()} -> {nifty.index.max()}")


def nifty_at(ts):
    s = nifty[nifty.index <= ts]
    return float(s.iloc[-1]) if len(s) else np.nan


def index_2h_ret(ts):
    """Trailing ~2h Nifty return; gap-as-1h for early-session anchors (percent)."""
    now = nifty_at(ts)
    if np.isnan(now):
        return np.nan
    ref_time = ts - LOOKBACK
    d = ts.date()
    if ref_time.time() >= SESSION_OPEN and ref_time.date() == d:
        ref = nifty_at(ref_time)
    else:
        ref = prev_close.get(d, np.nan)
    if not ref or np.isnan(ref):
        return np.nan
    return (now / ref - 1) * 100


def index_hold_ret(ts):
    """Nifty return over the 1h hold T -> T+1h (percent). Forward-looking diagnostic."""
    now = nifty_at(ts)
    fut = nifty_at(ts + HOLD)
    if np.isnan(now) or np.isnan(fut):
        return np.nan
    return (fut / now - 1) * 100


# ── join to top-1 short trades ───────────────────────────────────────────────
p = pd.read_csv(PANEL)
so = p[p.is_t1S].copy()
so["ts"] = pd.to_datetime(so.date + " " + so.anchor)
so["idx2h"] = so.ts.map(index_2h_ret)
so["idxHold"] = so.ts.map(index_hold_ret)
so["net6"] = so.Sg - COST6
so = so.dropna(subset=["idx2h"])
print(f"\ntop-1 shorts: {len(so)}  | baseline net@6 = {so.net6.mean():+.2f} bps "
      f"(WR {100*(so.Sg>0).mean():.0f}%)")
print(f"idx2h dist: min {so.idx2h.min():+.2f}%  med {so.idx2h.median():+.2f}%  "
      f"max {so.idx2h.max():+.2f}%  ( index up in {100*(so.idx2h>0).mean():.0f}% of anchors )")


def blk(x, label):
    if len(x) == 0:
        print(f"   {label:40s} n=  0"); return
    g = x.Sg; net = g.mean() - COST6
    t = g.mean() / (g.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 and g.std() > 0 else 0
    print(f"   {label:40s} n={len(x):3d}  WR={100*(g>0).mean():3.0f}%  net@6={net:+7.2f}  t/2~{t/2:+.1f}")


print("\n" + "=" * 92)
print("  A. SHORT net@6 by trailing-2h index direction  (does the short care about regime?)")
print("=" * 92)
for lo_b, hi_b, lbl in [(-99, -1.0, "idx2h < -1.0%  (index crashed)"), (-1.0, -0.5, "[-1.0,-0.5%)"),
                        (-0.5, 0.0, "[-0.5%, 0%)  index eased"), (0.0, 0.5, "[0%,+0.5%)"),
                        (0.5, 1.0, "[+0.5,+1.0%)"), (1.0, 99, ">= +1.0%  (index soared)")]:
    blk(so[(so.idx2h >= lo_b) & (so.idx2h < hi_b)], lbl)

print("\n" + "=" * 92)
print("  B. Index gates on the SHORT stream (keep subset, drop the rest)")
print("=" * 92)
blk(so, "BASELINE (all top-1 shorts)")
print("  " + "-" * 88)
print("  B1. cut shorts when index FELL hard (bounce risk): keep idx2h >= cutoff")
for cut in [-1.0, -0.5, -0.3, 0.0]:
    blk(so[so.idx2h >= cut], f"   KEEP idx2h >= {cut:+.1f}%")
print("  B2. cut shorts when index SOARED (short into strength?): keep idx2h <= cutoff")
for cut in [1.0, 0.5, 0.3, 0.0]:
    blk(so[so.idx2h <= cut], f"   KEEP idx2h <= {cut:+.1f}%")

print("\n" + "=" * 92)
print("  C. Trend-STRENGTH: |idx2h| buckets  (a mean-reversion engine should die on trend days)")
print("=" * 92)
so["abs2h"] = so.idx2h.abs()
for lo_b, hi_b, lbl in [(0.0, 0.15, "|idx2h| < 0.15%  (quiet/chop)"), (0.15, 0.35, "[0.15,0.35%)"),
                        (0.35, 0.6, "[0.35,0.6%)"), (0.6, 99, ">= 0.6%  (strong trend)")]:
    blk(so[(so.abs2h >= lo_b) & (so.abs2h < hi_b)], lbl)

print("\n" + "=" * 92)
print("  D. MECHANISM (diagnostic): index move DURING the 1h hold vs short P&L")
print("=" * 92)
sd = so.dropna(subset=["idxHold"])
if len(sd) > 3:
    r = np.corrcoef(sd.idxHold, sd.Sg)[0, 1]
    print(f"   corr(index move in hold, short gross bps) = {r:+.2f}  "
          f"(negative => shorts lose when index rises during the hold)")
for lo_b, hi_b, lbl in [(-99, -0.2, "index FELL in hold (<-0.2%)"), (-0.2, 0.2, "flat [-0.2,0.2%)"),
                        (0.2, 99, "index ROSE in hold (>=0.2%)")]:
    blk(sd[(sd.idxHold >= lo_b) & (sd.idxHold < hi_b)], lbl)

print("\n" + "=" * 92)
print("  E. Short conviction bands (re-confirm >0.04 over-extension fail, in this stream)")
print("=" * 92)
for lo_b, hi_b, lbl in [(0.0, 0.02, "conv [0,0.02)"), (0.02, 0.03, "[0.02,0.03) peak?"),
                        (0.03, 0.04, "[0.03,0.04)"), (0.04, 0.06, "[0.04,0.06) FAIL?"),
                        (0.06, 99, ">= 0.06")]:
    blk(so[(so.Sconv >= lo_b) & (so.Sconv < hi_b)], lbl)

print("\n" + "=" * 92)
print("  F. Time-of-day (short stream)")
print("=" * 92)
for a in sorted(so.anchor.unique()):
    blk(so[so.anchor == a], a)

out = f"data/backtests/short_regime_probe_{dt.date.today()}.csv"
so[["date", "anchor", "Sconv", "idx2h", "idxHold", "Sg", "net6"]].to_csv(out, index=False)
print(f"\n[SAVED] {out}")
print("Exploratory, no Gauntlet. Early anchors (<11:15) use gap-inclusive (prev-close) idx2h ref.")

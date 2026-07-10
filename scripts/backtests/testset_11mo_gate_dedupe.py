"""
v20 80/20 UNTOUCHED TEST (2025-08 -> 2026-06, 11 months) — gate x de-dupe x book
=================================================================================
Scores the DEPLOYED v20 long+short models on the panel rows the production model
never trained on (train ended 2025-06, val 2025-07, test 2025-08+), then runs the
same index-gate + 2h de-dupe analysis validated on the 17-day live window — now
over ~185 trading days.

Books:
  full   (K=1): top-1 pick per 15-min anchor (matches the live-window table)
  5-slot (K=5): the 5 best picks per anchor (repo Top-K convention)
De-dupe (2h per-ticker cooldown, resets each session):
  RAW      : take the top-K, no de-dupe
  SKIP     : take a ranked slot only if that name is off cooldown, else empty (no replacement)
  REPLACE  : fill K slots with the K highest-ranked names off cooldown
Gate (anchor-level, on Nifty-50 trailing-2h return; gap-as-1h before 11:15):
  SHORT : none | idx2h>=0 | 5-band  ([-0.05,+0.25) U idx2h<-0.5 U idx2h>=+0.5)
  LONG  : none | idx2h>=0 | idx2h>=+0.3 | idx2h>=+0.5   (user threshold = +0.5)

net@6 = pnl_bps - 6. Overlap handled with DAY-CLUSTERED t (t_d). Exploratory, no Gauntlet.
Run: python -m scripts.backtests.testset_11mo_gate_dedupe
"""
import os, sys, json, datetime as dt, warnings
sys.path.insert(0, os.getcwd()); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, xgboost as xgb
import pyarrow.compute as pc, pyarrow.dataset as pads

PANEL = "data/research/v20_rolling_1h/panel.parquet"
NIFTY = "data/raw_index_cache/nifty50_15m.csv"
TEST_FROM, TEST_TO = pd.Timestamp("2025-08-01"), pd.Timestamp("2026-06-05")
HOLD = pd.Timedelta(hours=1); COOL = pd.Timedelta(hours=2); COST = 6
NOTIONAL = 99517.68
v20 = json.load(open("models/research/v20_rolling_1h/metadata.json"))["features"]

# ── load test slice ──────────────────────────────────────────────────────────
cols = ["DateTime", "Ticker", "Next_Hour_Return"] + [f for f in v20]
cols = list(dict.fromkeys(cols))
dset = pads.dataset(PANEL)
tb = dset.to_table(columns=cols, filter=(pc.field("DateTime") >= TEST_FROM) & (pc.field("DateTime") < TEST_TO))
df = tb.to_pandas()
df["tod"] = df.DateTime.dt.strftime("%H:%M")
df = df[(df.tod >= "10:15") & (df.tod <= "14:15")].copy()
df = df[df.Next_Hour_Return.abs() <= 0.20]                       # drop data-glitch bars (few)
miss = [f for f in v20 if f not in df.columns]
assert not miss, f"missing features: {miss}"
print(f"test rows {len(df):,} | {df.DateTime.min()} -> {df.DateTime.max()} | {df.DateTime.dt.date.nunique()} days")

# ── score deployed models ────────────────────────────────────────────────────
bl = xgb.Booster(); bl.load_model("models/research/v20_rolling_1h/xgb_long_model.json")
bs = xgb.Booster(); bs.load_model("models/research/v20_rolling_1h/xgb_short_model.json")
X = xgb.DMatrix(np.nan_to_num(df[v20].values.astype(np.float32)), feature_names=v20)
df["ls"] = bl.predict(X); df["ss"] = bs.predict(X)
df["retbps"] = df.Next_Hour_Return * 1e4

# per-anchor ranked candidate lists
long_book, short_book = {}, {}
for ts, g in df.groupby("DateTime"):
    gl = g.sort_values("ls", ascending=False)
    long_book[ts] = list(zip(gl.Ticker.values, gl.retbps.values))       # long pnl = +ret
    gs = g.sort_values("ss", ascending=False)
    short_book[ts] = list(zip(gs.Ticker.values, (-gs.retbps).values))   # short pnl = -ret
anchors = sorted(long_book)

# ── Nifty idx2h ──────────────────────────────────────────────────────────────
nf = pd.read_csv(NIFTY, parse_dates=["ts"]).set_index("ts").close.sort_index()
sc = nf.groupby(nf.index.date).last(); ds = sorted(sc.index); pc_ = {d: sc[ds[i-1]] for i, d in enumerate(ds) if i > 0}
def nat(ts):
    s = nf[nf.index <= ts]; return float(s.iloc[-1]) if len(s) else np.nan
_idx = {}
def idx2h(ts):
    if ts in _idx: return _idx[ts]
    now = nat(ts); rt = ts - HOLD * 2
    ref = nat(rt) if (rt.time() >= dt.time(9, 15) and rt.date() == ts.date()) else pc_.get(ts.date(), np.nan)
    v = (now / ref - 1) * 100 if ref and not np.isnan(ref) and not np.isnan(now) else np.nan
    _idx[ts] = v; return v

def g_none(ts):  return True
def g_up(ts):    v = idx2h(ts); return (not np.isnan(v)) and v >= 0
def g_up3(ts):   v = idx2h(ts); return (not np.isnan(v)) and v >= 0.3
def g_up5(ts):   v = idx2h(ts); return (not np.isnan(v)) and v >= 0.5
def g_5band(ts):
    v = idx2h(ts)
    if np.isnan(v): return False
    return (-0.05 <= v < 0.25) or (v < -0.5) or (v >= 0.5)

# ── simulate ─────────────────────────────────────────────────────────────────
def simulate(bk, K, policy, gate):
    last = {}; cur = None; trades = []
    for ts in anchors:
        if ts.date() != cur: last = {}; cur = ts.date()
        if not gate(ts): continue
        cand = bk[ts]
        if policy == "RAW":
            picks = cand[:K]
        elif policy == "SKIP":
            picks = [(t, g) for (t, g) in cand[:K] if not (t in last and (ts - last[t]) < COOL)]
        else:  # REPLACE
            picks = []
            for t, g in cand:
                if t in last and (ts - last[t]) < COOL: continue
                picks.append((t, g))
                if len(picks) == K: break
        for t, g in picks: last[t] = ts; trades.append((ts, t, g))
    return trades

def row(trades, gate_lbl, pol):
    if not trades:
        return f"  {gate_lbl:10s} {pol:8s}  n=    0"
    d = pd.DataFrame(trades, columns=["ts", "tk", "pnl"]); g = d.pnl
    net6 = g - COST; net = net6.mean()
    dd = net6.groupby(d.ts.dt.date).mean()
    td = dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd))) if len(dd) > 1 and dd.std() > 0 else 0
    rs = (net6 / 1e4 * NOTIONAL).sum()
    return (f"  {gate_lbl:10s} {pol:8s}  n={len(d):5d}  WR={100*(g>0).mean():3.0f}%  "
            f"net@6={net:+6.2f}  CUM={net6.sum()/100:+7.2f}%  bookRs={rs:+9.0f}  t_d~{td:+.1f}")

def block(title, bk, gates, K):
    print("\n" + "=" * 104); print(f"  {title}"); print("=" * 104)
    print(f"  {'gate':10s} {'policy':8s}  {'n':>6s}  {'WR':>4s}  {'net@6':>7s}  {'CUM':>8s}  {'bookRs':>10s}  {'t_d':>5s}")
    for glbl, gate in gates:
        for pol in ["RAW", "SKIP", "REPLACE"]:
            print(row(simulate(bk, K, pol, gate), glbl, pol))

# idx2h coverage
iv = pd.Series({ts: idx2h(ts) for ts in anchors}).dropna()
print(f"idx2h anchors: {len(iv)} | >=0 in {100*(iv>=0).mean():.0f}% | >=+0.5 in {100*(iv>=0.5).mean():.0f}%")

SG = [("none", g_none), ("idx2h>=0", g_up), ("5-band", g_5band)]
LG = [("none", g_none), ("idx2h>=0", g_up), ("idx2h>=.3", g_up3), ("idx2h>=.5", g_up5)]
block("SHORT — FULL (top-1 / anchor)",  short_book, SG, 1)
block("SHORT — 5-SLOT (top-5 / anchor)", short_book, SG, 5)
block("LONG — FULL (top-1 / anchor)",   long_book,  LG, 1)
block("LONG — 5-SLOT (top-5 / anchor)",  long_book,  LG, 5)
print("\nDeployed v20 on its 80/20 UNTOUCHED test set. Overlapping 1h holds -> t_d = day-clustered. "
      "Exploratory, no Gauntlet. NIFTY-50 15m gap-as-1h before 11:15.")

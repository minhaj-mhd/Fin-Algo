"""
SHORT book: index-band filter + 2-hour per-ticker de-dupe (SKIP, no replacement)
=================================================================================
Mirrors scripts/backtests/long_dedupe_2h_oos.py but on the top-1 SHORT stream.
Short pick = highest short_score (v20 short model); Sg = -(fwd price return) bps.

De-dupe policies (2h cooldown from entry, resets each session):
  RAW      : top-1 short every gate-passing anchor (no de-dupe)
  SKIP     : take the short only if the top-1 pick is off cooldown, else NONE
             (user's request: no replacement — the slot just goes empty)
  REPLACE  : 1 short/anchor = highest-ranked short not entered in the last 2h

Gate applied FIRST (only gate-passing anchors are tradeable, so a vetoed anchor
consumes no cooldown), then de-dupe among the survivors:
  none        : full book
  idx2h>=0    : robust single-threshold index gate
  5-band      : the hand-picked positive bands
                [-0.05,+0.25) U (idx2h<-0.5) U (idx2h>=+0.5)

OOS 2026-06-16..07-09, 1h hold, net@6 = Sg - 6bps.
Run: python -m scripts.backtests.short_dedupe_2h_oos
"""
import os, sys, json, datetime as dt, warnings
import concurrent.futures as cf
sys.path.insert(0, os.getcwd()); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, xgboost as xgb, yfinance as yf
from scripts.tickers import TICKERS
from scripts.feature_utils import build_rolling_1h_ohlcv, compute_features

TODAY = dt.date.today(); OOS_FROM = dt.date(2026, 6, 16)
HOLD = pd.Timedelta(hours=1); COOL = pd.Timedelta(hours=2); COST6 = 6
NOTIONAL = 99517.68
CACHE = os.path.join(os.environ.get("TEMP", "/tmp"), "claude", "c--Users-loq-Desktop-Trading-finalgo", "today_top1_15m_cache")
INSTR = json.load(open("scripts/instrument_cache.json"))
v20 = json.load(open("models/research/v20_rolling_1h/metadata.json"))["features"]


def load(t):
    sym = t.replace(".NS", ""); p = os.path.join(CACHE, f"{sym}_{TODAY}.csv")
    if not os.path.exists(p) or sym not in INSTR: return t, None
    df = pd.read_csv(p, parse_dates=["timestamp"]).set_index("timestamp").rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
    return t, df[["Open", "High", "Low", "Close", "Volume"]].dropna()


bs = xgb.Booster(); bs.load_model("models/research/v20_rolling_1h/xgb_short_model.json")
feat, c1h = {}, {}
with cf.ThreadPoolExecutor(max_workers=8) as ex:
    for t, df in ex.map(load, TICKERS):
        if df is None or len(df) < 40: continue
        try:
            h = build_rolling_1h_ohlcv(df)
            if len(h) >= 20:
                feat[t] = compute_features(h[["Open", "High", "Low", "Close", "Volume"]].copy(), legacy=False)
                c1h[t] = h["Close"]
        except Exception:
            pass
anchors = sorted({ts for f in feat.values() for ts in f.index
                  if ts.date() >= OOS_FROM and "10:15" <= ts.strftime("%H:%M") <= "14:15"})

# per anchor: SHORT candidates [(ticker, Sg_bps), ...] desc by short_score
book = {}
for ts in anchors:
    fr, tks = [], []
    for t, f in feat.items():
        if ts in f.index and (ts + HOLD) in c1h[t].index:
            fr.append(f.loc[ts]); tks.append(t)
    if len(tks) < 10: continue
    X = pd.DataFrame(fr, index=tks)
    X["Market_Mean_Return"] = X["Return"].mean(); X["Relative_Return"] = X["Return"] - X["Market_Mean_Return"]
    X["Market_Mean_Volatility"] = X["HL_Range"].mean(); X["Relative_Volatility"] = X["HL_Range"] / (X["Market_Mean_Volatility"] + 1e-8)
    ss = bs.predict(xgb.DMatrix(np.nan_to_num(X[v20].values.astype(np.float32)), feature_names=v20))
    order = np.argsort(-ss)                        # highest short_score first
    cand = []
    for i in order:
        t = tks[i]; ep = float(c1h[t].loc[ts]); xp = float(c1h[t].loc[ts + HOLD])
        cand.append((t.replace(".NS", ""), -(xp - ep) / ep * 1e4))   # short P&L
    book[ts] = cand

# ── Nifty idx2h (gap-as-1h for early anchors) ────────────────────────────────
nf = yf.download("^NSEI", start="2026-06-15", end="2026-07-11", interval="15m", progress=False)
if isinstance(nf.columns, pd.MultiIndex): nf.columns = nf.columns.get_level_values(0)
nifty = nf["Close"].copy(); nifty.index = pd.to_datetime(nifty.index).tz_convert("Asia/Kolkata").tz_localize(None); nifty = nifty.sort_index()
sc = nifty.groupby(nifty.index.date).last(); ds = sorted(sc.index); pc = {d: sc[ds[i - 1]] for i, d in enumerate(ds) if i > 0}
def at(ts):
    s = nifty[nifty.index <= ts]; return float(s.iloc[-1]) if len(s) else np.nan
def idx2h(ts):
    now = at(ts); rt = ts - HOLD * 2
    ref = at(rt) if (rt.time() >= dt.time(9, 15) and rt.date() == ts.date()) else pc.get(ts.date(), np.nan)
    return (now / ref - 1) * 100 if ref and not np.isnan(ref) and not np.isnan(now) else np.nan

def gate_none(ts): return True
def gate_up(ts):   v = idx2h(ts); return (not np.isnan(v)) and v >= 0
def gate_5band(ts):
    v = idx2h(ts)
    if np.isnan(v): return False
    return (-0.05 <= v < 0.25) or (v < -0.5) or (v >= 0.5)


def simulate(policy, gate):
    """gate-first, then de-dupe; returns [(ts, ticker, Sg)]."""
    last = {}; cur_day = None; trades = []
    for ts in anchors:
        if ts not in book: continue
        if ts.date() != cur_day: last = {}; cur_day = ts.date()
        if not gate(ts): continue                      # anchor not tradeable -> no cooldown burned
        cand = book[ts]
        if policy == "RAW":
            pick = cand[0]
        elif policy == "SKIP":
            top = cand[0][0]
            if top in last and (ts - last[top]) < COOL: continue     # repeat -> empty slot
            pick = cand[0]
        else:  # REPLACE
            pick = next(((t, g) for t, g in cand if not (t in last and (ts - last[t]) < COOL)), None)
            if pick is None: continue
        t, g = pick; last[t] = ts; trades.append((ts, t, g))
    return trades


def rep(trades, label):
    if not trades:
        print(f"  {label:34s} n=  0"); return None
    df = pd.DataFrame(trades, columns=["ts", "ticker", "Sg"]); g = df.Sg
    net = g.mean() - COST6
    t = g.mean() / (g.std(ddof=1) / np.sqrt(len(df))) if len(df) > 1 and g.std() > 0 else 0
    rs = ((g - COST6) / 1e4 * NOTIONAL).sum()
    print(f"  {label:34s} n={len(df):3d}  WR={100*(g>0).mean():3.0f}%  net@6={net:+7.2f}  "
          f"CUM={(g-COST6).sum():+8.1f}bps={((g-COST6).sum()/100):+6.2f}%  bookRs={rs:+8.0f}  t/2~{t/2:+.1f}")
    return df


print("=" * 104)
print("  SHORT book: index gate  x  2h per-ticker de-dupe  (OOS 06-16..07-09, 1h hold)")
print("=" * 104)
for gname, gate in [("no gate (full)", gate_none), ("idx2h>=0", gate_up), ("5-band", gate_5band)]:
    print(f"\n-- gate: {gname} --")
    rep(simulate("RAW", gate),     "RAW top-1 (no de-dupe)")
    sk = rep(simulate("SKIP", gate), "DEDUPE-SKIP (no replacement)")
    rep(simulate("REPLACE", gate), "DEDUPE-REPLACE (next best)")

print("\n" + "=" * 104)
print("  DAILY EQUITY — DEDUPE-SKIP + idx2h>=0 gate  (the deployable short book)")
print("=" * 104)
sk = pd.DataFrame(simulate("SKIP", gate_up), columns=["ts", "ticker", "Sg"]); sk["date"] = sk.ts.dt.date.astype(str)
alld = sorted({ts.date().isoformat() for ts in anchors})
print(f"  {'date':12s} {'n':>3s} {'WR':>5s} {'avg net@6':>10s} {'day Rs':>9s} {'cum Rs':>9s}")
cum = 0
for day in alld:
    x = sk[sk.date == day]
    if len(x) == 0:
        print(f"  {day:12s} {0:3d} {'-':>5s} {'-':>10s} {0:+9.0f} {cum:+9.0f}"); continue
    net6 = x.Sg - COST6; rs = (net6 / 1e4 * NOTIONAL).sum(); cum += rs
    print(f"  {day:12s} {len(x):3d} {100*(x.Sg>0).mean():4.0f}% {net6.mean():+9.1f} {rs:+9.0f} {cum:+9.0f}")
tn = sk.Sg - COST6
print("  " + "-" * 56)
print(f"  TOTAL {len(sk)} trades | WR {100*(sk.Sg>0).mean():.0f}% | avg {tn.mean():+.1f} bps | "
      f"CUM {tn.sum()/100:+.2f}% | book Rs {(tn/1e4*NOTIONAL).sum():+,.0f}")
print("\nExploratory, no Gauntlet. Early anchors (<11:15) use gap-inclusive idx2h ref.")

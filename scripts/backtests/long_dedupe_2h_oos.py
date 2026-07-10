"""
LONG book with a 2-hour per-ticker de-dupe (shadow-trade realism)
=================================================================
Live behaviour: a ticker is entered at most ONCE per ~2h (no stacking the same
name across consecutive 15-min anchors). We re-score the full long ranking each
anchor and simulate two de-dupe policies vs the raw top-1 baseline:

  RAW       : top-1 long every anchor (no de-dupe)              -> baseline (-11.3)
  REPLACE   : 1 long/anchor = highest-ranked long NOT entered in the last 2h
  SKIP      : take the long only if the top-1 pick itself is off cooldown; else none

Cooldown = 2h from entry (resets each session). OOS 2026-06-16..07-09, 1h hold.
Also overlays the Nifty trailing-2h index gate (idx2h>=0) on the de-duped book.

Run: python -m scripts.backtests.long_dedupe_2h_oos
"""
import os, sys, json, datetime as dt, warnings
import concurrent.futures as cf
sys.path.insert(0, os.getcwd()); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, xgboost as xgb, yfinance as yf
from scripts.tickers import TICKERS
from scripts.feature_utils import build_rolling_1h_ohlcv, compute_features

TODAY = dt.date.today(); OOS_FROM = dt.date(2026, 6, 16)
HOLD = pd.Timedelta(hours=1); COOL = pd.Timedelta(hours=2); COST6 = 6
CACHE = os.path.join(os.environ.get("TEMP", "/tmp"), "claude", "c--Users-loq-Desktop-Trading-finalgo", "today_top1_15m_cache")
INSTR = json.load(open("scripts/instrument_cache.json"))
v20 = json.load(open("models/research/v20_rolling_1h/metadata.json"))["features"]


def load(t):
    sym = t.replace(".NS", ""); p = os.path.join(CACHE, f"{sym}_{TODAY}.csv")
    if not os.path.exists(p) or sym not in INSTR: return t, None
    df = pd.read_csv(p, parse_dates=["timestamp"]).set_index("timestamp").rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"})
    return t, df[["Open","High","Low","Close","Volume"]].dropna()


bl = xgb.Booster(); bl.load_model("models/research/v20_rolling_1h/xgb_long_model.json")
bs = xgb.Booster(); bs.load_model("models/research/v20_rolling_1h/xgb_short_model.json")
feat, c1h = {}, {}
with cf.ThreadPoolExecutor(max_workers=8) as ex:
    for t, df in ex.map(load, TICKERS):
        if df is None or len(df) < 40: continue
        try:
            h = build_rolling_1h_ohlcv(df)
            if len(h) >= 20: feat[t] = compute_features(h[["Open","High","Low","Close","Volume"]].copy(), legacy=False); c1h[t] = h["Close"]
        except Exception: pass
anchors = sorted({ts for f in feat.values() for ts in f.index if ts.date() >= OOS_FROM and "10:15" <= ts.strftime("%H:%M") <= "14:15"})

# per anchor: ranked long candidates [(ticker, fwd_bps), ...] desc by long_score
book = {}
for ts in anchors:
    fr, tks = [], []
    for t, f in feat.items():
        if ts in f.index and (ts+HOLD) in c1h[t].index: fr.append(f.loc[ts]); tks.append(t)
    if len(tks) < 10: continue
    X = pd.DataFrame(fr, index=tks)
    X["Market_Mean_Return"]=X["Return"].mean(); X["Relative_Return"]=X["Return"]-X["Market_Mean_Return"]
    X["Market_Mean_Volatility"]=X["HL_Range"].mean(); X["Relative_Volatility"]=X["HL_Range"]/(X["Market_Mean_Volatility"]+1e-8)
    ls = bl.predict(xgb.DMatrix(np.nan_to_num(X[v20].values.astype(np.float32)), feature_names=v20))
    order = np.argsort(-ls)
    cand = []
    for i in order:
        t = tks[i]; ep = float(c1h[t].loc[ts]); xp = float(c1h[t].loc[ts+HOLD])
        cand.append((t.replace(".NS",""), (xp-ep)/ep*1e4))
    book[ts] = cand

# ── Nifty idx2h ──────────────────────────────────────────────────────────────
nf = yf.download("^NSEI", start="2026-06-15", end="2026-07-11", interval="15m", progress=False)
if isinstance(nf.columns, pd.MultiIndex): nf.columns = nf.columns.get_level_values(0)
nifty = nf["Close"].copy(); nifty.index = pd.to_datetime(nifty.index).tz_convert("Asia/Kolkata").tz_localize(None); nifty = nifty.sort_index()
sc = nifty.groupby(nifty.index.date).last(); ds = sorted(sc.index); pc = {d: sc[ds[i-1]] for i,d in enumerate(ds) if i>0}
def at(ts):
    s = nifty[nifty.index <= ts]; return float(s.iloc[-1]) if len(s) else np.nan
def idx2h(ts):
    now = at(ts); rt = ts - HOLD*2; ref = at(rt) if (rt.time() >= dt.time(9,15) and rt.date()==ts.date()) else pc.get(ts.date(), np.nan)
    return (now/ref-1)*100 if ref and not np.isnan(ref) and not np.isnan(now) else np.nan


def simulate(policy):
    """returns list of (ts, ticker, fwd_bps)."""
    last = {}; cur_day = None; trades = []
    for ts in anchors:
        if ts not in book: continue
        if ts.date() != cur_day: last = {}; cur_day = ts.date()
        cand = book[ts]
        if policy == "RAW":
            pick = cand[0]
        elif policy == "SKIP":
            top, _ = cand[0]
            if top in last and (ts - last[top]) < COOL: continue     # top-1 cooled -> no long
            pick = cand[0]
        else:  # REPLACE
            pick = next(((t, g) for t, g in cand if not (t in last and (ts - last[t]) < COOL)), None)
            if pick is None: continue
        t, g = pick; last[t] = ts; trades.append((ts, t, g))
    return trades


NOTIONAL = 99517.68
def rep(trades, label, gate=None):
    df = pd.DataFrame(trades, columns=["ts", "ticker", "fwd"])
    if gate is not None:
        df = df[df.ts.map(idx2h) >= gate]
    g = df.fwd; net = g.mean() - COST6
    t = g.mean() / (g.std(ddof=1)/np.sqrt(len(df))) if len(df) > 1 and g.std() > 0 else 0
    rs = ((g - COST6) / 1e4 * NOTIONAL).sum()
    print(f"  {label:32s} n={len(df):3d}  WR={100*(g>0).mean():3.0f}%  net@6={net:+7.2f} bps  t/2~{t/2:+.1f}  bookRs={rs:+8.0f}")
    return df


print("=" * 88)
print("  LONG book: 2-hour per-ticker de-dupe vs raw top-1  (OOS 06-16..07-09, 1h hold)")
print("=" * 88)
raw = simulate("RAW"); rpl = simulate("REPLACE"); sk = simulate("SKIP")
print("\n-- FULL long book (all anchors) --")
rep(raw, "RAW top-1 (baseline)")
d_rpl = rep(rpl, "DEDUPE-REPLACE (next best)")
rep(sk, "DEDUPE-SKIP (skip repeats)")
for G in (0.0, 0.3):
    print(f"\n-- with Nifty idx2h >= +{G:.1f}% gate on top --")
    rep(raw, "RAW + gate", gate=G)
    rep(rpl, "DEDUPE-REPLACE + gate", gate=G)
    rep(sk, "DEDUPE-SKIP + gate", gate=G)

print("\n" + "=" * 88)
print("  DAILY RETURNS — DEDUPE-SKIP + idx2h>=+0.3% gate")
print("=" * 88)
sd = pd.DataFrame(sk, columns=["ts", "ticker", "fwd"])
sd = sd[sd.ts.map(idx2h) >= 0.3].copy()
sd["date"] = sd.ts.dt.date.astype(str)
alld = sorted({ts.date().isoformat() for ts in anchors})
print(f"  {'date':12s} {'nLong':>5s} {'WR':>5s} {'avg net@6':>10s} {'day Rs':>8s} {'cum Rs':>8s}")
cum = 0
for day in alld:
    x = sd[sd.date == day]
    if len(x) == 0:
        print(f"  {day:12s} {0:5d} {'-':>5s} {'-':>10s} {0:+8.0f} {cum:+8.0f}"); continue
    net6 = x.fwd - COST6; rs = (net6 / 1e4 * NOTIONAL).sum(); cum += rs
    print(f"  {day:12s} {len(x):5d} {100*(x.fwd>0).mean():4.0f}% {net6.mean():+9.1f} {rs:+8.0f} {cum:+8.0f}")
tn = sd.fwd - COST6
print("  " + "-" * 52)
print(f"  TOTAL {len(sd)} trades | WR {100*(sd.fwd>0).mean():.0f}% | avg {tn.mean():+.1f} bps | book Rs {(tn/1e4*NOTIONAL).sum():+,.0f}")

print("\n" + "=" * 88)
print("  Did de-dupe fix the two loser days?  (per-day net@6, REPLACE policy)")
print("=" * 88)
draw = pd.DataFrame(raw, columns=["ts","ticker","fwd"]); draw["date"]=draw.ts.dt.date.astype(str)
drpl = d_rpl.copy(); drpl["date"]=drpl.ts.dt.date.astype(str)
for day in ["2026-06-16","2026-06-22"]:
    a = draw[draw.date==day]; b = drpl[drpl.date==day]
    print(f"  {day}: RAW n={len(a)} net@6={a.fwd.mean()-6:+7.1f}  ->  DEDUPE n={len(b)} net@6={b.fwd.mean()-6:+7.1f}")
    print(f"     RAW picks : {list(a.ticker)}")
    print(f"     DEDUPE    : {list(b.ticker)}")

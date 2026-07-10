"""
Monthly and 17-day breakdown for the SHORT model
=================================================
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
df = df[df.Next_Hour_Return.abs() <= 0.20]

# ── score short model ────────────────────────────────────────────────────────
bs = xgb.Booster(); bs.load_model("models/research/v20_rolling_1h/xgb_short_model.json")
X = xgb.DMatrix(np.nan_to_num(df[v20].values.astype(np.float32)), feature_names=v20)
df["ss"] = bs.predict(X)
df["retbps"] = df.Next_Hour_Return * 1e4

# per-anchor ranked candidate lists
short_book = {}
for ts, g in df.groupby("DateTime"):
    gs = g.sort_values("ss", ascending=False)
    short_book[ts] = list(zip(gs.Ticker.values, (-gs.retbps).values)) # short pnl = -ret
anchors = sorted(short_book)
trading_days = sorted(list(set(ts.date() for ts in anchors)))
last_17_days = trading_days[-17:]

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

def g_none(ts): return True
def g_5band(ts):
    v = idx2h(ts)
    if np.isnan(v): return False
    return (-0.05 <= v < 0.25) or (v < -0.5) or (v >= 0.5)

def simulate(bk, K, policy, gate):
    last = {}; cur = None; trades = []
    for ts in anchors:
        if ts.date() != cur: last = {}; cur = ts.date()
        if not gate(ts): continue
        cand = bk[ts]
        if policy == "SKIP":
            picks = [(t, g) for (t, g) in cand[:K] if not (t in last and (ts - last[t]) < COOL)]
        else: # RAW
            picks = cand[:K]
        for t, g in picks: last[t] = ts; trades.append((ts, t, g))
    return trades

def print_analysis(trades, label):
    if not trades: return
    td = pd.DataFrame(trades, columns=["ts", "tk", "pnl"])
    td["net6"] = td.pnl - COST
    td["date"] = td.ts.dt.date
    td["month"] = td.ts.dt.to_period("M").astype(str)
    td["bookRs"] = td.net6 / 1e4 * NOTIONAL
    
    print(f"\n{'='*90}")
    print(f"  SHORT MODEL — {label} (K=1, SKIP Policy)")
    print(f"{'='*90}")
    
    # overall
    n = len(td); wr = 100*(td.pnl>0).mean(); net = td.net6.mean()
    tot_rs = td.bookRs.sum()
    print(f"  OVERALL 11 MONTHS: n={n} | WR={wr:.1f}% | net@6={net:+.2f} | Rs {tot_rs:+,.0f}")
    
    # 17 days
    td17 = td[td.date.isin(last_17_days)]
    if len(td17) > 0:
        n17 = len(td17); wr17 = 100*(td17.pnl>0).mean(); net17 = td17.net6.mean()
        rs17 = td17.bookRs.sum()
        print(f"  LAST 17 TRADING DAYS ({last_17_days[0]} to {last_17_days[-1]}):")
        print(f"  -> n={n17} | WR={wr17:.1f}% | net@6={net17:+.2f} | Rs {rs17:+,.0f}")
    else:
        print(f"  LAST 17 TRADING DAYS: 0 trades")
        
    print("\n  MONTHLY ROLL-UP:")
    print(f"  {'month':10s}  {'n':>5s}  {'WR':>6s}  {'net@6':>8s}  {'bookRs':>10s}")
    for mo, mdf in td.groupby("month"):
        mn = len(mdf); mwr = 100*(mdf.pnl>0).mean(); mnet = mdf.net6.mean(); mrs = mdf.bookRs.sum()
        print(f"  {mo:10s}  {mn:5d}  {mwr:5.1f}%  {mnet:+8.2f}  {mrs:+10,.0f}")

print_analysis(simulate(short_book, 1, "SKIP", g_none), "NO GATE")
print_analysis(simulate(short_book, 1, "SKIP", g_5band), "5-BAND GATE")

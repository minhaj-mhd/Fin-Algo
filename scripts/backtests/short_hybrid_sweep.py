"""
Sweep: Rank by Shadow Conviction, Filter by Raw SS
===================================================
"""
import os, sys, json, datetime as dt, warnings
sys.path.insert(0, os.getcwd()); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, xgboost as xgb
import pyarrow.compute as pc, pyarrow.dataset as pads

PANEL = "data/research/v20_rolling_1h/panel.parquet"
TEST_FROM, TEST_TO = pd.Timestamp("2025-08-01"), pd.Timestamp("2026-06-05")
HOLD = pd.Timedelta(hours=1); COOL = pd.Timedelta(hours=2); COST = 6
NOTIONAL = 99517.68
v20 = json.load(open("models/research/v20_rolling_1h/metadata.json"))["features"]

cols = ["DateTime", "Ticker", "Next_Hour_Return"] + [f for f in v20]
cols = list(dict.fromkeys(cols))
dset = pads.dataset(PANEL)
tb = dset.to_table(columns=cols, filter=(pc.field("DateTime") >= TEST_FROM) & (pc.field("DateTime") < TEST_TO))
df = tb.to_pandas()
df["tod"] = df.DateTime.dt.strftime("%H:%M")
df = df[(df.tod >= "10:15") & (df.tod <= "14:15")].copy()
df = df[df.Next_Hour_Return.abs() <= 0.20]

bl = xgb.Booster(); bl.load_model("models/research/v20_rolling_1h/xgb_long_model.json")
bs = xgb.Booster(); bs.load_model("models/research/v20_rolling_1h/xgb_short_model.json")
X = xgb.DMatrix(np.nan_to_num(df[v20].values.astype(np.float32)), feature_names=v20)
df["ls"] = bl.predict(X)
df["ss"] = bs.predict(X)
df["retbps"] = df.Next_Hour_Return * 1e4

short_book = {}
for ts, g in df.groupby("DateTime"):
    l_centered = g.ls.values - np.mean(g.ls.values)
    s_centered = g.ss.values - np.mean(g.ss.values)
    short_conviction = s_centered - l_centered
    
    g_new = pd.DataFrame({
        "Ticker": g.Ticker.values,
        "retbps": g.retbps.values,
        "short_conviction": short_conviction,
        "ss": g.ss.values
    })
    
    # rank by shadow conviction
    gs = g_new.sort_values("short_conviction", ascending=False)
    short_book[ts] = list(zip(gs.Ticker.values, (-gs.retbps).values, gs.short_conviction.values, gs.ss.values))

anchors = sorted(short_book)

def simulate(bk, policy, min_ss=None):
    last = {}; cur = None; trades = []
    for ts in anchors:
        if ts.date() != cur: last = {}; cur = ts.date()
        cand = bk[ts]
        
        # apply the raw ss filter 
        if min_ss is not None:
            cand = [(t, r, sc, ss) for (t, r, sc, ss) in cand if ss >= min_ss]
            if not cand: continue
            
        if policy == "SKIP":
            picks = [(t, r, sc, ss) for (t, r, sc, ss) in cand[:1] if not (t in last and (ts - last[t]) < COOL)]
        else:
            picks = cand[:1]
            
        for t, r, sc, ss in picks: 
            last[t] = ts
            trades.append((ts, t, r, sc))
    return trades

def eval_trades(trades):
    if not trades: return 0, 0, 0, 0, 0
    d = pd.DataFrame(trades, columns=["ts", "tk", "pnl", "sc"])
    d["net6"] = d.pnl - COST
    n = len(d); wr = 100*(d.pnl>0).mean(); net = d.net6.mean()
    rs = (d.net6 / 1e4 * NOTIONAL).sum()
    d = d.sort_values("ts").reset_index(drop=True)
    d["cum_rs"] = (d.net6 / 1e4 * NOTIONAL).cumsum()
    max_dd_rs = (d.cum_rs - d.cum_rs.cummax()).min()
    return n, wr, net, rs, max_dd_rs

print(f"{'='*90}")
print(f"  HYBRID TEST: Rank by Shadow Conviction, Filter by Raw SS")
print(f"{'='*90}")
print(f"  {'Min SS':>10s}  |  {'n':>5s}  {'WR':>5s}  {'net@6':>8s}  {'BookRs':>10s}  {'MaxDD':>10s}")

for p in [0.0, 0.05, 0.06, 0.07, 0.0762, 0.08, 0.0826, 0.09]:
    n, wr, net, rs, dd = eval_trades(simulate(short_book, "SKIP", min_ss=p))
    print(f"  {p:10.4f}  |  {n:5d}  {wr:4.0f}%  {net:+8.2f}  {rs:+10,.0f}  {dd:+10,.0f}")

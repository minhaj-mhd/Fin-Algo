"""
Sweep idx2h thresholds to find the optimal long gate
====================================================
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

# ── score deployed models ────────────────────────────────────────────────────
bl = xgb.Booster(); bl.load_model("models/research/v20_rolling_1h/xgb_long_model.json")
X = xgb.DMatrix(np.nan_to_num(df[v20].values.astype(np.float32)), feature_names=v20)
df["ls"] = bl.predict(X)
df["retbps"] = df.Next_Hour_Return * 1e4

# per-anchor ranked candidate lists
long_book = {}
for ts, g in df.groupby("DateTime"):
    gl = g.sort_values("ls", ascending=False)
    long_book[ts] = list(zip(gl.Ticker.values, gl.retbps.values))
anchors = sorted(long_book)

# ── Nifty idx2h ──────────────────────────────────────────────────────────────
nf = pd.read_csv(NIFTY, parse_dates=["ts"]).set_index("ts").close.sort_index()
sc = nf.groupby(nf.index.date).last(); ds = sorted(sc.index)
pc_ = {d: sc[ds[i-1]] for i, d in enumerate(ds) if i > 0}
def nat(ts):
    s = nf[nf.index <= ts]; return float(s.iloc[-1]) if len(s) else np.nan
_idx = {}
def idx2h(ts):
    if ts in _idx: return _idx[ts]
    now = nat(ts); rt = ts - HOLD * 2
    ref = nat(rt) if (rt.time() >= dt.time(9, 15) and rt.date() == ts.date()) else pc_.get(ts.date(), np.nan)
    v = (now / ref - 1) * 100 if ref and not np.isnan(ref) and not np.isnan(now) else np.nan
    _idx[ts] = v; return v

def simulate(bk, K, policy, thr):
    last = {}; cur = None; trades = []
    for ts in anchors:
        if ts.date() != cur: last = {}; cur = ts.date()
        
        # apply gate inline
        v = idx2h(ts)
        if np.isnan(v) or v < thr: continue
            
        cand = bk[ts]
        if policy == "SKIP":
            picks = [(t, g) for (t, g) in cand[:K] if not (t in last and (ts - last[t]) < COOL)]
        else: # RAW
            picks = cand[:K]
            
        for t, g in picks: last[t] = ts; trades.append((ts, t, g))
    return trades

def eval_trades(trades):
    if not trades: return 0, 0, 0, 0, 0, 0
    d = pd.DataFrame(trades, columns=["ts", "tk", "pnl"])
    d["net6"] = d.pnl - COST
    n = len(d); wr = 100*(d.pnl>0).mean(); net = d.net6.mean()
    rs = (d.net6 / 1e4 * NOTIONAL).sum()
    
    # drawdown
    d = d.sort_values("ts").reset_index(drop=True)
    d["cum_rs"] = d.bookRs = (d.net6 / 1e4 * NOTIONAL)
    d["cum_rs"] = d.bookRs.cumsum()
    max_dd_rs = (d.cum_rs - d.cum_rs.cummax()).min()
    
    # daily t_stat
    dd = d.net6.groupby(d.ts.dt.date).mean()
    td = dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd))) if len(dd) > 1 and dd.std() > 0 else 0
    return n, wr, net, rs, max_dd_rs, td

print(f"{'='*100}")
print(f"  IDX2H THRESHOLD SWEEP: LONG GATE (SKIP POLICY)")
print(f"{'='*100}")
print(f"  {'Thr':>5s}  |  {'n':>4s}  {'WR':>4s}  {'net@6':>6s}  {'BookRs':>8s}  {'MaxDD':>7s}  {'t_d':>5s}  |  {'n':>5s}  {'WR':>4s}  {'net@6':>6s}  {'BookRs':>8s}  {'MaxDD':>8s}  {'t_d':>5s}")
print(f"  {'':>5s}  |  {'--- Top-1 (Full) ---':^43s}  |  {'--- Top-5 (5-Slot) ---':^45s}")

thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
for thr in thresholds:
    n1, wr1, net1, rs1, dd1, td1 = eval_trades(simulate(long_book, 1, "SKIP", thr))
    n5, wr5, net5, rs5, dd5, td5 = eval_trades(simulate(long_book, 5, "SKIP", thr))
    
    s1 = f"{n1:4d}  {wr1:3.0f}%  {net1:+6.2f}  {rs1:+8,.0f}  {dd1:+7,.0f}  {td1:+5.1f}"
    s5 = f"{n5:5d}  {wr5:3.0f}%  {net5:+6.2f}  {rs5:+8,.0f}  {dd5:+8,.0f}  {td5:+5.1f}"
    print(f"  {thr:+.2f}  |  {s1}  |  {s5}")

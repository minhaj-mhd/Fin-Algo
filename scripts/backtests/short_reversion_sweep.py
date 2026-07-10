"""
Sweep short model mean-reversion gates and model score thresholds
=================================================================
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

# ── load test slice
cols = ["DateTime", "Ticker", "Next_Hour_Return"] + [f for f in v20]
cols = list(dict.fromkeys(cols))
dset = pads.dataset(PANEL)
tb = dset.to_table(columns=cols, filter=(pc.field("DateTime") >= TEST_FROM) & (pc.field("DateTime") < TEST_TO))
df = tb.to_pandas()
df["tod"] = df.DateTime.dt.strftime("%H:%M")
df = df[(df.tod >= "10:15") & (df.tod <= "14:15")].copy()
df = df[df.Next_Hour_Return.abs() <= 0.20]

# ── score short model
bs = xgb.Booster(); bs.load_model("models/research/v20_rolling_1h/xgb_short_model.json")
X = xgb.DMatrix(np.nan_to_num(df[v20].values.astype(np.float32)), feature_names=v20)
df["ss"] = bs.predict(X)
df["retbps"] = df.Next_Hour_Return * 1e4

# per-anchor ranked candidate lists, storing the model score (ss) as well
short_book = {}
for ts, g in df.groupby("DateTime"):
    gs = g.sort_values("ss", ascending=False)
    short_book[ts] = list(zip(gs.Ticker.values, (-gs.retbps).values, gs.ss.values))
anchors = sorted(short_book)

# ── Nifty idx2h
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

def simulate(bk, policy, idx_gate_min, min_score=None):
    last = {}; cur = None; trades = []
    for ts in anchors:
        if ts.date() != cur: last = {}; cur = ts.date()
        
        v = idx2h(ts)
        if np.isnan(v) or v < idx_gate_min: continue
        
        cand = bk[ts]
        # filter candidates by min_score if provided
        if min_score is not None:
            cand = [(t, g, s) for (t, g, s) in cand if s >= min_score]
            if not cand: continue
            
        if policy == "SKIP":
            picks = [(t, g, s) for (t, g, s) in cand[:1] if not (t in last and (ts - last[t]) < COOL)]
        else:
            picks = cand[:1]
            
        for t, g, s in picks: 
            last[t] = ts
            trades.append((ts, t, g, s))
    return trades

def eval_trades(trades):
    if not trades: return 0, 0, 0, 0, 0
    d = pd.DataFrame(trades, columns=["ts", "tk", "pnl", "score"])
    d["net6"] = d.pnl - COST
    n = len(d); wr = 100*(d.pnl>0).mean(); net = d.net6.mean()
    rs = (d.net6 / 1e4 * NOTIONAL).sum()
    d = d.sort_values("ts").reset_index(drop=True)
    d["cum_rs"] = (d.net6 / 1e4 * NOTIONAL).cumsum()
    max_dd_rs = (d.cum_rs - d.cum_rs.cummax()).min()
    return n, wr, net, rs, max_dd_rs

print(f"{'='*90}")
print(f"  SHORT MEAN-REVERSION: IDX2H UPSIDE GATE SWEEP (SKIP POLICY, Top-1)")
print(f"{'='*90}")
print(f"  Only shorting when index is up >= X% (mean reversion)")
print(f"  {'idx2h >=':>10s}  |  {'n':>5s}  {'WR':>5s}  {'net@6':>8s}  {'BookRs':>10s}  {'MaxDD':>10s}")
for thr in [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8]:
    n, wr, net, rs, dd = eval_trades(simulate(short_book, "SKIP", thr, min_score=None))
    print(f"  {thr:+10.2f}  |  {n:5d}  {wr:4.0f}%  {net:+8.2f}  {rs:+10,.0f}  {dd:+10,.0f}")

print(f"\n{'='*90}")
print(f"  SHORT CONVICTION SWEEP: MINIMUM SCORE (NO IDX2H GATE)")
print(f"{'='*90}")
print(f"  Filtering out low-conviction top picks")
print(f"  {'Min Score':>10s}  |  {'n':>5s}  {'WR':>5s}  {'net@6':>8s}  {'BookRs':>10s}  {'MaxDD':>10s}")
# find quantiles of the top-1 scores to set realistic sweep range
all_top_scores = [cand[0][2] for ts, cand in short_book.items() if cand]
pcts = np.percentile(all_top_scores, [25, 50, 75, 80, 85, 90, 95])
for p in pcts:
    n, wr, net, rs, dd = eval_trades(simulate(short_book, "SKIP", -999, min_score=p))
    print(f"  {p:10.4f}  |  {n:5d}  {wr:4.0f}%  {net:+8.2f}  {rs:+10,.0f}  {dd:+10,.0f}")

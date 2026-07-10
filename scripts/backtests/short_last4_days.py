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

bs = xgb.Booster(); bs.load_model("models/research/v20_rolling_1h/xgb_short_model.json")
X = xgb.DMatrix(np.nan_to_num(df[v20].values.astype(np.float32)), feature_names=v20)
df["ss"] = bs.predict(X)
df["retbps"] = df.Next_Hour_Return * 1e4

short_book = {}
for ts, g in df.groupby("DateTime"):
    gs = g.sort_values("ss", ascending=False)
    short_book[ts] = list(zip(gs.Ticker.values, (-gs.retbps).values, gs.ss.values))
anchors = sorted(short_book)

last = {}; cur = None; trades = []
for ts in anchors:
    if ts.date() != cur: last = {}; cur = ts.date()
    cand = short_book[ts]
    
    # filter candidates by 0.0826
    cand = [(t, r, s) for (t, r, s) in cand if s >= 0.0826]
    if not cand: continue
        
    picks = [(t, r, s) for (t, r, s) in cand[:1] if not (t in last and (ts - last[t]) < COOL)]
    for t, r, s in picks: 
        last[t] = ts
        trades.append((ts, t, r, s))

if trades:
    td = pd.DataFrame(trades, columns=["ts", "tk", "pnl", "score"])
    td["net6"] = td.pnl - COST
    td["bookRs"] = td.net6 / 1e4 * NOTIONAL
    td["date"] = td.ts.dt.date
    
    dates = sorted(td.date.unique())
    last_4 = dates[-4:]
    
    print("="*60)
    print("  SHORT MODEL (Min SS=0.0826) - LAST 4 TRADING DAYS")
    print("="*60)
    for d in last_4:
        d_trades = td[td.date == d]
        n = len(d_trades)
        net = d_trades.net6.mean()
        rs = d_trades.bookRs.sum()
        tickers = ", ".join(d_trades.tk.tolist())
        print(f"  {d}: n={n} | net@6={net:+.2f} | Rs {rs:+,.0f} | {tickers}")
else:
    print("No trades found.")

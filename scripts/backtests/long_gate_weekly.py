"""
LONG idx2h>=0.5 gate — WEEK-BY-WEEK breakdown on the 80/20 untouched test
==========================================================================
Top-1 (K=1), 15-min anchors 10:15–14:15, all 3 de-dupe policies.
Run: python -m scripts.backtests.long_gate_weekly
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
tb = dset.to_table(columns=cols,
                   filter=(pc.field("DateTime") >= TEST_FROM) & (pc.field("DateTime") < TEST_TO))
df = tb.to_pandas()
df["tod"] = df.DateTime.dt.strftime("%H:%M")
df = df[(df.tod >= "10:15") & (df.tod <= "14:15")].copy()
df = df[df.Next_Hour_Return.abs() <= 0.20]
print(f"test rows {len(df):,} | {df.DateTime.min()} -> {df.DateTime.max()} | {df.DateTime.dt.date.nunique()} days")

# ── score deployed models ────────────────────────────────────────────────────
bl = xgb.Booster(); bl.load_model("models/research/v20_rolling_1h/xgb_long_model.json")
X = xgb.DMatrix(np.nan_to_num(df[v20].values.astype(np.float32)), feature_names=v20)
df["ls"] = bl.predict(X)
df["retbps"] = df.Next_Hour_Return * 1e4

# per-anchor ranked candidate lists (long: pnl = +ret)
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

def g_up5(ts):
    v = idx2h(ts); return (not np.isnan(v)) and v >= 0.5

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

# ── run all 3 policies x both book sizes ─────────────────────────────────────
for K in [1, 5]:
    book_lbl = f"Top-{K}"
    for pol in ["RAW", "SKIP", "REPLACE"]:
        trades = simulate(long_book, K, pol, g_up5)
        if not trades:
            print(f"\n{pol}: 0 trades"); continue
    td = pd.DataFrame(trades, columns=["ts", "tk", "pnl"])
    td["net6"] = td.pnl - COST
    td["week"] = td.ts.dt.isocalendar().year.astype(str) + "-W" + td.ts.dt.isocalendar().week.astype(str).str.zfill(2)
    td["date"] = td.ts.dt.date
    td["bookRs"] = td.net6 / 1e4 * NOTIONAL

    # aggregate totals
    tot_n = len(td); tot_wr = 100*(td.pnl > 0).mean(); tot_net = td.net6.mean()
    tot_cum = td.net6.sum() / 100; tot_rs = td.bookRs.sum()
    print(f"\n{'='*110}")
    print(f"  LONG idx2h>=0.5 — Top-1, {pol}   |  TOTAL n={tot_n}  WR={tot_wr:.0f}%  net@6={tot_net:+.2f}  CUM={tot_cum:+.2f}%  bookRs={tot_rs:+,.0f}")
    print(f"{'='*110}")
    print(f"  {'week':10s}  {'n':>4s}  {'WR':>5s}  {'gross':>7s}  {'net@6':>7s}  {'CUM%':>8s}  {'bookRs':>10s}  {'days':>4s}  tickers")

    wg = td.groupby("week")
    cum = 0
    for wk, wdf in sorted(wg):
        n = len(wdf); wr = 100*(wdf.pnl > 0).mean()
        gross = wdf.pnl.mean(); net = wdf.net6.mean()
        cum += wdf.net6.sum() / 100
        rs = wdf.bookRs.sum()
        ndays = wdf.date.nunique()
        tks = ", ".join(wdf.tk.value_counts().head(3).index)
        print(f"  {wk:10s}  {n:4d}  {wr:4.0f}%  {gross:+7.2f}  {net:+7.2f}  {cum:+8.2f}%  {rs:+10,.0f}  {ndays:4d}  {tks}")

    # ── drawdown analysis ──────────────────────────────────────────────────────
    td_sorted = td.sort_values("ts").reset_index(drop=True)
    td_sorted["cum_rs"] = td_sorted.bookRs.cumsum()
    td_sorted["hwm"] = td_sorted.cum_rs.cummax()
    td_sorted["dd_rs"] = td_sorted.cum_rs - td_sorted.hwm  # always <= 0
    td_sorted["cum_pct"] = td_sorted.net6.cumsum() / 100
    td_sorted["hwm_pct"] = td_sorted.cum_pct.cummax()
    td_sorted["dd_pct"] = td_sorted.cum_pct - td_sorted.hwm_pct

    max_dd_rs = td_sorted.dd_rs.min()
    max_dd_pct = td_sorted.dd_pct.min()
    max_dd_idx = td_sorted.dd_rs.idxmin()
    max_dd_ts = td_sorted.loc[max_dd_idx, "ts"]
    # find the peak before the max drawdown
    peak_idx = td_sorted.loc[:max_dd_idx, "cum_rs"].idxmax()
    peak_ts = td_sorted.loc[peak_idx, "ts"]

    print(f"\n  {'-'*80}")
    print(f"  DRAWDOWN ANALYSIS")
    print(f"  {'-'*80}")
    print(f"  Max Drawdown:  {max_dd_pct:+.2f}%  /  Rs {max_dd_rs:+,.0f}")
    print(f"  Peak:          {peak_ts}  (Rs {td_sorted.loc[peak_idx, 'cum_rs']:+,.0f})")
    print(f"  Trough:        {max_dd_ts}  (Rs {td_sorted.loc[max_dd_idx, 'cum_rs']:+,.0f})")
    peak_hwm = td_sorted.loc[peak_idx, "cum_rs"]
    if peak_hwm > 0:
        print(f"  DD / Peak:     {100 * max_dd_rs / peak_hwm:.1f}%")

    # per-week drawdown
    td_sorted["week2"] = td_sorted.ts.dt.isocalendar().year.astype(str) + "-W" + td_sorted.ts.dt.isocalendar().week.astype(str).str.zfill(2)
    print(f"\n  Per-week max drawdown (from running equity curve):")
    print(f"  {'week':10s}  {'wk_dd_Rs':>10s}  {'wk_dd_%':>8s}  {'equity_end':>12s}")
    for wk2, wdf2 in td_sorted.groupby("week2"):
        wk_dd_rs = wdf2.dd_rs.min()
        wk_dd_pct = wdf2.dd_pct.min()
        eq_end = wdf2.cum_rs.iloc[-1]
        print(f"  {wk2:10s}  {wk_dd_rs:+10,.0f}  {wk_dd_pct:+8.2f}%  {eq_end:+12,.0f}")

    # consecutive losing trades
    streak = 0; max_streak = 0; streak_loss = 0; max_streak_loss = 0
    for _, r in td_sorted.iterrows():
        if r.net6 < 0:
            streak += 1; streak_loss += r.bookRs
            if streak > max_streak: max_streak = streak; max_streak_loss = streak_loss
        else:
            streak = 0; streak_loss = 0
    print(f"\n  Max consecutive losers: {max_streak} trades (Rs {max_streak_loss:+,.0f})")
    print(f"  Final equity:          Rs {td_sorted.cum_rs.iloc[-1]:+,.0f}")

    # monthly summary
    td["month"] = td.ts.dt.to_period("M").astype(str)
    print(f"\n  Monthly roll-up:")
    print(f"  {'month':8s}  {'n':>4s}  {'WR':>5s}  {'net@6':>7s}  {'CUM%':>8s}  {'bookRs':>10s}")
    cum = 0
    for mo, mdf in td.groupby("month"):
        n = len(mdf); wr = 100*(mdf.pnl > 0).mean()
        net = mdf.net6.mean(); cum += mdf.net6.sum()/100; rs = mdf.bookRs.sum()
        print(f"  {mo:8s}  {n:4d}  {wr:4.0f}%  {net:+7.2f}  {cum:+8.2f}%  {rs:+10,.0f}")

print("\nDeployed v20 on 80/20 untouched test. idx2h>=0.5 long gate. Exploratory, no Gauntlet.")

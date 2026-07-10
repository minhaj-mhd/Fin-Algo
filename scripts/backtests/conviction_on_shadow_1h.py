"""
Conviction-selection rule applied to the ACTUAL live shadow trades (ground truth)
=================================================================================
Cross-check fix: instead of reconstructing fills from fresh candles (which uses
the :15 candle-close as entry and diverges from live scan-time fills by ~20-40bps
on shorts), take the REAL logged shadow/live trades from data/vanguard_trades.db
(realized 1h gross = final_profit_pct) and attach each pick's CONVICTION score,
recomputed with live parity (model_inference centered long-short spread) from the
universe scoring at that pick's 15-min anchor.

Then apply the user's rule on the real book:
  BAND: keep only 0.011 <= conviction <= 0.04
  SELECT-ONE : take only the higher-conviction eligible side per anchor
  BOTH-if-SIM: both if |cL-cS|/max <= SIM (20% & 15%), else the higher
  ALWAYS-BOTH: every eligible side (band only)

Reports per-day and per-anchor on REAL P&L. Only anchors the live system actually
traded are included (07-08 was a partial-day outage -> fewer signals).

Run: python -m scripts.backtests.conviction_on_shadow_1h
"""
import os, sys, json, urllib.parse, datetime as dt, warnings, sqlite3
import concurrent.futures as cf
sys.path.insert(0, os.getcwd()); warnings.filterwarnings("ignore")
import requests, numpy as np, pandas as pd, xgboost as xgb
from scripts.tickers import TICKERS
from scripts.feature_utils import build_rolling_1h_ohlcv, compute_features

TODAY = dt.date.today()
DAYS  = ["2026-07-03", "2026-07-06", "2026-07-07", "2026-07-08", "2026-07-09"]
ENTRY_FROM, ENTRY_TO = "10:15", "14:15"
CONV_LO, CONV_HI = 0.011, 0.04
SIM_PRIMARY, SIM_TIGHT = 0.20, 0.15
COST6, COST10 = 6, 10
V20_META = "models/research/v20_rolling_1h/metadata.json"
CACHE = os.path.join(os.environ.get("TEMP", "/tmp"), "claude",
                     "c--Users-loq-Desktop-Trading-finalgo", "today_top1_15m_cache")
HDR = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Accept": "application/json"}
INSTR = json.load(open("scripts/instrument_cache.json"))


def fetch_15m(ticker):
    sym = ticker.replace(".NS", ""); ik = INSTR.get(sym)
    if not ik: return ticker, None
    cache_f = os.path.join(CACHE, f"{sym}_{TODAY}.csv")
    if os.path.exists(cache_f):
        df = pd.read_csv(cache_f, parse_dates=["timestamp"])
    else:
        enc = urllib.parse.quote(ik, safe=""); frm = TODAY - dt.timedelta(days=25); candles = []
        try:
            candles += requests.get(f"https://api.upstox.com/v3/historical-candle/{enc}/minutes/15/{TODAY}/{frm}", headers=HDR, timeout=25).json().get("data", {}).get("candles", [])
            candles += requests.get(f"https://api.upstox.com/v3/historical-candle/intraday/{enc}/minutes/15", headers=HDR, timeout=25).json().get("data", {}).get("candles", [])
        except Exception: return ticker, None
        if not candles: return ticker, None
        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        df = df.drop_duplicates("timestamp", keep="last").sort_values("timestamp"); df.to_csv(cache_f, index=False)
    df = df.set_index("timestamp").rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
    return ticker, df[["Open", "High", "Low", "Close", "Volume"]].dropna()


v20_feats = json.load(open(V20_META))["features"]
bst_l = xgb.Booster(); bst_l.load_model("models/research/v20_rolling_1h/xgb_long_model.json")
bst_s = xgb.Booster(); bst_s.load_model("models/research/v20_rolling_1h/xgb_short_model.json")
print("Fetching candles + building features (cache reuse)...")
raw15 = {}
with cf.ThreadPoolExecutor(max_workers=8) as ex:
    for tk, df in ex.map(fetch_15m, TICKERS):
        if df is not None and len(df) >= 40: raw15[tk] = df
feat = {}
for tk, df15 in raw15.items():
    try:
        h1 = build_rolling_1h_ohlcv(df15)
        if len(h1) >= 20: feat[tk] = compute_features(h1[["Open", "High", "Low", "Close", "Volume"]].copy(), legacy=False)
    except Exception: pass
print(f"  {len(feat)} tickers scored")

# conv_map[(date, 'HH:MM')] -> {ticker_noNS: (Long_Conviction, Short_Conviction)}
conv_map = {}
for tk, f in feat.items():
    pass
anchors_by_day = {}
for d in DAYS:
    dd = dt.date.fromisoformat(d)
    ancs = sorted({ts for f in feat.values() for ts in f.index if ts.date() == dd and ENTRY_FROM <= ts.strftime("%H:%M") <= ENTRY_TO})
    anchors_by_day[d] = ancs
    for ts in ancs:
        rows, tks = [], []
        for tk, f in feat.items():
            if ts in f.index: rows.append(f.loc[ts]); tks.append(tk)
        if len(tks) < 10: continue
        X = pd.DataFrame(rows, index=tks)
        X["Market_Mean_Return"] = X["Return"].mean(); X["Relative_Return"] = X["Return"] - X["Market_Mean_Return"]
        X["Market_Mean_Volatility"] = X["HL_Range"].mean(); X["Relative_Volatility"] = X["HL_Range"] / (X["Market_Mean_Volatility"] + 1e-8)
        dm = xgb.DMatrix(np.nan_to_num(X[v20_feats].values.astype(np.float32)), feature_names=v20_feats)
        ls, ss = bst_l.predict(dm), bst_s.predict(dm)
        lc, sc = ls - ls.mean(), ss - ss.mean(); longc = lc - sc
        key = (d, ts.strftime("%H:%M"))
        conv_map[key] = {tk.replace(".NS", ""): (float(longc[i]), float(-longc[i])) for i, tk in enumerate(tks)}

# ── LOAD REAL SHADOW TRADES ─────────────────────────────────────────────────
c = sqlite3.connect("data/vanguard_trades.db")
db = pd.read_sql(f"select timestamp, side, ticker, final_profit_pct pct, status from trades "
                 f"where substr(timestamp,1,10) in ({','.join(repr(x) for x in DAYS)}) and final_profit_pct is not null", c)
db["ticker"] = db.ticker.str.replace(".NS", "", regex=False)
db["date"] = db.timestamp.str[:10]
db["anchor"] = pd.to_datetime(db.timestamp).dt.floor("15min").dt.strftime("%H:%M")
db["gross_bps"] = db.pct * 100

def lookup_conv(r):
    m = conv_map.get((r.date, r.anchor))
    if not m or r.ticker not in m: return np.nan
    return m[r.ticker][0 if r.side == "LONG" else 1]
db["conv"] = db.apply(lookup_conv, axis=1)
matched = db.conv.notna().mean() * 100
print(f"  conviction matched for {matched:.0f}% of {len(db)} shadow trades\n")

# one pick per (date, anchor, side): if the live log has >1, keep the highest-conviction
db = db.dropna(subset=["conv"]).sort_values("conv", ascending=False).drop_duplicates(["date", "anchor", "side"])

# pivot to per-anchor long/short pair
piv = db.pivot_table(index=["date", "anchor"], columns="side", values=["conv", "gross_bps"], aggfunc="first")
rows = []
for (d, a), r in piv.iterrows():
    cL, cS = r.get(("conv", "LONG"), np.nan), r.get(("conv", "SHORT"), np.nan)
    gL, gS = r.get(("gross_bps", "LONG"), np.nan), r.get(("gross_bps", "SHORT"), np.nan)
    for side, cc, gg in (("LONG", cL, gL), ("SHORT", cS, gS)):
        if not np.isnan(cc):
            rows.append(dict(date=d, anchor=a, side=side, conv=cc, gross_bps=gg,
                             cL=cL, cS=cS, has_both=(not np.isnan(cL) and not np.isnan(cS))))
panel = pd.DataFrame(rows)
panel["net6"] = panel.gross_bps - COST6; panel["net10"] = panel.gross_bps - COST10
inb = lambda c: (c >= CONV_LO) & (c <= CONV_HI)
panel["elig"] = inb(panel.conv)
mx = panel[["cL", "cS"]].max(axis=1)
panel["reldiff"] = (panel.cL - panel.cS).abs() / mx.replace(0, np.nan)
panel["is_higher"] = np.where(panel.side == "LONG", panel.cL >= panel.cS.fillna(-9), panel.cS > panel.cL.fillna(-9))
bothok = inb(panel.cL) & inb(panel.cS)

def cond(sim): return panel.elig & ((bothok & (panel.reldiff <= sim)) | (panel.elig & panel.is_higher))
panel["take_one"]     = panel.elig & panel.is_higher
panel["take_both20"]  = cond(SIM_PRIMARY)
panel["take_both15"]  = cond(SIM_TIGHT)
panel["take_allboth"] = panel.elig

# ── REPORT ──────────────────────────────────────────────────────────────────
def s(x):
    if len(x) == 0: return "n=0"
    return (f"n={len(x):3d} WR={100*(x.gross_bps>0).mean():3.0f}% gross={x.gross_bps.mean():+6.1f} "
            f"net@6={x.net6.mean():+6.1f} net@10={x.net10.mean():+6.1f} bps")

print("=" * 78)
print("  REAL SHADOW-TRADE P&L (ground truth) + user's conviction rule")
print("=" * 78)
print("\n-- BAND CHECK (real shadow picks) --")
for side in ("LONG", "SHORT"):
    c = panel[panel.side == side].conv
    print(f"   {side:5s} n={len(c):3d} in-band={100*inb(c).mean():3.0f}%  below={100*(c<CONV_LO).mean():3.0f}%  above={100*(c>CONV_HI).mean():3.0f}%")

print("\n-- REFERENCE: raw top-1 L/S, NO band, NO selection (all real shadow trades) --")
print("   ALL  ", s(panel)); print("   LONG ", s(panel[panel.side=="LONG"])); print("   SHORT", s(panel[panel.side=="SHORT"]))

for label, col in [("SELECT-ONE (higher only)", "take_one"),
                   ("BOTH if within 20%", "take_both20"),
                   ("BOTH if within 15%", "take_both15"),
                   ("ALWAYS-BOTH (band only)", "take_allboth")]:
    b = panel[panel[col]]
    print("\n" + "=" * 78); print(f"  POLICY: {label}   (trades: {len(b)})"); print("=" * 78)
    print("   ALL  ", s(b)); print("   LONG ", s(b[b.side=="LONG"])); print("   SHORT", s(b[b.side=="SHORT"]))
    print("   -- per day --")
    for d in DAYS: print(f"     {d}  {s(b[b.date==d])}")
    print("   -- per anchor --")
    for a in sorted(panel.anchor.unique()): print(f"     {a}  {s(b[b.anchor==a])}")

out = f"data/backtests/conviction_on_shadow_{TODAY}.csv"
panel.to_csv(out, index=False)
print(f"\n[SAVED] {out}")
print("NOTE: P&L = REAL logged 1h gross (final_profit_pct); conviction reconstructed")
print("      with live parity. Coverage = anchors the live system actually scanned.")

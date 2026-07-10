"""
Conviction-Selection Top-1 L/S Backtest  |  v20_rolling_1h  |  1h hold  |  multi-day
====================================================================================
Extends today_top1_1h_backtest.py to the LAST N trading sessions and adds the
user's conviction-based entry-selection rule on top of the pure top-1 signal
stream (hourly model, NO veto layers — as requested).

At every 15-min anchor [10:15..14:15] we form the live top-1 LONG (max long_score)
and top-1 SHORT (max short_score) exactly like the live signal layer
(SIGNAL_RAW_SCORE_ONLY, ENTRY_TOP_K=1). Each carries a CONVICTION score computed
with full parity to scripts/vanguard/model_inference.py:218-222:

    l_centered  = long_score  - mean(long_score)      # cross-sectional, per anchor
    s_centered  = short_score - mean(short_score)
    Long_Conviction  =  l_centered - s_centered
    Short_Conviction = -Long_Conviction

We then decide which of the two per-anchor trades to actually take:

  BAND (eligibility):  a side is eligible only if  0.011 <= conviction <= 0.04
                       (user rule: reject conviction < 0.011 or > 0.04).

  Three policies are scored side-by-side:
    * SELECT-ONE           : among eligible sides, take ONLY the higher-conviction one.
    * SELECT-BOTH-IF-SIMILAR: if both eligible AND relative gap |cL-cS|/max <= SIM,
                              take BOTH; else take only the higher one. (SIM=20% & 15%)
    * ALWAYS-BOTH (ref)    : take every eligible side (band filter only, no comparison).

Each taken trade is held exactly 1 hour and scored gross / net@6bps / net@10bps and
rupee P&L at a uniform live per-trade notional. Reported overall, LONG/SHORT,
per-day, and per-anchor-time (the requested time-interval split). NO slot cap,
NO ATR stop, NO veto layers — pure signal + the selection rule.

Run:  python -m scripts.backtests.conviction_select_1h_backtest [N_SESSIONS]
"""

import os, sys, json, urllib.parse, datetime as dt, warnings
import concurrent.futures as cf
sys.path.insert(0, os.getcwd())
warnings.filterwarnings("ignore")

import requests
import numpy as np
import pandas as pd

import xgboost as xgb
from scripts.tickers import TICKERS
from scripts.feature_utils import build_rolling_1h_ohlcv, compute_features

# ── CONFIG ──────────────────────────────────────────────────────────────────
TODAY       = dt.date.today()
N_SESSIONS  = int(sys.argv[1]) if len(sys.argv) > 1 else 5   # last week (5 trading days)
ENTRY_FROM  = "10:15"
ENTRY_TO    = "14:15"
HOLD        = pd.Timedelta(hours=1)

CONV_LO     = 0.011      # user band: reject conviction below this
CONV_HI     = 0.04       # user band: reject conviction above this
SIM_PRIMARY = 0.20       # "somewhat similar" = within 20%  -> take both
SIM_TIGHT   = 0.15       # sensitivity: within 15%

V20_LONG  = "models/research/v20_rolling_1h/xgb_long_model.json"
V20_SHORT = "models/research/v20_rolling_1h/xgb_short_model.json"
V20_META  = "models/research/v20_rolling_1h/metadata.json"

COST_BPS_PRIMARY = 6
COST_BPS_CONSERV = 10

# uniform live per-trade notional (config.py: CAPITAL/MAX_SLOTS * MARGIN)
CAPITAL, MAX_SLOTS, MARGIN = 99517.68, 5, 5.0
NOTIONAL     = (CAPITAL / MAX_SLOTS) * MARGIN
BROKERAGE_RT = 20.0
STT_RATE     = 0.00025
SLIP_PCT     = 0.0003

CACHE = os.path.join(os.environ.get("TEMP", "/tmp"), "claude",
                     "c--Users-loq-Desktop-Trading-finalgo", "today_top1_15m_cache")
os.makedirs(CACHE, exist_ok=True)
HDR = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Accept": "application/json"}
INSTR = json.load(open("scripts/instrument_cache.json"))


def fetch_15m(ticker):
    """~25d history + today intraday, 15-min. Reuses today_top1's day cache."""
    sym = ticker.replace(".NS", "")
    ik = INSTR.get(sym)
    if not ik:
        return ticker, None
    cache_f = os.path.join(CACHE, f"{sym}_{TODAY}.csv")
    if os.path.exists(cache_f):
        df = pd.read_csv(cache_f, parse_dates=["timestamp"])
    else:
        enc = urllib.parse.quote(ik, safe="")
        frm = TODAY - dt.timedelta(days=25)
        candles = []
        try:
            r = requests.get(f"https://api.upstox.com/v3/historical-candle/{enc}/minutes/15/{TODAY}/{frm}",
                             headers=HDR, timeout=25).json()
            candles += r.get("data", {}).get("candles", [])
            r2 = requests.get(f"https://api.upstox.com/v3/historical-candle/intraday/{enc}/minutes/15",
                              headers=HDR, timeout=25).json()
            candles += r2.get("data", {}).get("candles", [])
        except Exception:
            return ticker, None
        if not candles:
            return ticker, None
        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        df = df.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
        df.to_csv(cache_f, index=False)
    df = df.set_index("timestamp").rename(columns={
        "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
    return ticker, df[["Open", "High", "Low", "Close", "Volume"]].dropna()


print("=" * 82)
print(f"  CONVICTION-SELECT TOP-1 L/S  |  v20_rolling_1h  |  1h hold  |  last {N_SESSIONS} sessions")
print("=" * 82)
v20_feats = json.load(open(V20_META))["features"]
bst_l = xgb.Booster(); bst_l.load_model(V20_LONG)
bst_s = xgb.Booster(); bst_s.load_model(V20_SHORT)
print(f"[1/5] Models loaded — v20 features: {len(v20_feats)}")

print(f"[2/5] Fetching 15-min candles for {len(TICKERS)} tickers (cache reuse)...")
raw15 = {}
with cf.ThreadPoolExecutor(max_workers=8) as ex:
    for tk, df in ex.map(fetch_15m, TICKERS):
        if df is not None and len(df) >= 40:
            raw15[tk] = df
print(f"      {len(raw15)}/{len(TICKERS)} tickers loaded")

print("[3/5] Building rolling-1h features (live pipeline parity)...")
feat, close_1h = {}, {}
for tk, df15 in raw15.items():
    try:
        h1 = build_rolling_1h_ohlcv(df15)
        if len(h1) < 20:
            continue
        feat[tk] = compute_features(h1[["Open", "High", "Low", "Close", "Volume"]].copy(), legacy=False)
        close_1h[tk] = h1["Close"]
    except Exception:
        continue
print(f"      {len(feat)} tickers with v20 features ready")

# session dates present in the data, within the entry window; take the last N
all_dates = set()
for f in feat.values():
    for tsx in f.index:
        if ENTRY_FROM <= tsx.strftime("%H:%M") <= ENTRY_TO:
            all_dates.add(tsx.date())
sessions = sorted(all_dates)[-N_SESSIONS:]
print(f"[4/5] Sessions: {[d.isoformat() for d in sessions]}")


def score_anchor(ts):
    rows, tickers = [], []
    for tk, f in feat.items():
        if ts in f.index:
            rows.append(f.loc[ts]); tickers.append(tk)
    if len(tickers) < 10:
        return None
    X = pd.DataFrame(rows, index=tickers)
    X["Market_Mean_Return"] = X["Return"].mean()
    X["Relative_Return"] = X["Return"] - X["Market_Mean_Return"]
    X["Market_Mean_Volatility"] = X["HL_Range"].mean()
    X["Relative_Volatility"] = X["HL_Range"] / (X["Market_Mean_Volatility"] + 1e-8)
    Xm = np.nan_to_num(X[v20_feats].values.astype(np.float32))
    dm = xgb.DMatrix(Xm, feature_names=v20_feats)
    ls = bst_l.predict(dm); ss = bst_s.predict(dm)
    l_c = ls - ls.mean(); s_c = ss - ss.mean()
    return pd.DataFrame({"ticker": tickers, "long_score": ls, "short_score": ss,
                         "Long_Conviction": l_c - s_c, "Short_Conviction": s_c - l_c})


def exit_price(tk, ts):
    s = close_1h[tk]; ex_ts = ts + HOLD
    if ex_ts in s.index:
        return float(s.loc[ex_ts]), ex_ts
    after = s[s.index >= ex_ts]
    if not after.empty:
        return float(after.iloc[0]), after.index[0]
    same = s[s.index.date == ts.date()]
    return float(same.iloc[-1]), same.index[-1]


def cost_inr(side, qty, ep, xp, notional):
    stt = STT_RATE * (qty * xp if side == "LONG" else notional)
    return BROKERAGE_RT + stt + SLIP_PCT * notional * 2


# ── BUILD PER-ANCHOR CANDIDATE PANEL (top-1 long + top-1 short) ──────────────
print("[5/5] Scoring anchors + building top-1 L/S candidates...\n")
cand = []   # one row per (session, anchor, side)
for d in sessions:
    anchors = sorted({tsx for f in feat.values() for tsx in f.index
                      if tsx.date() == d and ENTRY_FROM <= tsx.strftime("%H:%M") <= ENTRY_TO})
    for ts in anchors:
        sc = score_anchor(ts)
        if sc is None:
            continue
        picks = {}
        for side, raw_col, conv_col in (("LONG", "long_score", "Long_Conviction"),
                                        ("SHORT", "short_score", "Short_Conviction")):
            best = sc.sort_values(raw_col, ascending=False).iloc[0]     # live raw-score pick
            tk = best["ticker"]
            ep = float(close_1h[tk].loc[ts])
            if ep <= 0:
                continue
            xp, xts = exit_price(tk, ts)
            gross = (xp - ep) / ep if side == "LONG" else (ep - xp) / ep
            qty = int(NOTIONAL / ep); notional = qty * ep
            net_inr = gross * notional - cost_inr(side, qty, ep, xp, notional)
            picks[side] = {
                "date": d, "anchor": ts.strftime("%H:%M"), "entry_time": ts, "exit_time": xts,
                "side": side, "ticker": tk.replace(".NS", ""), "conv": float(best[conv_col]),
                "entry_px": round(ep, 2), "exit_px": round(xp, 2),
                "gross_bps": gross * 1e4, "net6_bps": gross * 1e4 - COST_BPS_PRIMARY,
                "net10_bps": gross * 1e4 - COST_BPS_CONSERV, "net_inr": net_inr,
            }
        # pair-level fields (conviction comparison) attached to each side row
        cL = picks.get("LONG", {}).get("conv", np.nan)
        cS = picks.get("SHORT", {}).get("conv", np.nan)
        for side, row in picks.items():
            row["conv_long"], row["conv_short"] = cL, cS
            cand.append(row)

panel = pd.DataFrame(cand)
if panel.empty:
    print("No candidates generated."); sys.exit(0)

# ── SELECTION POLICIES ──────────────────────────────────────────────────────
in_band = lambda c: (c >= CONV_LO) & (c <= CONV_HI)
panel["eligible"] = in_band(panel["conv"])
panel["ok_long"]  = in_band(panel["conv_long"])
panel["ok_short"] = in_band(panel["conv_short"])
mx = panel[["conv_long", "conv_short"]].max(axis=1)
panel["reldiff"] = (panel["conv_long"] - panel["conv_short"]).abs() / mx.replace(0, np.nan)
# is THIS row's side the higher-conviction one of the pair?
panel["is_higher"] = np.where(panel["side"] == "LONG",
                              panel["conv_long"] >= panel["conv_short"],
                              panel["conv_short"] >  panel["conv_long"])
both_ok = panel["ok_long"] & panel["ok_short"]

def taken_conditional(sim):
    # both eligible & similar -> take both; else take only the higher eligible side
    take_both = both_ok & (panel["reldiff"] <= sim)
    take_high = panel["eligible"] & panel["is_higher"]
    return panel["eligible"] & (take_both | take_high)

panel["take_one"]      = panel["eligible"] & panel["is_higher"]          # SELECT-ONE
panel["take_both20"]   = taken_conditional(SIM_PRIMARY)                  # both if within 20%
panel["take_both15"]   = taken_conditional(SIM_TIGHT)                    # both if within 15%
panel["take_allboth"]  = panel["eligible"]                              # ALWAYS-BOTH (band only)


# ── REPORTING ───────────────────────────────────────────────────────────────
def stats(sdf):
    if len(sdf) == 0:
        return dict(n=0, wr=float("nan"), g=float("nan"), n6=float("nan"),
                    n10=float("nan"), rs=0.0)
    return dict(n=len(sdf), wr=(sdf.gross_bps > 0).mean() * 100,
                g=sdf.gross_bps.mean(), n6=sdf.net6_bps.mean(),
                n10=sdf.net10_bps.mean(), rs=sdf.net_inr.sum())

def line(label, s):
    if s["n"] == 0:
        print(f"    {label:24s} n= 0"); return
    print(f"    {label:24s} n={s['n']:3d}  WR={s['wr']:4.0f}%  gross={s['g']:+7.2f}  "
          f"net@6={s['n6']:+7.2f}  net@10={s['n10']:+7.2f} bps   netRs={s['rs']:+10.0f}")

SEP = "-" * 82
POLICIES = [("SELECT-ONE  (higher only)", "take_one"),
            ("BOTH if within 20%", "take_both20"),
            ("BOTH if within 15%", "take_both15"),
            ("ALWAYS-BOTH (band only)", "take_allboth")]

print("\n" + "=" * 82)
print("  CONVICTION BAND CHECK  (top-1 pick conviction distribution, last week)")
print("=" * 82)
for side in ("LONG", "SHORT"):
    c = panel.loc[panel.side == side, "conv"]
    print(f"    {side:5s}  n={len(c)}  min={c.min():+.4f}  med={c.median():+.4f}  "
          f"max={c.max():+.4f}   in-band[0.011,0.04]={in_band(c).mean()*100:.0f}%  "
          f"below={ (c<CONV_LO).mean()*100:.0f}%  above={ (c>CONV_HI).mean()*100:.0f}%")

for label, col in POLICIES:
    book = panel[panel[col]]
    print("\n" + "=" * 82)
    print(f"  POLICY: {label}   (trades taken: {len(book)})")
    print("=" * 82)
    line("ALL", stats(book))
    line("LONG", stats(book[book.side == "LONG"]))
    line("SHORT", stats(book[book.side == "SHORT"]))
    print("  " + SEP)
    print("  Per-DAY:")
    for d in sessions:
        line(d.isoformat(), stats(book[book.date == d]))
    print("  " + SEP)
    print("  Per-TIME-INTERVAL (anchor, across week):")
    for a in sorted(panel.anchor.unique()):
        line(a, stats(book[book.anchor == a]))

# ── SAVE ─────────────────────────────────────────────────────────────────────
out = f"data/backtests/conviction_select_1h_last{N_SESSIONS}_{TODAY}.csv"
keep = ["date", "anchor", "entry_time", "exit_time", "ticker", "side", "conv",
        "conv_long", "conv_short", "reldiff", "eligible", "is_higher",
        "entry_px", "exit_px", "gross_bps", "net6_bps", "net10_bps", "net_inr",
        "take_one", "take_both20", "take_both15", "take_allboth"]
panel.sort_values(["date", "anchor", "side"])[keep].to_csv(out, index=False)
print("\n[SAVED] " + out)
print("NOTE: pure top-1 signal (raw-score pick, live parity) + conviction selection;")
print("      no veto layers, no ATR stop, no slot cap. Uniform Rs{:.0f} notional/trade.".format(NOTIONAL))

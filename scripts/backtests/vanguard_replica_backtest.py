"""
v20 + v3 15min  |  TRUE OOS Backtest  |  June 5-18, 2026
==========================================================
Fetches fresh 15-min data from Upstox for all 172 tickers,
builds rolling 1H candles, scores with v20 + v3 15min models,
and simulates the live trading system's dual-model logic.

Signal : v20_rolling_1h  top-K by Long/Short Conviction
Gate   : v3_15min_clean  top/bottom 15% by score_15m per scan
Hold   : 1 hour
Cost   : 10 bps round-trip

Run: python -m scripts.backtests.v20_v3_true_oos_backtest
"""

import sys, os, json, time, warnings
sys.path.insert(0, os.getcwd())
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime, timedelta, date
from pathlib import Path
from unittest.mock import patch

import upstox_client
from dotenv import load_dotenv
load_dotenv()

from scripts.tickers import TICKERS
from scripts.feature_utils import compute_features
from scripts.vanguard.orchestrator import VanguardOrchestrator

# ── CONFIG ─────────────────────────────────────────────────────────────────────
OOS_FROM     = date(2026, 6, 18)
OOS_TO       = date(2026, 6, 18)

V20_LONG     = "models/research/v20_rolling_1h/xgb_long_model.json"
V20_SHORT    = "models/research/v20_rolling_1h/xgb_short_model.json"
V20_META     = "models/research/v20_rolling_1h/metadata.json"
V3_LONG      = "models/v3_15min_clean/xgb_long_model.json"
V3_SHORT     = "models/v3_15min_clean/xgb_short_model.json"
V3_META      = "models/v3_15min_clean/metadata.json"

CACHE_15M    = Path("data/oos_cache_15min_jun2026")
CACHE_15M.mkdir(parents=True, exist_ok=True)

GATE_PCT     = 0.10
# Dynamic daily macro gate using VanguardOrchestrator to faithfully replicate live system bugs.
DAILY_GATE_FILE = None
TOP_K        = 3
COST_RT      = 10 / 10000   # 10 bps
STOP_PCT     = None         # 0.30% intrabar stop: first 15-min candle in the hold whose
                            # adverse extreme (LONG=Low, SHORT=High) breaches the level
                            # exits at THAT candle's Close. Set None to disable.
MAX_SLOTS    = 6
CAPITAL      = 99517.68
MARGIN       = 5.0
SLOT_CAP     = CAPITAL / MAX_SLOTS

RATE_PAUSE   = 0.25          # seconds between API calls
VALID_TODS_1H = {
    "09:15", "10:15", "11:15", "12:15", "13:15", "14:15"
}
# Live system only ENTERS trades from 10:15 (matches orchestrator is_eligible_time)
ENTRY_START = "10:15"
ENTRY_END   = "15:05"

def get_daily_gate(target_date):
    target_date_str = target_date.strftime("%Y-%m-%d")
    target_dt = datetime.strptime(f"{target_date_str} 09:00:00", "%Y-%m-%d %H:%M:%S")
    cutoff = pd.Timestamp(target_date)  # strictly < target_date (yesterday's close only)

    print(f"[REPLICA] Daily gate for {target_date_str} | data cutoff: < {target_date_str}")

    import yfinance as yf_orig
    _orig_download = yf_orig.download

    def _patched_download(*args, **kwargs):
        df = _orig_download(*args, **kwargs)
        if not df.empty:
            # Strip timezone and truncate to strictly before target_date
            if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df = df[df.index < cutoff]
        return df

    with patch('scripts.vanguard.orchestrator.datetime') as mock_dt, \
         patch('yfinance.download', side_effect=_patched_download):
        mock_dt.now.return_value = target_dt
        mock_dt.strptime = datetime.strptime
        mock_dt.strftime = datetime.strftime
        orch = VanguardOrchestrator()
        orch.update_daily_macro_filters()
        return orch.long_eligible_tickers, orch.short_eligible_tickers

# ── BANNER ─────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  VANGUARD REPLICA BACKTEST  |  Jun 18, 2026 ONLY")
print(f"  Tickers : {len(TICKERS)}   Gate : top/bottom {int(GATE_PCT*100)}%   Top-K : {TOP_K}")
print(f"  Cost    : 10 bps RT   Capital : Rs {CAPITAL:,.0f}   Margin : {MARGIN}x")
print("=" * 70)

# ── LOAD MODELS ────────────────────────────────────────────────────────────────
print("\n[1/4] Loading models ...")
with open(V20_META) as f: v20_feats = json.load(f)["features"]
with open(V3_META)  as f: v3_feats  = json.load(f)["features"]

bst_v20l = xgb.Booster(); bst_v20l.load_model(V20_LONG)
bst_v20s = xgb.Booster(); bst_v20s.load_model(V20_SHORT)
bst_v3l  = xgb.Booster(); bst_v3l.load_model(V3_LONG)
bst_v3s  = xgb.Booster(); bst_v3s.load_model(V3_SHORT)
print(f"     v20 features : {len(v20_feats)}   v3 15min features : {len(v3_feats)}")

# ── DAILY MACRO GATE ──────────────────────────────────────────────────────────
# Dynamic now!
long_elig = short_elig = None
current_gate_date = None

# ── PHASE 1: LOAD 15-MIN CACHED DATA ──────────────────────────────────────────
print(f"\n[2/4] Loading cached 15-min data for {len(TICKERS)} tickers ...")
print(f"      Period : {OOS_FROM} to {OOS_TO} (market open only)")

def load_cached_15min(ticker):
    sym   = ticker.replace(".NS", "")
    cache = CACHE_15M / f"{sym}.csv"
    if cache.exists():
        try:
            df = pd.read_csv(cache)
            if not df.empty and "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                return df
        except Exception:
            pass
    return None

ticker_15m = {}   # ticker -> DataFrame (15min OHLCV, IST tz-naive)
ok_count   = 0
for ticker in TICKERS:
    raw = load_cached_15min(ticker)
    if raw is None or len(raw) < 5:
        continue
    # Convert to IST tz-naive
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    raw["DateTime"]  = raw["timestamp"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    raw = raw.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume"
    })
    raw = (raw.drop_duplicates("DateTime")
              .sort_values("DateTime")
              .set_index("DateTime"))
    raw = raw[["Open","High","Low","Close","Volume"]].dropna()
    # Filter to OOS window
    raw = raw[(raw.index.date >= OOS_FROM) & (raw.index.date <= OOS_TO)]
    if len(raw) >= 5:
        ticker_15m[ticker] = raw
        ok_count += 1

print(f"     Loaded  : {ok_count}/{len(TICKERS)} tickers OK from cache")

# ── PHASE 2: COMPUTE FEATURES FOR EVERY SCAN TIMESTAMP ────────────────────────
print("\n[3/4] Computing features and building scan panels ...")

# We need lookback context for feature computation.
# Load the existing 15min_3y cache for the warm-up period (up to Jun 4).
WARMUP_CACHE = Path("data/raw_upstox_cache_15min_3y")

def load_warmup(ticker):
    sym   = ticker.replace(".NS","")
    fpath = WARMUP_CACHE / f"{sym}.csv"
    if not fpath.exists():
        return None
    df = pd.read_csv(fpath)
    if len(df) > 1000:
        df = df.iloc[-1000:]
    dt_col = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()][0]
    df[dt_col] = pd.to_datetime(df[dt_col], utc=True)
    df["DateTime"] = df[dt_col].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    df = df.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume"
    })
    df = (df.drop_duplicates("DateTime")
            .sort_values("DateTime")
            .set_index("DateTime"))
    return df[["Open","High","Low","Close","Volume"]].dropna()


def build_rolling_1h_ohlcv(df_15m):
    """Build overlapping trailing-1H candles from 15-min bars (4-bar rolling window) using vectorized Pandas."""
    if len(df_15m) < 4:
        return pd.DataFrame()
    out = pd.DataFrame(index=df_15m.index)
    out["Open"] = df_15m["Open"].shift(3)
    out["High"] = df_15m["High"].rolling(4).max()
    out["Low"] = df_15m["Low"].rolling(4).min()
    out["Close"] = df_15m["Close"]
    out["Volume"] = df_15m["Volume"].rolling(4).sum()
    out = out.dropna()
    # Keep only intraday 1H TODs matching v20 training grid
    tods = out.index.strftime("%H:%M")
    out  = out[pd.Series(tods).isin(VALID_TODS_1H).values]
    return out


# Per-ticker feature DataFrames keyed by model
feat_v20 = {}   # ticker -> df with v20_feats, indexed by DateTime
feat_v3  = {}   # ticker -> df with v3_feats,  indexed by DateTime

for ticker, df_oos in ticker_15m.items():
    warmup = load_warmup(ticker)
    if warmup is not None:
        # Concatenate warmup + OOS 15min for full feature context
        df_full_15m = pd.concat([warmup, df_oos])
        df_full_15m = df_full_15m[~df_full_15m.index.duplicated(keep="last")]
        df_full_15m = df_full_15m.sort_index()
    else:
        df_full_15m = df_oos.copy()

    # ── v3 15min features ──────────────────────────────────────────────────
    try:
        f3 = compute_features(df_full_15m[["Open","High","Low","Close","Volume"]].copy(),
                              legacy=False)
        # Keep only OOS rows
        f3_oos = f3[f3.index.date >= OOS_FROM]
        if len(f3_oos) >= 2:
            feat_v3[ticker] = f3_oos
    except Exception:
        pass

    # ── v20 rolling 1H features ────────────────────────────────────────────
    try:
        df_1h = build_rolling_1h_ohlcv(df_full_15m)
        if len(df_1h) < 10:
            continue
        f20 = compute_features(df_1h[["Open","High","Low","Close","Volume"]].copy(),
                               legacy=False)
        # Filter to valid TODs in OOS range
        f20 = f20[f20.index.strftime("%H:%M").isin(VALID_TODS_1H)]
        f20_oos = f20[f20.index.date >= OOS_FROM]
        if len(f20_oos) >= 2:
            feat_v20[ticker] = f20_oos
    except Exception:
        pass

print(f"     v20 features ready : {len(feat_v20)} tickers")
print(f"     v3  features ready : {len(feat_v3)} tickers")

# ── SCORING FUNCTIONS ──────────────────────────────────────────────────────────
def _add_cross_sectional(df):
    """Cross-sectional market-relative features, matching the live orchestrator
    (scripts/vanguard/orchestrator.py:1081-1084). Computed per scan over the
    current cross-section — these are NOT produced by compute_features()."""
    if "Return" in df.columns:
        df["Market_Mean_Return"] = df["Return"].mean()
        df["Relative_Return"]    = df["Return"] - df["Market_Mean_Return"]
    if "HL_Range" in df.columns:
        df["Market_Mean_Volatility"] = df["HL_Range"].mean()
        df["Relative_Volatility"]    = df["HL_Range"] / (df["Market_Mean_Volatility"] + 1e-8)
    return df


def score_v20_cs(cs_dict):
    """cs_dict: {ticker: feature_row_Series}. Returns DataFrame with scores."""
    rows = []
    for ticker, row in cs_dict.items():
        rows.append({"Ticker": ticker, **row.to_dict()})
    df = pd.DataFrame(rows).set_index("Ticker")
    df = _add_cross_sectional(df)
    missing = [c for c in v20_feats if c not in df.columns]
    for m in missing:
        df[m] = 0.0
    X = np.nan_to_num(df[v20_feats].values.astype(np.float32))
    dm = xgb.DMatrix(X, feature_names=v20_feats)
    l  = bst_v20l.predict(dm)
    s  = bst_v20s.predict(dm)
    lc = (l - l.mean()) - (s - s.mean())
    sc = (s - s.mean()) - (l - l.mean())
    out = pd.DataFrame({
        "Ticker"     : df.index,
        "long_conv"  : lc,
        "short_conv" : sc,
        "long_score" : l,
        "short_score": s,
    })
    return out.reset_index(drop=True)


def score_v3_cs(cs_dict):
    """cs_dict: {ticker: feature_row_Series}. Returns DataFrame with score_15m."""
    rows = []
    for ticker, row in cs_dict.items():
        rows.append({"Ticker": ticker, **row.to_dict()})
    df = pd.DataFrame(rows).set_index("Ticker")
    df = _add_cross_sectional(df)
    missing = [c for c in v3_feats if c not in df.columns]
    for m in missing:
        df[m] = 0.0
    X = np.nan_to_num(df[v3_feats].values.astype(np.float32))
    dm = xgb.DMatrix(X, feature_names=v3_feats)
    l  = bst_v3l.predict(dm)
    s  = bst_v3s.predict(dm)
    lstd = l.std(); sstd = s.std()
    ln = (l - l.mean()) / lstd if lstd > 1e-9 else (l - l.mean())
    sn = (s - s.mean()) / sstd if sstd > 1e-9 else (s - s.mean())
    out = pd.DataFrame({
        "Ticker"   : df.index,
        "score_15m": ln - sn,
    })
    return out.reset_index(drop=True)


# ── EXIT RESOLUTION (intrabar stop + time exit) ─────────────────────────────────
def resolve_exit(pos, df_15m):
    """Resolve a position's exit price/time/reason.

    Intrabar stop (if STOP_PCT is set): scan the raw 15-min candles strictly
    after entry up to the 1h mark; the FIRST candle whose adverse extreme
    breaches the stop level closes the trade at THAT candle's Close.
        LONG  -> stop if candle Low  <= entry * (1 - STOP_PCT)
        SHORT -> stop if candle High >= entry * (1 + STOP_PCT)
    Otherwise time exit: first 15-min Close at/after exit_ts (fallback: last bar).
    Returns (exit_px, exit_time, reason).
    """
    ep = pos["entry_px"]
    if df_15m is None or df_15m.empty:
        return ep, pos["exit_ts"], "NONE"

    if STOP_PCT is not None:
        hold = df_15m[(df_15m.index > pos["entry_ts"]) & (df_15m.index <= pos["exit_ts"])]
        if not hold.empty:
            if pos["side"] == "LONG":
                hit = hold[hold["Low"] <= ep * (1 - STOP_PCT)]
            else:
                hit = hold[hold["High"] >= ep * (1 + STOP_PCT)]
            if not hit.empty:
                return float(hit.iloc[0]["Close"]), hit.index[0], "STOP"

    # time exit — clamped to the SAME trading day (never roll into the next session)
    day      = pos["entry_ts"].date()
    same_day = df_15m[df_15m.index.date == day]
    after    = same_day[same_day.index >= pos["exit_ts"]]
    if not after.empty:
        return float(after.iloc[0]["Close"]), after.index[0], "TIME"
    if not same_day.empty:
        return float(same_day.iloc[-1]["Close"]), same_day.index[-1], "TIME"
    return float(df_15m.iloc[-1]["Close"]), df_15m.index[-1], "TIME"


# ── PHASE 3: SIMULATION ────────────────────────────────────────────────────────
print("\n[4/4] Running simulation ...")

# Build set of all unique v20 timestamps in OOS window
all_ts = sorted(set(
    ts
    for df in feat_v20.values()
    for ts in df.index
))
print(f"     Total scan timestamps : {len(all_ts)}")

trades   = []
open_pos = {}

for i, ts in enumerate(all_ts):
    if i % 100 == 0:
        print(f"     scan {i:4d}/{len(all_ts)}  {ts.strftime('%Y-%m-%d %H:%M')}"
              f"  open={len(open_pos)}")

    # Update daily gate if day changed
    ts_date = ts.date()
    if current_gate_date != ts_date:
        current_gate_date = ts_date
        l_elig, s_elig = get_daily_gate(ts_date)
        long_elig = set(l_elig)
        short_elig = set(s_elig)
        print(f"     [DAILY GATE] {ts_date} | Longs: {len(long_elig)}, Shorts: {len(short_elig)}")

    # ── close expired positions ────────────────────────────────────────────
    to_close = [t for t, p in open_pos.items() if ts >= p["exit_ts"]]
    for ticker in to_close:
        pos = open_pos.pop(ticker)
        df_15m = ticker_15m.get(ticker)
        exit_px, exit_time, reason = resolve_exit(pos, df_15m)
        ep  = pos["entry_px"]
        ret = (exit_px - ep) / ep if pos["side"] == "LONG" else (ep - exit_px) / ep
        net = ret - COST_RT
        qty = int((SLOT_CAP * MARGIN) / ep) if ep > 0 else 0
        trades.append({
            "entry_time" : pos["entry_ts"],
            "exit_time"  : exit_time,
            "date"       : pos["entry_ts"].date(),
            "ticker"     : ticker,
            "side"       : pos["side"],
            "exit_reason": reason,
            "entry_px"   : round(ep, 2),
            "exit_px"    : round(exit_px, 2),
            "pnl_pct"    : round(ret * 100, 4),
            "net_pnl_pct": round(net * 100, 4),
            "qty"        : qty,
            "net_pnl_inr": round(net * ep * qty, 2),
            "long_conv"  : pos["long_conv"],
            "short_conv" : pos["short_conv"],
            "score_15m"  : pos["score_15m"],
        })

    # ── build cross-section at this timestamp ──────────────────────────────
    # v20 cross-section
    cs_v20 = {}
    for ticker, df in feat_v20.items():
        if ts in df.index:
            cs_v20[ticker] = df.loc[ts]

    # v3 cross-section: nearest 15min <= ts (within 20 min)
    cs_v3 = {}
    for ticker, df in feat_v3.items():
        cands = df[df.index <= ts]
        if cands.empty:
            continue
        nearest = cands.index[-1]
        if (ts - nearest).total_seconds() > 1200:
            continue
        cs_v3[ticker] = cands.iloc[-1]

    if len(cs_v20) < 10 or len(cs_v3) < 10:
        continue

    # ── score ──────────────────────────────────────────────────────────────
    try:
        sc20 = score_v20_cs(cs_v20)
        sc3  = score_v3_cs(cs_v3)
    except Exception:
        continue

    merged = sc20.merge(sc3, on="Ticker", how="inner")
    if len(merged) < 10:
        continue

    # Attach Close prices from feat_v20
    closes = {t: float(feat_v20[t].loc[ts, "Close"])
              for t in merged["Ticker"]
              if t in feat_v20 and ts in feat_v20[t].index}
    merged["Close"] = merged["Ticker"].map(closes)
    merged = merged.dropna(subset=["Close"])

    s15       = merged["score_15m"]
    long_thr  = s15.quantile(1.0 - GATE_PCT)
    short_thr = s15.quantile(GATE_PCT)

    # Live system only enters trades 10:15–15:05
    ts_tod = ts.strftime("%H:%M")
    if not (ENTRY_START <= ts_tod < ENTRY_END):
        continue

    for side in ("LONG", "SHORT"):
        # 1. Daily macro gate: keep only tickers eligible for this side
        elig = long_elig if side == "LONG" else short_elig
        cands = merged.copy()
        if elig is not None:
            cands = cands[cands["Ticker"].isin(elig)]

        # 2. Extract Top Net and Top Raw AI candidates (matches signal_generation.py exactly)
        if side == "LONG":
            top_net = cands.sort_values("long_conv", ascending=False).head(TOP_K)
            rem = cands[~cands["Ticker"].isin(top_net["Ticker"])]
            top_raw = rem.sort_values("long_score", ascending=False).head(TOP_K)
            ai_cands = pd.concat([top_net, top_raw])
        else:
            top_net = cands.sort_values("short_conv", ascending=False).head(TOP_K)
            rem = cands[~cands["Ticker"].isin(top_net["Ticker"])]
            top_raw = rem.sort_values("short_score", ascending=False).head(TOP_K)
            ai_cands = pd.concat([top_net, top_raw])

        picked = 0
        for _, row in ai_cands.iterrows():
            # 3. 15m Gate: reject if NOT in top/bottom 10% of ENTIRE universe
            if side == "LONG" and row["score_15m"] < long_thr:
                continue
            if side == "SHORT" and row["score_15m"] > short_thr:
                continue
                
            ticker = row["Ticker"]
            if ticker in open_pos or len(open_pos) >= MAX_SLOTS:
                continue
            px = row["Close"]
            if px <= 0:
                continue
            open_pos[ticker] = {
                "side"      : side,
                "entry_px"  : float(px),
                "entry_ts"  : ts,
                "exit_ts"   : ts + timedelta(hours=1),
                "long_conv" : float(row["long_conv"]),
                "short_conv": float(row["short_conv"]),
                "score_15m" : float(row["score_15m"]),
            }
            picked += 1

# flush remaining positions
for ticker, pos in open_pos.items():
    df_15m  = ticker_15m.get(ticker)
    exit_px, exit_time, reason = resolve_exit(pos, df_15m)
    ep  = pos["entry_px"]
    ret = (exit_px - ep) / ep if pos["side"] == "LONG" else (ep - exit_px) / ep
    net = ret - COST_RT
    qty = int((SLOT_CAP * MARGIN) / ep) if ep > 0 else 0
    trades.append({
        "entry_time" : pos["entry_ts"],
        "exit_time"  : exit_time,
        "date"       : pos["entry_ts"].date(),
        "ticker"     : ticker,
        "side"       : pos["side"],
        "exit_reason": reason,
        "entry_px"   : round(ep, 2),
        "exit_px"    : round(exit_px, 2),
        "pnl_pct"    : round(ret * 100, 4),
        "net_pnl_pct": round(net * 100, 4),
        "qty"        : qty,
        "net_pnl_inr": round(net * ep * qty, 2),
        "long_conv"  : pos["long_conv"],
        "short_conv" : pos["short_conv"],
        "score_15m"  : pos["score_15m"],
    })

# ── RESULTS ────────────────────────────────────────────────────────────────────
df = pd.DataFrame(trades)
if df.empty:
    print("\nNo trades generated.")
    sys.exit(0)

df["entry_time"] = pd.to_datetime(df["entry_time"])
df = df.sort_values("entry_time").reset_index(drop=True)
df.index += 1
df["cum_pnl_inr"] = df["net_pnl_inr"].cumsum()

N        = len(df)
wins     = (df["net_pnl_pct"] > 0).sum()
wr       = wins / N * 100
tot_net  = df["net_pnl_inr"].sum()
avg_pct  = df["net_pnl_pct"].mean()
avg_win  = df[df["net_pnl_pct"] > 0]["net_pnl_pct"].mean() if wins else 0.0
avg_loss = df[df["net_pnl_pct"] <= 0]["net_pnl_pct"].mean() if (N - wins) else 0.0
roc      = tot_net / CAPITAL * 100
cum      = df["net_pnl_inr"].cumsum()
max_dd   = (cum - cum.cummax()).min()
gross_w  = df[df["net_pnl_pct"] > 0]["net_pnl_inr"].sum()
gross_l  = abs(df[df["net_pnl_pct"] <= 0]["net_pnl_inr"].sum())
pf       = gross_w / (gross_l + 1e-9)

daily = df.groupby("date").agg(
    trades      = ("net_pnl_pct", "count"),
    winners     = ("net_pnl_pct", lambda x: (x > 0).sum()),
    net_pnl_inr = ("net_pnl_inr", "sum"),
    avg_pct     = ("net_pnl_pct", "mean"),
    best_pct    = ("pnl_pct",     "max"),
    worst_pct   = ("pnl_pct",     "min"),
).reset_index()
daily["win_rate"] = (daily["winners"] / daily["trades"] * 100).round(1)
daily["cum_pnl"]  = daily["net_pnl_inr"].cumsum()

long_df  = df[df["side"] == "LONG"]
short_df = df[df["side"] == "SHORT"]

SEP = "=" * 72
print(f"\n{SEP}")
print(f"  TRUE OOS RESULTS  |  v20 1H + v3 15min top-{int(GATE_PCT*100)}% gate  |  Jun 5-18, 2026")
print(SEP)
print(f"  Period           : {OOS_FROM}  to  {OOS_TO}")
print(f"  Total Trades     : {N}")
print(f"  Win Rate         : {wr:.1f}%")
print(f"  Avg Net PnL/trade: {avg_pct:+.4f}%")
print(f"  Avg Win          : {avg_win:+.4f}%    Avg Loss : {avg_loss:+.4f}%")
print(f"  Profit Factor    : {pf:.2f}")
print(f"  Total Net PnL    : Rs {tot_net:+,.2f}")
print(f"  Return on Cap    : {roc:+.3f}%  (on Rs {CAPITAL:,.0f})")
print(f"  Max Drawdown     : Rs {max_dd:,.2f}")
if "exit_reason" in df.columns and STOP_PCT is not None:
    n_stop  = int((df["exit_reason"] == "STOP").sum())
    stop_df = df[df["exit_reason"] == "STOP"]
    hold_df = df[df["exit_reason"] != "STOP"]
    print(f"  Stop ({STOP_PCT*100:.2f}%)      : {n_stop}/{N} stopped ({n_stop/N*100:.1f}%)  "
          f"avg net% stop:{stop_df['net_pnl_pct'].mean():+.4f}  "
          f"avg net% hold:{hold_df['net_pnl_pct'].mean():+.4f}")
print()

print("--- By Side " + "-" * 60)
for label, sdf in [("LONG", long_df), ("SHORT", short_df)]:
    if sdf.empty:
        continue
    sw = (sdf["net_pnl_pct"] > 0).sum()
    print(f"  {label:5s}  Trades:{len(sdf):3d}  WR:{sw/len(sdf)*100:5.1f}%  "
          f"Avg%:{sdf['net_pnl_pct'].mean():+.4f}%  "
          f"Net Rs:{sdf['net_pnl_inr'].sum():+,.2f}  "
          f"Best:{sdf['pnl_pct'].max():+.4f}%  Worst:{sdf['pnl_pct'].min():+.4f}%")
print()

print("--- Daily PnL " + "-" * 57)
print(f"  {'Date':<12} {'Trd':>4} {'WR%':>6} {'Net Rs':>11} {'Avg%':>9}"
      f" {'Best%':>9} {'Worst%':>9} {'Cum Rs':>11}")
print("  " + "-" * 77)
for _, r in daily.iterrows():
    print(f"  {str(r.date):<12} {r.trades:>4d} {r.win_rate:>5.1f}%"
          f" {r.net_pnl_inr:>+11,.2f} {r.avg_pct:>+9.4f}%"
          f" {r.best_pct:>+9.4f}% {r.worst_pct:>+9.4f}%"
          f" {r.cum_pnl:>+11,.2f}")
print("  " + "-" * 77)
print(f"  {'TOTAL':<12} {daily.trades.sum():>4d} {wr:>5.1f}% {tot_net:>+11,.2f}")
print()

print("--- All Trades " + "-" * 57)
show = df[["entry_time","ticker","side","entry_px","exit_px",
           "pnl_pct","net_pnl_pct","qty","net_pnl_inr","cum_pnl_inr","score_15m"]].copy()
show["entry_time"] = show["entry_time"].dt.strftime("%Y-%m-%d %H:%M")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")
print(show.to_string())

OUT = Path("data/backtests")
OUT.mkdir(parents=True, exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d_%H%M")
df.to_csv(OUT / f"v20_v3_true_oos_trades_{stamp}.csv")
daily.to_csv(OUT / f"v20_v3_true_oos_daily_{stamp}.csv", index=False)
print(f"\n[SAVED] Trades -> data/backtests/v20_v3_true_oos_trades_{stamp}.csv")
print(f"[SAVED] Daily  -> data/backtests/v20_v3_true_oos_daily_{stamp}.csv")

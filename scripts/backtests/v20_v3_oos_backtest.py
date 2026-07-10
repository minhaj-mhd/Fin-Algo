"""
v20 + v3 15min  |  Walk-Forward OOS Backtest  |  Last Fold (Fold 8)
====================================================================
OOS Window  : 2025-11-03 to 2026-06-04  (fold 8 - model never trained on this)
Signal      : v20_rolling_1h  -> top-K by Long/Short Conviction
Gate        : v3_15min_clean  -> top/bottom 15% by score_15m
Hold        : 1 hour
Cost        : 10 bps round-trip (Gauntlet standard)
Capital     : Rs 99,517.68  |  5x margin  |  max 5 slots

Run: python -m scripts.backtests.v20_v3_oos_backtest
"""

import sys, os, json
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime, timedelta
from pathlib import Path

# ── CONFIG ─────────────────────────────────────────────────────────────────────
V20_PANEL   = "data/research/v20_rolling_1h/panel.parquet"
V3_PANEL    = "data/ranking_data_upstox_15min_3y_clean.csv"
V20_META    = "models/research/v20_rolling_1h/metadata.json"
V3_META     = "models/v3_15min_clean/metadata.json"
V20_LONG    = "models/research/v20_rolling_1h/xgb_long_model.json"
V20_SHORT   = "models/research/v20_rolling_1h/xgb_short_model.json"
V3_LONG     = "models/v3_15min_clean/xgb_long_model.json"
V3_SHORT    = "models/v3_15min_clean/xgb_short_model.json"

# Fold 8 OOS dates (last fold - model never saw this data)
OOS_START   = "2025-11-03"
OOS_END     = "2026-06-04"

GATE_PCT    = 0.15        # top/bottom 15% of score_15m per scan
TOP_K       = 3           # picks per direction per scan
COST_RT     = 10 / 10000 # 10 bps round-trip as fraction
MAX_SLOTS   = 5
CAPITAL     = 99517.68
MARGIN      = 5.0
SLOT_CAP    = CAPITAL / MAX_SLOTS

# ── BANNER ─────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  v20 Rolling-1H + v3 15min  |  WF OOS Backtest  |  Fold 8")
print(f"  OOS  : {OOS_START} to {OOS_END}")
print(f"  Gate : top/bottom {int(GATE_PCT*100)}% by score_15m  |  Top-K : {TOP_K}")
print(f"  Cost : {int(COST_RT*10000)} bps RT  |  Capital : Rs{CAPITAL:,.0f}  |  Margin : {MARGIN}x")
print("=" * 70)

# ── LOAD MODELS ────────────────────────────────────────────────────────────────
print("\n[1/5] Loading models ...")
with open(V20_META) as f: v20_feats = json.load(f)["features"]
with open(V3_META)  as f: v3_feats  = json.load(f)["features"]

bst_v20l = xgb.Booster(); bst_v20l.load_model(V20_LONG)
bst_v20s = xgb.Booster(); bst_v20s.load_model(V20_SHORT)
bst_v3l  = xgb.Booster(); bst_v3l.load_model(V3_LONG)
bst_v3s  = xgb.Booster(); bst_v3s.load_model(V3_SHORT)
print(f"     v20 features : {len(v20_feats)}   v3 features : {len(v3_feats)}")

# ── LOAD OOS PANELS ────────────────────────────────────────────────────────────
print("[2/5] Loading panels (please wait) ...")

v20p = pd.read_parquet(V20_PANEL)
v20p["DateTime"] = pd.to_datetime(v20p["DateTime"])
v20p = v20p.sort_values(["DateTime", "Ticker"]).reset_index(drop=True)
v20_oos = v20p[(v20p["DateTime"] >= OOS_START) & (v20p["DateTime"] <= OOS_END)].copy()
print(f"     v20 OOS rows : {len(v20_oos):,}  |  timestamps : {v20_oos['DateTime'].nunique()}")
del v20p

print("     Loading v3 15min panel ...")
usecols = ["DateTime", "Ticker"] + v3_feats
v3p = pd.read_csv(V3_PANEL, usecols=usecols)
v3p["DateTime"] = pd.to_datetime(v3p["DateTime"])
v3p = v3p.sort_values(["DateTime", "Ticker"]).reset_index(drop=True)
v3_oos = v3p[(v3p["DateTime"] >= OOS_START) & (v3p["DateTime"] <= OOS_END)].copy()
print(f"     v3 OOS rows  : {len(v3_oos):,}  |  timestamps : {v3_oos['DateTime'].nunique()}")
del v3p

# ── SCORING HELPERS ────────────────────────────────────────────────────────────
print("[3/5] Setting up scoring pipeline ...")

def score_v20(cs):
    X = np.nan_to_num(cs[v20_feats].values.astype(np.float32))
    dm = xgb.DMatrix(X, feature_names=v20_feats)
    l = bst_v20l.predict(dm); s = bst_v20s.predict(dm)
    lc = (l - l.mean()) - (s - s.mean())
    sc = (s - s.mean()) - (l - l.mean())
    out = cs[["Ticker", "Close"]].copy()
    out["long_conv"]  = lc
    out["short_conv"] = sc
    out["long_score"] = l
    out["short_score"] = s
    return out

def score_v3(cs):
    X = np.nan_to_num(cs[v3_feats].values.astype(np.float32))
    dm = xgb.DMatrix(X, feature_names=v3_feats)
    l = bst_v3l.predict(dm); s = bst_v3s.predict(dm)
    lstd = l.std(); sstd = s.std()
    ln = (l - l.mean()) / lstd if lstd > 1e-9 else (l - l.mean())
    sn = (s - s.mean()) / sstd if sstd > 1e-9 else (s - s.mean())
    out = cs[["Ticker"]].copy()
    out["score_15m"] = ln - sn
    return out

# ── SIMULATION ─────────────────────────────────────────────────────────────────
print("[4/5] Running walk-forward simulation ...")

scan_ts  = sorted(v20_oos["DateTime"].unique())
v3_ts    = pd.Series(sorted(v3_oos["DateTime"].unique()))
trades   = []
open_pos = {}   # ticker -> dict
n_ts     = len(scan_ts)

for i, ts in enumerate(scan_ts):
    if i % 300 == 0:
        pct = i / n_ts * 100
        print(f"     scan {i:5d}/{n_ts}  ({pct:.0f}%)  {ts.date()} {str(ts.time())[:5]}"
              f"  open={len(open_pos)}")

    # ── close positions at their scheduled exit time ───────────────────────────
    to_close = [t for t, p in open_pos.items() if ts >= p["exit_ts"]]
    for ticker in to_close:
        pos = open_pos.pop(ticker)
        exit_rows = v20_oos[(v20_oos["DateTime"] == pos["exit_ts"]) &
                             (v20_oos["Ticker"] == ticker)]
        if exit_rows.empty:
            # fallback: next available row
            exit_rows = v20_oos[(v20_oos["DateTime"] > pos["exit_ts"]) &
                                 (v20_oos["Ticker"] == ticker)]
        exit_px = float(exit_rows.iloc[0]["Close"]) if not exit_rows.empty else pos["entry_px"]

        ep  = pos["entry_px"]
        ret = (exit_px - ep) / ep if pos["side"] == "LONG" else (ep - exit_px) / ep
        net = ret - COST_RT
        qty = int((SLOT_CAP * MARGIN) / ep) if ep > 0 else 0
        pnl_inr = net * ep * qty

        trades.append({
            "entry_time" : pos["entry_ts"],
            "exit_time"  : ts,
            "date"       : pos["entry_ts"].date(),
            "ticker"     : ticker,
            "side"       : pos["side"],
            "entry_px"   : round(ep, 2),
            "exit_px"    : round(exit_px, 2),
            "pnl_pct"    : round(ret * 100, 4),
            "net_pnl_pct": round(net * 100, 4),
            "qty"        : qty,
            "net_pnl_inr": round(pnl_inr, 2),
            "long_conv"  : pos["long_conv"],
            "short_conv" : pos["short_conv"],
            "score_15m"  : pos["score_15m"],
            "long_score" : pos["long_score"],
            "short_score": pos["short_score"],
        })

    # ── get cross-sections ─────────────────────────────────────────────────────
    cs_v20 = v20_oos[v20_oos["DateTime"] == ts]
    if len(cs_v20) < 10:
        continue

    # nearest v3 ts <= current ts (within 20 min)
    v3_candidates = v3_ts[v3_ts <= ts]
    if v3_candidates.empty:
        continue
    v3_nearest = v3_candidates.iloc[-1]
    if (ts - v3_nearest).total_seconds() > 1200:
        continue
    cs_v3 = v3_oos[v3_oos["DateTime"] == v3_nearest]
    if len(cs_v3) < 10:
        continue

    # ── score ──────────────────────────────────────────────────────────────────
    try:
        sc20 = score_v20(cs_v20)
        sc3  = score_v3(cs_v3)
    except Exception:
        continue

    merged = sc20.merge(sc3, on="Ticker", how="inner")
    if len(merged) < 10:
        continue

    s15        = merged["score_15m"]
    long_thr   = s15.quantile(1.0 - GATE_PCT)
    short_thr  = s15.quantile(GATE_PCT)

    # ── pick signals ───────────────────────────────────────────────────────────
    for side in ("LONG", "SHORT"):
        if side == "LONG":
            cands = merged[merged["score_15m"] >= long_thr].sort_values(
                "long_conv", ascending=False)
        else:
            cands = merged[merged["score_15m"] <= short_thr].sort_values(
                "short_conv", ascending=False)

        picked = 0
        for _, row in cands.iterrows():
            if picked >= TOP_K:
                break
            ticker = row["Ticker"]
            if ticker in open_pos or len(open_pos) >= MAX_SLOTS:
                continue
            px = row["Close"]
            if pd.isna(px) or px <= 0:
                continue

            open_pos[ticker] = {
                "side"      : side,
                "entry_px"  : float(px),
                "entry_ts"  : ts,
                "exit_ts"   : ts + timedelta(hours=1),
                "long_conv" : float(row["long_conv"]),
                "short_conv": float(row["short_conv"]),
                "score_15m" : float(row["score_15m"]),
                "long_score": float(row["long_score"]),
                "short_score":float(row["short_score"]),
            }
            picked += 1

# flush remaining open positions
for ticker, pos in open_pos.items():
    last = v20_oos[v20_oos["Ticker"] == ticker].sort_values("DateTime").tail(1)
    exit_px = float(last["Close"].iloc[0]) if not last.empty else pos["entry_px"]
    ep = pos["entry_px"]
    ret = (exit_px - ep) / ep if pos["side"] == "LONG" else (ep - exit_px) / ep
    net = ret - COST_RT
    qty = int((SLOT_CAP * MARGIN) / ep) if ep > 0 else 0
    trades.append({
        "entry_time" : pos["entry_ts"],
        "exit_time"  : None,
        "date"       : pos["entry_ts"].date(),
        "ticker"     : ticker,
        "side"       : pos["side"],
        "entry_px"   : round(ep, 2),
        "exit_px"    : round(exit_px, 2),
        "pnl_pct"    : round(ret * 100, 4),
        "net_pnl_pct": round(net * 100, 4),
        "qty"        : qty,
        "net_pnl_inr": round(net * ep * qty, 2),
        "long_conv"  : pos["long_conv"],
        "short_conv" : pos["short_conv"],
        "score_15m"  : pos["score_15m"],
        "long_score" : pos["long_score"],
        "short_score": pos["short_score"],
    })

# ── RESULTS ────────────────────────────────────────────────────────────────────
print("[5/5] Computing results ...")

df = pd.DataFrame(trades)
if df.empty:
    print("No trades generated.")
    sys.exit(0)

df["entry_time"] = pd.to_datetime(df["entry_time"])
df = df.sort_values("entry_time").reset_index(drop=True)
df.index += 1
df["cum_pnl_inr"] = df["net_pnl_inr"].cumsum()

N         = len(df)
wins      = (df["net_pnl_pct"] > 0).sum()
wr        = wins / N * 100
tot_net   = df["net_pnl_inr"].sum()
avg_pct   = df["net_pnl_pct"].mean()
avg_win   = df[df["net_pnl_pct"] > 0]["net_pnl_pct"].mean()
avg_loss  = df[df["net_pnl_pct"] <= 0]["net_pnl_pct"].mean()
roc       = tot_net / CAPITAL * 100
cum       = df["net_pnl_inr"].cumsum()
max_dd    = (cum - cum.cummax()).min()
profit_factor = (df[df["net_pnl_pct"] > 0]["net_pnl_inr"].sum() /
                 abs(df[df["net_pnl_pct"] <= 0]["net_pnl_inr"].sum() + 1e-9))

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

# ── PRINT ──────────────────────────────────────────────────────────────────────
SEP = "=" * 70
print(f"\n{SEP}")
print(f"  RESULTS  |  v20 1H + v3 15min (top {int(GATE_PCT*100)}% gate)  |  WF Fold 8 OOS")
print(SEP)
print(f"  Period          : {OOS_START}  to  {OOS_END}")
print(f"  Total Trades    : {N:,}")
print(f"  Win Rate        : {wr:.1f}%")
print(f"  Avg Net PnL/trade: {avg_pct:+.4f}%")
print(f"  Avg Win         : {avg_win:+.4f}%   Avg Loss : {avg_loss:+.4f}%")
print(f"  Profit Factor   : {profit_factor:.2f}")
print(f"  Total Net PnL   : Rs {tot_net:+,.2f}")
print(f"  Return on Cap   : {roc:+.3f}%  (on Rs {CAPITAL:,.0f})")
print(f"  Max Drawdown    : Rs {max_dd:,.2f}")
print()

print("--- By Side " + "-" * 58)
for label, sdf in [("LONG", long_df), ("SHORT", short_df)]:
    if sdf.empty:
        continue
    sw = (sdf["net_pnl_pct"] > 0).sum()
    print(f"  {label:5s}  Trades:{len(sdf):4d}  WR:{sw/len(sdf)*100:5.1f}%  "
          f"Avg%:{sdf['net_pnl_pct'].mean():+.4f}%  "
          f"Net: Rs {sdf['net_pnl_inr'].sum():+,.2f}  "
          f"Best:{sdf['pnl_pct'].max():+.4f}%  Worst:{sdf['pnl_pct'].min():+.4f}%")
print()

print("--- Daily PnL " + "-" * 56)
hdr = f"  {'Date':<12} {'Trades':>6} {'WR%':>6} {'Net PnL Rs':>13} {'Avg%':>9} {'Best%':>9} {'Worst%':>9} {'CumPnL Rs':>13}"
print(hdr)
print("  " + "-" * (len(hdr) - 2))
for _, r in daily.iterrows():
    print(f"  {str(r['date']):<12} {r.trades:>6d} {r.win_rate:>5.1f}%"
          f" {r.net_pnl_inr:>+13,.2f} {r.avg_pct:>+9.4f}%"
          f" {r.best_pct:>+9.4f}% {r.worst_pct:>+9.4f}%"
          f" {r.cum_pnl:>+13,.2f}")
print("  " + "-" * (len(hdr) - 2))
print(f"  {'TOTAL':<12} {daily.trades.sum():>6d} {wr:>5.1f}% {tot_net:>+13,.2f}")
print()

print("--- All Trades " + "-" * 55)
show = df[["entry_time", "ticker", "side", "entry_px", "exit_px",
           "pnl_pct", "net_pnl_pct", "qty", "net_pnl_inr", "cum_pnl_inr",
           "score_15m", "long_conv", "short_conv"]].copy()
show["entry_time"] = show["entry_time"].dt.strftime("%Y-%m-%d %H:%M")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")
print(show.to_string())

# ── SAVE ───────────────────────────────────────────────────────────────────────
OUT = Path("data/backtests")
OUT.mkdir(parents=True, exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d_%H%M")
trades_path = OUT / f"v20_v3_fold8_trades_{stamp}.csv"
daily_path  = OUT / f"v20_v3_fold8_daily_{stamp}.csv"
df.to_csv(trades_path)
daily.to_csv(daily_path, index=False)
print(f"\n[SAVED] Trades -> {trades_path}")
print(f"[SAVED] Daily  -> {daily_path}")

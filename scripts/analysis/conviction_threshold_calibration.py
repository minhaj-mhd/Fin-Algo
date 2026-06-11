"""Threshold calibration analysis for the live model lineup.
Reconstructs the live engine's conviction quantity (per-query mean-centered
long score minus centered short score, as in model_inference.py L221) from
genuine OOS gauntlet predictions, and derives rank-based gate equivalents.
"""
import numpy as np
import pandas as pd

RUNS = {
    "v10_native_1h (1H)":   "data/gauntlet/20260610T183204Z-d795438c",
    "v3_15min_clean (15M)": "data/gauntlet/20260610T113721Z-5f7d069f",
    "daily_macro_v2 (D3d)": "data/gauntlet/20260610T135608Z-5f7d069f",
    "daily_macro_v3 (D1d)": "data/gauntlet/20260610T144343Z-5f7d069f",
}

for name, run_dir in RUNS.items():
    try:
        npz = np.load(f"{run_dir}/preds.npz")
    except Exception as e:
        print(f"{name}: preds load failed -> {e}")
        continue
    q = npz["q"]
    rl = npz["rl"].astype(np.float64)
    rs = npz["rs"].astype(np.float64)

    df = pd.DataFrame({"q": q, "rl": rl, "rs": rs})
    g = df.groupby("q")
    df["rl_c"] = df["rl"] - g["rl"].transform("mean")
    df["rs_c"] = df["rs"] - g["rs"].transform("mean")
    df["conv_long"] = df["rl_c"] - df["rs_c"]

    # distribution of long conviction across all OOS rows
    pct = np.percentile(df["conv_long"], [50, 75, 90, 95, 98, 99, 100])
    # per-query top-3 boundary (3rd highest conviction in each query)
    top3_bound = g["conv_long"].apply(lambda s: s.nlargest(3).min() if len(s) >= 3 else np.nan).dropna()
    # per-query top-1 boundary
    top1_bound = g["conv_long"].max()
    # how many rows would pass the hardcoded 0.08 gate
    pass08 = (df["conv_long"] >= 0.08).mean()
    n_q = df["q"].nunique()
    avg_per_q = len(df) / n_q

    print(f"=== {name} ===")
    print(f"  rows {len(df):,} | queries {n_q:,} | avg tickers/query {avg_per_q:.0f}")
    print(f"  conviction percentiles  p50 {pct[0]:+.4f}  p75 {pct[1]:+.4f}  p90 {pct[2]:+.4f}  "
          f"p95 {pct[3]:+.4f}  p98 {pct[4]:+.4f}  p99 {pct[5]:+.4f}  max {pct[6]:+.4f}")
    print(f"  top-3 boundary per query: median {top3_bound.median():+.4f}  "
          f"p25 {top3_bound.quantile(.25):+.4f}  p75 {top3_bound.quantile(.75):+.4f}")
    print(f"  top-1 boundary per query: median {top1_bound.median():+.4f}")
    print(f"  rows passing hardcoded 0.08 gate: {pass08:.2%}")
    print()

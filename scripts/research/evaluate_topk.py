"""
evaluate_topk.py — Top-K Precision Evaluator for Ranking Models

Answers: "Of the top-3 stocks the model ranks highest each hour,
          how often do they actually outperform the median stock?"

This is MORE relevant than Spearman rho for the Vanguard use case
because we only trade top-3, not rank all 170 stocks.

Usage:
  python scripts/evaluate_topk.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

sys.path.append(os.getcwd())

# ============================================================
# CONFIG
# ============================================================
# Use the latest data file available (Upstox > v3 > v2)
if os.path.exists("data/ranking_data_upstox_3y.csv"):
    DATA_FILE = "data/ranking_data_upstox_3y.csv"
elif os.path.exists("data/ranking_data_upstox.csv"):
    DATA_FILE = "data/ranking_data_upstox.csv"
elif os.path.exists("data/ranking_data_v3.csv"):
    DATA_FILE = "data/ranking_data_v3.csv"
elif os.path.exists("data/ranking_data_v2.csv"):
    DATA_FILE = "data/ranking_data_v2.csv"
else:
    DATA_FILE = "data/ranking_data_full.csv"

TOPK      = [1, 3, 5]   # evaluate precision at these cut-offs

MODELS = {
    "v1_yfinance": {
        "long":  "models/xgb_long_model.json",
        "short": "models/xgb_short_model.json",
        "meta":  "models/model_metadata.json",
    },
    "v2_feature_fix": {
        "long":  "models/v2_feature_fix/xgb_long_model.json",
        "short": "models/v2_feature_fix/xgb_short_model.json",
        "meta":  "models/v2_feature_fix/metadata.json",
    },
    "v3_sector_momentum": {
        "long":  "models/v3_sector_momentum/xgb_long_model.json",
        "short": "models/v3_sector_momentum/xgb_short_model.json",
        "meta":  "models/v3_sector_momentum/metadata.json",
    },
    "v6_feature_fixed": {
        "long":  "models/v6_feature_fixed/xgb_long_model.json",
        "short": "models/v6_feature_fixed/xgb_short_model.json",
        "meta":  "models/v6_feature_fixed/metadata.json",
    },
    "v8_upstox_3y": {
        "long":  "models/v8_upstox_3y/xgb_long_model.json",
        "short": "models/v8_upstox_3y/xgb_short_model.json",
        "meta":  "models/v8_upstox_3y/metadata.json",
    },
}

print("=" * 65)
print("TOP-K PRECISION EVALUATOR")
print("=" * 65)

# ============================================================
# LOAD DATA
# ============================================================
print(f"\nLoading {DATA_FILE}...")
df = pd.read_csv(DATA_FILE)
print(f"  {df.shape[0]:,} rows | {df['Query_ID'].nunique():,} queries")

EXCLUDE = {
    'DateTime', 'DateTime_Hour', 'Query_ID', 'Ticker', 'Next_Hour_Return',
    'Open', 'High', 'Low', 'Close', 'Volume'
}

# Temporal split — only evaluate on TEST set (last 20%)
unique_qids = np.sort(df['Query_ID'].unique())
split_idx   = int(len(unique_qids) * 0.8)
test_qids   = unique_qids[split_idx:]
df_test     = df[df['Query_ID'].isin(test_qids)].copy()

print(f"  Test set: {df_test.shape[0]:,} rows | {len(test_qids):,} queries")

# ============================================================
# EVALUATE EACH MODEL
# ============================================================
results = {}

for model_name, paths in MODELS.items():
    # Skip if model files don't exist
    if not all(os.path.exists(p) for p in paths.values()):
        print(f"\n[SKIP] {model_name} — files not found")
        continue

    print(f"\n{'=' * 65}")
    print(f"EVALUATING: {model_name}")
    print(f"{'=' * 65}")

    # Load feature list from metadata
    with open(paths["meta"]) as f:
        meta = json.load(f)
    feature_cols = meta["features"]

    # Check all features exist in v2 data
    missing = [c for c in feature_cols if c not in df_test.columns]
    if missing:
        print(f"  [WARN] {len(missing)} features missing in test data: {missing[:5]}")
        feature_cols = [c for c in feature_cols if c in df_test.columns]

    X_test = df_test[feature_cols].values
    X_test = np.nan_to_num(X_test)

    bst_long  = xgb.Booster(); bst_long.load_model(paths["long"])
    bst_short = xgb.Booster(); bst_short.load_model(paths["short"])

    dmatrix = xgb.DMatrix(X_test)
    long_scores  = bst_long.predict(dmatrix)
    short_scores = bst_short.predict(dmatrix)

    df_test = df_test.copy()
    df_test[f"{model_name}_long_score"]  = long_scores
    df_test[f"{model_name}_short_score"] = short_scores

    # ── TOP-K PRECISION ───────────────────────────────────────────────────────
    # For each hour (query), find top-K ranked stocks by model score.
    # A "hit" = the selected stock's Next_Hour_Return > median return for that hour.
    # Precision@K = hits / total top-K selections across all hours.

    print(f"\n  TOP-K PRECISION (LONG MODEL):")
    print(f"  {'K':>4}  {'Precision':>10}  {'Random':>10}  {'Lift':>8}  {'Queries':>8}")
    print(f"  " + "-" * 48)

    long_precisions  = {}
    short_precisions = {}

    for k in TOPK:
        long_hits   = 0
        short_hits  = 0
        total_picks = 0
        valid_q     = 0

        for qid in test_qids:
            q_mask = df_test['Query_ID'] == qid
            q_df   = df_test[q_mask]

            if len(q_df) < k + 1:
                continue

            actual_returns  = q_df['Next_Hour_Return'].values
            median_return   = np.median(actual_returns)

            # LONG: top-K by long score
            long_sc  = q_df[f"{model_name}_long_score"].values
            short_sc = q_df[f"{model_name}_short_score"].values

            top_long_idx  = np.argsort(long_sc)[::-1][:k]
            top_short_idx = np.argsort(short_sc)[::-1][:k]

            # Long hit: selected stock's return > median (beat the market)
            long_hits  += (actual_returns[top_long_idx]  > median_return).sum()
            short_hits += (actual_returns[top_short_idx] < median_return).sum()  # short wants below-median
            total_picks += k
            valid_q += 1

        prec_long  = long_hits  / total_picks if total_picks > 0 else 0
        prec_short = short_hits / total_picks if total_picks > 0 else 0
        random_p   = 0.5  # random = 50% chance of beating median

        lift_long  = prec_long  / random_p - 1
        lift_short = prec_short / random_p - 1

        long_precisions[k]  = prec_long
        short_precisions[k] = prec_short

        print(f"  {k:>4}  {prec_long:>9.1%}  {random_p:>9.1%}  {lift_long:>+7.1%}  {valid_q:>8,}")

    print(f"\n  TOP-K PRECISION (SHORT MODEL):")
    print(f"  {'K':>4}  {'Precision':>10}  {'Random':>10}  {'Lift':>8}")
    print(f"  " + "-" * 40)
    for k in TOPK:
        prec_short = short_precisions[k]
        lift_short = prec_short / 0.5 - 1
        print(f"  {k:>4}  {prec_short:>9.1%}  {50.0:>9.1%}  {lift_short:>+7.1%}")

    # ── AVERAGE RETURN OF TOP-K VS RANDOM ────────────────────────────────────
    print(f"\n  AVERAGE RETURN: Top-3 Long selections vs Random pick")
    top3_returns = []
    random_returns = []
    short3_returns = []

    for qid in test_qids:
        q_mask = df_test['Query_ID'] == qid
        q_df   = df_test[q_mask]
        if len(q_df) < 4:
            continue

        actual  = q_df['Next_Hour_Return'].values
        long_sc = q_df[f"{model_name}_long_score"].values
        short_sc = q_df[f"{model_name}_short_score"].values

        top3_idx   = np.argsort(long_sc)[::-1][:3]
        short3_idx = np.argsort(short_sc)[::-1][:3]

        top3_returns.append(actual[top3_idx].mean())
        random_returns.append(actual.mean())
        short3_returns.append(-actual[short3_idx].mean())  # short = negative return is profit

    if top3_returns:
        print(f"    Top-3 Long  avg return per hour : {np.mean(top3_returns)*100:+.4f}%")
        print(f"    Random pick avg return per hour : {np.mean(random_returns)*100:+.4f}%")
        print(f"    Top-3 Short avg return per hour : {np.mean(short3_returns)*100:+.4f}% (short P&L)")
        print(f"    Long  edge over random          : {(np.mean(top3_returns) - np.mean(random_returns))*100:+.4f}%")

    results[model_name] = {
        "long_prec@1":  long_precisions.get(1),
        "long_prec@3":  long_precisions.get(3),
        "long_prec@5":  long_precisions.get(5),
        "short_prec@3": short_precisions.get(3),
        "avg_top3_long_return": float(np.mean(top3_returns)) if top3_returns else None,
        "avg_random_return": float(np.mean(random_returns)) if random_returns else None,
    }

# ============================================================
# COMPARISON SUMMARY
# ============================================================
print(f"\n{'=' * 65}")
print("SUMMARY COMPARISON")
print(f"{'=' * 65}")
fmt = "{:<20} {:>12} {:>12} {:>12} {:>12}"
print(fmt.format("Model", "Long P@1", "Long P@3", "Long P@5", "Short P@3"))
print("-" * 65)
for name, r in results.items():
    print(fmt.format(
        name,
        f"{r['long_prec@1']:.1%}"  if r['long_prec@1']  else "N/A",
        f"{r['long_prec@3']:.1%}"  if r['long_prec@3']  else "N/A",
        f"{r['long_prec@5']:.1%}"  if r['long_prec@5']  else "N/A",
        f"{r['short_prec@3']:.1%}" if r['short_prec@3'] else "N/A",
    ))

print(f"\nRandom baseline: 50.0% at all K")
print(f"\nInterpretation:")
print(f"  P@K > 52% = model adds edge over random stock picking")
print(f"  P@K > 55% = strong edge, worth live testing")
print(f"  P@K > 60% = very strong edge")
print(f"{'=' * 65}")

# Save results
os.makedirs("data", exist_ok=True)
with open("data/topk_eval_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: data/topk_eval_results.json")

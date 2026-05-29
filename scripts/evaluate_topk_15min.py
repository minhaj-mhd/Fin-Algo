"""
evaluate_topk_15min.py — Top-K Precision and Win Rate Evaluator for the 15-Min Model

Evaluates the trained final production model on the final out-of-sample month (2026-05).
Answers: "Of the top stocks the model ranks highest every 15 minutes,
          how often do they actually outperform the median stock?
          What is the average 15-minute return edge?"

Usage:
  python scripts/evaluate_topk_15min.py
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
DATA_FILE = "data/ranking_data_upstox_15min_1y.csv"
MODEL_DIR = "models/v1_15min"
LONG_MODEL_PATH = f"{MODEL_DIR}/xgb_long_model.json"
SHORT_MODEL_PATH = f"{MODEL_DIR}/xgb_short_model.json"
META_PATH = f"{MODEL_DIR}/metadata.json"

TOPK = [1, 3, 5]

print("=" * 65)
print("15-MIN MODEL WIN RATE & TOP-K PRECISION EVALUATOR")
print("Evaluating Final Production Model on Out-of-Sample Month (2026-05)")
print("=" * 65)

# ============================================================
# LOAD DATA & MODELS
# ============================================================
if not os.path.exists(DATA_FILE):
    print(f"[FATAL] Data file not found: {DATA_FILE}")
    sys.exit(1)

if not os.path.exists(LONG_MODEL_PATH) or not os.path.exists(SHORT_MODEL_PATH) or not os.path.exists(META_PATH):
    print(f"[FATAL] Model files not found in {MODEL_DIR}. Please train the model first.")
    sys.exit(1)

print(f"\nLoading data from {DATA_FILE}...")
df = pd.read_csv(DATA_FILE)
print(f"  {df.shape[0]:,} rows | {df['Query_ID'].nunique():,} queries")

# Load metadata
with open(META_PATH) as f:
    meta = json.load(f)
feature_cols = meta["features"]
print(f"  Features loaded: {len(feature_cols)}")

# Temporal split — evaluate on out-of-sample holdout month (May 2026)
df['YearMonth'] = df['DateTime'].str[:7]
df_test = df[df['YearMonth'] == '2026-05'].copy()
test_qids = np.sort(df_test['Query_ID'].unique())

print(f"  Test set (2026-05): {df_test.shape[0]:,} rows | {len(test_qids):,} queries")

if len(test_qids) == 0:
    print("[FATAL] No test queries found for month 2026-05.")
    sys.exit(1)

# ============================================================
# PREDICT
# ============================================================
X_test = df_test[feature_cols].values
X_test = np.nan_to_num(X_test)

print("\nLoading models and predicting on test set...")
bst_long  = xgb.Booster()
bst_long.load_model(LONG_MODEL_PATH)

bst_short = xgb.Booster()
bst_short.load_model(SHORT_MODEL_PATH)

dmatrix = xgb.DMatrix(X_test)
long_scores  = bst_long.predict(dmatrix)
short_scores = bst_short.predict(dmatrix)

df_test = df_test.copy()
df_test["long_score"]  = long_scores
df_test["short_score"] = short_scores

# ============================================================
# TOP-K WIN RATE (PRECISION)
# ============================================================
# Win rate = how often the top-K selections beat the median return of that 15-min block.
print(f"\n{'=' * 65}")
print(f"WIN RATE (PRECISION) ON TEMPORAL TEST SET")
print(f"{'=' * 65}")

long_precisions = {}
short_precisions = {}

print(f"\n  LONG MODEL WIN RATES (Beats Median Return):")
print(f"  {'K':>4}  {'Win Rate':>10}  {'Random':>10}  {'Lift':>8}  {'Total Picks':>12}  {'Queries':>8}")
print(f"  " + "-" * 58)

for k in TOPK:
    long_hits = 0
    total_picks = 0
    valid_q = 0

    for qid in test_qids:
        q_mask = df_test['Query_ID'] == qid
        q_df   = df_test[q_mask]

        if len(q_df) < k + 1:
            continue

        actual_returns  = q_df['Next_15Min_Return'].values
        median_return   = np.median(actual_returns)

        long_sc  = q_df["long_score"].values
        top_long_idx  = np.argsort(long_sc)[::-1][:k]

        # Long win: selected stock's return > median (beat the cross-sectional average)
        long_hits  += (actual_returns[top_long_idx] > median_return).sum()
        total_picks += k
        valid_q += 1

    win_rate = long_hits / total_picks if total_picks > 0 else 0
    random_rate = 0.50
    lift = win_rate / random_rate - 1
    long_precisions[k] = win_rate

    print(f"  {k:>4}  {win_rate:>9.1%}  {random_rate:>9.1%}  {lift:>+7.1%}  {total_picks:>12,}  {valid_q:>8,}")

print(f"\n  SHORT MODEL WIN RATES (Below Median Return):")
print(f"  {'K':>4}  {'Win Rate':>10}  {'Random':>10}  {'Lift':>8}  {'Total Picks':>12}  {'Queries':>8}")
print(f"  " + "-" * 58)

for k in TOPK:
    short_hits = 0
    total_picks = 0
    valid_q = 0

    for qid in test_qids:
        q_mask = df_test['Query_ID'] == qid
        q_df   = df_test[q_mask]

        if len(q_df) < k + 1:
            continue

        actual_returns  = q_df['Next_15Min_Return'].values
        median_return   = np.median(actual_returns)

        short_sc = q_df["short_score"].values
        top_short_idx = np.argsort(short_sc)[::-1][:k]

        # Short win: selected stock's return < median (fell below the cross-sectional average)
        short_hits += (actual_returns[top_short_idx] < median_return).sum()
        total_picks += k
        valid_q += 1

    win_rate = short_hits / total_picks if total_picks > 0 else 0
    random_rate = 0.50
    lift = win_rate / random_rate - 1
    short_precisions[k] = win_rate

    print(f"  {k:>4}  {win_rate:>9.1%}  {random_rate:>9.1%}  {lift:>+7.1%}  {total_picks:>12,}  {valid_q:>8,}")

# ============================================================
# EXPECTED 15-MINUTE RETURNS
# ============================================================
print(f"\n{'=' * 65}")
print(f"EXPECTED 15-MIN RETURNS: TOP-3 SELECTIONS VS BASES")
print(f"{'=' * 65}")

top3_long_returns = []
top3_short_returns = []
random_returns = []

for qid in test_qids:
    q_mask = df_test['Query_ID'] == qid
    q_df   = df_test[q_mask]
    if len(q_df) < 4:
        continue

    actual  = q_df['Next_15Min_Return'].values
    long_sc = q_df["long_score"].values
    short_sc = q_df["short_score"].values

    top3_long_idx = np.argsort(long_sc)[::-1][:3]
    top3_short_idx = np.argsort(short_sc)[::-1][:3]

    top3_long_returns.append(actual[top3_long_idx].mean())
    top3_short_returns.append(-actual[top3_short_idx].mean()) # Shorting captures negative returns
    random_returns.append(actual.mean())

if top3_long_returns:
    avg_long  = np.mean(top3_long_returns)
    avg_short = np.mean(top3_short_returns)
    avg_rand  = np.mean(random_returns)

    print(f"  Top-3 Long Selections  Avg Return: {avg_long*100:+.4f}% per 15-min bar")
    print(f"  Random Pick (Market)   Avg Return: {avg_rand*100:+.4f}% per 15-min bar")
    print(f"  Top-3 Short Selections Avg Return: {avg_short*100:+.4f}% per 15-min bar (Short P&L)")
    print()
    print(f"  Long Edge over Market  : {(avg_long - avg_rand)*100:+.4f}% per bar")
    print(f"  Short Edge over Market : {(avg_short - (-avg_rand))*100:+.4f}% per bar")
    print(f"  Combined Long/Short Edge: {(avg_long + avg_short)*100:+.4f}% per bar")

# Save results
eval_results = {
    "long_win_rates": {str(k): float(v) for k, v in long_precisions.items()},
    "short_win_rates": {str(k): float(v) for k, v in short_precisions.items()},
    "top3_long_avg_return": float(np.mean(top3_long_returns)) if top3_long_returns else None,
    "top3_short_avg_return": float(np.mean(top3_short_returns)) if top3_short_returns else None,
    "market_avg_return": float(np.mean(random_returns)) if random_returns else None,
}

output_results_path = "data/topk_eval_15min_results.json"
with open(output_results_path, "w") as f:
    json.dump(eval_results, f, indent=2)
print(f"\nSaved evaluation summary to: {output_results_path}")
print("=" * 65)
print()

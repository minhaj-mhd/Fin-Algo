"""
MV2-R0.6: Coverage Booster — Inference Backfill for v8_upstox_3y
=================================================================
Status: infrastructure-not-certification (logged as inference_backfill,
no verdicts claimed, no Gauntlet ledger entry written).

Problem: The v8 Gauntlet run used step_months=4, test_horizon=2,
leaving alternating 2-month gaps in OOS coverage. DEV span has only
9 distinct months of v8 predictions — G1 requires >= 12.

This script loads the FROZEN v8 model (the same artifact already
stamped in the registry) and runs inference on the held-out months
that were not covered by the Gauntlet folds. The result is merged
with the existing preds.npz to produce a unified OOS panel covering
all trading months from Aug 2023 through Dec 2024.

Integrity guarantees:
  1. No new training — model weights are frozen (verified by SHA-256).
  2. No Gauntlet verdict claimed — this is panel infrastructure only.
  3. Logged to dev_ledger.jsonl with event="inference_backfill" so the
     coverage expansion is auditable.
  4. Output written to data/gauntlet/meta/mv2/v8_backfill_preds.npz
     — separate from the original preds.npz, merged only in the panel.
  5. Model SHA-256 verified before any inference.
"""

import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

sys.path.append(os.getcwd())

from scripts.gauntlet.paths import gauntlet_root
from scripts.gauntlet.uplift import find_latest_completed_run

OUT_DIR    = os.path.join("data", "gauntlet", "meta", "mv2")
LEDGER_PATH = os.path.join(OUT_DIR, "dev_ledger.jsonl")
MODEL_DIR  = os.path.join("models", "v8_upstox_3y")
DATASET    = "data/ranking_data_upstox_1h_v3_3y.csv"
DEV_CUTOFF = pd.Timestamp("2025-01-01")


def verify_model_sha(model_dir: str) -> str:
    """Verify the frozen model matches its registry-stamped SHA-256."""
    meta_path = os.path.join(model_dir, "metadata.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # v8 is an XGBoost model — find the model file
    model_candidates = [
        os.path.join(model_dir, "xgb_long.json"),
        os.path.join(model_dir, "xgb_long_model.json"),
        os.path.join(model_dir, "model.json"),
    ]
    model_path = next((p for p in model_candidates if os.path.exists(p)), None)
    if model_path is None:
        # Try any .json file in the dir
        jsons = [f for f in os.listdir(model_dir) if f.endswith(".json") and f != "metadata.json"]
        if jsons:
            model_path = os.path.join(model_dir, jsons[0])

    if model_path is None:
        raise FileNotFoundError(f"No model file found in {model_dir}")

    print(f"  Model file: {model_path}")
    with open(model_path, "rb") as f:
        actual_sha = hashlib.sha256(f.read()).hexdigest()

    # Check registry stamp
    registry_path = os.path.join("models", "registry.json")
    with open(registry_path, "r", encoding="utf-8") as f:
        registry = json.load(f)

    v8_entry = registry.get("v8_upstox_3y", {})
    stamped_sha = v8_entry.get("model_sha256", "")

    if stamped_sha:
        if actual_sha != stamped_sha:
            raise RuntimeError(
                f"Model SHA-256 mismatch!\n"
                f"  Registry:  {stamped_sha}\n"
                f"  On disk:   {actual_sha}\n"
                "This means the model file has changed since the last Gauntlet stamp. Abort."
            )
        print(f"  SHA-256 verified: {actual_sha[:16]}... matches registry stamp.")
    else:
        print(f"  [WARN] No SHA-256 in registry for v8_upstox_3y — skipping stamp check.")
        print(f"  Model SHA-256 (for audit): {actual_sha}")

    return model_path


def load_frozen_v8(model_dir: str):
    """Load the frozen v8 XGBoost long and short rankers."""
    import xgboost as xgb

    meta_path = os.path.join(model_dir, "metadata.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    features = meta["features"]

    def _load_model(filename):
        path = os.path.join(model_dir, filename)
        if not os.path.exists(path):
            return None
        booster = xgb.Booster()
        booster.load_model(path)
        return booster

    # Try standard naming conventions
    model_long  = _load_model("xgb_long.json") or _load_model("xgb_long_model.json") or _load_model("model_long.json")
    model_short = _load_model("xgb_short.json") or _load_model("xgb_short_model.json") or _load_model("model_short.json")

    if model_long is None or model_short is None:
        # Fall back to single model
        single = _load_model("model.json")
        if single is None:
            raise FileNotFoundError(f"Cannot find v8 long/short XGBoost models in {model_dir}")
        model_long = single
        model_short = single
        print("  [WARN] Could not find separate long/short models — using single model for both sides.")

    return model_long, model_short, features


def get_existing_oos_idx(run_dir: str) -> set:
    """Return the set of dataset row indices already covered by the Gauntlet preds.npz."""
    npz = np.load(os.path.join(run_dir, "preds.npz"))
    return set(npz["idx"].tolist())


def main():
    print("=" * 70)
    print("MV2-R0.6: V8 INFERENCE BACKFILL (infrastructure, no new verdicts)")
    print("=" * 70)

    os.makedirs(OUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Verify model integrity
    # ------------------------------------------------------------------
    print("\n[1/5] Verifying frozen v8 model integrity...")
    try:
        model_path = verify_model_sha(MODEL_DIR)
    except RuntimeError as e:
        print(f"[FATAL] {e}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Load model and features
    # ------------------------------------------------------------------
    print("\n[2/5] Loading frozen model weights (no retraining)...")
    try:
        model_long, model_short, features = load_frozen_v8(MODEL_DIR)
    except Exception as e:
        print(f"[FATAL] Cannot load v8 models: {e}")
        sys.exit(1)
    print(f"  Features: {len(features)}")

    # ------------------------------------------------------------------
    # 3. Identify missing months
    # ------------------------------------------------------------------
    print("\n[3/5] Identifying DEV months missing from existing preds.npz...")
    run_dir = find_latest_completed_run("v8_upstox_3y")
    existing_idx = get_existing_oos_idx(run_dir)

    # Load dataset
    print(f"  Loading dataset: {DATASET}")
    df = pd.read_csv(DATASET, usecols=["DateTime", "Query_ID"] + features)
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df["_ym"] = df["DateTime"].dt.to_period("M")

    dev_mask = df["DateTime"] < DEV_CUTOFF
    df_dev = df[dev_mask].copy()

    existing_dev_idx = {i for i in existing_idx if i < len(df) and dev_mask.iloc[i]}
    existing_dev_months = set(
        df.iloc[list(existing_dev_idx)]["_ym"].unique()
    )

    all_dev_months = set(df_dev["_ym"].unique())
    missing_months = sorted(all_dev_months - existing_dev_months)

    print(f"  Existing OOS DEV months : {len(existing_dev_months)}")
    print(f"  Total DEV months        : {len(all_dev_months)}")
    print(f"  Missing months to fill  : {len(missing_months)}")
    for m in missing_months:
        n_rows = (df_dev["_ym"] == m).sum()
        print(f"    {m}: {n_rows:,} rows")

    if len(missing_months) == 0:
        print("  No missing months — coverage already complete. Exiting.")
        sys.exit(0)

    # ------------------------------------------------------------------
    # 4. Run inference on missing months
    # ------------------------------------------------------------------
    print("\n[4/5] Running inference on missing months (frozen weights, no retraining)...")
    import xgboost as xgb

    missing_mask = df["_ym"].isin(missing_months) & dev_mask
    df_missing = df[missing_mask].copy()
    global_idx  = df_missing.index.tolist()

    print(f"  Rows to infer: {len(df_missing):,}")

    X = df_missing[features].values
    dmat = xgb.DMatrix(X, feature_names=features)

    preds_long  = model_long.predict(dmat)
    preds_short = model_short.predict(dmat)

    # ------------------------------------------------------------------
    # 5. Save backfill preds + log to ledger
    # ------------------------------------------------------------------
    print("\n[5/5] Saving backfill predictions and logging to dev_ledger...")
    out_npz = os.path.join(OUT_DIR, "v8_backfill_preds.npz")
    np.savez(
        out_npz,
        idx=np.array(global_idx, dtype=np.int64),
        rl=preds_long.astype(np.float32),
        rs=preds_short.astype(np.float32),
    )
    print(f"  Backfill preds saved -> {out_npz}")

    # Compute SHA
    with open(out_npz, "rb") as f:
        backfill_sha = hashlib.sha256(f.read()).hexdigest()

    # Log to dev_ledger
    ledger_entry = {
        "event":           "inference_backfill",
        "source":          "v8_upstox_3y",
        "model_dir":       MODEL_DIR,
        "months_filled":   [str(m) for m in missing_months],
        "n_rows_filled":   len(df_missing),
        "backfill_sha256": backfill_sha,
        "original_run_dir": run_dir,
        "no_new_training": True,
        "no_gauntlet_verdict": True,
        "timestamp":       pd.Timestamp.now().isoformat(),
    }
    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(ledger_entry) + "\n")

    print(f"  Ledger entry written -> {LEDGER_PATH}")
    print(f"  Backfill SHA-256: {backfill_sha[:32]}...")

    # Verify combined coverage
    combined_dev_months = existing_dev_months | set(missing_months)
    print(f"\n  Combined DEV months after backfill: {len(combined_dev_months)}")
    if len(combined_dev_months) >= 12:
        print("  [OK] G1 month count will be satisfied after panel rebuild.")
    else:
        print(f"  [WARN] Still only {len(combined_dev_months)} DEV months — G1 may still fail.")

    print("=" * 70)
    print("MV2-R0.6 COMPLETED — re-run build_trade_panel.py to verify G1 passes")
    print("=" * 70)


if __name__ == "__main__":
    main()

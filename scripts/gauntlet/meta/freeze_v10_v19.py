"""
MV2-R1: freeze_v10_v19.py — Freeze the Winning Candidate from dev_ledger
===========================================================================
Trains the final model on the full DEV panel of v10-v19 models and writes
a hash-chained candidate to models/meta_veto_v10_v19/.
"""

import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
import joblib

sys.path.append(os.getcwd())

import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONFIG_DIR    = os.path.join("scripts", "gauntlet", "meta", "config")
MODEL_SPEC    = os.path.join(CONFIG_DIR, "meta_model_spec_v10_v19.yaml")
PANEL_PATH    = os.path.join("data", "gauntlet", "meta", "mv_v10_v19", "trade_panel.parquet")
LEDGER_PATH   = os.path.join("data", "gauntlet", "meta", "mv_v10_v19", "dev_ledger.jsonl")
CANDIDATE_DIR = os.path.join("models", "meta_veto_v10_v19")
COST_RATE     = 0.0010

COMPLEXITY_RANK = {"logistic": 1, "gbm_shallow": 2, "mlp_small": 3}


def load_ledger(ledger_path: str) -> list:
    if not os.path.exists(ledger_path):
        raise FileNotFoundError(f"dev_ledger.jsonl not found at {ledger_path}. Run dev_run_v10_v19.py first.")
    entries = []
    with open(ledger_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    e = json.loads(line)
                    if e.get("event", "experiment") == "experiment":
                        entries.append(e)
                except Exception:
                    pass
    return entries


def select_winner(entries: list) -> dict:
    if not entries:
        raise RuntimeError("dev_ledger.jsonl has no model-training entries. Run dev_run_v10_v19.py first.")

    def sort_key(e):
        return (-e.get("dev_oof_kept_net_bps", -9999),
                COMPLEXITY_RANK.get(e.get("class", "logistic"), 99))

    sorted_entries = sorted(entries, key=sort_key)
    winner = sorted_entries[0]
    return winner


def build_model(model_class: str, params: dict, seed: int = 42):
    if model_class == "logistic":
        return LogisticRegression(
            penalty="l2", C=params.get("C", 0.1),
            solver="lbfgs", max_iter=1000, random_state=seed
        )
    elif model_class == "gbm_shallow":
        return GradientBoostingClassifier(
            max_depth=params.get("max_depth", 2),
            n_estimators=params.get("n_estimators", 100),
            learning_rate=params.get("learning_rate", 0.05),
            subsample=0.8, random_state=seed
        )
    elif model_class == "mlp_small":
        return MLPClassifier(
            hidden_layer_sizes=tuple(params.get("hidden_layer_sizes", [32])),
            alpha=params.get("alpha", 0.01),
            early_stopping=True,
            random_state=seed,
            max_iter=500,
        )
    raise ValueError(f"Unknown model class: {model_class}")


def main():
    print("=" * 70)
    print("MV2-R1: FREEZE WINNING CANDIDATE FOR v10-v19")
    print("=" * 70)

    with open(MODEL_SPEC, "r") as f:
        mspec = yaml.safe_load(f)

    primary_endpoint = mspec["primary_endpoint"]
    print(f"  Primary endpoint: {primary_endpoint}")

    # 1. Load ledger and select winner
    print("\n[1/5] Loading dev_ledger and selecting winner...")
    entries = load_ledger(LEDGER_PATH)
    n_experiments = len(entries)
    print(f"  Total experiments logged: {n_experiments}")

    winner = select_winner(entries)
    print(f"\n  Winner:")
    print(f"    Class    : {winner['class']}")
    print(f"    Params   : {winner['params']}")
    print(f"    Net bps  : {winner['dev_oof_kept_net_bps']:+.2f} bps")
    print(f"    Keep pct : {winner['dev_oof_keep_pct']:.1%}")
    print(f"    Theta    : {winner['theta']:.2f}")

    # G2 pre-check
    if winner["dev_oof_kept_net_bps"] <= 0.0 or winner["dev_oof_keep_pct"] > 0.90:
        raise RuntimeError(
            f"[G2 PRE-CHECK FAILED] Winning candidate has net={winner['dev_oof_kept_net_bps']:+.2f} bps "
            f"and keep={winner['dev_oof_keep_pct']:.1%}. Certifier will refuse this candidate."
        )
    print(f"\n  [G2 PRE-CHECK OK] Candidate has positive net and keep <= 90%.")

    # 2. Load DEV panel and train final model
    print("\n[2/5] Training final model on the full DEV panel...")
    df = pd.read_parquet(PANEL_PATH)
    dev_df = df[df["span"] == "DEV"].copy().reset_index(drop=True)

    META_COLS    = {"model", "datetime", "ticker", "side", "trade_return", "Query_ID", "span", "y", "proposed_by"}
    feature_cols = [c for c in df.columns if c not in META_COLS]
    feature_cols = [
        c for c in feature_cols
        if pd.api.types.is_numeric_dtype(dev_df[c]) and dev_df[c].std() > 1e-8
    ]

    X_dev = dev_df[feature_cols].values
    y_dev = dev_df["y"].values

    # Impute missing values with full DEV median
    col_medians = np.nanmedian(X_dev, axis=0)
    col_medians = np.nan_to_num(col_medians, nan=0.0)
    for j in range(X_dev.shape[1]):
        X_dev[np.isnan(X_dev[:, j]), j] = col_medians[j]

    # Scale features
    scaler = StandardScaler()
    X_dev_scaled = scaler.fit_transform(X_dev)

    # Train model
    model = build_model(winner["class"], winner["params"], seed=42)
    model.fit(X_dev_scaled, y_dev)
    print(f"  Model trained on {len(dev_df):,} DEV rows.")

    # 3. Create models candidate directory and save files
    print(f"\n[3/5] Saving to {CANDIDATE_DIR}...")
    os.makedirs(CANDIDATE_DIR, exist_ok=True)

    model_path  = os.path.join(CANDIDATE_DIR, "model.joblib")
    scaler_path = os.path.join(CANDIDATE_DIR, "scaler.joblib")
    medians_path = os.path.join(CANDIDATE_DIR, "col_medians.npy")
    meta_path   = os.path.join(CANDIDATE_DIR, "candidate_metadata.json")

    joblib.dump(model,  model_path)
    joblib.dump(scaler, scaler_path)
    np.save(medians_path, col_medians)

    with open(model_path, "rb") as f:
        model_sha = hashlib.sha256(f.read()).hexdigest()

    # Get panel hash for audit trail
    with open(PANEL_PATH, "rb") as f:
        panel_sha = hashlib.sha256(f.read()).hexdigest()

    # 4. Build hash-chained candidate metadata (G3: endpoint sealed here)
    candidate_hash_input = f"{model_sha}|{panel_sha}|{primary_endpoint}"
    candidate_hash       = hashlib.sha256(candidate_hash_input.encode()).hexdigest()

    metadata = {
        "mv_version":                "mv_v10_v19",
        "model_class":               winner["class"],
        "model_params":              winner["params"],
        "features":                  feature_cols,
        "theta":                     winner["theta"],
        "dev_oof_kept_net_bps":      winner["dev_oof_kept_net_bps"],
        "dev_oof_keep_pct":          winner["dev_oof_keep_pct"],
        "n_dev_experiments_tried":   n_experiments,
        "primary_endpoint":          primary_endpoint,   # G3: SEALED HERE
        "panel_sha256":              panel_sha,
        "model_sha256":              model_sha,
        "candidate_hash":            candidate_hash,     # hash chain
        "n_dev_rows":                int(len(dev_df)),
        "timestamp":                 pd.Timestamp.now().isoformat(),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Model saved    : {model_path}")
    print(f"  Scaler saved   : {scaler_path}")
    print(f"  Metadata saved : {meta_path}")
    print(f"  Model SHA-256  : {model_sha[:32]}...")
    print(f"  Candidate hash : {candidate_hash[:32]}...")

    # 5. Print freeze summary
    print("\n" + "=" * 60)
    print("FREEZE SUMMARY")
    print("=" * 60)
    print(f"  Primary endpoint (sealed): {primary_endpoint}")
    print(f"  Model class              : {winner['class']}")
    print(f"  DEV OOF kept-net         : {winner['dev_oof_kept_net_bps']:+.2f} bps")
    print(f"  DEV experiments logged   : {n_experiments}")
    print(f"  Candidate hash           : {candidate_hash}")
    print("\nNext step:")
    print(f"  python scripts/gauntlet/meta/certify_meta_veto_v10_v19.py --primary-endpoint {primary_endpoint}")
    print("=" * 70)
    print("MV2-R1 FREEZE COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()

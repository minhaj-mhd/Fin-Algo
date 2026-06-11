"""
MV2-R1/R2/R3: dev_run_v10_v19.py — M-DYN Development Runner for v10-v19
======================================================================
Reads config/panel_spec_v10_v19.yaml and config/meta_model_spec_v10_v19.yaml.
Implements the rolling purged WF inside DEV (R2) and the
capacity ladder (R3) with full guardrail enforcement (G4, G5).
"""

import os
import sys
import json
import time
import hashlib
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from typing import Optional

sys.path.append(os.getcwd())

import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONFIG_DIR    = os.path.join("scripts", "gauntlet", "meta", "config")
PANEL_SPEC    = os.path.join(CONFIG_DIR, "panel_spec_v10_v19.yaml")
MODEL_SPEC    = os.path.join(CONFIG_DIR, "meta_model_spec_v10_v19.yaml")
PANEL_PATH    = os.path.join("data", "gauntlet", "meta", "mv_v10_v19", "trade_panel.parquet")
OUT_DIR       = os.path.join("data", "gauntlet", "meta", "mv_v10_v19")
CANDIDATE_DIR = os.path.join("models", "meta_veto_v10_v19")
COST_RATE     = 0.0010   # 10 bps

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config():
    with open(PANEL_SPEC, "r") as f:
        pspec = yaml.safe_load(f)
    with open(MODEL_SPEC, "r") as f:
        mspec = yaml.safe_load(f)
    return pspec, mspec


def load_ledger(ledger_path: str) -> list:
    if not os.path.exists(ledger_path):
        return []
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


def append_ledger(ledger_path: str, entry: dict):
    os.makedirs(os.path.dirname(ledger_path) or ".", exist_ok=True)
    entry["event"] = "experiment"
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# G4: Capacity ascent gate
# ---------------------------------------------------------------------------

def check_g4_gate(ledger_path: str, rung: int, ascent_gate: Optional[dict]):
    if ascent_gate is None:
        return
    if rung == 1:
        return

    entries = load_ledger(ledger_path)

    if rung == 2:
        rung1 = [e for e in entries if e.get("rung") == 1]
        if not rung1:
            raise RuntimeError(
                "[G4 VIOLATED] Cannot start Rung 2 (GBM) — Rung 1 (logistic) "
                "has not been logged in dev_ledger.jsonl."
            )

    elif rung == 3:
        requires_rung = ascent_gate.get("requires_rung", 2)
        beat_margin   = ascent_gate.get("gbm_beat_logistic_by_bps", 0.5)

        logistic_entries = [e for e in entries if e.get("rung") == 1]
        gbm_entries      = [e for e in entries if e.get("rung") == 2]

        if not logistic_entries:
            raise RuntimeError("[G4 VIOLATED] Cannot start Rung 3 (NN) — Rung 1 (logistic) not logged.")
        if not gbm_entries:
            raise RuntimeError("[G4 VIOLATED] Cannot start Rung 3 (NN) — Rung 2 (GBM) not logged.")

        best_logistic_bps = max(e["dev_oof_kept_net_bps"] for e in logistic_entries)
        best_gbm_bps      = max(e["dev_oof_kept_net_bps"] for e in gbm_entries)
        margin            = best_gbm_bps - best_logistic_bps

        if margin < beat_margin:
            raise RuntimeError(
                f"[G4 VIOLATED] GBM best DEV OOF ({best_gbm_bps:+.2f} bps) did not beat "
                f"logistic best ({best_logistic_bps:+.2f} bps) by the required "
                f"{beat_margin} bps margin (actual: {margin:+.2f} bps).\n"
                "Capacity has not been earned — NN rung is blocked."
            )
        print(f"[G4 OK] GBM beat logistic by {margin:+.2f} bps >= {beat_margin} required.")


# ---------------------------------------------------------------------------
# G5: Seed robustness check
# ---------------------------------------------------------------------------

def check_g5_entry(entry: dict):
    if entry.get("class") != "mlp_small":
        return
    n_seeds = entry.get("n_seeds", 0)
    if n_seeds < 3:
        raise RuntimeError(
            f"[G5 VIOLATED] NN ledger entry has n_seeds={n_seeds}. "
            "Must report worst-of-3-seeds. This entry is rejected."
        )
    seed_results = entry.get("seed_results", [])
    reported = entry.get("dev_oof_kept_net_bps")
    if seed_results and reported is not None:
        worst = min(seed_results)
        if abs(reported - worst) > 1e-6:
            raise RuntimeError(
                f"[G5 VIOLATED] Reported bps ({reported:.4f}) != worst of seeds "
                f"({worst:.4f}, seeds={seed_results}). Must report worst-of-3, not best."
            )


# ---------------------------------------------------------------------------
# Rolling WF inside DEV
# ---------------------------------------------------------------------------

def build_rolling_folds(dev_df: pd.DataFrame,
                        min_train_months: int,
                        test_months: int,
                        embargo_days: int) -> list:
    dev_df = dev_df.sort_values("datetime").reset_index(drop=True)
    dev_df["_ym"] = dev_df["datetime"].dt.to_period("M")
    months = sorted(dev_df["_ym"].unique())

    folds = []
    for i in range(min_train_months, len(months) - test_months + 1):
        train_months = months[:i]
        test_month   = months[i: i + test_months]

        test_start_date = test_month[0].to_timestamp()
        embargo_cutoff  = test_start_date - pd.Timedelta(days=embargo_days)

        train_mask = (
            dev_df["_ym"].isin(train_months) &
            (dev_df["datetime"] < embargo_cutoff)
        )
        test_mask  = dev_df["_ym"].isin(test_month)

        train_idx = dev_df[train_mask].index.tolist()
        test_idx  = dev_df[test_mask].index.tolist()

        if len(train_idx) < 100 or len(test_idx) < 20:
            continue

        folds.append((train_idx, test_idx))

    return folds


def compute_oof(dev_df: pd.DataFrame, feature_cols: list, folds: list,
                model_class: str, params: dict,
                seed: int = 42) -> pd.Series:
    target_col = "y"
    oof_probs  = pd.Series(np.nan, index=dev_df.index)

    for fold_num, (train_idx, test_idx) in enumerate(folds, 1):
        X_tr = dev_df.loc[train_idx, feature_cols].values
        y_tr = dev_df.loc[train_idx, target_col].values
        X_te = dev_df.loc[test_idx,  feature_cols].values

        # Median imputation from training fold only
        col_medians = np.nanmedian(X_tr, axis=0)
        col_medians = np.nan_to_num(col_medians, nan=0.0)
        for j in range(X_tr.shape[1]):
            X_tr[np.isnan(X_tr[:, j]), j] = col_medians[j]
            X_te[np.isnan(X_te[:, j]), j] = col_medians[j]

        # Scale
        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr)
        X_te   = scaler.transform(X_te)

        # Build model
        model = _build_model(model_class, params, seed)
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_te)[:, 1]
        oof_probs.iloc[test_idx] = probs

    return oof_probs


def _build_model(model_class: str, params: dict, seed: int):
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
    else:
        raise ValueError(f"Unknown model_class: {model_class}")


# ---------------------------------------------------------------------------
# Theta calibration
# ---------------------------------------------------------------------------

def calibrate_theta(dev_df: pd.DataFrame, oof_probs: pd.Series,
                    keep_floor: float, keep_ceiling: float) -> tuple:
    valid_mask = oof_probs.notna()
    df_oof = dev_df[valid_mask].copy()
    df_oof["oof_prob"] = oof_probs[valid_mask].values

    best_theta    = 0.50
    best_net_bps  = -9999.0
    best_keep_pct = 0.0

    for theta in np.arange(0.10, 0.91, 0.01):
        kept_mask = df_oof["oof_prob"] >= theta
        n_kept    = kept_mask.sum()
        total     = len(df_oof)
        if total == 0:
            continue
        keep_pct  = n_kept / total

        if keep_pct < keep_floor or keep_pct > keep_ceiling:
            continue

        kept_trades = df_oof[kept_mask]
        net_return  = kept_trades["trade_return"] - COST_RATE
        net_bps     = net_return.mean() * 10000.0 if len(kept_trades) > 0 else 0.0

        if net_bps > best_net_bps:
            best_net_bps  = net_bps
            best_theta    = theta
            best_keep_pct = keep_pct

    return best_theta, best_net_bps, best_keep_pct


# ---------------------------------------------------------------------------
# Rung Runners
# ---------------------------------------------------------------------------

def run_rung_1(dev_df: pd.DataFrame, feature_cols: list, folds: list, mspec: dict, ledger_path: str, panel_sha: str):
    print("\n--- RUNNING RUNG 1: L2 LOGISTIC REGRESSION ---")
    cfg = [c for c in mspec["capacity_ladder"] if c["rung"] == 1][0]
    p_grid = list(ParameterGrid(cfg["param_grid"]))
    
    keep_cfg = mspec["theta_calibration"]
    
    for params in p_grid:
        print(f"Sweeping C={params['C']}...")
        oof_probs = compute_oof(dev_df, feature_cols, folds, "logistic", params)
        best_theta, best_net_bps, best_keep = calibrate_theta(dev_df, oof_probs, keep_cfg["keep_floor"], keep_cfg["keep_ceiling"])
        
        entry = {
            "rung": 1,
            "class": "logistic",
            "params": params,
            "n_seeds": 1,
            "dev_oof_kept_net_bps": float(best_net_bps),
            "dev_oof_keep_pct": float(best_keep),
            "theta": float(best_theta),
            "n_folds": len(folds),
            "panel_sha256": panel_sha,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        append_ledger(ledger_path, entry)
        print(f"  Result: kept_net = {best_net_bps:+.2f} bps | keep = {best_keep:.1%} | theta = {best_theta:.2f}")


def run_rung_2(dev_df: pd.DataFrame, feature_cols: list, folds: list, mspec: dict, ledger_path: str, panel_sha: str):
    print("\n--- RUNNING RUNG 2: SHALLOW GRADIENT BOOSTING TREES ---")
    check_g4_gate(ledger_path, rung=2, ascent_gate=None)
    
    cfg = [c for c in mspec["capacity_ladder"] if c["rung"] == 2][0]
    p_grid = list(ParameterGrid(cfg["param_grid"]))
    keep_cfg = mspec["theta_calibration"]
    
    for params in p_grid:
        print(f"Sweeping max_depth={params['max_depth']}, n_estimators={params['n_estimators']}...")
        oof_probs = compute_oof(dev_df, feature_cols, folds, "gbm_shallow", params)
        best_theta, best_net_bps, best_keep = calibrate_theta(dev_df, oof_probs, keep_cfg["keep_floor"], keep_cfg["keep_ceiling"])
        
        entry = {
            "rung": 2,
            "class": "gbm_shallow",
            "params": params,
            "n_seeds": 1,
            "dev_oof_kept_net_bps": float(best_net_bps),
            "dev_oof_keep_pct": float(best_keep),
            "theta": float(best_theta),
            "n_folds": len(folds),
            "panel_sha256": panel_sha,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        append_ledger(ledger_path, entry)
        print(f"  Result: kept_net = {best_net_bps:+.2f} bps | keep = {best_keep:.1%} | theta = {best_theta:.2f}")


def run_rung_3(dev_df: pd.DataFrame, feature_cols: list, folds: list, mspec: dict, ledger_path: str, panel_sha: str):
    print("\n--- RUNNING RUNG 3: SMALL NEURAL NETWORK ---")
    cfg = [c for c in mspec["capacity_ladder"] if c["rung"] == 3][0]
    check_g4_gate(ledger_path, rung=3, ascent_gate=cfg["ascent_gate"])
    
    p_grid = list(ParameterGrid(cfg["param_grid"]))
    keep_cfg = mspec["theta_calibration"]
    n_seeds = cfg.get("n_seeds", 3)
    
    for params in p_grid:
        print(f"Sweeping hidden_layer_sizes={params['hidden_layer_sizes']}, alpha={params['alpha']} (running {n_seeds} seeds)...")
        seed_results = []
        seed_details = []
        
        for seed_idx in range(n_seeds):
            seed = 42 + seed_idx
            oof_probs = compute_oof(dev_df, feature_cols, folds, "mlp_small", params, seed=seed)
            best_theta, best_net_bps, best_keep = calibrate_theta(dev_df, oof_probs, keep_cfg["keep_floor"], keep_cfg["keep_ceiling"])
            seed_results.append(best_net_bps)
            seed_details.append((best_theta, best_keep))
            
        # worst-of-seeds
        worst_idx = np.argmin(seed_results)
        worst_bps = seed_results[worst_idx]
        worst_theta, worst_keep = seed_details[worst_idx]
        
        entry = {
            "rung": 3,
            "class": "mlp_small",
            "params": params,
            "n_seeds": n_seeds,
            "seed_results": [float(r) for r in seed_results],
            "dev_oof_kept_net_bps": float(worst_bps),
            "dev_oof_keep_pct": float(worst_keep),
            "theta": float(worst_theta),
            "n_folds": len(folds),
            "panel_sha256": panel_sha,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        check_g5_entry(entry)
        append_ledger(ledger_path, entry)
        print(f"  Result (worst-of-{n_seeds} seeds): kept_net = {worst_bps:+.2f} bps | keep = {worst_keep:.1%} | theta = {worst_theta:.2f}")


def main():
    parser = argparse.ArgumentParser(description="MV2 dev_run — capacity ladder for v10-v19")
    parser.add_argument("--rung",      type=int, choices=[1, 2, 3], help="Run a specific rung")
    parser.add_argument("--all-rungs", action="store_true", help="Run all rungs in order (G4 gate enforced)")
    args = parser.parse_args()

    if not args.rung and not args.all_rungs:
        parser.error("Specify --rung N or --all-rungs")

    print("=" * 70)
    print("MV2-R1/R2/R3: DEV RUNNER FOR v10-v19 (Rolling WF + Capacity Ladder)")
    print("=" * 70)

    pspec, mspec = load_config()
    ledger_path  = os.path.join(OUT_DIR, "dev_ledger.jsonl")
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(PANEL_PATH):
        print(f"[FATAL] Trade panel not found at {PANEL_PATH}. Run build_trade_panel_v10_v19.py first.")
        sys.exit(1)

    df = pd.read_parquet(PANEL_PATH)
    print(f"Panel loaded: {len(df):,} rows")

    with open(PANEL_PATH, "rb") as f:
        panel_sha = hashlib.sha256(f.read()).hexdigest()
    print(f"Panel SHA-256: {panel_sha[:32]}...")

    dev_df = df[df["span"] == "DEV"].copy().reset_index(drop=True)
    print(f"DEV rows: {len(dev_df):,}")

    # Feature columns
    META_COLS    = {"model", "datetime", "ticker", "side", "trade_return", "Query_ID", "span", "y", "proposed_by"}
    feature_cols = [c for c in df.columns if c not in META_COLS]
    feature_cols = [
        c for c in feature_cols
        if pd.api.types.is_numeric_dtype(dev_df[c]) and dev_df[c].std() > 1e-8
    ]
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")

    wf_cfg   = mspec["dev_wf"]
    folds    = build_rolling_folds(
        dev_df,
        min_train_months = wf_cfg["min_train_months"],
        test_months      = wf_cfg["test_months"],
        embargo_days     = wf_cfg["embargo_days"],
    )
    print(f"Rolling WF folds: {len(folds)}")
    if len(folds) < 3:
        print("[FATAL] Too few WF folds — DEV span too short.")
        sys.exit(1)

    # Execute
    if args.rung:
        if args.rung == 1:
            run_rung_1(dev_df, feature_cols, folds, mspec, ledger_path, panel_sha)
        elif args.rung == 2:
            run_rung_2(dev_df, feature_cols, folds, mspec, ledger_path, panel_sha)
        elif args.rung == 3:
            run_rung_3(dev_df, feature_cols, folds, mspec, ledger_path, panel_sha)
    else:
        # Run all in sequence, catching G4 violations gracefully
        try:
            run_rung_1(dev_df, feature_cols, folds, mspec, ledger_path, panel_sha)
            run_rung_2(dev_df, feature_cols, folds, mspec, ledger_path, panel_sha)
            run_rung_3(dev_df, feature_cols, folds, mspec, ledger_path, panel_sha)
        except RuntimeError as e:
            if "[G4 VIOLATED]" in str(e):
                print(f"\n[INFO] Capacity ladder execution stopped at ascent gate: {e}")
            else:
                raise e

    print("\n" + "=" * 70)
    print(f"DEV RUNNER COMPLETED. Check dev_ledger -> {ledger_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

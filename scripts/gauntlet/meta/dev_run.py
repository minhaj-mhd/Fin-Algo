"""
MV2-R1/R2/R3: dev_run.py — M-DYN Development Runner
======================================================
Reads config/panel_spec.yaml and config/meta_model_spec.yaml.
Implements the rolling purged WF inside DEV (R2) and the
capacity ladder (R3) with full guardrail enforcement (G4, G5).

Usage:
  # Run a single rung:
  python scripts/gauntlet/meta/dev_run.py --rung 1
  python scripts/gauntlet/meta/dev_run.py --rung 2
  python scripts/gauntlet/meta/dev_run.py --rung 3

  # Run all rungs (stops at G4 gate if applicable):
  python scripts/gauntlet/meta/dev_run.py --all-rungs

Every experiment is logged to dev_ledger.jsonl.
G4 and G5 are code assertions — they raise RuntimeError if violated.
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
PANEL_SPEC    = os.path.join(CONFIG_DIR, "panel_spec.yaml")
MODEL_SPEC    = os.path.join(CONFIG_DIR, "meta_model_spec.yaml")
PANEL_PATH    = os.path.join("data", "gauntlet", "meta", "mv2_clean", "trade_panel.parquet")
OUT_DIR       = os.path.join("data", "gauntlet", "meta", "mv2_clean")
CANDIDATE_DIR = os.path.join("models", "meta_veto_v2")
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
                    # Only count model-training entries (not backfill/infrastructure)
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
    """
    G4 enforcement: raises RuntimeError if the rung's ascent gate is not satisfied.
    """
    if ascent_gate is None:
        return   # no gate for this rung
    if rung == 1:
        return   # logistic always runs first

    entries = load_ledger(ledger_path)

    if rung == 2:
        # GBM: only check that rung 1 has been logged
        rung1 = [e for e in entries if e.get("rung") == 1]
        if not rung1:
            raise RuntimeError(
                f"[G4 VIOLATED] Cannot start Rung 2 (GBM) — Rung 1 (logistic) "
                "has not been logged in dev_ledger.jsonl. Run --rung 1 first."
            )

    elif rung == 3:
        # NN: GBM must be logged AND must beat logistic by >= beat_by_bps
        requires_rung = ascent_gate.get("requires_rung", 2)
        beat_margin   = ascent_gate.get("gbm_beat_logistic_by_bps", 0.5)

        logistic_entries = [e for e in entries if e.get("rung") == 1]
        gbm_entries      = [e for e in entries if e.get("rung") == 2]

        if not logistic_entries:
            raise RuntimeError(
                "[G4 VIOLATED] Cannot start Rung 3 (NN) — Rung 1 (logistic) not logged."
            )
        if not gbm_entries:
            raise RuntimeError(
                "[G4 VIOLATED] Cannot start Rung 3 (NN) — Rung 2 (GBM) not logged. Run --rung 2 first."
            )

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
    """
    G5 enforcement: NN ledger entries must have n_seeds >= 3 and report worst-of-seeds.
    """
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
    """
    Generate rolling monthly folds over the DEV span.
    Each fold: train on [i - min_train_months, i), embargo gap, test on month i+1.
    Returns list of (train_idx, test_idx) integer index arrays.
    """
    dev_df = dev_df.sort_values("datetime").reset_index(drop=True)
    dev_df["_ym"] = dev_df["datetime"].dt.to_period("M")
    months = sorted(dev_df["_ym"].unique())

    folds = []
    for i in range(min_train_months, len(months) - test_months + 1):
        train_months = months[:i]
        test_month   = months[i: i + test_months]

        # Embargo: exclude rows within embargo_days of the test window start
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
            continue   # skip degenerate folds

        folds.append((train_idx, test_idx))

    return folds


def compute_oof(dev_df: pd.DataFrame, feature_cols: list, folds: list,
                model_class: str, params: dict,
                scaler_fit_on_train: bool = True,
                seed: int = 42) -> pd.Series:
    """
    Run rolling WF and return OOF probability predictions for the full DEV panel.
    Returns a pd.Series indexed by dev_df.index with OOF probs (NaN for train-only rows).
    """
    target_col = "y"
    oof_probs  = pd.Series(np.nan, index=dev_df.index)

    for fold_num, (train_idx, test_idx) in enumerate(folds, 1):
        X_tr = dev_df.loc[train_idx, feature_cols].values
        y_tr = dev_df.loc[train_idx, target_col].values
        X_te = dev_df.loc[test_idx,  feature_cols].values

        # Median imputation from training fold only
        col_medians = np.nanmedian(X_tr, axis=0)
        # If any median is NaN (entire column was NaN in train fold), default to 0.0
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
            # Note: early_stopping uses a random internal val split inside the
            # training fold — acceptable (never touches the test month).
            # Do NOT report the internal validation score as a result (G5 rule).
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
    """
    Grid search theta to maximise DEV OOF kept-net bps.
    Returns (best_theta, best_net_bps, best_keep_pct).
    """
    valid_mask = oof_probs.notna()
    df_oof = dev_df[valid_mask].copy()
    df_oof["oof_prob"] = oof_probs[valid_mask].values

    best_theta    = 0.50
    best_net_bps  = -9999.0
    best_keep_pct = 0.0

    for theta in np.arange(0.10, 0.91, 0.01):
        kept_mask = df_oof["oof_prob"] >= theta
        n_kept    = kept_mask.sum()
        keep_pct  = n_kept / len(df_oof)

        if keep_pct < keep_floor or keep_pct > keep_ceiling:
            continue

        net_rets   = (df_oof.loc[kept_mask, "trade_return"] - COST_RATE).values
        mean_net   = net_rets.mean() * 10000.0  # in bps

        if mean_net > best_net_bps:
            best_net_bps  = mean_net
            best_theta    = float(theta)
            best_keep_pct = float(keep_pct)

    return best_theta, best_net_bps, best_keep_pct


# ---------------------------------------------------------------------------
# G2 pre-check: DEV-promise
# ---------------------------------------------------------------------------

def check_g2_promise(net_bps: float, keep_pct: float):
    """
    G2: refuse to freeze a candidate with DEV OOF net <= 0 bps OR keep > 90%.
    This is checked at freeze time, but dev_run.py logs it for visibility.
    """
    if net_bps <= 0.0 or keep_pct > 0.90:
        print(
            f"[G2 WARN] Candidate violates DEV promise: "
            f"net={net_bps:+.2f} bps, keep={keep_pct:.1%}. "
            "This candidate will be REFUSED by the certifier."
        )
    else:
        print(f"[G2 OK] DEV promise met: net={net_bps:+.2f} bps, keep={keep_pct:.1%}")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_rung(rung_config: dict, dev_df: pd.DataFrame, feature_cols: list,
             folds: list, ledger_path: str, mspec: dict, panel_sha: str):
    """Run one capacity ladder rung, log results to ledger."""
    rung        = rung_config["rung"]
    model_class = rung_config["class"]
    param_grid  = rung_config.get("param_grid", {})
    ascent_gate = rung_config.get("ascent_gate")
    n_seeds     = rung_config.get("n_seeds", 1)
    early_stop  = rung_config.get("early_stopping", False)

    theta_cfg    = mspec["theta_calibration"]
    keep_floor   = theta_cfg["keep_floor"]
    keep_ceiling = theta_cfg["keep_ceiling"]

    # G4 gate
    check_g4_gate(ledger_path, rung, ascent_gate)

    print(f"\n{'='*60}")
    print(f"RUNG {rung}: {model_class.upper()}")
    print(f"{'='*60}")

    # Enumerate param combinations
    if param_grid:
        param_combos = list(ParameterGrid(param_grid))
    else:
        param_combos = [{}]

    print(f"  Param combinations: {len(param_combos)}")
    print(f"  Seeds: {n_seeds}")
    print(f"  Folds: {len(folds)}")

    best_combo_result = None
    all_combo_results = []

    for combo_idx, params in enumerate(param_combos, 1):
        print(f"\n  [{combo_idx}/{len(param_combos)}] Params: {params}")

        if n_seeds >= 3:
            # NN rung: worst-of-3-seeds (G5)
            seed_net_bps = []
            seed_keeps   = []
            seed_thetas  = []

            for seed in [42, 123, 7]:
                oof_probs = compute_oof(
                    dev_df, feature_cols, folds, model_class, params, seed=seed
                )
                theta, net_bps, keep_pct = calibrate_theta(
                    dev_df, oof_probs, keep_floor, keep_ceiling
                )
                seed_net_bps.append(net_bps)
                seed_keeps.append(keep_pct)
                seed_thetas.append(theta)
                print(f"    seed={seed}: net={net_bps:+.2f} bps, keep={keep_pct:.1%}, theta={theta:.2f}")

            # G5: report worst-of-3
            worst_idx  = int(np.argmin(seed_net_bps))
            net_bps    = seed_net_bps[worst_idx]    # worst seed
            keep_pct   = seed_keeps[worst_idx]
            theta      = seed_thetas[worst_idx]
            print(f"    Worst-of-3: net={net_bps:+.2f} bps (seed {[42,123,7][worst_idx]})")

            ledger_entry = {
                "rung":                  rung,
                "class":                 model_class,
                "params":                params,
                "n_seeds":               3,
                "seed_results":          seed_net_bps,
                "dev_oof_kept_net_bps":  net_bps,    # worst seed (G5)
                "dev_oof_keep_pct":      keep_pct,
                "theta":                 theta,
                "n_folds":               len(folds),
                "panel_sha256":          panel_sha,
                "timestamp":             pd.Timestamp.now().isoformat(),
            }
            check_g5_entry(ledger_entry)

        else:
            # Logistic / GBM: single seed
            oof_probs = compute_oof(
                dev_df, feature_cols, folds, model_class, params, seed=42
            )
            theta, net_bps, keep_pct = calibrate_theta(
                dev_df, oof_probs, keep_floor, keep_ceiling
            )
            print(f"    net={net_bps:+.2f} bps, keep={keep_pct:.1%}, theta={theta:.2f}")

            ledger_entry = {
                "rung":                  rung,
                "class":                 model_class,
                "params":                params,
                "n_seeds":               1,
                "dev_oof_kept_net_bps":  net_bps,
                "dev_oof_keep_pct":      keep_pct,
                "theta":                 theta,
                "n_folds":               len(folds),
                "panel_sha256":          panel_sha,
                "timestamp":             pd.Timestamp.now().isoformat(),
            }

        append_ledger(ledger_path, ledger_entry)
        check_g2_promise(net_bps, keep_pct)
        all_combo_results.append(ledger_entry)

        # Track best for this rung
        if best_combo_result is None or net_bps > best_combo_result["dev_oof_kept_net_bps"]:
            best_combo_result = ledger_entry

    print(f"\n  --- Rung {rung} Summary ---")
    print(f"  Best params: {best_combo_result['params']}")
    print(f"  Best DEV OOF kept-net: {best_combo_result['dev_oof_kept_net_bps']:+.2f} bps")
    print(f"  Keep pct: {best_combo_result['dev_oof_keep_pct']:.1%}")
    print(f"  Theta: {best_combo_result['theta']:.2f}")

    return best_combo_result


def main():
    parser = argparse.ArgumentParser(description="MV2 dev_run — capacity ladder with rolling WF")
    parser.add_argument("--rung",      type=int, choices=[1, 2, 3], help="Run a specific rung")
    parser.add_argument("--all-rungs", action="store_true", help="Run all rungs in order (G4 gate enforced)")
    args = parser.parse_args()

    if not args.rung and not args.all_rungs:
        parser.error("Specify --rung N or --all-rungs")

    print("=" * 70)
    print("MV2-R1/R2/R3: DEV RUNNER (Rolling WF + Capacity Ladder)")
    print("=" * 70)

    # Load configs
    pspec, mspec = load_config()
    ledger_path  = os.path.join(OUT_DIR, "dev_ledger.jsonl")
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load panel
    if not os.path.exists(PANEL_PATH):
        print(f"[FATAL] Trade panel not found at {PANEL_PATH}. Run build_trade_panel.py first.")
        sys.exit(1)

    df = pd.read_parquet(PANEL_PATH)
    print(f"Panel loaded: {len(df):,} rows")

    # Panel SHA for audit
    with open(PANEL_PATH, "rb") as f:
        panel_sha = hashlib.sha256(f.read()).hexdigest()
    print(f"Panel SHA-256: {panel_sha[:32]}...")

    # DEV span only
    dev_df = df[df["span"] == "DEV"].copy().reset_index(drop=True)
    print(f"DEV rows: {len(dev_df):,}")

    # Feature columns
    META_COLS    = {"model", "datetime", "ticker", "side", "trade_return", "Query_ID", "span", "y"}
    feature_cols = [c for c in df.columns if c not in META_COLS]
    # Only keep numeric, non-zero-variance features
    feature_cols = [
        c for c in feature_cols
        if pd.api.types.is_numeric_dtype(dev_df[c]) and dev_df[c].std() > 1e-8
    ]
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")

    # Build rolling WF folds
    wf_cfg   = mspec["dev_wf"]
    folds    = build_rolling_folds(
        dev_df,
        min_train_months = wf_cfg["min_train_months"],
        test_months      = wf_cfg["test_months"],
        embargo_days     = wf_cfg["embargo_days"],
    )
    print(f"Rolling WF folds: {len(folds)}")
    if len(folds) < 3:
        print("[FATAL] Too few WF folds — DEV span too short for rolling WF.")
        sys.exit(1)

    # Load capacity ladder
    ladder = mspec["capacity_ladder"]

    # Determine which rungs to run
    if args.all_rungs:
        rungs_to_run = [r["rung"] for r in ladder]
    else:
        rungs_to_run = [args.rung]

    # Prior experiment count (for report)
    prior_entries = load_ledger(ledger_path)
    print(f"\nExisting dev_ledger entries: {len(prior_entries)}")

    # Run each rung
    for rung_num in rungs_to_run:
        rung_config = next((r for r in ladder if r["rung"] == rung_num), None)
        if rung_config is None:
            print(f"[WARN] Rung {rung_num} not found in ladder config — skipping.")
            continue

        run_rung(rung_config, dev_df, feature_cols, folds, ledger_path, mspec, panel_sha)

    # Summary
    all_entries = load_ledger(ledger_path)
    print(f"\n{'='*70}")
    print(f"DEV RUNNER COMPLETED")
    print(f"Total dev_ledger experiments logged: {len(all_entries)}")
    print(f"Ledger: {ledger_path}")
    print(f"{'='*70}")
    print("\nNext step: run freeze.py to freeze the best candidate, then")
    print("           run certify_meta_veto.py --primary-endpoint v8_upstox_3y_long")


if __name__ == "__main__":
    main()

"""
MV2-R4: certify_meta_veto.py — Elevated-Bar Certification
===========================================================
Single-run VAULT evaluation of the frozen MV2 candidate.

Guardrails enforced here:
  G2: refuses candidates with DEV OOF net <= 0 bps OR keep > 90%
  G3: --primary-endpoint is REQUIRED, no default; must match the
      endpoint sealed inside candidate_metadata.json at freeze time.

Elevated thresholds (VAULT was seen once):
  uplift >= +3.0 bps, t >= 2.5   (NOT 2.0)
  absolute kept-net > 0, t >= 2.5 (NOT 2.0)

Verdict map (pre-declared, no changes allowed):
  Both pass -> META_VETO_CERTIFIED (CONDITIONAL) + 3-month shadow
  Uplift only -> FILTER_GRADE (sizing modifier, no capital gate)
  Neither -> LINE CLOSED permanently for price/volume/macro inputs

Usage:
  python scripts/gauntlet/meta/certify_meta_veto.py \\
    --primary-endpoint v8_upstox_3y_long
"""

import os
import sys
import json
import hashlib
import argparse
import numpy as np
import pandas as pd
import joblib
from scipy.stats import ttest_ind, ttest_1samp

sys.path.append(os.getcwd())

import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONFIG_DIR    = os.path.join("scripts", "gauntlet", "meta", "config")
MODEL_SPEC    = os.path.join(CONFIG_DIR, "meta_model_spec.yaml")
PANEL_PATH    = os.path.join("data", "gauntlet", "meta", "mv2_clean", "trade_panel.parquet")
CANDIDATE_DIR = os.path.join("models", "meta_veto_v2")
LEDGER_PATH   = os.path.join("data", "gauntlet", "meta", "mv2_clean", "dev_ledger.jsonl")
REPORT_DIR    = os.path.join("finalgo-memory-layer", "finalgo", "08. Model Analysis", "Meta-Veto")


def bootstrap_uplift_ci(fav_rets: np.ndarray, unfav_rets: np.ndarray,
                         cost: float, n_reps: int = 1000, seed: int = 42) -> tuple:
    rng     = np.random.default_rng(seed)
    uplifts = []
    fav_net  = fav_rets - cost
    unfav_net = unfav_rets - cost
    for _ in range(n_reps):
        f = rng.choice(fav_net,   size=len(fav_net),   replace=True)
        u = rng.choice(unfav_net, size=len(unfav_net), replace=True)
        uplifts.append((f.mean() - u.mean()) * 10000.0)
    return float(np.percentile(uplifts, 2.5)), float(np.percentile(uplifts, 97.5))


def count_ledger_experiments(ledger_path: str) -> int:
    """Count model-training experiments in dev_ledger (not backfill entries)."""
    if not os.path.exists(ledger_path):
        return 0
    n = 0
    with open(ledger_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    e = json.loads(line)
                    if e.get("event", "experiment") == "experiment":
                        n += 1
                except Exception:
                    pass
    return n


def main():
    # ------------------------------------------------------------------
    # G3: --primary-endpoint is REQUIRED, no default
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="MV2 Certifier — single-run VAULT certification (G2/G3 enforced)"
    )
    parser.add_argument(
        "--primary-endpoint",
        type=str,
        required=True,                    # G3: REQUIRED, no default
        choices=["v8_upstox_3y_long", "v8_upstox_3y_short",
                 "v2_15min_3y_long",  "v2_15min_3y_short"],
        help="Pre-registered primary endpoint (REQUIRED — no default per G3)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("MV2-R4: META-VETO CERTIFICATION (ELEVATED BAR)")
    print(f"Primary Endpoint: {args.primary_endpoint.upper()}")
    print("=" * 70)

    # Load config for thresholds
    with open(MODEL_SPEC, "r", encoding="utf-8") as f:
        mspec = yaml.safe_load(f)
    cert_cfg = mspec["certification"]
    COST_BPS        = cert_cfg["cost_bps"]       # 10.0
    UPLIFT_MIN_BPS  = cert_cfg["uplift_min_bps"] # 3.0
    UPLIFT_MIN_T    = cert_cfg["uplift_min_t"]   # 2.5 (elevated)
    ABS_MIN_T       = cert_cfg["abs_min_t"]       # 2.5 (elevated)
    COST_RATE       = COST_BPS / 10000.0
    SEALED_ENDPOINT = mspec["primary_endpoint"]

    print(f"\nCertification thresholds (from meta_model_spec.yaml):")
    print(f"  Uplift min bps : >= {UPLIFT_MIN_BPS} bps")
    print(f"  Uplift min t   : >= {UPLIFT_MIN_T}  [ELEVATED from 2.0]")
    print(f"  Absolute min t : >= {ABS_MIN_T}  [ELEVATED from 2.0]")
    print(f"  Binding cost   : {COST_BPS} bps")

    # ------------------------------------------------------------------
    # G3: Verify --primary-endpoint matches the sealed value in spec
    # ------------------------------------------------------------------
    if args.primary_endpoint != SEALED_ENDPOINT:
        raise RuntimeError(
            f"[G3 VIOLATED] --primary-endpoint '{args.primary_endpoint}' does not match "
            f"the sealed primary endpoint in meta_model_spec.yaml: '{SEALED_ENDPOINT}'.\n"
            "The endpoint is locked at spec time. To change it, update the YAML and "
            "re-run the entire dev_run -> freeze -> certify pipeline."
        )
    print(f"\n[G3 OK] Primary endpoint matches sealed spec: {SEALED_ENDPOINT}")

    # ------------------------------------------------------------------
    # Load frozen candidate
    # ------------------------------------------------------------------
    meta_path   = os.path.join(CANDIDATE_DIR, "candidate_metadata.json")
    model_path  = os.path.join(CANDIDATE_DIR, "model.joblib")
    scaler_path = os.path.join(CANDIDATE_DIR, "scaler.joblib")
    medians_path = os.path.join(CANDIDATE_DIR, "col_medians.npy")

    for p in [meta_path, model_path, scaler_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Frozen candidate file not found: {p}. Run freeze.py first."
            )

    print("\n[Loading frozen candidate...]")
    with open(meta_path, "r", encoding="utf-8") as f:
        candidate_meta = json.load(f)

    # Verify candidate version
    if candidate_meta.get("mv_version") != "mv2":
        raise RuntimeError(
            f"Candidate is from mv_version='{candidate_meta.get('mv_version')}', "
            "expected 'mv2'. Do not use MV1 candidates with the MV2 certifier."
        )

    # ------------------------------------------------------------------
    # G3: Also verify the primary_endpoint sealed inside the candidate
    # ------------------------------------------------------------------
    candidate_endpoint = candidate_meta.get("primary_endpoint", "")
    if candidate_endpoint != args.primary_endpoint:
        raise RuntimeError(
            f"[G3 VIOLATED] Candidate's sealed endpoint '{candidate_endpoint}' "
            f"!= --primary-endpoint '{args.primary_endpoint}'.\n"
            "This candidate was frozen for a different primary endpoint. "
            "Re-run freeze.py with the correct spec."
        )
    print(f"[G3 OK] Candidate sealed endpoint: {candidate_endpoint}")

    # ------------------------------------------------------------------
    # G2: DEV-promise gate (refuse dead candidates before touching VAULT)
    # ------------------------------------------------------------------
    dev_oof_net_bps = candidate_meta.get("dev_oof_kept_net_bps", 0.0)
    dev_oof_keep_pct = candidate_meta.get("dev_oof_keep_pct", 0.0)

    if dev_oof_net_bps <= 0.0 or dev_oof_keep_pct > 0.90:
        raise RuntimeError(
            f"[G2 VIOLATED] Frozen candidate has DEV OOF net={dev_oof_net_bps:+.2f} bps "
            f"and keep={dev_oof_keep_pct:.1%}.\n"
            "G2 refuses to certify candidates without a positive DEV promise "
            "or with a no-op veto (keep > 90%). Do not burn the VAULT endpoint.\n"
            "Review dev_ledger.jsonl and freeze a better candidate."
        )
    print(f"\n[G2 OK] Candidate DEV promise: net={dev_oof_net_bps:+.2f} bps, keep={dev_oof_keep_pct:.1%}")

    # Verify panel hash
    print("\n[Verifying panel integrity...]")
    if not os.path.exists(PANEL_PATH):
        raise FileNotFoundError(f"Panel not found at {PANEL_PATH}")
    with open(PANEL_PATH, "rb") as f:
        current_panel_sha = hashlib.sha256(f.read()).hexdigest()
    if current_panel_sha != candidate_meta["panel_sha256"]:
        raise RuntimeError(
            f"[FATAL] Panel SHA-256 mismatch!\n"
            f"  Candidate sealed: {candidate_meta['panel_sha256']}\n"
            f"  Current panel:    {current_panel_sha}\n"
            "The panel has changed since the candidate was frozen. "
            "Re-run dev_run.py and freeze.py."
        )
    print(f"[OK] Panel SHA-256 verified: {current_panel_sha[:32]}...")

    # Verify model hash
    with open(model_path, "rb") as f:
        current_model_sha = hashlib.sha256(f.read()).hexdigest()
    if current_model_sha != candidate_meta["model_sha256"]:
        raise RuntimeError(
            f"[FATAL] Model SHA-256 mismatch!\n"
            f"  Candidate sealed: {candidate_meta['model_sha256']}\n"
            f"  Current model:    {current_model_sha}\n"
            "Model file has changed since freezing."
        )
    print(f"[OK] Model SHA-256 verified: {current_model_sha[:32]}...")

    # ------------------------------------------------------------------
    # Load model, scaler, panel — VAULT span only
    # ------------------------------------------------------------------
    print("\n[Loading model and VAULT panel...]")
    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    col_medians = np.load(medians_path) if os.path.exists(medians_path) else None

    df      = pd.read_parquet(PANEL_PATH)
    vault_df = df[df["span"] == "VAULT"].copy().reset_index(drop=True)
    print(f"VAULT rows: {len(vault_df):,}")

    if len(vault_df) == 0:
        raise RuntimeError("VAULT span is empty — cannot run certification.")

    # Hard time-firewall assertion
    min_vault_date = vault_df["datetime"].min()
    if min_vault_date < pd.to_datetime("2025-01-01"):
        raise RuntimeError(
            f"[TIME FIREWALL VIOLATED] VAULT span contains dates before 2025-01-01 "
            f"(min: {min_vault_date}). Abort."
        )
    print(f"[OK] Time firewall: VAULT starts {min_vault_date.date()}")

    # Feature columns
    features = candidate_meta["features"]
    theta    = candidate_meta["theta"]

    X_vault = vault_df[features].values

    # Impute missing values using training medians
    if col_medians is not None:
        for j in range(X_vault.shape[1]):
            mask = np.isnan(X_vault[:, j])
            if mask.any():
                X_vault[mask, j] = col_medians[j]
    else:
        # Fallback: impute from VAULT column medians (not ideal but safe)
        for j in range(X_vault.shape[1]):
            mask = np.isnan(X_vault[:, j])
            if mask.any():
                X_vault[mask, j] = np.nanmedian(X_vault[:, j])

    X_vault_scaled = scaler.transform(X_vault)
    vault_df["pred_prob"] = model.predict_proba(X_vault_scaled)[:, 1]
    vault_df["kept"]      = vault_df["pred_prob"] >= theta

    # ------------------------------------------------------------------
    # Evaluate all families
    # ------------------------------------------------------------------
    families = [
        ("v8_upstox_3y",  "long"),
        ("v8_upstox_3y",  "short"),
        ("v2_15min_3y",   "long"),
        ("v2_15min_3y",   "short"),
    ]

    results = {}
    primary_passed   = False
    primary_summary  = {}
    n_experiments    = count_ledger_experiments(LEDGER_PATH)

    print(f"\n{'='*60}")
    print(f"EVALUATING ON VAULT SPAN (n_dev_experiments={n_experiments})")
    print(f"{'='*60}")

    for model_name, side in families:
        family_name = f"{model_name}_{side}"
        is_primary  = family_name == args.primary_endpoint

        fam_df = vault_df[
            (vault_df["model"] == model_name) & (vault_df["side"] == side)
        ].copy()
        n_total = len(fam_df)

        label = "[PRIMARY]" if is_primary else "[SECONDARY]"
        print(f"\n--- {label} {family_name.upper()} ---")

        if n_total == 0:
            print(f"  No trades in VAULT span.")
            results[family_name] = {"n_total": 0, "certified": False, "reason": "no trades"}
            continue

        kept_df   = fam_df[fam_df["kept"]]
        vetoed_df = fam_df[~fam_df["kept"]]
        n_kept    = len(kept_df)
        n_vetoed  = len(vetoed_df)
        keep_pct  = n_kept / n_total

        print(f"  Total: {n_total:,} | Kept: {n_kept:,} ({keep_pct:.1%}) | Vetoed: {n_vetoed:,}")

        if n_kept < 10 or n_vetoed < 10:
            print(f"  [SKIP] Too few trades for t-tests.")
            results[family_name] = {
                "n_total": n_total, "n_kept": n_kept, "n_vetoed": n_vetoed,
                "certified": False, "reason": "insufficient samples"
            }
            continue

        kept_rets   = (kept_df["trade_return"]   - COST_RATE).values
        vetoed_rets = (vetoed_df["trade_return"] - COST_RATE).values

        mean_kept_bps   = kept_rets.mean()   * 10000.0
        mean_vetoed_bps = vetoed_rets.mean() * 10000.0
        uplift          = mean_kept_bps - mean_vetoed_bps

        t_uplift, _ = ttest_ind(kept_rets, vetoed_rets, equal_var=False)
        t_abs,    _ = ttest_1samp(kept_rets, 0.0)
        ci_lo, ci_hi = bootstrap_uplift_ci(
            kept_df["trade_return"].values,
            vetoed_df["trade_return"].values,
            COST_RATE
        )

        # Certification conditions (ELEVATED thresholds)
        passed_uplift = (uplift >= UPLIFT_MIN_BPS) and (t_uplift >= UPLIFT_MIN_T)
        passed_abs    = (mean_kept_bps > 0.0) and (t_abs >= ABS_MIN_T)
        certified     = passed_uplift and passed_abs

        print(f"  Kept net (bps)    : {mean_kept_bps:+.2f}")
        print(f"  Vetoed net (bps)  : {mean_vetoed_bps:+.2f}")
        print(f"  Uplift (bps)      : {uplift:+.2f}  [need >= {UPLIFT_MIN_BPS}]  CI: [{ci_lo:+.2f}, {ci_hi:+.2f}]")
        print(f"  Uplift t          : {t_uplift:+.2f}  [need >= {UPLIFT_MIN_T}]")
        print(f"  Absolute t        : {t_abs:+.2f}  [need >= {ABS_MIN_T}]")
        print(f"  Uplift condition  : {'PASS' if passed_uplift else 'FAIL'}")
        print(f"  Absolute condition: {'PASS' if passed_abs else 'FAIL'}")
        print(f"  Verdict           : {'[CERTIFIED]' if certified else '[FAILED]'}")

        result_row = {
            "n_total": int(n_total), "n_kept": int(n_kept), "n_vetoed": int(n_vetoed),
            "keep_pct": float(keep_pct), "kept_net_bps": float(mean_kept_bps),
            "vetoed_net_bps": float(mean_vetoed_bps), "uplift_bps": float(uplift),
            "uplift_t": float(t_uplift), "abs_t": float(t_abs),
            "ci_lower_bps": float(ci_lo), "ci_upper_bps": float(ci_hi),
            "passed_uplift": bool(passed_uplift), "passed_abs": bool(passed_abs),
            "certified": bool(certified),
        }
        results[family_name] = result_row

        if is_primary:
            primary_passed  = bool(certified)
            primary_summary = result_row

    # ------------------------------------------------------------------
    # Pre-declared verdict map
    # ------------------------------------------------------------------
    passed_uplift_primary = primary_summary.get("passed_uplift", False)
    passed_abs_primary    = primary_summary.get("passed_abs",    False)

    if passed_uplift_primary and passed_abs_primary:
        final_verdict = "META_VETO_CERTIFIED (CONDITIONAL)"
        verdict_implication = (
            "Both conditions met. Integration permitted as an ML veto stage AFTER "
            "mandatory 3-month live shadow via start_vetoed_tracking(). "
            "No capital gating until shadow re-confirms."
        )
    elif passed_uplift_primary:
        final_verdict = "FILTER_GRADE"
        verdict_implication = (
            "Uplift only — absolute kept-net not significant. "
            "Permissible as a sizing modifier only (reduce, not gate). "
            "No live capital gating."
        )
    else:
        final_verdict = "LINE CLOSED"
        verdict_implication = (
            "Neither condition met. Per the pre-registered protocol, this line "
            "is now permanently closed for price/volume/macro inputs. "
            "Next bps must come from new information sources "
            "(options OI, depth/order-flow, news-sentiment) or from "
            "deploying the already-certified 3-day positional sleeve."
        )

    print(f"\n{'='*60}")
    print("FINAL CERTIFICATION VERDICT")
    print(f"  Primary endpoint  : {args.primary_endpoint.upper()}")
    print(f"  Verdict           : {final_verdict}")
    print(f"  n_dev_experiments : {n_experiments}")
    print(f"  Implication       : {verdict_implication}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # Save JSON report
    # ------------------------------------------------------------------
    os.makedirs(CANDIDATE_DIR, exist_ok=True)
    cert_json_path = os.path.join(CANDIDATE_DIR, "certification_report.json")
    cert_data = {
        "mv_version":               "mv2",
        "primary_endpoint":         args.primary_endpoint,
        "primary_passed":           primary_passed,
        "final_verdict":            final_verdict,
        "primary_results":          primary_summary,
        "all_results":              results,
        "n_dev_experiments_tried":  n_experiments,
        "uplift_threshold_bps":     UPLIFT_MIN_BPS,
        "t_threshold":              UPLIFT_MIN_T,
        "cost_bps":                 COST_BPS,
        "timestamp":                pd.Timestamp.now().isoformat(),
    }
    with open(cert_json_path, "w", encoding="utf-8") as f:
        json.dump(cert_data, f, indent=2)
    print(f"\nJSON report saved: {cert_json_path}")

    # ------------------------------------------------------------------
    # Write markdown report to vault
    # ------------------------------------------------------------------
    os.makedirs(REPORT_DIR, exist_ok=True)
    report_path = os.path.join(REPORT_DIR, "Certification Report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Meta-Veto MV2 Certification Report\n\n")
        f.write(f"> [!{'NOTE' if 'CERTIFIED' in final_verdict else 'WARNING'}]\n")
        f.write(f"> **MV2 Verdict**: {final_verdict}\n\n")
        f.write(f"- **Primary Endpoint**: `{args.primary_endpoint}`\n")
        f.write(f"- **Model class**: `{candidate_meta.get('model_class', '?')}`\n")
        f.write(f"- **DEV OOF net**: `{dev_oof_net_bps:+.2f} bps` (keep: {dev_oof_keep_pct:.1%})\n")
        f.write(f"- **Dev experiments tried**: `{n_experiments}`\n")
        f.write(f"- **Model SHA-256**: `{candidate_meta['model_sha256'][:32]}...`\n")
        f.write(f"- **Panel SHA-256**: `{candidate_meta['panel_sha256'][:32]}...`\n")
        f.write(f"- **Timestamp**: `{pd.Timestamp.now().isoformat()}`\n\n")

        f.write(f"## Primary Endpoint Results\n\n")
        if primary_summary:
            f.write(f"| Metric | Value | Threshold | Status |\n")
            f.write(f"|---|---|---|---|\n")
            f.write(f"| Kept count | {primary_summary['n_kept']:,} / {primary_summary['n_total']:,} ({primary_summary['keep_pct']:.1%}) | >= 25% | {'OK' if primary_summary['keep_pct'] >= 0.25 else 'LOW'} |\n")
            f.write(f"| Uplift (bps) | {primary_summary['uplift_bps']:+.2f} | >= {UPLIFT_MIN_BPS} | {'PASS' if primary_summary['passed_uplift'] else 'FAIL'} |\n")
            f.write(f"| Uplift t-stat | {primary_summary['uplift_t']:+.2f} | >= {UPLIFT_MIN_T} | {'PASS' if primary_summary['passed_uplift'] else 'FAIL'} |\n")
            f.write(f"| Kept net (bps) | {primary_summary['kept_net_bps']:+.2f} | > 0 | {'PASS' if primary_summary['passed_abs'] else 'FAIL'} |\n")
            f.write(f"| Absolute t-stat | {primary_summary['abs_t']:+.2f} | >= {ABS_MIN_T} | {'PASS' if primary_summary['passed_abs'] else 'FAIL'} |\n\n")

        f.write(f"## All Families\n\n")
        f.write("| Family | Total | Kept | Keep% | Kept Net | Uplift | Uplift t | Abs t | Certified |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for fam, r in results.items():
            if "reason" in r:
                f.write(f"| {fam} | {r['n_total']} | {r['n_kept']} | - | - | - | - | - | {r['reason']} |\n")
            else:
                f.write(f"| {fam} | {r['n_total']:,} | {r['n_kept']:,} | {r['keep_pct']:.1%} | "
                        f"{r['kept_net_bps']:+.2f} | {r['uplift_bps']:+.2f} | "
                        f"{r['uplift_t']:+.2f} | {r['abs_t']:+.2f} | "
                        f"{'YES' if r['certified'] else 'NO'} |\n")

        f.write(f"\n## Verdict Implication\n\n{verdict_implication}\n\n")
        f.write(f"## Backlinks\n\n")
        f.write(f"- Plan: [[02. Model Suite/Meta-Veto Rectification Plan MV2]]\n")
        f.write(f"- Current Context: [[06. Context & Logs/Current Context]]\n")

    print(f"Vault report saved: {report_path}")

    print("=" * 70)
    print("MV2-R4 CERTIFICATION EXECUTED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()

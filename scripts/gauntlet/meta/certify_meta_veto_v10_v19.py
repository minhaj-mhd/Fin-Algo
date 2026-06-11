"""
MV2-R4: certify_meta_veto_v10_v19.py — Elevated-Bar Certification for v10-v19
================================================================================
Single-run VAULT evaluation of the frozen MV_V10_V19 candidate.
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
MODEL_SPEC    = os.path.join(CONFIG_DIR, "meta_model_spec_v10_v19.yaml")
PANEL_PATH    = os.path.join("data", "gauntlet", "meta", "mv_v10_v19", "trade_panel.parquet")
CANDIDATE_DIR = os.path.join("models", "meta_veto_v10_v19")
LEDGER_PATH   = os.path.join("data", "gauntlet", "meta", "mv_v10_v19", "dev_ledger.jsonl")
REPORT_DIR    = os.path.join("finalgo-memory-layer", "finalgo", "08. Model Analysis", "Meta-Veto")

MODELS_1H = [
    "v10_native_1h",
    "v10_depth4_1h",
    "v11_utility_1h",
    "v12_lambdamart_1h",
    "v13_ndcg_raw_1h",
    "v14_lambdamart_no_es_1h",
    "v15_lambdamart_es_1h",
    "v15_lambdamart_map5_1h",
    "v16_binary_breakout_1h",
    "v17_random_forest_1h",
    "v18_random_forest_1h",
    "v19_catboost_1h",
]

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
    # Generate dynamic choices based on available models
    endpoint_choices = []
    for m in MODELS_1H:
        endpoint_choices.extend([f"{m}_long", f"{m}_short"])

    parser = argparse.ArgumentParser(
        description="MV_V10_V19 Certifier — single-run VAULT certification"
    )
    parser.add_argument(
        "--primary-endpoint",
        type=str,
        required=True,
        choices=endpoint_choices,
        help="Pre-registered primary endpoint (REQUIRED — no default per G3)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("MV2-R4: META-VETO CERTIFICATION FOR v10-v19 (ELEVATED BAR)")
    print(f"Primary Endpoint: {args.primary_endpoint.upper()}")
    print("=" * 70)

    with open(MODEL_SPEC, "r", encoding="utf-8") as f:
        mspec = yaml.safe_load(f)
    cert_cfg = mspec["certification"]
    COST_BPS        = cert_cfg["cost_bps"]
    UPLIFT_MIN_BPS  = cert_cfg["uplift_min_bps"]
    UPLIFT_MIN_T    = cert_cfg["uplift_min_t"]
    ABS_MIN_T       = cert_cfg["abs_min_t"]
    COST_RATE       = COST_BPS / 10000.0
    SEALED_ENDPOINT = mspec["primary_endpoint"]

    print(f"\nCertification thresholds:")
    print(f"  Uplift min bps : >= {UPLIFT_MIN_BPS} bps")
    print(f"  Uplift min t   : >= {UPLIFT_MIN_T}")
    print(f"  Absolute min t : >= {ABS_MIN_T}")
    print(f"  Binding cost   : {COST_BPS} bps")

    # G3: Verify --primary-endpoint matches the sealed value in spec
    if args.primary_endpoint != SEALED_ENDPOINT:
        raise RuntimeError(
            f"[G3 VIOLATED] --primary-endpoint '{args.primary_endpoint}' does not match "
            f"the sealed primary endpoint in meta_model_spec_v10_v19.yaml: '{SEALED_ENDPOINT}'."
        )
    print(f"\n[G3 OK] Primary endpoint matches sealed spec: {SEALED_ENDPOINT}")

    # Load frozen candidate files
    meta_path   = os.path.join(CANDIDATE_DIR, "candidate_metadata.json")
    model_path  = os.path.join(CANDIDATE_DIR, "model.joblib")
    scaler_path = os.path.join(CANDIDATE_DIR, "scaler.joblib")
    medians_path = os.path.join(CANDIDATE_DIR, "col_medians.npy")

    for p in [meta_path, model_path, scaler_path, medians_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Frozen candidate file not found: {p}. Run freeze_v10_v19.py first.")

    print("\n[Loading frozen candidate...]")
    with open(meta_path, "r", encoding="utf-8") as f:
        candidate_meta = json.load(f)

    # Verify candidate version matches spec
    c_version = candidate_meta.get("mv_version", "")
    if c_version != mspec["version"]:
        raise RuntimeError(f"Candidate version '{c_version}' does not match spec version '{mspec['version']}'")

    model    = joblib.load(model_path)
    scaler   = joblib.load(scaler_path)
    medians  = np.load(medians_path)
    print(f"  Loaded model class : {candidate_meta['model_class']}")
    print(f"  Sealed theta       : {candidate_meta['theta']:.2f}")
    print(f"  DEV kept-net bps   : {candidate_meta['dev_oof_kept_net_bps']:+.2f} bps")
    print(f"  DEV keep pct       : {candidate_meta['dev_oof_keep_pct']:.1%}")

    # G2: Certifier-level double-enforcement
    if candidate_meta["dev_oof_kept_net_bps"] <= 0.0 or candidate_meta["dev_oof_keep_pct"] > 0.90:
        raise RuntimeError(
            f"[G2 VIOLATED] Certifier rejected candidate with DEV OOF net = {candidate_meta['dev_oof_kept_net_bps']:.2f} bps "
            f"and keep = {candidate_meta['dev_oof_keep_pct']:.1%}"
        )
    print(f"  [G2 OK] DEV promise check passed.")

    # Load panel parquet
    print(f"\n[Loading trade panel parquet...]")
    df = pd.read_parquet(PANEL_PATH)
    
    # Verify panel SHA matches sealed metadata
    with open(PANEL_PATH, "rb") as f:
        panel_sha = hashlib.sha256(f.read()).hexdigest()
    if panel_sha != candidate_meta["panel_sha256"]:
        raise RuntimeError(
            f"[HYGIENE VIOLATED] Panel SHA '{panel_sha}' does not match candidate's "
            f"sealed panel SHA '{candidate_meta['panel_sha256']}'. Stale panel file detected."
        )
    print(f"  [SHA OK] Panel matched candidate's sealed SHA.")

    vault_df = df[df["span"] == "VAULT"].copy().reset_index(drop=True)
    print(f"  VAULT trades available: {len(vault_df):,}")
    if len(vault_df) == 0:
        raise RuntimeError("VAULT span is empty. Cannot certify.")

    feature_cols = candidate_meta["features"]
    
    # Filter to endpoint (e.g. v10_native_1h long or short)
    endpoint_model, endpoint_side = args.primary_endpoint.rsplit("_", 1)
    ep_mask = (vault_df["model"] == endpoint_model) & (vault_df["side"] == endpoint_side)
    ep_df = vault_df[ep_mask].copy().reset_index(drop=True)
    print(f"  Endpoint '{args.primary_endpoint}' trades on VAULT: {len(ep_df):,}")
    if len(ep_df) == 0:
        raise RuntimeError(f"No trades matching endpoint '{args.primary_endpoint}' found in VAULT panel.")

    # 5. Evaluate on VAULT span (Single-Run)
    print("\n[Evaluating model on VAULT endpoint trades...]")
    X_vault = ep_df[feature_cols].values
    
    # Impute missing features using frozen medians
    for j in range(X_vault.shape[1]):
        X_vault[np.isnan(X_vault[:, j]), j] = medians[j]

    X_vault_scaled = scaler.transform(X_vault)
    
    # Get probabilities
    probs = model.predict_proba(X_vault_scaled)[:, 1]
    theta = candidate_meta["theta"]
    
    kept_mask = probs >= theta
    n_kept = kept_mask.sum()
    n_total = len(ep_df)
    keep_pct = n_kept / n_total if n_total > 0 else 0.0

    kept_df = ep_df[kept_mask]
    vetoed_df = ep_df[~kept_mask]

    # kept-net
    kept_net_rets = kept_df["trade_return"].values - COST_RATE
    kept_net_bps = kept_net_rets.mean() * 10000.0 if len(kept_net_rets) > 0 else 0.0
    
    # vetoed-net
    vetoed_net_rets = vetoed_df["trade_return"].values - COST_RATE
    vetoed_net_bps = vetoed_net_rets.mean() * 10000.0 if len(vetoed_net_rets) > 0 else 0.0

    # Uplift (kept net minus vetoed net)
    actual_uplift_bps = kept_net_bps - vetoed_net_bps

    # Statistical tests
    # 1. Uplift t-stat
    if len(kept_df) > 1 and len(vetoed_df) > 1:
        t_stat_uplift, p_val_uplift = ttest_ind(kept_net_rets, vetoed_net_rets, equal_var=False)
    else:
        t_stat_uplift, p_val_uplift = 0.0, 1.0

    # 2. Absolute t-stat (kept net > 0)
    if len(kept_df) > 1:
        t_stat_abs, p_val_abs = ttest_1samp(kept_net_rets, 0.0)
    else:
        t_stat_abs, p_val_abs = 0.0, 1.0

    # Bootstrap Uplift CI
    ci_lower, ci_upper = bootstrap_uplift_ci(kept_df["trade_return"].values, vetoed_df["trade_return"].values, COST_RATE)

    print("\n" + "=" * 60)
    print("VAULT PERFORMANCE EVALUATION")
    print("=" * 60)
    print(f"  Kept trades    : {n_kept} / {n_total} ({keep_pct:.1%})")
    print(f"  Kept net return: {kept_net_bps:+.2f} bps  (t-stat = {t_stat_abs:+.2f})")
    print(f"  Veto net return: {vetoed_net_bps:+.2f} bps")
    print(f"  Uplift         : {actual_uplift_bps:+.2f} bps  (t-stat = {t_stat_uplift:+.2f})")
    print(f"  Uplift 95% CI  : [{ci_lower:+.2f}, {ci_upper:+.2f}] bps")

    # Certification decisions
    passed_uplift = actual_uplift_bps >= UPLIFT_MIN_BPS and t_stat_uplift >= UPLIFT_MIN_T
    passed_absolute = kept_net_bps > 0.0 and t_stat_abs >= ABS_MIN_T

    # 6. Apply pre-declared verdict map
    print("\n[Applying pre-declared verdict map...]")
    if passed_uplift and passed_absolute:
        verdict = "META_VETO_CERTIFIED"
        status_text = "🟢 CERTIFIED (Passed both Uplift and Absolute gates)"
    elif passed_uplift:
        verdict = "FILTER_GRADE"
        status_text = "🟡 FILTER GRADE ONLY (Passed Uplift but failed Absolute kept-net)"
    else:
        verdict = "REJECTED"
        status_text = "🔴 LINE CLOSED PERMANENTLY (Failed uplift check)"

    print(f"  Verdict: {verdict}")
    print(f"  Status : {status_text}")

    # Log to Obsidian report
    os.makedirs(REPORT_DIR, exist_ok=True)
    report_path = os.path.join(REPORT_DIR, "Certification_Report_v10_v19.md")
    
    n_dev_exp = count_ledger_experiments(LEDGER_PATH)
    
    report_content = f"""# 🛡️ Meta-Veto Stacking Certification: v10-v19 Panel

- **Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Sealed Primary Endpoint**: `{SEALED_ENDPOINT}`
- **Verdict**: `{verdict}`
- **Status Summary**: {status_text}

## 📊 Performance Metrics (VAULT out-of-sample)
| Metric | Observed | Threshold | Gate Status |
| :--- | :---: | :---: | :---: |
| **Uplift Return** | `{actual_uplift_bps:+.2f} bps` | `>= {UPLIFT_MIN_BPS} bps` | {"✅ PASS" if actual_uplift_bps >= UPLIFT_MIN_BPS else "❌ FAIL"} |
| **Uplift t-stat** | `{t_stat_uplift:+.2f}` | `>= {UPLIFT_MIN_T}` | {"✅ PASS" if t_stat_uplift >= UPLIFT_MIN_T else "❌ FAIL"} |
| **Kept-net Return** | `{kept_net_bps:+.2f} bps` | `> 0.00 bps` | {"✅ PASS" if kept_net_bps > 0.0 else "❌ FAIL"} |
| **Kept-net t-stat** | `{t_stat_abs:+.2f}` | `>= {ABS_MIN_T}` | {"✅ PASS" if t_stat_abs >= ABS_MIN_T else "❌ FAIL"} |
| **Veto Keep %** | `{keep_pct:.1%}` | `25.0% - 90.0%` | {"✅ PASS" if 0.25 <= keep_pct <= 0.90 else "❌ FAIL"} |

## ⚙️ Audit Details
- **Base Models**: `v10` through `v19`
- **Total Trades evaluated**: {n_total:,} trades
- **Kept Trades count**: {n_kept:,} trades
- **Uplift 95% Bootstrap CI**: `[{ci_lower:+.2f}, {ci_upper:+.2f}] bps`
- **DEV Experiments Tried**: {n_dev_exp}
- **Panel Parquet SHA-256**: `{panel_sha}`
- **Model Checksum**: `{candidate_meta.get("checksum", "N/A")[:16]}`
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"  Obsidian certification report saved -> {report_path}")

    # Append to dev_ledger
    ledger_entry = {
        "event": "certification",
        "verdict": verdict,
        "primary_endpoint": args.primary_endpoint,
        "vault_trades": int(n_total),
        "kept_trades": int(n_kept),
        "keep_pct": float(keep_pct),
        "kept_net_bps": float(kept_net_bps),
        "kept_t_stat": float(t_stat_abs),
        "uplift_bps": float(actual_uplift_bps),
        "uplift_t_stat": float(t_stat_uplift),
        "panel_sha256": panel_sha,
        "timestamp": pd.Timestamp.now().isoformat()
    }
    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(ledger_entry) + "\n")
    print(f"  Appended certification record to dev_ledger.")
    print("=" * 70)


if __name__ == "__main__":
    main()

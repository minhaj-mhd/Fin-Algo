"""
MV2-R0 (post-panel): M1 Orthogonality Kill-Gate
================================================
Runs on the rectified MV2 DEV panel (data/gauntlet/meta/mv2/trade_panel.parquet).

Kill-gate question: does anything BEYOND the own-model score add information?
If no qualifying feature has incremental Rank-IC >= 0.005, stacking has no
premise and the line closes before R1. Per 2026-06-10 ruling:

  Qualifying signals = everything EXCEPT:
    own_score, own_z, own_pct    (own-model score — the baseline)
    model_is_v8, side_is_long    (family identity indicators)

  Hour/ToD, daily scores, cross-TF, VIX/macro all qualify.

Output:
  - Console table of raw + partial rank-IC per feature per model family
  - Report written to finalgo-memory-layer/finalgo/08. Model Analysis/Meta-Veto/
  - Hard sys.exit(1) if kill-gate fires (per spec — this is a binding stop)

Usage:
  python scripts/gauntlet/meta/orthogonality_audit.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from pathlib import Path

sys.path.append(os.getcwd())

# ---------------------------------------------------------------------------
# Constants (per 2026-06-10 spec ruling)
# ---------------------------------------------------------------------------
PANEL_PATH  = os.path.join("data", "gauntlet", "meta", "mv2_clean", "trade_panel.parquet")
LEDGER_PATH = os.path.join("data", "gauntlet", "meta", "mv2_clean", "dev_ledger.jsonl")
REPORT_DIR  = os.path.join("finalgo-memory-layer", "finalgo", "08. Model Analysis", "Meta-Veto")

# These columns are EXCLUDED from the kill-gate (own-model baseline + identity)
EXCLUDED_FROM_GATE = {
    "own_score", "own_z", "own_pct",
    "model_is_v8", "side_is_long"
}

# All non-feature metadata columns
META_COLS = {
    "model", "datetime", "ticker", "side", "trade_return",
    "Query_ID", "span", "y"
}

# Minimum incremental IC for the gate to pass
MIN_INCREMENTAL_IC = 0.005


# ---------------------------------------------------------------------------
# Partial rank correlation: IC(feature, return) controlling for own_pct
# ---------------------------------------------------------------------------

def partial_rank_ic(df: pd.DataFrame, y_col: str, x_col: str, z_col: str) -> float:
    """
    Partial rank correlation of x_col with y_col, controlling for z_col.
    Implements the residual approach on ranks.
    """
    ry = df[y_col].rank().values.astype(float)
    rx = df[x_col].rank().values.astype(float)
    rz = df[z_col].rank().values.astype(float)

    A    = np.vstack([np.ones_like(rz), rz]).T
    beta_y = np.linalg.lstsq(A, ry, rcond=None)[0]
    res_y  = ry - (beta_y[0] + beta_y[1] * rz)

    beta_x = np.linalg.lstsq(A, rx, rcond=None)[0]
    res_x  = rx - (beta_x[0] + beta_x[1] * rz)

    corr, _ = pearsonr(res_y, res_x)
    return corr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("M1: ORTHOGONALITY KILL-GATE (MV2 panel)")
    print("=" * 70)

    # Verify we're running on the MV2 panel
    if not os.path.exists(PANEL_PATH):
        print(f"[FATAL] MV2 panel not found at {PANEL_PATH}. Run build_trade_panel.py first.")
        sys.exit(1)

    df = pd.read_parquet(PANEL_PATH)

    # Load panel metadata for the SHA check
    meta_path = os.path.join("data", "gauntlet", "meta", "mv2_clean", "panel_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            panel_meta = json.load(f)
        print(f"  Panel version: {panel_meta.get('mv_version', '?')}")
        print(f"  DEV months: {panel_meta.get('n_dev_months', '?')}")
        print(f"  DEV trades: {panel_meta.get('n_dev_trades', '?'):,}")
    else:
        print("  [WARN] panel_metadata.json not found — continuing without version check.")

    dev_df = df[df["span"] == "DEV"].copy()
    print(f"\nLoaded DEV panel: {len(dev_df):,} rows")

    if len(dev_df) == 0:
        print("[FATAL] DEV span is empty.")
        sys.exit(1)

    # Determine feature columns and gate-qualifying columns
    all_feature_cols = [c for c in df.columns if c not in META_COLS and c not in EXCLUDED_FROM_GATE]
    gate_cols        = [c for c in all_feature_cols
                        if pd.api.types.is_numeric_dtype(dev_df[c])
                        and c not in EXCLUDED_FROM_GATE
                        and not c.endswith("_missing")]  # exclude binary indicators from IC calc

    print(f"\nGate-qualifying features ({len(gate_cols)}):")
    for gc in gate_cols:
        print(f"  {gc}")

    print(f"\nExcluded from gate (own-model + identity): {sorted(EXCLUDED_FROM_GATE)}")

    # Compute pairwise Spearman among gate features + own_pct (for reference only)
    present_gate_cols = [c for c in gate_cols if c in dev_df.columns]

    # ---------------------
    # Incremental Rank-IC
    # ---------------------
    print(f"\n{'='*60}")
    print("INCREMENTAL RANK-IC (partial rank correlation vs trade_return,")
    print("controlling for own_pct as the baseline own-model signal)")
    print(f"{'='*60}")

    partial_results = []
    max_abs_partial_ic = 0.0
    best_feature = None
    best_model   = None

    for model_name in sorted(dev_df["model"].unique()):
        model_df = dev_df[dev_df["model"] == model_name].copy()
        print(f"\n--- Model family: {model_name} ({len(model_df):,} trades) ---")

        for feat in present_gate_cols:
            sub_df = model_df.dropna(subset=[feat, "trade_return", "own_pct"])
            if len(sub_df) < 50:
                print(f"  {feat:<30}: too few non-NaN rows ({len(sub_df)}) -- skipped")
                continue

            # Skip constant columns (would give NaN correlation)
            if sub_df[feat].std() < 1e-10:
                print(f"  {feat:<30}: constant in this family -- skipped")
                continue

            raw_ic, _ = spearmanr(sub_df[feat], sub_df["trade_return"])
            part_ic   = partial_rank_ic(sub_df, "trade_return", feat, "own_pct")

            flag = " <-- QUALIFIES" if abs(part_ic) >= MIN_INCREMENTAL_IC else ""
            print(f"  {feat:<30}: raw_IC={raw_ic:+.4f}  |  partial_IC={part_ic:+.4f}{flag}")

            partial_results.append({
                "model":      model_name,
                "feature":    feat,
                "n":          int(len(sub_df)),
                "raw_ic":     round(float(raw_ic), 5),
                "partial_ic": round(float(part_ic), 5),
            })

            if abs(part_ic) > max_abs_partial_ic:
                max_abs_partial_ic = abs(part_ic)
                best_feature       = feat
                best_model         = model_name

    partial_df = pd.DataFrame(partial_results) if partial_results else pd.DataFrame()

    # ---------------------
    # Kill-gate verdict
    # ---------------------
    gate_passed = max_abs_partial_ic >= MIN_INCREMENTAL_IC

    print(f"\n{'='*60}")
    print("M1 KILL-GATE VERDICT")
    print(f"  Max |partial IC| : {max_abs_partial_ic:.5f}")
    print(f"  Best feature     : {best_feature}")
    print(f"  Best model       : {best_model}")
    print(f"  Threshold        : {MIN_INCREMENTAL_IC}")
    print(f"  Gate status      : {'PASS -- proceed to R1' if gate_passed else 'FAIL -- LINE CLOSED'}")
    print(f"{'='*60}")

    # ---------------------
    # Save report
    # ---------------------
    os.makedirs(REPORT_DIR, exist_ok=True)
    report_path = os.path.join(REPORT_DIR, "M1 Orthogonality Audit MV2.md")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# M1 Orthogonality Kill-Gate Report (MV2)\n\n")
        status_str = "PASS - Proceed to R1" if gate_passed else "FAIL - LINE CLOSED"
        f.write(f"> [!{'NOTE' if gate_passed else 'CAUTION'}]\n")
        f.write(f"> **Gate status**: {status_str}  \n")
        f.write(f"> **Max |partial IC|**: `{max_abs_partial_ic:.5f}` "
                f"(feature: `{best_feature}` on `{best_model}`)  \n")
        f.write(f"> **Required**: >= {MIN_INCREMENTAL_IC}  \n")
        f.write(f"> **Panel**: `{PANEL_PATH}`  \n")
        f.write(f"> **Timestamp**: `{pd.Timestamp.now().isoformat()}`\n\n")

        f.write("## Gate Qualifying Features\n\n")
        f.write("Features qualify if they are NOT own_score/own_z/own_pct or identity "
                "indicators (model_is_v8, side_is_long). Hour/ToD, cross-TF, daily "
                "scores, VIX/macro all qualify.\n\n")

        if not partial_df.empty:
            f.write("## Incremental Rank-IC Table\n\n")
            top_df = partial_df.sort_values("partial_ic", key=abs, ascending=False)
            f.write(top_df.to_markdown(index=False) + "\n\n")

        if gate_passed:
            f.write("## Finding\n\n")
            f.write(f"> [!NOTE]\n")
            f.write(f"> Gate **PASSED**: `{best_feature}` on `{best_model}` has "
                    f"incremental IC = `{max_abs_partial_ic:.5f}` >= threshold. "
                    "Stacking has a demonstrated premise. Proceed to R1 (dev_run.py).\n\n")
            f.write("## Backlinks\n\n")
            f.write("- [[02. Model Suite/Meta-Veto Rectification Plan MV2]]\n")
            f.write("- [[06. Context & Logs/Current Context]]\n")
        else:
            f.write("## Finding\n\n")
            f.write("> [!CAUTION]\n")
            f.write("> Gate **FAILED**: no qualifying feature exceeds the incremental IC "
                    f"threshold of {MIN_INCREMENTAL_IC}. Per the pre-registered protocol "
                    "(binding, not optional), the meta-veto line is **permanently closed** "
                    "for price/volume/macro inputs. Next edge must come from new information "
                    "sources: options OI, depth/order-flow, or news-sentiment.\n\n")

    print(f"\nReport saved -> {report_path}")

    # Save raw IC table as CSV
    if not partial_df.empty:
        csv_path = os.path.join("data", "gauntlet", "meta", "mv2_clean", "m1_incremental_ic.csv")
        partial_df.to_csv(csv_path, index=False)
        print(f"IC table saved -> {csv_path}")

    # Log to dev_ledger
    ledger_entry = {
        "event":                "m1_kill_gate",
        "gate_passed":          bool(gate_passed),
        "max_abs_partial_ic":   float(max_abs_partial_ic),
        "best_feature":         str(best_feature) if best_feature is not None else None,
        "best_model":           str(best_model) if best_model is not None else None,
        "min_incremental_ic":   float(MIN_INCREMENTAL_IC),
        "n_gate_features":      int(len(present_gate_cols)),
        "timestamp":            pd.Timestamp.now().isoformat(),
    }
    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(ledger_entry) + "\n")

    print(f"Gate result logged -> {LEDGER_PATH}")
    print("=" * 70)
    print("M1 COMPLETED")
    print("=" * 70)

    if not gate_passed:
        print("\n[ABORT] M1 kill-gate triggered. Line is closed.")
        print("        Per spec: halting pipeline before any model training.")
        sys.exit(1)

    print("\nGate passed. Ready to run R1: python scripts/gauntlet/meta/dev_run.py --rung 1")


if __name__ == "__main__":
    main()

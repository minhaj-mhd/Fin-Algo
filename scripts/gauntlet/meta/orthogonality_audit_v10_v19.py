"""
MV2-R0 (post-panel): M1 Orthogonality Kill-Gate for v10 to v19
==============================================================
Runs on the rectified v10-v19 DEV panel.
Checks if any feature (beyond the anchor v10_native_1h score) adds incremental Rank-IC >= 0.005.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from pathlib import Path

sys.path.append(os.getcwd())

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PANEL_PATH  = os.path.join("data", "gauntlet", "meta", "mv_v10_v19", "trade_panel.parquet")
LEDGER_PATH = os.path.join("data", "gauntlet", "meta", "mv_v10_v19", "dev_ledger.jsonl")
REPORT_DIR  = os.path.join("finalgo-memory-layer", "finalgo", "08. Model Analysis", "Meta-Veto")

# These columns are EXCLUDED from the kill-gate (anchor-model baseline + identity/indicators)
EXCLUDED_FROM_GATE = {
    "own_score", "own_z", "own_pct",
    "v10_native_1h_score", "v10_native_1h_z", "v10_native_1h_pct",
    "side_is_long"
}

# All non-feature metadata columns
META_COLS = {
    "model", "datetime", "ticker", "side", "trade_return",
    "Query_ID", "span", "y", "proposed_by"
}

MIN_INCREMENTAL_IC = 0.005

# ---------------------------------------------------------------------------
# Partial rank correlation
# ---------------------------------------------------------------------------

def partial_rank_ic(df: pd.DataFrame, y_col: str, x_col: str, z_col: str) -> float:
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
    print("M1: ORTHOGONALITY KILL-GATE (v10-v19 panel)")
    print("=" * 70)

    if not os.path.exists(PANEL_PATH):
        print(f"[FATAL] Panel not found at {PANEL_PATH}. Run build_trade_panel_v10_v19.py first.")
        sys.exit(1)

    df = pd.read_parquet(PANEL_PATH)

    meta_path = os.path.join("data", "gauntlet", "meta", "mv_v10_v19", "panel_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            panel_meta = json.load(f)
        print(f"  Panel version: {panel_meta.get('mv_version', '?')}")
        print(f"  DEV months: {panel_meta.get('n_dev_months', '?')}")
        print(f"  DEV trades: {panel_meta.get('n_dev_trades', '?'):,}")

    dev_df = df[df["span"] == "DEV"].copy()
    print(f"\nLoaded DEV panel: {len(dev_df):,} rows")

    if len(dev_df) == 0:
        print("[FATAL] DEV span is empty.")
        sys.exit(1)

    # Determine feature columns and gate-qualifying columns
    all_feature_cols = [c for c in df.columns if c not in META_COLS and c not in EXCLUDED_FROM_GATE]
    # Exclude proposed_by flags and missingness indicators from IC calc
    gate_cols = [c for c in all_feature_cols
                 if pd.api.types.is_numeric_dtype(dev_df[c])
                 and not c.startswith("proposed_by_")
                 and not c.endswith("_missing")]

    print(f"\nGate-qualifying features ({len(gate_cols)}):")
    for gc in gate_cols:
        print(f"  {gc}")

    print(f"\nExcluded from gate (own-model + identity): {sorted(EXCLUDED_FROM_GATE)}")

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

    # Audit features on the whole DEV set or per model
    # Since all trades are 1H trades proposed by the models, we can audit across the full DEV set
    for feat in gate_cols:
        sub_df = dev_df.dropna(subset=[feat, "trade_return", "own_pct"])
        if len(sub_df) < 100:
            continue
        try:
            p_ic = partial_rank_ic(sub_df, "trade_return", feat, "own_pct")
            raw_corr, _ = pearsonr(sub_df[feat].rank(), sub_df["trade_return"].rank())
            
            partial_results.append({
                "feature": feat,
                "raw_ic": raw_corr,
                "partial_ic": p_ic
            })
            
            if abs(p_ic) > max_abs_partial_ic:
                max_abs_partial_ic = abs(p_ic)
                best_feature = feat
        except Exception as e:
            print(f"  Error calculating IC for {feat}: {e}")

    # Display results sorted by partial IC magnitude
    partial_results_df = pd.DataFrame(partial_results)
    if not partial_results_df.empty:
        partial_results_df["abs_partial_ic"] = partial_results_df["partial_ic"].abs()
        partial_results_df = partial_results_df.sort_values("abs_partial_ic", ascending=False)
        print("\n  Top 15 Features by Partial Rank-IC:")
        print(partial_results_df.head(15).to_string(index=False))
        
        # Save to CSV
        csv_path = os.path.join("data", "gauntlet", "meta", "mv_v10_v19", "m1_incremental_ic.csv")
        partial_results_df.to_csv(csv_path, index=False)
        print(f"\n  Incremental IC saved -> {csv_path}")

    print(f"\nMax Absolute Partial IC: {max_abs_partial_ic:.6f}")
    if best_feature:
        print(f"Best feature: {best_feature}")

    # ---------------------
    # Kill-Gate Decision
    # ---------------------
    passed = max_abs_partial_ic >= MIN_INCREMENTAL_IC
    verdict_str = "PASS" if passed else "KILL"
    print(f"\nVERDICT: {verdict_str} (max |partial IC| = {max_abs_partial_ic:.6f} vs threshold = {MIN_INCREMENTAL_IC})")

    # Append to dev_ledger
    ledger_entry = {
        "event": "m1_kill_gate",
        "gate_passed": bool(passed),
        "max_abs_partial_ic": float(max_abs_partial_ic),
        "best_feature": best_feature,
        "min_incremental_ic": MIN_INCREMENTAL_IC,
        "n_gate_features": len(gate_cols),
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    os.makedirs(os.path.dirname(LEDGER_PATH), exist_ok=True)
    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(ledger_entry) + "\n")
    print(f"  Appended event to dev_ledger -> {LEDGER_PATH}")

    # Generate Markdown Report in Obsidian
    os.makedirs(REPORT_DIR, exist_ok=True)
    md_path = os.path.join(REPORT_DIR, "M1_Orthogonality_Audit_v10_v19.md")
    
    md_content = f"""# 📊 M1 Orthogonality Audit: v10-v19 Stacking Panel

- **Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Verdict**: {"🟢 PASS" if passed else "🔴 KILL"}
- **Max Absolute Partial IC**: `{max_abs_partial_ic:.6f}`
- **Best Feature**: `{best_feature}`
- **Threshold**: `{MIN_INCREMENTAL_IC}`
- **Total Features Audited**: {len(gate_cols)}

## 📈 Top 15 Features by Partial Rank-IC
| Feature | Raw Rank-IC | Partial Rank-IC (controlling for own_pct) |
| :--- | :---: | :---: |
"""
    if not partial_results_df.empty:
        for _, r in partial_results_df.head(15).iterrows():
            md_content += f"| `{r['feature']}` | {r['raw_ic']:+.4f} | {r['partial_ic']:+.4f} |\n"
            
    md_content += f"\n\n*This audit verifies whether features beyond the anchor model (`v10_native_1h`) possess orthogonal predictive signal.*"
    
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"  Report written to Obsidian memory layer -> {md_path}")

    if not passed:
        print("\n[KILL-GATE FIRED] Stacking has no empirical premise (no orthogonal signal). Aborting.")
        sys.exit(1)

    print("\n[M1 PASS] Stacking framework verified with empirical premise. Ready for dev_run capacity ladder.")


if __name__ == "__main__":
    main()

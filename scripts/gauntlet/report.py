import os
import json
import datetime
from typing import Dict, Any, List, Optional
from dataclasses import asdict
from .contracts import DatasetSpec, ModelSpec, GauntletConfig
from .paths import gauntlet_root

def get_git_commit() -> str:
    """
    Attempts to get the current git commit hash.
    """
    try:
        import subprocess
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("ascii").strip()
        return commit
    except Exception:
        return "unknown"

def generate_markdown_report(
    run_id: str,
    verdicts: Dict[str, str],
    dataset_spec: DatasetSpec,
    model_spec: ModelSpec,
    config: GauntletConfig,
    fold_spearman: List[Dict[str, Any]],
    topk_results: Dict[str, Any],
    tod_results: Dict[str, Any],
    git_commit: str,
    n_prior_runs: int,
    deflated_t_threshold: float,
    audit_stats: Optional[Dict[str, Any]] = None,
    baselines: Optional[Dict[str, Dict[str, float]]] = None
) -> str:
    """
    Generates a structured, readable Markdown report for the Gauntlet run.
    """
    now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
    
    # Verdict badges
    verdict_md = ""
    for side, grade in verdicts.items():
        color = "red" if grade == "DEAD" else "orange" if grade == "FILTER_GRADE" else "green"
        verdict_md += f"- **{side.upper()} Side**: <span style='color:{color};font-weight:bold'>{grade}</span>\n"
        
    # Fold table
    fold_rows = ""
    for f in fold_spearman:
        fold_rows += (
            f"| {f['fold']} | {f['test_months']} | {f['long_rho']:+.4f} | {f['short_rho']:+.4f} | "
            f"{f['best_iter_long']} | {f['best_iter_short']} |\n"
        )
        
    # Top-K table
    topk_rows = ""
    for period, p_val in topk_results.items():
        for k_val, k_data in p_val["K"].items():
            for cost_name, c_data in k_data.items():
                for side in model_spec.sides:
                    stats = c_data[side]
                    t_val = stats.get("t_stat", 0.0)
                    t_marker = " ✷" if abs(t_val) >= deflated_t_threshold else ""
                    b_wr_str = "N/A"
                    if baselines and period in baselines and side in baselines[period]:
                        b_wr_str = f"{baselines[period][side]:.1%}"
                    topk_rows += (
                        f"| {period} | Top-{k_val} | {cost_name} | {side.upper()} | {stats['n']} | "
                        f"{stats['raw_bps']:+.2f} | {stats['net_bps']:+.2f} | {stats['raw_win']:.1%} | "
                        f"{stats['net_win']:.1%} | {b_wr_str} | {t_val}{t_marker} |\n"
                    )

    # Time of day table
    tod_rows = ""
    for time_str, t_data in sorted(tod_results.items()):
        l_n = t_data.get("long_n", 0)
        l_wr = f"{t_data.get('long_raw_win', 0.0):.1%}" if l_n else "N/A"
        l_net = f"{t_data.get('long_net_bps', 0.0):+.2f}" if l_n else "N/A"
        
        s_n = t_data.get("short_n", 0)
        s_wr = f"{t_data.get('short_raw_win', 0.0):.1%}" if s_n else "N/A"
        s_net = f"{t_data.get('short_net_bps', 0.0):+.2f}" if s_n else "N/A"
        
        tod_rows += (
            f"| {time_str} | {l_n} | {l_wr} | {l_net} | {s_n} | {s_wr} | {s_net} |\n"
        )

    audit_stats_md = ""
    if audit_stats is not None:
        audit_stats_md = f"""
## 📊 Dataset Label Verification Stats
- **In-File Verified (INTRA) Rows**: {audit_stats.get('pct_verified', 1.0):.2%}
- **Unverifiable (Missing Target Bar) Rows**: {audit_stats.get('pct_unverifiable', 0.0):.2%}
- **Boundary (Session Terminal) Rows**: {audit_stats.get('pct_boundary', 0.0):.2%}
- **Unverified Label Waiver Reason**: {audit_stats.get('unverified_label_waiver_reason') or 'N/A'}
- **Prefix Invariance Waiver Reason**: {dataset_spec.prefix_invariance_waiver_reason or 'N/A'}
"""

    md = f"""# 🛡️ Validation Gauntlet Report: `{model_spec.name}`

## 📌 Metadata
- **Run ID**: `{run_id}`
- **Evaluated At (UTC)**: `{now_str}`
- **Dataset Path**: `{dataset_spec.path}`
- **Model Adapter**: `{model_spec.adapter}`
- **Git Commit**: `{git_commit}`
- **Multiple Testing Context**: Prior runs for dataset family = `{n_prior_runs}`
- **Deflated t-Threshold**: `{deflated_t_threshold:.4f}` (corrected for `{n_prior_runs + 1}` total tests)

{audit_stats_md}

## ⚖️ Final Verdicts
{verdict_md}

---

## 📈 Fold-Level Spearman Correlation
| Fold | Test Segment | Long Rho | Short Rho | Best Iter Long | Best Iter Short |
|---|---|---|---|---|---|
{fold_rows}

---

## 💻 Top-K Returns (Walk-Forward Pooled)
> [!IMPORTANT]
> Deflated t-threshold is **`{deflated_t_threshold:.2f}`**. Values exceeding this are marked with ✷.

| Period | Config | Cost Level | Side | Trades | Raw bps | Net bps | Raw WR | Net WR | Baseline WR | t-stat |
|---|---|---|---|---|---|---|---|---|---|---|
{topk_rows}

---

## 🕒 Time-of-Day Performance Breakdown (Top-3, Pooled OOS)
> [!NOTE]
> Time-of-day tables are **diagnostic only** and do not feed the verdict engine unless a specific slice was pre-registered in the config. Cost level applied is informational `@6bps`.

| Time | Long Trades | Long Raw WR | Long Net bps | Short Trades | Short Raw WR | Short Net bps |
|---|---|---|---|---|---|---|
{tod_rows}
"""
    return md

def save_report(
    run_id: str,
    dataset_spec: DatasetSpec,
    model_spec: ModelSpec,
    config: GauntletConfig,
    fold_spearman: List[Dict[str, Any]],
    topk_results: Dict[str, Any],
    tod_results: Dict[str, Any],
    verdicts: Dict[str, str],
    row_count: int,
    dataset_sha256: str,
    config_hash: str,
    n_prior_runs: int,
    deflated_t_threshold: float,
    harness_res: Optional[Dict[str, Any]] = None,
    audit_stats: Optional[Dict[str, Any]] = None,
    baselines: Optional[Dict[str, Dict[str, float]]] = None
) -> str:
    """
    Saves the JSON, Markdown, and config lock files to the run output directory.
    Appends the run to the central ledger.jsonl file.
    Returns the report output directory path.
    """
    out_dir = os.path.join(gauntlet_root(), run_id)
    os.makedirs(out_dir, exist_ok=True)
    
    git_commit = get_git_commit()
    
    # 2. Build JSON report
    report_json = {
        "run_id": run_id,
        "evaluated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": git_commit,
        "dataset": {
            "path": dataset_spec.path,
            "sha256": dataset_sha256,
            "row_count": row_count,
            "pct_verified": audit_stats.get("pct_verified", 1.0) if audit_stats else 1.0,
            "pct_unverifiable": audit_stats.get("pct_unverifiable", 0.0) if audit_stats else 0.0,
            "pct_boundary": audit_stats.get("pct_boundary", 0.0) if audit_stats else 0.0,
            "unverified_label_waiver_reason": audit_stats.get("unverified_label_waiver_reason") if audit_stats else None,
            "prefix_invariance_waiver_reason": dataset_spec.prefix_invariance_waiver_reason
        },
        "model": {
            "name": model_spec.name,
            "adapter": model_spec.adapter,
            "params": model_spec.params,
            "features": model_spec.features,
            "binary_threshold": model_spec.params.get("binary_threshold", 0.0020) if model_spec.adapter == "xgb_binary" else None
        },
        "config_hash": config_hash,
        "n_prior_runs": n_prior_runs,
        "deflated_t_threshold": deflated_t_threshold,
        "verdicts": verdicts,
        "baselines": baselines,
        "fold_spearman": fold_spearman,
        "topk": topk_results,
        "time_of_day": tod_results
    }
    
    json_path = os.path.join(out_dir, "report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=2)
        
    # 3. Build Markdown report
    md_content = generate_markdown_report(
        run_id=run_id,
        verdicts=verdicts,
        dataset_spec=dataset_spec,
        model_spec=model_spec,
        config=config,
        fold_spearman=fold_spearman,
        topk_results=topk_results,
        tod_results=tod_results,
        git_commit=git_commit,
        n_prior_runs=n_prior_runs,
        deflated_t_threshold=deflated_t_threshold,
        audit_stats=audit_stats,
        baselines=baselines
    )
    
    md_path = os.path.join(out_dir, "report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
        
    # 3.5 Save preds.npz
    if harness_res is not None:
        import numpy as np
        npz_fields = {
            "idx": harness_res["idx"],
            "ym": harness_res["ym"],
            "q": harness_res["q"],
            "y": harness_res["y"],
            "time": harness_res["time"]
        }
        if "long" in harness_res["preds"]:
            npz_fields["rl"] = harness_res["preds"]["long"]
        if "short" in harness_res["preds"]:
            npz_fields["rs"] = harness_res["preds"]["short"]
        np.savez_compressed(os.path.join(out_dir, "preds.npz"), **npz_fields)
        
    # 4. Append to central ledger
    ledger_path = os.path.join(gauntlet_root(), "ledger.jsonl")
    ledger_record = {
        "event": "completed",
        "run_id": run_id,
        "evaluated_at": report_json["evaluated_at"],
        "model_name": model_spec.name,
        "dataset_path": dataset_spec.path,
        "dataset_sha256": dataset_sha256,
        "config_hash": config_hash,
        "verdicts": verdicts,
        "git_commit": git_commit
    }
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(ledger_record) + "\n")
        
    print(f"Report saved to: {out_dir}")
    return out_dir

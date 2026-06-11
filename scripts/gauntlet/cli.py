import os
import sys
import datetime
import hashlib
import json
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List
from dataclasses import asdict

from .contracts import DatasetSpec, ModelSpec, GauntletConfig
from .paths import gauntlet_root

# Pre-registered datasets
REGISTERED_DATASETS = {
    "1h_v3_3y": DatasetSpec(
        path="data/ranking_data_upstox_1h_v3_3y.csv",
        label_col="Next_Hour_Return",
        bar_minutes=60,
        bar_label_side="left",
        label_horizon_bars=1,
        label_may_cross_session=False,
        raw_close_col="Close",
        unverified_label_waiver_reason="Pre-drop 14:15 target bars omitted from 3y training file but verified consistent."
    ),
    "15min_3y": DatasetSpec(
        path="data/ranking_data_upstox_15min_3y.csv",
        label_col="Next_15Min_Return",
        bar_minutes=15,
        bar_label_side="left",
        label_horizon_bars=1,
        label_may_cross_session=False,
        raw_close_col="Close",
        unverified_label_waiver_reason="15m older intraday parquet source files not available in raw history directory."
    ),
    "15min_3y_clean": DatasetSpec(
        path="data/ranking_data_upstox_15min_3y_clean.csv",
        label_col="Next_15Min_Return",
        bar_minutes=15,
        bar_label_side="left",
        label_horizon_bars=1,
        label_may_cross_session=False,
        raw_close_col="Close",
        unverified_label_waiver_reason="15m older intraday parquet source files not available in raw history directory."
    ),
    "daily_5y": DatasetSpec(
        path="data/ranking_data_upstox_daily_5y.csv",
        label_col="Next_Day_Return",
        bar_minutes=1440,
        bar_label_side="left",
        label_horizon_bars=1,
        label_may_cross_session=True,
        raw_close_col="Close",
        unverified_label_waiver_reason="Daily close-to-close returns have no intraday target bars."
    ),
    "daily_macro_v2": DatasetSpec(
        path="data/ranking_data_daily_macro_v2.csv",
        label_col="Label_3D",
        bar_minutes=1440,
        bar_label_side="left",
        label_horizon_bars=3,
        label_may_cross_session=True,
        raw_close_col="Close",
        feature_pipeline=None,
        prefix_invariance_waiver_reason="Daily macro dataset contains cross-asset and global features that cannot be computed per-ticker in isolation.",
        unverified_label_waiver_reason="Daily close-to-close returns have no intraday target bars."
    ),
    "daily_macro_v3": DatasetSpec(
        path="data/ranking_data_daily_macro_v3.csv",
        label_col="Label_1D",
        bar_minutes=1440,
        bar_label_side="left",
        label_horizon_bars=1,
        label_may_cross_session=True,
        raw_close_col="Close",
        feature_pipeline=None,
        prefix_invariance_waiver_reason="Daily macro dataset contains cross-asset and global features that cannot be computed per-ticker in isolation.",
        unverified_label_waiver_reason="Daily close-to-close returns have no intraday target bars."
    )
}

from scripts.feature_utils import compute_features
PIPELINE_REGISTRY = {
    "ranking_v3": lambda df: compute_features(df, legacy=False)
}


def get_canonical_hash(config: GauntletConfig) -> str:
    """
    Computes a canonical SHA-256 hash of the GauntletConfig.
    The dictionary is serialized to JSON with sorted keys to ensure stability.
    """
    d = asdict(config)
    j = json.dumps(d, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(j.encode('utf-8')).hexdigest()

def load_model_spec(model_name: str, model_dir: str) -> ModelSpec:
    """
    Loads ModelSpec dynamically from the model registry metadata.json.
    """
    meta_path = os.path.join(model_dir, "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Model metadata not found at {meta_path}")
        
    with open(meta_path, "r") as f:
        meta = json.load(f)
        
    features = meta["features"]
    # Default parameters if missing
    params = meta.get("params", {
        "objective": "rank:pairwise",
        "eta": 0.03,
        "max_depth": 5,
        "random_state": 42
    })
    
    # Infer adapter from parameters or type
    objective = params.get("objective", "")
    meta_type = meta.get("type", "")
    
    if "rank:pairwise" in objective or "rank" in meta_type:
        adapter = "xgb_ranker"
    elif "binary:logistic" in objective or "binary" in meta_type:
        adapter = "xgb_binary"
    elif "catboost" in model_name or "cb" in model_name or "Logloss" in params.get("loss_function", ""):
        adapter = "catboost"
    else:
        adapter = "xgb_ranker"
        
    return ModelSpec(
        name=model_name,
        adapter=adapter,
        params=params,
        features=features,
        num_boost_round=meta.get("num_boost_round", 500),
        early_stopping_rounds=meta.get("early_stopping_rounds", 50)
    )

def run_gauntlet(
    dataset_spec: DatasetSpec,
    model_spec: ModelSpec,
    config: GauntletConfig,
    run_id: Optional[str] = None,
    tolerance: float = 0.005,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Runs the 6-stage Validation Gauntlet end-to-end.
    """
    if not run_id:
        config_hash = get_canonical_hash(config)
        run_id = f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{config_hash[:8]}"
        
    out_dir = os.path.join(gauntlet_root(), run_id)
    os.makedirs(out_dir, exist_ok=True)
    
    lock_data = {
        "config": asdict(config),
        "config_hash": get_canonical_hash(config),
        "dataset_path": dataset_spec.path,
        "model_name": model_spec.name,
        "locked_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }
    lock_path = os.path.join(out_dir, "config.lock.json")
    with open(lock_path, "w", encoding="utf-8") as f:
        json.dump(lock_data, f, indent=2)
        
    ledger_path = os.path.join(gauntlet_root(), "ledger.jsonl")
    started_record = {
        "event": "started",
        "run_id": run_id,
        "model_name": model_spec.name,
        "dataset_path": dataset_spec.path,
        "config_hash": lock_data["config_hash"],
        "evaluated_at": lock_data["locked_at"]
    }
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(started_record) + "\n")
        
    if dry_run:
        print(f"[DRY-RUN] Config locked and pre-registered. Run ID: {run_id}")
        return {"run_id": run_id, "verdicts": {}, "output_dir": out_dir, "flagged_features": []}

    # 1. Load dataset with cache (Stage 0)
    from .data_audit import load_dataset_with_cache, audit_dataset
    
    # Optimize memory footprint by loading only required columns
    required_cols = [dataset_spec.label_col, dataset_spec.qid_col, dataset_spec.ticker_col, dataset_spec.datetime_col]
    if dataset_spec.raw_close_col:
        required_cols.append(dataset_spec.raw_close_col)
    required_cols.extend(model_spec.features)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in required_cols:
            required_cols.append(col)
            
    df, sha256_hash = load_dataset_with_cache(dataset_spec, columns=required_cols)
    
    # 2. Audit dataset (Stage 0 assertions)
    audit_stats = audit_dataset(df, dataset_spec, model_spec.features)
    
    # 3. Leakage checks (Stage 1)
    from .leakage import check_prefix_invariance, check_same_bar_correlation, check_within_query_label_shuffle
    
    # A1.1 Prefix Invariance Check
    run_prefix_check = False
    pipeline_fn = None
    if dataset_spec.feature_pipeline is not None:
        pipeline_name = dataset_spec.feature_pipeline
        assert pipeline_name in PIPELINE_REGISTRY, f"Feature pipeline '{pipeline_name}' not registered in PIPELINE_REGISTRY"
        pipeline_fn = PIPELINE_REGISTRY[pipeline_name]
        
        # Check OHLCV columns
        required_ohlcv = ["Open", "High", "Low", "Close", "Volume"]
        missing_ohlcv = [col for col in required_ohlcv if col not in df.columns]
        if missing_ohlcv:
            if dataset_spec.prefix_invariance_waiver_reason:
                print(f"[WARNING] Missing OHLCV columns {missing_ohlcv} for prefix invariance check, but waiver is provided. Reason: {dataset_spec.prefix_invariance_waiver_reason}")
            else:
                raise AssertionError(
                    f"Dataset missing required OHLCV columns {missing_ohlcv} for feature pipeline prefix invariance check. "
                    f"Either add these columns or provide a prefix_invariance_waiver_reason in the DatasetSpec."
                )
        else:
            run_prefix_check = True
    else:
        if dataset_spec.prefix_invariance_waiver_reason:
            print(f"Prefix invariance check skipped. Waiver reason: {dataset_spec.prefix_invariance_waiver_reason}")
        else:
            raise AssertionError(
                "DatasetSpec has feature_pipeline=None but no prefix_invariance_waiver_reason was provided."
            )
            
    if run_prefix_check and pipeline_fn is not None:
        # Select >= 10 tickers stratified by span
        ticker_spans = df.groupby(dataset_spec.ticker_col).size()
        sorted_tickers = ticker_spans.sort_values().index.tolist()
        if len(sorted_tickers) <= 10:
            selected_tickers = sorted_tickers
        else:
            indices = np.linspace(0, len(sorted_tickers) - 1, 10, dtype=int)
            selected_tickers = [sorted_tickers[i] for i in indices]
            
        print(f"Selecting {len(selected_tickers)} tickers for prefix invariance check: {selected_tickers}")
        check_prefix_invariance(df, selected_tickers, pipeline_fn, n_cuts=5)
        
    flagged_features = check_same_bar_correlation(df, dataset_spec, model_spec.features)
    check_within_query_label_shuffle(df, dataset_spec, model_spec, config, tolerance=tolerance)
    
    # 4. Harness (Stage 3 walk-forward loop)
    from .harness import run_harness
    harness_res = run_harness(df, dataset_spec, model_spec, config)
    
    # 5. Metrics & Costs & Verdict (Stage 4 & 5)
    from .metrics import compute_query_spearman, compute_topk_returns, calculate_trade_stats, query_bootstrap_ci, compute_decay_diagnostics, calculate_uplift_t_stat

    
    # Get number of prior runs from central ledger
    ledger_path = os.path.join(gauntlet_root(), "ledger.jsonl")
    n_prior_runs = 0
    if os.path.exists(ledger_path):
        with open(ledger_path, "r") as f:
            for line_no, line in enumerate(f, 1):
                try:
                    record = json.loads(line)
                    if record.get("event") == "started":
                        if record.get("dataset_path") == dataset_spec.path and record.get("run_id") != run_id:
                            n_prior_runs += 1
                except Exception as e:
                    print(f"[WARNING] Corrupt line {line_no} in ledger: {e}")
                    
    # Calculate deflated t-threshold
    from scipy.stats import norm
    deflated_alpha = 0.025 / (n_prior_runs + 1)
    deflated_t_threshold = float(norm.ppf(1.0 - deflated_alpha))
    
    # Setup output structures
    topk_results = {}
    tod_results = {}
    verdicts = {}
    
    # Determine recent window (cadence-aware: 12mo for intraday <= 60m, 24mo for daily >= 1440m)
    recent_window_months = 24 if dataset_spec.bar_minutes >= 1440 else config.recent_window_months
    unique_ym = sorted(list(set(harness_res["ym"])))
    recent_months = unique_ym[-recent_window_months:]
    recent_mask = np.isin(harness_res["ym"], recent_months)
    full_mask = np.ones(len(harness_res["idx"]), dtype=bool)
    
    recent_key = f"recent_{recent_window_months}mo"
    periods = {
        "full_OOS": full_mask,
        recent_key: recent_mask
    }
    
    # Stage 5 entry: Verify config pre-registration hash matches
    current_hash = get_canonical_hash(config)
    with open(lock_path, "r", encoding="utf-8") as f:
        locked_info = json.load(f)
    locked_hash = locked_info["config_hash"]
    assert current_hash == locked_hash, (
        f"Config hash mismatch at Stage 5 verdict time! "
        f"In-memory: {current_hash}, Locked: {locked_hash}. Mutation is prohibited."
    )

    # Calculate universe-baseline WR per side per period
    baselines = {}
    for period, mask in periods.items():
        y_period = harness_res["y"][mask]
        baselines[period] = {
            "long": float((y_period > 0).mean()) if len(y_period) > 0 else 0.5,
            "short": float((-y_period > 0).mean()) if len(y_period) > 0 else 0.5
        }

    # Calculate Top-K returns per side and cost
    results_per_k_side = {side: {} for side in model_spec.sides}
    
    for period in periods:
        topk_results[period] = {"K": {}}
        for k_val in config.top_k:
            topk_results[period]["K"][k_val] = {}
            for cost in config.costs_bps:
                topk_results[period]["K"][k_val][f"{cost}bps"] = {}
                
    for side in model_spec.sides:
        invert = (side == "short")
        
        # Calculate OOS Spearman over all test data
        fold_rhos = [f[f"{side}_rho"] for f in harness_res["fold_stats"]]
        decay_stats = compute_decay_diagnostics(fold_rhos)
        
        # Calculate per-fold Top-K net bps for decay diagnostics
        fold_topk_net_bps = []
        for f_stat in harness_res["fold_stats"]:
            months = [m.strip() for m in f_stat["test_months"].split(",")]
            mask = np.isin(harness_res["ym"], months)
            
            f_preds = harness_res["preds"][side][mask]
            f_y = harness_res["y"][mask]
            f_q = harness_res["q"][mask]
            f_time = harness_res["time"][mask]
            
            if len(f_preds) > 0:
                f_rets, _, _ = compute_topk_returns(
                    f_preds, f_y, f_q, f_time, K=config.primary_k, invert=invert
                )
                cost_val = config.binding_cost_bps / 10000.0
                f_trade_stats = calculate_trade_stats(f_rets, cost_val)
                fold_topk_net_bps.append(f_trade_stats["net_bps"])
            else:
                fold_topk_net_bps.append(0.0)
                
        decay_stats_perf = compute_decay_diagnostics(fold_topk_net_bps)
        
        for k_val in config.top_k:
            verdict_stats_pooled = None
            verdict_stats_recent = None
            
            for cost in config.costs_bps:
                cost_val = cost / 10000.0
                
                # Pooled full returns
                full_rets, _, full_q_map = compute_topk_returns(
                    harness_res["preds"][side],
                    harness_res["y"],
                    harness_res["q"],
                    harness_res["time"],
                    K=k_val,
                    invert=invert
                )
                full_stats = calculate_trade_stats(full_rets, cost_val)
                if cost == config.binding_cost_bps:
                    ci_lower, ci_upper = query_bootstrap_ci(full_q_map, cost_val, seed=config.seed)
                    full_stats["ci_lower"] = ci_lower
                    full_stats["ci_upper"] = ci_upper
                    verdict_stats_pooled = full_stats
                    
                topk_results["full_OOS"]["K"][k_val][f"{cost}bps"][side] = full_stats
                
                # Recent OOS
                recent_preds = harness_res["preds"][side][recent_mask]
                recent_y = harness_res["y"][recent_mask]
                recent_q = harness_res["q"][recent_mask]
                recent_time = harness_res["time"][recent_mask]
                
                rec_rets, _, rec_q_map = compute_topk_returns(
                    recent_preds,
                    recent_y,
                    recent_q,
                    recent_time,
                    K=k_val,
                    invert=invert
                )
                rec_stats = calculate_trade_stats(rec_rets, cost_val)
                
                # Compute magnitude-based recent uplift t-stat
                uplift_t = calculate_uplift_t_stat(
                    recent_preds,
                    recent_y,
                    recent_q,
                    recent_time,
                    K=k_val,
                    invert=invert
                )
                rec_stats["uplift_t_stat"] = uplift_t
                
                if cost == config.binding_cost_bps:
                    verdict_stats_recent = rec_stats
                    
                topk_results[recent_key]["K"][k_val][f"{cost}bps"][side] = rec_stats
                
            results_per_k_side[side][k_val] = {
                "pooled": verdict_stats_pooled,
                "recent": verdict_stats_recent
            }
            
        from .verdict import compute_verdict
        baseline_wr = baselines[recent_key][side]
        verdicts[side] = compute_verdict(
            side=side,
            results_per_k=results_per_k_side[side],
            fold_rhos=fold_rhos,
            decay_stats=decay_stats,
            decay_stats_perf=decay_stats_perf,
            baseline_wr=baseline_wr,
            config=config
        )
        
    # Time of Day (diagnostic, Top-3, full_OOS, 6bps)
    for side in model_spec.sides:
        invert = (side == "short")
        tod_rets, tod_times, _ = compute_topk_returns(
            harness_res["preds"][side],
            harness_res["y"],
            harness_res["q"],
            harness_res["time"],
            K=3,
            invert=invert
        )
        df_tod = pd.DataFrame({"ret": tod_rets, "time": tod_times})
        for t_str, group in df_tod.groupby("time"):
            if t_str not in tod_results:
                tod_results[t_str] = {}
            n_val = len(group)
            raw_wr = (group["ret"] > 0).mean() if n_val else 0.0
            net_val = (group["ret"].mean() - 0.0006) * 10000.0 if n_val else 0.0
            
            tod_results[t_str][f"{side}_n"] = n_val
            tod_results[t_str][f"{side}_raw_win"] = raw_wr
            tod_results[t_str][f"{side}_net_bps"] = net_val
            
    # Save report
    from .report import save_report
    out_dir = save_report(
        run_id=run_id,
        dataset_spec=dataset_spec,
        model_spec=model_spec,
        config=config,
        fold_spearman=harness_res["fold_stats"],
        topk_results=topk_results,
        tod_results=tod_results,
        verdicts=verdicts,
        row_count=len(df),
        dataset_sha256=sha256_hash,
        config_hash=get_canonical_hash(config),
        n_prior_runs=n_prior_runs,
        deflated_t_threshold=deflated_t_threshold,
        harness_res=harness_res,
        audit_stats=audit_stats,
        baselines=baselines
    )
    
    # Stamp model metadata in registry
    from .registry import stamp_model_metadata
    from .report import get_git_commit
    git_commit = get_git_commit()
    try:
        stamp_model_metadata(
            model_name=model_spec.name,
            run_id=run_id,
            verdicts=verdicts,
            binding_cost_bps=config.binding_cost_bps,
            dataset_sha256=sha256_hash,
            config_hash=get_canonical_hash(config),
            git_commit=git_commit,
            audit_stats=audit_stats
        )
    except Exception as stamp_err:
        print(f"[WARNING] Failed to stamp model metadata: {stamp_err}")
    
    return {
        "run_id": run_id,
        "verdicts": verdicts,
        "output_dir": out_dir,
        "flagged_features": flagged_features
    }

def print_ledger():
    ledger_path = os.path.join(gauntlet_root(), "ledger.jsonl")
    if not os.path.exists(ledger_path):
        print("No ledger found.")
        return
    with open(ledger_path, "r") as f:
        for line_no, line in enumerate(f, 1):
            try:
                record = json.loads(line)
                print(f"[{record['evaluated_at'][:19]}] Event: {record.get('event', 'completed')} | Run: {record['run_id']} | Model: {record['model_name']} | Verdict: {record.get('verdicts', 'N/A')}")
            except Exception as e:
                print(f"[WARNING] Corrupt line {line_no} in ledger: {e}")

def main():
    import argparse
    parser = argparse.ArgumentParser(prog="scripts.gauntlet")
    subparsers = parser.add_subparsers(dest="command")
    
    # Run Parser
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--model", type=str, required=True, help="Name of model directory inside models/")
    run_parser.add_argument("--dataset", type=str, required=True, help="Pre-registered dataset name or path to CSV")
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--step-months", type=int, default=4, help="Step months for walk-forward")
    run_parser.add_argument("--test-horizon-months", type=int, default=2, help="Test horizon months for walk-forward")
    
    # Selftest Parser
    selftest_parser = subparsers.add_parser("selftest")
    selftest_parser.add_argument("--full", action="store_true", help="Run full self-tests including slow T8 regression tests")
    
    # Ledger Parser
    subparsers.add_parser("ledger")
    
    # Leakage Audit Parser
    leakage_parser = subparsers.add_parser("leakage-audit")
    leakage_parser.add_argument("--dataset", type=str, required=True, help="Pre-registered dataset name")
    
    args = parser.parse_args()
    
    if args.command == "run":
        # Load dataset spec
        if args.dataset in REGISTERED_DATASETS:
            dataset_spec = REGISTERED_DATASETS[args.dataset]
        else:
            raise ValueError(f"Dataset '{args.dataset}' is not a pre-registered dataset.")
            
        model_dir = os.path.join("models", args.model)
        model_spec = load_model_spec(args.model, model_dir)
        config = GauntletConfig(
            step_months=args.step_months,
            test_horizon_months=args.test_horizon_months
        )
        
        config_hash = get_canonical_hash(config)
        print(f"Pre-registering config... Hash: {config_hash}")
        
        res = run_gauntlet(dataset_spec, model_spec, config, dry_run=args.dry_run)
        if args.dry_run:
            return
        print(f"\n[SUCCESS] Run completed: {res['run_id']}")
        print(f"Verdicts: {res['verdicts']}")
        print(f"Report: {res['output_dir']}/report.md")
        
    elif args.command == "selftest":
        import pytest
        if args.full:
            print("Running full self-tests including T8 regression tests...")
            ret = pytest.main(["tests/gauntlet/", "-v", "-m", "t8 or not t8"])
        else:
            print("Running synthetic self-tests via pytest...")
            ret = pytest.main(["tests/gauntlet/", "-v"])
        sys.exit(ret)
        
    elif args.command == "ledger":
        print_ledger()
        
    elif args.command == "leakage-audit":
        if args.dataset in REGISTERED_DATASETS:
            dataset_spec = REGISTERED_DATASETS[args.dataset]
        else:
            raise ValueError(f"Dataset '{args.dataset}' is not a pre-registered dataset.")
            
        from .data_audit import load_dataset_with_cache
        required_cols = [dataset_spec.ticker_col, dataset_spec.datetime_col, "Open", "High", "Low", "Close", "Volume"]
        df, _ = load_dataset_with_cache(dataset_spec, columns=required_cols)
        
        # Run check_prefix_invariance
        if dataset_spec.feature_pipeline is None:
            print("No feature pipeline registered for this dataset.")
            sys.exit(0)
            
        pipeline_name = dataset_spec.feature_pipeline
        assert pipeline_name in PIPELINE_REGISTRY, f"Feature pipeline '{pipeline_name}' not registered in PIPELINE_REGISTRY"
        pipeline_fn = PIPELINE_REGISTRY[pipeline_name]
        
        from .leakage import check_prefix_invariance
        # Select >= 10 tickers stratified by span
        ticker_spans = df.groupby(dataset_spec.ticker_col).size()
        sorted_tickers = ticker_spans.sort_values().index.tolist()
        if len(sorted_tickers) <= 10:
            selected_tickers = sorted_tickers
        else:
            indices = np.linspace(0, len(sorted_tickers) - 1, 10, dtype=int)
            selected_tickers = [sorted_tickers[i] for i in indices]
            
        print(f"Running prefix-invariance audit on {len(selected_tickers)} tickers...")
        check_prefix_invariance(df, selected_tickers, pipeline_fn, n_cuts=5)
        print("[SUCCESS] prefix-invariance audit completed successfully.")

if __name__ == "__main__":
    main()

import os
import json
import pytest
import numpy as np
import pandas as pd
from unittest import mock

from scripts.gauntlet.contracts import DatasetSpec, ModelSpec, GauntletConfig
from scripts.gauntlet.cli import run_gauntlet
from scripts.gauntlet.verdict import evaluate_verdict, compute_verdict

def test_recent_window_exact():
    # Generate OOS months from 2024-01 to 2026-06 (30 months)
    ym_list = []
    for y in [2024, 2025, 2026]:
        for m in range(1, 13):
            ym_list.append(f"{y}-{m:02d}")
    ym_list = sorted(ym_list)[:30] # First 30 months
    
    ym_array = np.array(ym_list)
    config = GauntletConfig(recent_window_months=12)
    
    unique_ym = sorted(list(set(ym_array)))
    recent_months = unique_ym[-config.recent_window_months:]
    recent_mask = np.isin(ym_array, recent_months)
    
    assert len(recent_months) == 12
    assert recent_months[0] == "2025-07"
    assert recent_months[-1] == "2026-06"
    assert recent_mask.sum() == 12

def test_verdict_universe_baseline():
    config = GauntletConfig(filter_min_recent_z=2.0)
    
    # Top-K WR is exactly equal to universe baseline WR (no selection skill)
    recent_raw_win = 0.55
    baseline_wr = 0.55
    recent_n = 1000
    
    grade = evaluate_verdict(
        pooled_net_bps=-1.0,
        pooled_t=0.0,
        recent_net_bps=-1.0,
        recent_raw_wr=recent_raw_win,
        recent_n=recent_n,
        baseline_wr=baseline_wr,
        fold_rhos=[0.01, 0.02],
        decay_slope=0.0,
        decay_p_val=1.0,
        decay_slope_perf=0.0,
        decay_p_val_perf=1.0,
        config=config
    )
    # Since z-stat is 0.0, it must be DEAD
    assert grade == "DEAD"

def test_verdict_no_best_of_k():
    config = GauntletConfig(primary_k=3)
    
    # Stats where K=1 passes, but K=3 fails
    results_per_k = {
        1: {
            "pooled": {"net_bps": 5.0, "t_stat": 3.0},
            "recent": {"net_bps": 4.0, "raw_win": 0.60, "n": 500}
        },
        3: {
            "pooled": {"net_bps": -1.0, "t_stat": 0.0},
            "recent": {"net_bps": -2.0, "raw_win": 0.45, "n": 500}
        }
    }
    
    grade = compute_verdict(
        side="long",
        results_per_k=results_per_k,
        fold_rhos=[0.05, 0.04],
        decay_stats={"slope": 0.0, "p_value": 1.0},
        decay_stats_perf={"slope": 0.0, "p_value": 1.0},
        baseline_wr=0.50,
        config=config
    )
    assert grade == "DEAD"

def test_verdict_filter_real_skill():
    config = GauntletConfig(filter_min_recent_z=2.0)
    
    # Top-K WR is 60%, baseline is 50%, large sample size -> z-score > 2.0
    recent_raw_win = 0.60
    baseline_wr = 0.50
    recent_n = 500
    
    grade = evaluate_verdict(
        pooled_net_bps=-1.0,
        pooled_t=0.0,
        recent_net_bps=-1.0,
        recent_raw_wr=recent_raw_win,
        recent_n=recent_n,
        baseline_wr=baseline_wr,
        fold_rhos=[0.05, 0.06] * 3,
        decay_slope=0.0,
        decay_p_val=1.0,
        decay_slope_perf=0.0,
        decay_p_val_perf=1.0,
        config=config
    )
    assert grade == "FILTER_GRADE"

def test_r4_lock_before_train(tmp_path):
    csv_path = os.path.join(tmp_path, "tiny.csv")
    from scripts.gauntlet.synth import generate_synthetic_panel
    df = generate_synthetic_panel(
        path=csv_path,
        n_tickers=5,
        n_years=1,
        planted_rho=0.0,
        seed=42
    )
    
    spec = DatasetSpec(
        path=csv_path,
        label_col="Next_Hour_Return",
        bar_minutes=60,
        bar_label_side="left",
        label_horizon_bars=1,
        label_may_cross_session=False,
        session_close="15:15",
        raw_close_col="Close",
        prefix_invariance_waiver_reason="Skip"
    )
    
    model_spec = ModelSpec(
        name="tiny_model",
        adapter="xgb_ranker",
        params={"objective": "rank:pairwise", "eta": 0.1, "max_depth": 2},
        features=["noise_feature_1"],
        sides=("long",)
    )
    
    config = GauntletConfig(
        min_train_months=6,
        test_horizon_months=1,
        step_months=2,
        top_k=(1,)
    )
    
    def mock_run_harness(df_arg, spec_arg, model_spec_arg, config_arg):
        from scripts.gauntlet.paths import gauntlet_root
        lock_path = os.path.join(gauntlet_root(), "run_lock_test", "config.lock.json")
        assert os.path.exists(lock_path), "config.lock.json must exist before training"
        
        return {
            "idx": np.array([0]),
            "ym": np.array(["2024-01"]),
            "q": np.array([0]),
            "y": np.array([0.01]),
            "time": np.array(["09:15:00"]),
            "preds": {"long": np.array([0.5])},
            "fold_stats": [{"fold": 1, "test_months": "2024-01", "best_iter_long": 5, "best_iter_short": 0, "long_rho": 0.0, "short_rho": 0.0}]
        }
        
    with mock.patch("scripts.gauntlet.harness.run_harness", side_effect=mock_run_harness):
        run_gauntlet(spec, model_spec, config, run_id="run_lock_test")

def test_r4_tamper_aborts(tmp_path):
    csv_path = os.path.join(tmp_path, "tiny.csv")
    from scripts.gauntlet.synth import generate_synthetic_panel
    df = generate_synthetic_panel(
        path=csv_path,
        n_tickers=5,
        n_years=1,
        planted_rho=0.0,
        seed=42
    )
    
    spec = DatasetSpec(
        path=csv_path,
        label_col="Next_Hour_Return",
        bar_minutes=60,
        bar_label_side="left",
        label_horizon_bars=1,
        label_may_cross_session=False,
        session_close="15:15",
        raw_close_col="Close",
        prefix_invariance_waiver_reason="Skip"
    )
    
    model_spec = ModelSpec(
        name="tiny_model",
        adapter="xgb_ranker",
        params={"objective": "rank:pairwise", "eta": 0.1},
        features=["noise_feature_1"],
        sides=("long",)
    )
    
    config = GauntletConfig(
        min_train_months=6,
        test_horizon_months=1,
        step_months=2,
        top_k=(1,)
    )
    
    def mock_run_harness_mutate(df_arg, spec_arg, model_spec_arg, config_arg):
        object.__setattr__(config_arg, "filter_min_recent_z", 9.9)
        return {
            "idx": np.array([0]),
            "ym": np.array(["2024-01"]),
            "q": np.array([0]),
            "y": np.array([0.01]),
            "time": np.array(["09:15:00"]),
            "preds": {"long": np.array([0.5])},
            "fold_stats": [{"fold": 1, "test_months": "2024-01", "best_iter_long": 5, "best_iter_short": 0, "long_rho": 0.0, "short_rho": 0.0}]
        }
        
    with mock.patch("scripts.gauntlet.harness.run_harness", side_effect=mock_run_harness_mutate):
        with pytest.raises(AssertionError, match="Config hash mismatch at Stage 5 verdict time"):
            run_gauntlet(spec, model_spec, config, run_id="run_tamper_test")

def test_r4_ledger_started_counted(tmp_path):
    csv_path = os.path.join(tmp_path, "tiny.csv")
    from scripts.gauntlet.synth import generate_synthetic_panel
    df = generate_synthetic_panel(
        path=csv_path,
        n_tickers=5,
        n_years=1,
        planted_rho=0.0,
        seed=42
    )
    
    spec = DatasetSpec(
        path=csv_path,
        label_col="Next_Hour_Return",
        bar_minutes=60,
        bar_label_side="left",
        label_horizon_bars=1,
        label_may_cross_session=False,
        session_close="15:15",
        raw_close_col="Close",
        prefix_invariance_waiver_reason="Skip"
    )
    
    model_spec = ModelSpec(
        name="tiny_model",
        adapter="xgb_ranker",
        params={"objective": "rank:pairwise", "eta": 0.1},
        features=["noise_feature_1"],
        sides=("long",)
    )
    
    config = GauntletConfig(
        min_train_months=6,
        test_horizon_months=1,
        step_months=2,
        top_k=(1,)
    )
    
    run_gauntlet(spec, model_spec, config, run_id="run1", tolerance=0.05)
    
    from scripts.gauntlet.paths import gauntlet_root
    ledger_path = os.path.join(gauntlet_root(), "ledger.jsonl")
    
    started_count = 0
    with open(ledger_path, "r") as f:
        for line in f:
            record = json.loads(line)
            if record.get("event") == "started" and record.get("dataset_path") == spec.path:
                started_count += 1
                
    assert started_count == 1

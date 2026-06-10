import os
import shutil
import pytest
import numpy as np
import pandas as pd
from unittest import mock

from scripts.gauntlet.contracts import DatasetSpec, ModelSpec, GauntletConfig
from scripts.gauntlet.synth import generate_synthetic_panel
from scripts.gauntlet.cli import run_gauntlet
from scripts.gauntlet.data_audit import audit_dataset
from scripts.gauntlet.leakage import check_prefix_invariance, check_same_bar_correlation
from scripts.gauntlet.costs import verify_cost_invariants
from scripts.gauntlet.splits import generate_folds

@pytest.fixture
def temp_dir(tmp_path):
    # Setup temporary directory for test runs
    d = tmp_path / "gauntlet_test"
    d.mkdir()
    yield str(d)

def test_t1_power(temp_dir):
    # T1 power: Planted rho = 0.30 (strong signal to ensure detection in small dataset)
    csv_path = os.path.join(temp_dir, "synthetic_t1.csv")
    
    # Generate 10 tickers, 2 years of daily data (approx 250*5 = 1250 bars)
    df = generate_synthetic_panel(
        path=csv_path,
        n_tickers=10,
        n_years=2,
        planted_rho=0.30,
        seed=100
    )
    
    dataset_spec = DatasetSpec(
        path=csv_path,
        label_col="Next_Hour_Return",
        bar_minutes=60,
        bar_label_side="left",
        label_horizon_bars=1,
        label_may_cross_session=False,
        raw_close_col="Close"
    )
    
    model_spec = ModelSpec(
        name="t1_model",
        adapter="xgb_ranker",
        params={
            "objective": "rank:pairwise",
            "eta": 0.1,
            "max_depth": 3,
            "verbosity": 0,
            "tree_method": "hist"
        },
        features=["signal_feature"],
        sides=("long",),
        num_boost_round=20,
        early_stopping_rounds=5
    )
    
    config = GauntletConfig(
        min_train_months=12,
        test_horizon_months=1,
        step_months=2,
        top_k=(1,),
        trigger_min_net_bps=0.5,
        trigger_min_t=1.0
    )
    
    # Run gauntlet
    res = run_gauntlet(dataset_spec, model_spec, config, run_id="run_t1", tolerance=0.03)
    
    # Verdict should not be DEAD because planted signal is very strong
    assert res["verdicts"]["long"] in ["TRIGGER_GRADE", "FILTER_GRADE"]
    
    # Verify preds.npz was correctly generated and contains expected fields
    from scripts.gauntlet.paths import gauntlet_root
    npz_path = os.path.join(gauntlet_root(), "run_t1", "preds.npz")
    assert os.path.exists(npz_path), f"preds.npz not found at {npz_path}"
    
    npz_data = np.load(npz_path)
    for k in ["idx", "ym", "q", "y", "time", "rl"]:
        assert k in npz_data, f"Key {k} missing from preds.npz"

def test_t2_false_positive(temp_dir):
    # T2 false-positive: Noise features (planted_rho = 0.0)
    csv_path = os.path.join(temp_dir, "synthetic_t2.csv")
    
    df = generate_synthetic_panel(
        path=csv_path,
        n_tickers=10,
        n_years=2,
        planted_rho=0.00,
        seed=200
    )
    
    dataset_spec = DatasetSpec(
        path=csv_path,
        label_col="Next_Hour_Return",
        bar_minutes=60,
        bar_label_side="left",
        label_horizon_bars=1,
        label_may_cross_session=False,
        raw_close_col="Close"
    )
    
    model_spec = ModelSpec(
        name="t2_model",
        adapter="xgb_ranker",
        params={
            "objective": "rank:pairwise",
            "eta": 0.1,
            "max_depth": 3,
            "verbosity": 0,
            "tree_method": "hist"
        },
        features=["noise_feature_1"],
        sides=("long",),
        num_boost_round=10,
        early_stopping_rounds=3
    )
    
    config = GauntletConfig(
        min_train_months=12,
        test_horizon_months=1,
        step_months=2,
        top_k=(1,),
        trigger_min_net_bps=2.0,
        trigger_min_t=2.0
    )
    
    res = run_gauntlet(dataset_spec, model_spec, config, run_id="run_t2", tolerance=0.03)
    
    # With no signal, verdict must be DEAD
    assert res["verdicts"]["long"] == "DEAD"

def test_t3_leak_detection():
    # T3 target leak detection: Feature is equal to the label
    df = pd.DataFrame({
        "Next_Hour_Return": [0.01, -0.02, 0.03, -0.01, 0.02],
        "leaky_feature": [0.01, -0.02, 0.03, -0.01, 0.02]
    })
    
    spec = DatasetSpec(
        path="dummy_path",
        label_col="Next_Hour_Return",
        bar_minutes=60,
        bar_label_side="left",
        label_horizon_bars=1,
        label_may_cross_session=False
    )
    
    with pytest.raises(AssertionError, match="Target leakage detected"):
        check_same_bar_correlation(df, spec, features=["leaky_feature"])

def test_t4_lookahead_detection():
    # T4 prefix invariance check: feature uses future data
    # Create simple time-series with 60 rows for 5 tickers to pass length check (min 50) and checked count (min 5)
    tickers = [f"TKR{i}" for i in range(5)]
    rows = []
    for t in tickers:
        dts = pd.date_range("2026-06-01 09:15:00", periods=60, freq="h")
        closes = np.random.uniform(100.0, 110.0, 60)
        for dt, cl in zip(dts, closes):
            rows.append({"Ticker": t, "DateTime": dt, "Close": cl})
    df_raw = pd.DataFrame(rows)
    
    # We define a custom leaky pipeline that shifts Close backward (lookahead)
    def compute_features_leaky(df_ticker):
        res = pd.DataFrame(index=df_ticker.index)
        res["leaky_indicator"] = df_ticker["Close"].shift(-1)
        return res
        
    with pytest.raises(AssertionError, match="Prefix invariance violation"):
        check_prefix_invariance(df_raw, tickers=tickers, pipeline_fn=compute_features_leaky, n_cuts=2)

def test_t4b_centered_window():
    # Centered window indicator (uses future information)
    tickers = [f"TKR{i}" for i in range(5)]
    rows = []
    for t in tickers:
        dts = pd.date_range("2026-06-01 09:15:00", periods=60, freq="h")
        closes = np.random.uniform(100.0, 110.0, 60)
        for dt, cl in zip(dts, closes):
            rows.append({"Ticker": t, "DateTime": dt, "Close": cl})
    df_raw = pd.DataFrame(rows)
    
    def pipeline_centered(df_ticker):
        res = pd.DataFrame(index=df_ticker.index)
        # rolling with center=True uses future data
        res["centered_avg"] = df_ticker["Close"].rolling(5, center=True).mean()
        return res
        
    with pytest.raises(AssertionError, match="Prefix invariance violation"):
        check_prefix_invariance(df_raw, tickers=tickers, pipeline_fn=pipeline_centered, n_cuts=2)

def test_a11_exception_is_failure():
    tickers = [f"TKR{i}" for i in range(5)]
    rows = []
    for t in tickers:
        dts = pd.date_range("2026-06-01 09:15:00", periods=60, freq="h")
        closes = np.random.uniform(100.0, 110.0, 60)
        for dt, cl in zip(dts, closes):
            rows.append({"Ticker": t, "DateTime": dt, "Close": cl})
    df_raw = pd.DataFrame(rows)
    
    def pipeline_raising(df_ticker):
        # Raise exception on sliced input if length is short
        if len(df_ticker) < 40:
            raise ValueError("Exception on sliced input")
        res = pd.DataFrame(index=df_ticker.index)
        res["val"] = df_ticker["Close"]
        return res
        
    with pytest.raises(AssertionError, match="Feature pipeline failed"):
        check_prefix_invariance(df_raw, tickers=tickers, pipeline_fn=pipeline_raising, n_cuts=2)

def test_a11_real_pipeline_clean(temp_dir):
    csv_path = os.path.join(temp_dir, "synthetic_real_pipeline.csv")
    
    # Generate 10 tickers, 1 year of daily data (approx 250*6 = 1500 bars)
    df = generate_synthetic_panel(
        path=csv_path,
        n_tickers=10,
        n_years=1,
        planted_rho=0.05,
        seed=100
    )
    
    from scripts.feature_utils import compute_features
    pipeline_fn = lambda d: compute_features(d, legacy=False)
    
    tickers = df["Ticker"].unique()[:10].tolist()
    # Should pass cleanly without raising any exception or lookahead warning
    res = check_prefix_invariance(df, tickers=tickers, pipeline_fn=pipeline_fn, n_cuts=2)
    assert res is True

def test_t5_cost_sign_tamper():
    # T5 cost-sign tamper check: Net return = Raw return + Cost (cost was added!)
    raw = np.array([0.01, -0.02, 0.03])
    tampered_net = raw + 0.0010  # Cost added instead of subtracted
    
    with pytest.raises(AssertionError, match="Cost invariant violation"):
        verify_cost_invariants(raw, tampered_net, cost=0.0010)

def test_t6_split_tamper():
    # T6 split tamper check: Val and Test overlap
    ym = np.array(["2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06"])
    qids = np.arange(len(ym))
    
    # Mock np.intersect1d to simulate an overlap between splits
    with mock.patch("numpy.intersect1d", return_value=np.array([42])):
        with pytest.raises(AssertionError, match="overlap"):
            generate_folds(ym, qids, min_train_months=2, test_horizon_months=1, step_months=1, label_horizon_bars=0)

def test_t7_overnight_real(temp_dir):
    # Synth panel with plant_overnight_labels=True -> audit must FAIL on overnight returns check
    csv_path = os.path.join(temp_dir, "synthetic_t7_dirty.csv")
    df_dirty = generate_synthetic_panel(
        path=csv_path,
        n_tickers=6,
        n_years=1,
        planted_rho=0.05,
        seed=42,
        plant_overnight_labels=True
    )
    
    spec_dirty = DatasetSpec(
        path=csv_path,
        label_col="Next_Hour_Return",
        bar_minutes=60,
        bar_label_side="left",
        label_horizon_bars=1,
        label_may_cross_session=False,
        session_close="15:15", # Synthetic panel close bar starts at 14:15 and ends at 15:15
        raw_close_col="Close"
    )
    
    with pytest.raises(AssertionError, match="Overnight label leakage detected"):
        audit_dataset(df_dirty, spec_dirty, features=["signal_feature"])
        
    # Clean panel -> must pass
    csv_path_clean = os.path.join(temp_dir, "synthetic_t7_clean.csv")
    df_clean = generate_synthetic_panel(
        path=csv_path_clean,
        n_tickers=6,
        n_years=1,
        planted_rho=0.05,
        seed=42,
        plant_overnight_labels=False
    )
    
    spec_clean = DatasetSpec(
        path=csv_path_clean,
        label_col="Next_Hour_Return",
        bar_minutes=60,
        bar_label_side="left",
        label_horizon_bars=1,
        label_may_cross_session=False,
        session_close="15:15",
        raw_close_col="Close"
    )
    
    # Should pass without overnight return error
    audit_dataset(df_clean, spec_clean, features=["signal_feature"])

def test_a04_coverage_reported(temp_dir):
    csv_path = os.path.join(temp_dir, "synthetic_a04.csv")
    df = generate_synthetic_panel(
        path=csv_path,
        n_tickers=15,
        n_years=1,
        planted_rho=0.05,
        seed=42,
        plant_overnight_labels=False
    )
    
    df_corrupted = df.copy()
    df_corrupted = df_corrupted.sort_values(["Ticker", "DateTime"]).reset_index(drop=True)
    
    # Drop every 3rd row to make target bars missing, reducing verified rate below 80%
    drop_indices = [idx for idx in range(2, len(df_corrupted), 3)]
    df_corrupted = df_corrupted.drop(index=drop_indices).reset_index(drop=True)
    
    csv_corrupted_path = os.path.join(temp_dir, "synthetic_a04_corrupted.csv")
    df_corrupted.to_csv(csv_corrupted_path, index=False)
    
    # 1. No waiver -> should raise AssertionError because verified rate is low
    spec_no_waiver = DatasetSpec(
        path=csv_corrupted_path,
        label_col="Next_Hour_Return",
        bar_minutes=60,
        bar_label_side="left",
        label_horizon_bars=1,
        label_may_cross_session=False,
        session_close="15:15",
        raw_close_col="Close"
    )
    
    with pytest.raises(AssertionError, match="unverifiable rows|below 80% threshold"):
        audit_dataset(df_corrupted, spec_no_waiver, features=["signal_feature"])
        
    # 2. With waiver -> should pass and report low coverage
    spec_with_waiver = DatasetSpec(
        path=csv_corrupted_path,
        label_col="Next_Hour_Return",
        bar_minutes=60,
        bar_label_side="left",
        label_horizon_bars=1,
        label_may_cross_session=False,
        session_close="15:15",
        raw_close_col="Close",
        unverified_label_waiver_reason="Testing low coverage waiver"
    )
    stats = audit_dataset(df_corrupted, spec_with_waiver, features=["signal_feature"])
    assert stats["pct_verified"] < 0.80
    assert stats["unverified_label_waiver_reason"] == "Testing low coverage waiver"

def test_a04_full_recompute(temp_dir):
    csv_path = os.path.join(temp_dir, "synthetic_a04_rec.csv")
    df = generate_synthetic_panel(
        path=csv_path,
        n_tickers=6,
        n_years=1,
        planted_rho=0.05,
        seed=42,
        plant_overnight_labels=False
    )
    
    # Corrupt one INTRA label by 1e-4
    df_corrupted = df.copy()
    df_corrupted.loc[0, "Next_Hour_Return"] += 0.0001
    
    spec = DatasetSpec(
        path=csv_path,
        label_col="Next_Hour_Return",
        bar_minutes=60,
        bar_label_side="left",
        label_horizon_bars=1,
        label_may_cross_session=False,
        session_close="15:15",
        raw_close_col="Close"
    )
    
    with pytest.raises(AssertionError, match="Label mismatch for ticker"):
        audit_dataset(df_corrupted, spec, features=["signal_feature"])

import os
import json
import pytest
import numpy as np

from scripts.gauntlet.contracts import DatasetSpec, ModelSpec, GauntletConfig
from scripts.gauntlet.cli import run_gauntlet, load_model_spec, REGISTERED_DATASETS

@pytest.mark.t8
def test_t8_regression():
    """
    T8 regression test: Runs the real v8 model spec against the real 1h dataset
    and asserts metrics against the verified-correct first-run baseline.
    """
    model_name = "v8_upstox_3y"
    model_dir = os.path.join("models", model_name)
    assert os.path.exists(model_dir), f"Model directory {model_dir} not found"
    
    # Load model and dataset specs
    model_spec = load_model_spec(model_name, model_dir)
    dataset_spec = REGISTERED_DATASETS["1h_v3_3y"]
    
    # Ensure real CSV exists
    assert os.path.exists(dataset_spec.path), f"Dataset CSV {dataset_spec.path} not found"
    
    # Standard GauntletConfig
    config = GauntletConfig()
    
    # Run gauntlet
    res = run_gauntlet(dataset_spec, model_spec, config, run_id="run_t8_regression")
    
    # Load report.json to verify metrics
    report_json_path = os.path.join(res["output_dir"], "report.json")
    assert os.path.exists(report_json_path), f"report.json not found at {report_json_path}"
    
    with open(report_json_path, "r") as f:
        report = json.load(f)
        
    # Assertions
    # 1. Fold count must be exactly 9
    assert len(report["fold_spearman"]) == 9, f"Expected 9 folds, got {len(report['fold_spearman'])}"
    
    # 2. Mean fold rho
    long_rhos = [fold["long_rho"] for fold in report["fold_spearman"]]
    short_rhos = [fold["short_rho"] for fold in report["fold_spearman"]]
    mean_long_rho = np.mean(long_rhos)
    mean_short_rho = np.mean(short_rhos)
    
    # References from audited run 20260610T074638Z-c7de73f9:
    # long: 0.02524, short: 0.02434
    # Specification accepts ±0.004 from 0.0261/0.0245, let's test against gauntlet baseline with ±0.002
    expected_long_rho = 0.02524
    expected_short_rho = 0.02434
    
    assert np.isclose(mean_long_rho, expected_long_rho, atol=0.002), (
        f"Mean long fold rho {mean_long_rho:.5f} is not close to expected {expected_long_rho:.5f}"
    )
    assert np.isclose(mean_short_rho, expected_short_rho, atol=0.002), (
        f"Mean short fold rho {mean_short_rho:.5f} is not close to expected {expected_short_rho:.5f}"
    )
    
    # 3. Top-3 and Top-1 net returns @6bps (cost is config.costs_bps[0])
    # audited first-run values: Top-3 long -1.71, short -3.74; Top-1 long -1.0, short -2.95
    # with ±0.5 bps tolerance
    topk_oos = report["topk"]["full_OOS"]["K"]
    
    # Top-3 @6bps
    top3_long_net = topk_oos["3"]["6.0bps"]["long"]["net_bps"]
    top3_short_net = topk_oos["3"]["6.0bps"]["short"]["net_bps"]
    assert np.isclose(top3_long_net, -1.71, atol=0.5), f"Top-3 long net bps {top3_long_net:.2f} is not within 0.5 bps of -1.71"
    assert np.isclose(top3_short_net, -3.74, atol=0.5), f"Top-3 short net bps {top3_short_net:.2f} is not within 0.5 bps of -3.74"
    
    # Top-1 @6bps
    top1_long_net = topk_oos["1"]["6.0bps"]["long"]["net_bps"]
    top1_short_net = topk_oos["1"]["6.0bps"]["short"]["net_bps"]
    assert np.isclose(top1_long_net, -1.00, atol=0.5), f"Top-1 long net bps {top1_long_net:.2f} is not within 0.5 bps of -1.00"
    assert np.isclose(top1_short_net, -2.95, atol=0.5), f"Top-1 short net bps {top1_short_net:.2f} is not within 0.5 bps of -2.95"

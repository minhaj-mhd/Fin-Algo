import os
import glob
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from scipy.stats import spearmanr

from .contracts import DatasetSpec, ModelSpec, GauntletConfig

def check_prefix_invariance(raw_df: pd.DataFrame, tickers: List[str], pipeline_fn, n_cuts: int = 5) -> bool:
    """
    A1.1 Prefix Invariance Check:
    Verifies that computing features on a sliced dataset data[:t] yields the
    exact same values at and before t as when computed on the full dataset.
    This catches future leaks (lookahead bias, centered window indicators, etc.).
    """
    print("Running A1.1: Prefix Invariance check...")
    
    checked_count = 0
    for ticker in tickers:
        ticker_data = raw_df[raw_df["Ticker"] == ticker].copy()
        ticker_data["DateTime"] = pd.to_datetime(ticker_data["DateTime"])
        ticker_data = ticker_data.set_index("DateTime").sort_index()
        ticker_data = ticker_data[~ticker_data.index.duplicated(keep="first")]
        
        if len(ticker_data) < 50:
            continue
            
        dts = ticker_data.index
        # Cut timestamps spread across the timeline
        cut_indices = np.linspace(len(dts) // 3, len(dts) - 5, n_cuts, dtype=int)
        
        try:
            feats_full = pipeline_fn(ticker_data)
        except Exception as e:
            raise AssertionError(f"Feature pipeline failed for ticker '{ticker}' on full data: {e}") from e
            
        for cut_idx in cut_indices:
            cut_dt = dts[cut_idx]
            sliced_data = ticker_data.loc[:cut_dt]
            
            try:
                feats_sliced = pipeline_fn(sliced_data)
            except Exception as e:
                raise AssertionError(f"Feature pipeline failed for ticker '{ticker}' on sliced data at cut {cut_dt}: {e}") from e
                
            assert len(feats_sliced) > 0, f"Sliced output is empty for ticker '{ticker}' cut at {cut_dt}"
            assert feats_sliced.index[-1] == cut_dt, f"Sliced output's index tail {feats_sliced.index[-1]} does not reach cut_dt {cut_dt} for ticker '{ticker}'"
            
            common_idx = feats_sliced.index.intersection(feats_full.index)
            common_idx = [idx for idx in common_idx if idx <= cut_dt]
            
            if len(common_idx) == 0:
                continue
                
            f_full = feats_full.loc[common_idx]
            f_sliced = feats_sliced.loc[common_idx]
            
            numeric_cols = f_sliced.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col in f_full.columns:
                    val_full = f_full[col].values
                    val_sliced = f_sliced[col].values
                    diff = np.abs(val_full - val_sliced)
                    
                    # We assert they are equal within 1e-9 tolerance
                    assert np.allclose(val_full, val_sliced, atol=1e-9, equal_nan=True), (
                        f"Prefix invariance violation in feature '{col}' for ticker '{ticker}' "
                        f"at timestamp {common_idx[np.argmax(diff)]}: sliced={val_sliced[np.argmax(diff)]}, full={val_full[np.argmax(diff)]}"
                    )
        checked_count += 1
                    
    assert checked_count >= 5, f"Prefix invariance checked only {checked_count} tickers (must check at least 5)"
    print(f"  Prefix invariance check passed on {checked_count} tickers.")
    return True

def check_within_query_label_shuffle(
    df: pd.DataFrame,
    spec: DatasetSpec,
    model_spec: ModelSpec,
    config: GauntletConfig,
    tolerance: float = 0.005
) -> bool:
    """
    A1.2 Within-Query Label Shuffle Check:
    Shuffles labels within each query group on train+val.
    Asserts average OOS Spearman correlation falls in [-0.005, 0.005] (since signal should be zero).
    """
    print("Running A1.2: Within-Query Label Shuffle check...")
    from .harness import run_harness
    from .metrics import compute_query_spearman
    
    # Copy dataset
    df_shuffled = df.copy()
    
    # Shuffle labels within each query group
    rng = np.random.default_rng(42)
    for qid in df_shuffled[spec.qid_col].unique():
        mask = df_shuffled[spec.qid_col] == qid
        labels = df_shuffled.loc[mask, spec.label_col].values.copy()
        rng.shuffle(labels)
        df_shuffled.loc[mask, spec.label_col] = labels
    
    # Define a fast 10-round training spec for shuffle test
    quick_model_spec = ModelSpec(
        name=model_spec.name + "_shuffle_test",
        adapter=model_spec.adapter,
        params=model_spec.params,
        features=model_spec.features,
        sides=model_spec.sides,
        num_boost_round=10,
        early_stopping_rounds=5
    )
    
    # Run harness on shuffled data
    res = run_harness(df_shuffled, spec, quick_model_spec, config)
    
    rhos = []
    for side in quick_model_spec.sides:
        preds = res["preds"][side]
        y = res["y"]
        qids = res["q"]
        corrs = compute_query_spearman(preds, y, qids, invert=(side == "short"))
        if corrs:
            rhos.append(np.mean(corrs))
            
    avg_rho = np.mean(rhos) if rhos else 0.0
    print(f"  Shuffled average OOS Spearman correlation: {avg_rho:.6f}")
    
    assert -tolerance <= avg_rho <= tolerance, (
        f"Within-query label shuffle check failed: "
        f"shuffled OOS Spearman correlation is {avg_rho:.6f} (not in [-{tolerance}, {tolerance}])"
    )
    
    print("  Within-query label shuffle check passed.")
    return True

def check_same_bar_correlation(
    df: pd.DataFrame,
    spec: DatasetSpec,
    features: List[str]
) -> List[str]:
    """
    A1.4 Same-Bar Correlation Screen:
    Computes Spearman correlation of each feature vs the current bar's return.
    Features with correlation > 0.95 are flagged (warning).
    Also checks for direct target leakage (correlation with the target label > 0.95),
    which raises an AssertionError.
    """
    print("Running A1.4: Same-bar correlation screen...")
    flagged = []
    
    # 1. Target leakage check (correlation with the label column)
    y_target = df[spec.label_col].values
    for feat in features:
        feat_val = df[feat].values
        if np.std(feat_val) == 0:
            continue
        corr_target, _ = spearmanr(feat_val, y_target, nan_policy="omit")
        if not np.isnan(corr_target):
            assert abs(corr_target) < 0.95, (
                f"Target leakage detected: Feature '{feat}' has correlation {corr_target:.4f} "
                f"with the label column '{spec.label_col}' (must be < 0.95)"
            )
            
    # 2. Current bar return check
    ret_col = None
    for col in ["Return", "Log_Return", "close_return"]:
        if col in df.columns:
            ret_col = col
            break
            
    if not ret_col:
        print("  Current bar return column not found; skipping same-bar correlation screen.")
        return flagged
        
    y_current = df[ret_col].values
    for feat in features:
        feat_val = df[feat].values
        if np.std(feat_val) == 0:
            continue
            
        corr, _ = spearmanr(feat_val, y_current, nan_policy="omit")
        if not np.isnan(corr) and abs(corr) > 0.95:
            print(f"  [WARNING] Feature '{feat}' has high correlation with current bar return: {corr:.4f}")
            flagged.append(feat)
            
    print(f"  Same-bar correlation screen completed. Flagged {len(flagged)} features.")
    return flagged

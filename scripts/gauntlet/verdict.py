import numpy as np
from typing import List, Dict, Any
from scipy.stats import ttest_1samp
from .contracts import GauntletConfig

def evaluate_verdict(
    pooled_net_bps: float,
    pooled_t: float,
    recent_net_bps: float,
    recent_raw_wr: float,
    recent_n: int,
    baseline_wr: float,
    fold_rhos: List[float],
    decay_slope: float,
    decay_p_val: float,
    decay_slope_perf: float,
    decay_p_val_perf: float,
    config: GauntletConfig
) -> str:
    """
    Evaluates the 3-tier verdict (TRIGGER_GRADE, FILTER_GRADE, DEAD) for a single side and K.
    """
    # Check TRIGGER_GRADE
    # Decay slope is significantly negative if slope < 0 and one-sided p-value < 0.05 (two-sided p-value < 0.10)
    significant_decay_rho = (decay_slope < 0) and (decay_p_val < 0.10)
    significant_decay_perf = (decay_slope_perf < 0) and (decay_p_val_perf < 0.10)
    significant_decay = significant_decay_rho or significant_decay_perf
    
    passes_trigger = (
        pooled_net_bps >= config.trigger_min_net_bps and
        pooled_t >= config.trigger_min_t and
        recent_net_bps > 0.0 and
        not significant_decay
    )
    
    if passes_trigger:
        return "TRIGGER_GRADE"
        
    # Check FILTER_GRADE
    # paired t-test on fold rhos vs 0. We check if they are significantly positive (p < 0.01 one-sided)
    if len(fold_rhos) >= 2:
        t_stat, p_val = ttest_1samp(fold_rhos, 0.0)
        passes_rho_test = (t_stat > 0) and (p_val < 0.02) # two-sided p < 0.02 is equivalent to one-sided p < 0.01
    else:
        passes_rho_test = False
        
    # Win rate z-score for recent window against universe-baseline WR
    if recent_n > 0:
        denom = np.sqrt(max(baseline_wr * (1.0 - baseline_wr), 1e-8) / recent_n)
        z_stat = (recent_raw_wr - baseline_wr) / denom
    else:
        z_stat = 0.0
        
    passes_filter = passes_rho_test and (z_stat >= config.filter_min_recent_z)
    
    if passes_filter:
        return "FILTER_GRADE"
        
    return "DEAD"

def compute_verdict(
    side: str,
    results_per_k: Dict[int, Dict[str, Any]],
    fold_rhos: List[float],
    decay_stats: Dict[str, Any],
    decay_stats_perf: Dict[str, Any],
    baseline_wr: float,
    config: GauntletConfig
) -> str:
    """
    Computes the verdict for a single side at the primary_k.
    """
    k = config.primary_k
    if k not in results_per_k:
        k = list(results_per_k.keys())[0] if results_per_k else 3
        
    stats = results_per_k.get(k)
    if not stats:
        return "DEAD"
        
    return evaluate_verdict(
        pooled_net_bps=stats["pooled"]["net_bps"],
        pooled_t=stats["pooled"]["t_stat"],
        recent_net_bps=stats["recent"]["net_bps"],
        recent_raw_wr=stats["recent"]["raw_win"],
        recent_n=stats["recent"]["n"],
        baseline_wr=baseline_wr,
        fold_rhos=fold_rhos,
        decay_slope=decay_stats["slope"],
        decay_p_val=decay_stats["p_value"],
        decay_slope_perf=decay_stats_perf["slope"],
        decay_p_val_perf=decay_stats_perf["p_value"],
        config=config
    )

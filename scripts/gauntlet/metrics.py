import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from scipy.stats import spearmanr, ttest_1samp, linregress

from .costs import verify_cost_invariants

def compute_query_spearman(preds: np.ndarray, y: np.ndarray, qids: np.ndarray, invert: bool = False) -> List[float]:
    """
    Computes Spearman correlation per query.
    """
    corrs = []
    for qid in np.unique(qids):
        m = qids == qid
        if m.sum() < 2:
            continue
        pred_q = preds[m]
        y_q = -y[m] if invert else y[m]
        rho, _ = spearmanr(pred_q, y_q)
        if not np.isnan(rho):
            corrs.append(float(rho))
    return corrs

def compute_topk_returns(
    preds: np.ndarray,
    y: np.ndarray,
    qids: np.ndarray,
    times: np.ndarray,
    K: int,
    invert: bool = False
) -> Tuple[np.ndarray, List[str], Dict[Any, List[float]]]:
    """
    Selects Top-K items per query group based on scores.
    Returns (pooled_returns, pooled_times, query_to_returns_map).
    For short models, returns are inverted (-y) to represent profit.
    """
    pooled_rets = []
    pooled_times = []
    query_to_rets = {}
    
    unique_qids = np.unique(qids)
    for qid in unique_qids:
        m = qids == qid
        if m.sum() < max(3, K):
            continue
        
        preds_q = preds[m]
        y_q = y[m]
        times_q = times[m]
        
        # Sort indices ascending, so top K are the last K elements
        top_idx = np.argsort(preds_q)[-K:]
        
        rets_selected = []
        for idx in top_idx:
            ret = -y_q[idx] if invert else y_q[idx]
            pooled_rets.append(ret)
            pooled_times.append(times_q[idx])
            rets_selected.append(ret)
            
        query_to_rets[qid] = rets_selected
        
    return np.array(pooled_rets), pooled_times, query_to_rets

def calculate_trade_stats(returns: np.ndarray, cost: float) -> Dict[str, Any]:
    """
    Computes trade statistics for an array of returns under a given cost.
    """
    r = np.asarray(returns, dtype=float)
    if len(r) == 0:
        return dict(n=0, raw_bps=0.0, net_bps=0.0, raw_win=0.0, net_win=0.0, t_stat=0.0)
        
    net = r - cost
    
    # Cost Invariant Verification
    verify_cost_invariants(r, net, cost)
    
    # Calculate t-statistic
    if len(r) > 1 and np.std(net) > 0:
        t_stat = float(ttest_1samp(net, 0.0).statistic)
    else:
        t_stat = 0.0
        
    return dict(
        n=int(len(r)),
        raw_bps=round(float(r.mean()) * 10000, 2),
        net_bps=round(float(net.mean()) * 10000, 2),
        raw_win=round(float((r > 0).mean()), 4),
        net_win=round(float((net > 0).mean()), 4),
        t_stat=round(t_stat, 2)
    )

def query_bootstrap_ci(
    query_returns: Dict[Any, List[float]],
    cost: float,
    n_reps: int = 1000,
    seed: int = 42
) -> Tuple[float, float]:
    """
    Performs per-query bootstrap resampling.
    Resamples query groups with replacement, pooling their returns.
    """
    qids = list(query_returns.keys())
    if not qids:
        return 0.0, 0.0
        
    rng = np.random.default_rng(seed)
    means = []
    
    for _ in range(n_reps):
        resampled_qids = rng.choice(qids, size=len(qids), replace=True)
        pooled_rets = []
        for qid in resampled_qids:
            pooled_rets.extend(query_returns[qid])
            
        if pooled_rets:
            pooled_arr = np.array(pooled_rets, dtype=float)
            means.append((pooled_arr.mean() - cost) * 10000)
        else:
            means.append(0.0)
            
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

def compute_decay_diagnostics(fold_values: List[float]) -> Dict[str, float]:
    """
    Fits an OLS regression on fold values vs fold index to measure performance decay.
    """
    n = len(fold_values)
    if n < 3:
        return {"slope": 0.0, "p_value": 1.0}
        
    x = np.arange(n)
    slope, intercept, r_val, p_val, std_err = linregress(x, fold_values)
    
    return {
        "slope": float(slope),
        "p_value": float(p_val)
    }

def calculate_uplift_t_stat(
    preds: np.ndarray,
    y: np.ndarray,
    qids: np.ndarray,
    times: np.ndarray,
    K: int,
    invert: bool = False
) -> float:
    """
    Computes the t-statistic of the top-K trades' raw returns against
    the cross-sectional mean returns of their respective query groups.
    """
    # 1. Compute query group means
    query_means = {}
    for qid in np.unique(qids):
        m = qids == qid
        if m.sum() > 0:
            query_means[qid] = np.mean(y[m])

    # 2. Get the top-K trades
    unique_qids = np.unique(qids)
    uplifts = []
    
    for qid in unique_qids:
        m = qids == qid
        if m.sum() < max(3, K):
            continue
        
        preds_q = preds[m]
        y_q = y[m]
        
        top_idx = np.argsort(preds_q)[-K:]
        
        for idx in top_idx:
            # Raw return of this trade (adjusted for side)
            ret = -y_q[idx] if invert else y_q[idx]
            # Group mean (also adjusted for side)
            grp_mean = -query_means[qid] if invert else query_means[qid]
            uplifts.append(ret - grp_mean)
            
    if len(uplifts) > 1 and np.std(uplifts) > 0:
        t_stat = float(ttest_1samp(uplifts, 0.0).statistic)
    else:
        t_stat = 0.0
        
    return t_stat

import numpy as np
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class FoldPlan:
    fold_idx: int
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    test_months: List[str]

def validate_fold_plan(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    test_months: List[str],
    covered_test_months: set,
    train_months: List[str],
    min_train_months: int,
    sliding_window_months: Optional[int] = None
) -> None:
    assert len(np.intersect1d(train_idx, val_idx)) == 0, "Train and Val overlap"
    assert len(np.intersect1d(val_idx, test_idx)) == 0, "Val and Test overlap"
    assert len(np.intersect1d(train_idx, test_idx)) == 0, "Train and Test overlap"
    
    assert train_idx.max() < val_idx.min(), "Chronological ordering violated between train and val"
    assert val_idx.max() < test_idx.min(), "Chronological ordering violated between val and test"
    
    for tm in test_months:
        assert tm not in covered_test_months, f"Test month {tm} covered multiple times"
        covered_test_months.add(tm)
        
    expected_train_months = sliding_window_months if sliding_window_months is not None else min_train_months
    assert len(train_months) >= expected_train_months, "Not enough train months"

def generate_folds(
    ym: np.ndarray,
    qids: np.ndarray,
    min_train_months: int,
    test_horizon_months: int,
    step_months: int,
    label_horizon_bars: int,
    embargo_bars: Optional[int] = None,
    sliding_window_months: Optional[int] = None
) -> List[FoldPlan]:
    if embargo_bars is None:
        embargo_bars = label_horizon_bars
        
    months = sorted(list(np.unique(ym)))
    folds = []
    
    # helper to drop last N unique qids
    def purge_end(indices, n_bars):
        if n_bars <= 0 or len(indices) == 0:
            return indices
        q = qids[indices]
        uq = np.unique(q)
        if len(uq) <= n_bars:
            return np.array([], dtype=int)
        cutoff = uq[-n_bars - 1]
        # Keep where qids[indices] <= cutoff
        return indices[q <= cutoff]

    # helper to drop first N unique qids
    def embargo_start(indices, n_bars):
        if n_bars <= 0 or len(indices) == 0:
            return indices
        q = qids[indices]
        uq = np.unique(q)
        if len(uq) <= n_bars:
            return np.array([], dtype=int)
        cutoff = uq[n_bars]
        return indices[q >= cutoff]

    covered_test_months = set()
    
    fold_idx = 1
    for i in range(min_train_months, len(months) - test_horizon_months, step_months):
        if sliding_window_months:
            start_i = max(0, i - sliding_window_months)
            train_months = months[start_i:i]
        else:
            train_months = months[:i]
            
        val_months = [months[i]]
        test_months = months[i+1 : i+1+test_horizon_months]
        
        trm = np.isin(ym, train_months)
        vam = np.isin(ym, val_months)
        tem = np.isin(ym, test_months)
        
        train_idx = np.where(trm)[0]
        val_idx = np.where(vam)[0]
        test_idx = np.where(tem)[0]
        
        # Purge train end
        train_idx = purge_end(train_idx, label_horizon_bars)
        
        # Embargo val start, Purge val end
        val_idx = embargo_start(val_idx, embargo_bars)
        val_idx = purge_end(val_idx, label_horizon_bars)
        
        # Embargo test start
        test_idx = embargo_start(test_idx, embargo_bars)
        
        if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
            continue
            
        validate_fold_plan(
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            test_months=test_months,
            covered_test_months=covered_test_months,
            train_months=train_months,
            min_train_months=min_train_months,
            sliding_window_months=sliding_window_months
        )
        
        folds.append(FoldPlan(
            fold_idx=fold_idx,
            train_indices=train_idx,
            val_indices=val_idx,
            test_indices=test_idx,
            test_months=test_months
        ))
        fold_idx += 1
        
    return folds

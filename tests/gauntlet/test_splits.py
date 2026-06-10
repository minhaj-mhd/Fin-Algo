import numpy as np
import pytest
from scripts.gauntlet.splits import generate_folds, FoldPlan

def test_generate_folds_basic():
    # 10 months, 100 rows per month
    # We use qids 0..99 for month 1, 100..199 for month 2, etc.
    ym = []
    qids = []
    
    current_qid = 0
    for m in range(1, 11):
        month_str = f"2026-{m:02d}"
        for i in range(100):
            ym.append(month_str)
            qids.append(current_qid)
            current_qid += 1
            
    ym = np.array(ym)
    qids = np.array(qids)
    
    folds = generate_folds(
        ym=ym,
        qids=qids,
        min_train_months=3,
        test_horizon_months=1,
        step_months=1,
        label_horizon_bars=5,
        embargo_bars=5
    )
    
    assert len(folds) > 0
    for f in folds:
        assert len(np.intersect1d(f.train_indices, f.val_indices)) == 0
        assert len(np.intersect1d(f.val_indices, f.test_indices)) == 0
        assert f.train_indices.max() < f.val_indices.min()
        assert f.val_indices.max() < f.test_indices.min()
        
        # Train should drop last 5
        # The last qid of train before purge would be e.g., 299 for fold 1
        # so max qid in train should be 294
        
        # Wait, if val is month 4 (qids 300..399)
        # train max qid originally 299, after purge it drops last 5 qids (299, 298, 297, 296, 295)
        # so max train qid is 294.
        
        # let's verify exact boundaries for fold 1
        if f.fold_idx == 1:
            # train: months 1,2,3 -> qids 0..299
            # purge 5 -> max train qid = 294
            assert qids[f.train_indices].max() == 294
            
            # val: month 4 -> qids 300..399
            # embargo 5 -> drops 300..304
            # purge 5 -> drops 395..399
            assert qids[f.val_indices].min() == 305
            assert qids[f.val_indices].max() == 394
            
            # test: month 5 -> qids 400..499
            # embargo 5 -> drops 400..404
            assert qids[f.test_indices].min() == 405
            assert qids[f.test_indices].max() == 499

def test_multiple_tickers_per_qid():
    ym = []
    qids = []
    
    # 5 tickers per qid
    current_qid = 0
    for m in range(1, 11):
        month_str = f"2026-{m:02d}"
        for i in range(100):  # 100 queries per month
            for _ in range(5):
                ym.append(month_str)
                qids.append(current_qid)
            current_qid += 1
            
    ym = np.array(ym)
    qids = np.array(qids)
    
    folds = generate_folds(
        ym=ym,
        qids=qids,
        min_train_months=3,
        test_horizon_months=1,
        step_months=1,
        label_horizon_bars=2,
        embargo_bars=3
    )
    
    f = folds[0]
    # train qids: 0..299
    # purge 2 -> drops 298, 299. max qid = 297
    assert qids[f.train_indices].max() == 297
    # there should be 5 rows with qid 297
    assert (qids[f.train_indices] == 297).sum() == 5
    
    # val qids: 300..399
    # embargo 3 -> drops 300, 301, 302. min qid = 303
    # purge 2 -> drops 398, 399. max qid = 397
    assert qids[f.val_indices].min() == 303
    assert qids[f.val_indices].max() == 397

def test_sliding_window():
    ym = []
    qids = []
    
    current_qid = 0
    for m in range(1, 11):
        month_str = f"2026-{m:02d}"
        for i in range(100):
            ym.append(month_str)
            qids.append(current_qid)
            current_qid += 1
            
    ym = np.array(ym)
    qids = np.array(qids)
    
    folds = generate_folds(
        ym=ym,
        qids=qids,
        min_train_months=3,
        test_horizon_months=1,
        step_months=1,
        label_horizon_bars=0,
        embargo_bars=0,
        sliding_window_months=2
    )
    
    # fold 1: i=3 -> val=month 4. train should be months 2,3 (since sliding=2)
    f = folds[0]
    assert "2026-02" in ym[f.train_indices]
    assert "2026-01" not in ym[f.train_indices]

def test_validate_fold_plan():
    from scripts.gauntlet.splits import validate_fold_plan
    
    train_idx = np.array([0, 1, 2])
    val_idx = np.array([3, 4])
    test_idx = np.array([5, 6])
    test_months = ["2026-06"]
    covered_test_months = set()
    train_months = ["2026-01", "2026-02", "2026-03"]
    
    # Clean run
    validate_fold_plan(
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        test_months=test_months,
        covered_test_months=covered_test_months,
        train_months=train_months,
        min_train_months=3
    )
    
    # Overlap train and val
    with pytest.raises(AssertionError, match="Train and Val overlap"):
        validate_fold_plan(
            train_idx=np.array([0, 1, 3]),
            val_idx=val_idx,
            test_idx=test_idx,
            test_months=test_months,
            covered_test_months=set(),
            train_months=train_months,
            min_train_months=3
        )
        
    # Chronology violation
    with pytest.raises(AssertionError, match="Chronological ordering violated between train and val"):
        validate_fold_plan(
            train_idx=np.array([0, 4, 1]),
            val_idx=np.array([2, 3]),
            test_idx=test_idx,
            test_months=test_months,
            covered_test_months=set(),
            train_months=train_months,
            min_train_months=3
        )



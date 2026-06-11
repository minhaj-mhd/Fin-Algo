import os
import time
import datetime
from functools import lru_cache
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from scipy.stats import rankdata

from .contracts import DatasetSpec, ModelSpec, GauntletConfig
from .splits import generate_folds, FoldPlan

# Device detection for XGBoost
@lru_cache(maxsize=1)
def detect_device() -> str:
    """
    Detects if CUDA is available for XGBoost hist tree method.
    """
    try:
        import xgboost as xgb
        d = xgb.DMatrix(np.random.randn(10, 2), label=np.arange(10))
        d.set_group([10])
        xgb.train({'objective': 'rank:pairwise', 'device': 'cuda', 'tree_method': 'hist'},
                  d, num_boost_round=1)
        return 'cuda'
    except Exception:
        return 'cpu'

# Label helper
def int_ranks(y: np.ndarray, qids: np.ndarray, invert: bool = False) -> np.ndarray:
    """
    Computes integer ranks per query group. Replicates rankdata ordinal ranking.
    """
    out = np.zeros_like(y, dtype=int)
    for qid in np.unique(qids):
        m = qids == qid
        vals = -y[m] if invert else y[m]
        out[m] = rankdata(vals, method='ordinal') - 1
    return out

# Group helper
def group_sizes(qids: np.ndarray) -> np.ndarray:
    """
    Computes contiguous group sizes for ranking models.
    """
    _, idx, counts = np.unique(qids, return_index=True, return_counts=True)
    order = np.argsort(idx)
    return counts[order]

# Adapters Registry
ADAPTERS = {}

def register_adapter(name: str):
    def decorator(cls):
        ADAPTERS[name] = cls()
        return cls
    return decorator

@register_adapter("xgb_ranker")
class XGBRankerAdapter:
    def fit(self, Xtr: np.ndarray, ytr: np.ndarray, qtr: np.ndarray,
            Xva: np.ndarray, yva: np.ndarray, qva: np.ndarray,
            side: str, params: Dict[str, Any],
            num_boost_round: int, early_stopping_rounds: int,
            seed: int = 42) -> Any:
        import xgboost as xgb
        
        ytr_ranks = int_ranks(ytr, qtr, invert=(side == "short"))
        yva_ranks = int_ranks(yva, qva, invert=(side == "short"))
        
        g_tr = group_sizes(qtr)
        g_va = group_sizes(qva)
        
        dtrain = xgb.DMatrix(Xtr, label=ytr_ranks)
        dtrain.set_group(g_tr)
        dval = xgb.DMatrix(Xva, label=yva_ranks)
        dval.set_group(g_va)
        
        run_params = params.copy()
        if "device" not in run_params:
            run_params["device"] = detect_device()
        if "tree_method" not in run_params:
            run_params["tree_method"] = "hist"
        if "seed" not in run_params and "random_state" not in run_params:
            run_params["seed"] = seed
            
        # Ensure ndcg_exp_gain is False to support ranking more than 31 items
        if "ndcg_exp_gain" not in run_params:
            run_params["ndcg_exp_gain"] = False
            
        model = xgb.train(
            run_params, dtrain,
            num_boost_round=num_boost_round,
            evals=[(dval, 'val')],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False
        )
        return model

    def predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        import xgboost as xgb
        dtest = xgb.DMatrix(X)
        return model.predict(dtest)

@register_adapter("xgb_binary")
class XGBBinaryAdapter:
    def fit(self, Xtr: np.ndarray, ytr: np.ndarray, qtr: np.ndarray,
            Xva: np.ndarray, yva: np.ndarray, qva: np.ndarray,
            side: str, params: Dict[str, Any],
            num_boost_round: int, early_stopping_rounds: int,
            seed: int = 42) -> Any:
        """
        XGBoost binary classification adapter.
        Uses binary_threshold parameter from params (default: 0.0020, i.e. 20bps) to classify returns.
        """
        import xgboost as xgb
        
        # Binary target: > threshold for long, < -threshold for short
        threshold = params.get("binary_threshold", 0.0020)
        ytr_bin = (ytr > threshold).astype(int) if side == "long" else (ytr < -threshold).astype(int)
        yva_bin = (yva > threshold).astype(int) if side == "long" else (yva < -threshold).astype(int)
        
        dtrain = xgb.DMatrix(Xtr, label=ytr_bin)
        dval = xgb.DMatrix(Xva, label=yva_bin)
        
        run_params = params.copy()
        if "device" not in run_params:
            run_params["device"] = detect_device()
        if "tree_method" not in run_params:
            run_params["tree_method"] = "hist"
        if "seed" not in run_params and "random_state" not in run_params:
            run_params["seed"] = seed
        
        # Dynamically calculate scale_pos_weight if not explicitly provided
        if "scale_pos_weight" not in run_params:
            pos_rate = ytr_bin.mean()
            if pos_rate > 0.0:
                run_params["scale_pos_weight"] = (1.0 - pos_rate) / pos_rate
                
        model = xgb.train(
            run_params, dtrain,
            num_boost_round=num_boost_round,
            evals=[(dval, 'val')],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False
        )
        return model

    def predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        import xgboost as xgb
        dtest = xgb.DMatrix(X)
        return model.predict(dtest)

@register_adapter("catboost")
class CatBoostAdapter:
    def fit(self, Xtr: np.ndarray, ytr: np.ndarray, qtr: np.ndarray,
            Xva: np.ndarray, yva: np.ndarray, qva: np.ndarray,
            side: str, params: Dict[str, Any],
            num_boost_round: int, early_stopping_rounds: int,
            seed: int = 42) -> Any:
        """
        CatBoost classifier adapter.
        Uses random_seed parameter from params or seed from config to ensure reproducibility.
        """
        from catboost import CatBoostClassifier
        
        # Target: > 0 for long, < 0 for short
        ytr_bin = (ytr > 0.0).astype(np.int32) if side == "long" else (ytr < 0.0).astype(np.int32)
        yva_bin = (yva > 0.0).astype(np.int32) if side == "long" else (yva < 0.0).astype(np.int32)
        
        run_params = params.copy()
        if "iterations" not in run_params:
            run_params["iterations"] = num_boost_round
        if "early_stopping_rounds" not in run_params:
            run_params["early_stopping_rounds"] = early_stopping_rounds
        if "random_seed" not in run_params:
            run_params["random_seed"] = seed
            
        model = CatBoostClassifier(**run_params)
        model.fit(Xtr, ytr_bin, eval_set=(Xva, yva_bin), verbose=False)
        return model

    def predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        return model.predict_proba(X)[:, 1]

def run_harness(df: pd.DataFrame, spec: DatasetSpec, model_spec: ModelSpec, config: GauntletConfig) -> Dict[str, Any]:
    """
    Runs the walk-forward harness (Stage 3).
    Sorts df by Query_ID, runs splits, trains per fold, and returns OOS predictions.
    """
    print("=" * 60)
    print(f"STAGE 3: RUNNING HARNESS FOR MODEL: {model_spec.name}...")
    print("=" * 60)
    
    # Sort dataset by Query_ID to ensure query groups are contiguous
    df = df.sort_values(by=spec.qid_col).reset_index(drop=True)
    
    # Calculate YearMonth (for folds) and Time (for diagnostics)
    dts = pd.to_datetime(df[spec.datetime_col])
    ym = dts.dt.to_period('M').astype(str).values
    times = dts.dt.time.astype(str).str[:5].values
    
    X = df[model_spec.features].values.astype(np.float64)
    y = df[spec.label_col].values.astype(np.float64)
    qids = df[spec.qid_col].values
    
    # Generate folds
    folds = generate_folds(
        ym=ym,
        qids=qids,
        min_train_months=config.min_train_months,
        test_horizon_months=config.test_horizon_months,
        step_months=config.step_months,
        label_horizon_bars=spec.label_horizon_bars,
        embargo_bars=config.embargo_bars,
    )
    print(f"Generated {len(folds)} walk-forward folds.")
    
    # Accumulate OOS predictions and indices
    acc = {
        "idx": [],
        "ym": [],
        "q": [],
        "y": [],
        "time": []
    }
    for side in model_spec.sides:
        acc[side] = []
        
    adapter_name = model_spec.adapter
    assert adapter_name in ADAPTERS, f"Adapter '{adapter_name}' not registered in harness"
    adapter = ADAPTERS[adapter_name]
    
    fold_stats = []
    
    for fi, fold in enumerate(folds, 1):
        t0 = time.time()
        
        trm = fold.train_indices
        vam = fold.val_indices
        tem = fold.test_indices
        
        print(f"  Fold {fi}/{len(folds)}: test {fold.test_months[0]}..{fold.test_months[-1]} ({len(tem)} rows)")
        
        Xtr, Xva, Xte = X[trm].copy(), X[vam].copy(), X[tem].copy()
        ytr, yva = y[trm], y[vam]
        qtr, qva = qids[trm], qids[vam]
        
        # Per-fold NaN fill using train-only column means
        col_means = np.nanmean(Xtr, axis=0)
        col_means = np.nan_to_num(col_means)
        for arr in (Xtr, Xva, Xte):
            inds = np.where(~np.isfinite(arr))
            if len(inds[0]):
                arr[inds] = np.take(col_means, inds[1])
                
        # Train and predict per side
        fold_res = {
            "fold": fi,
            "test_months": ", ".join(fold.test_months),
            "best_iter_long": 0,
            "best_iter_short": 0,
            "long_rho": 0.0,
            "short_rho": 0.0
        }
        
        preds_fold = {}
        for side in model_spec.sides:
            model = adapter.fit(
                Xtr=Xtr, ytr=ytr, qtr=qtr,
                Xva=Xva, yva=yva, qva=qva,
                side=side, params=model_spec.params,
                num_boost_round=model_spec.num_boost_round,
                early_stopping_rounds=model_spec.early_stopping_rounds,
                seed=config.seed
            )
            
            # Record best iteration
            best_iter = 0
            if hasattr(model, "best_iteration"):
                best_iter = int(model.best_iteration)
            elif hasattr(model, "get_best_iteration"):
                best_iter = int(model.get_best_iteration())
                
            if side == "long":
                fold_res["best_iter_long"] = best_iter
            else:
                fold_res["best_iter_short"] = best_iter
                
            pred_side = adapter.predict(model, Xte)
            preds_fold[side] = pred_side
            
            # Compute fold Spearman
            from .metrics import compute_query_spearman
            corrs = compute_query_spearman(pred_side, y[tem], qids[tem], invert=(side == "short"))
            fold_rho = float(np.mean(corrs)) if corrs else 0.0
            
            if side == "long":
                fold_res["long_rho"] = fold_rho
            else:
                fold_res["short_rho"] = fold_rho
            
        # Accumulate
        acc["idx"].append(tem)
        acc["ym"].append(ym[tem])
        acc["q"].append(qids[tem])
        acc["y"].append(y[tem])
        acc["time"].append(times[tem])
        
        for side in model_spec.sides:
            acc[side].append(preds_fold[side])
            
        fold_stats.append(fold_res)
        print(f"    Completed fold in {time.time()-t0:.1f}s | L_rho={fold_res['long_rho']:+.4f} S_rho={fold_res['short_rho']:+.4f}")
        
    # Concatenate results
    res_dict = {
        "idx": np.concatenate(acc["idx"]),
        "ym": np.concatenate(acc["ym"]),
        "q": np.concatenate(acc["q"]),
        "y": np.concatenate(acc["y"]),
        "time": np.concatenate(acc["time"]),
        "preds": {},
        "fold_stats": fold_stats
    }
    
    for side in model_spec.sides:
        res_dict["preds"][side] = np.concatenate(acc[side])
        
    return res_dict

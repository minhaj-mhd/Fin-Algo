"""
Train and evaluate LGBMRanker on:
1. Hand-crafted features only (baseline)
2. Retrieval features only (ablation)
3. Hand-crafted + Retrieval features (hybrid)
Supports walk-forward folds and outputs Spearman rho, net PnL (6 bps cost), and Sharpe.
"""
import os, sys, json, time, argparse
import numpy as np
import lightgbm as lgb
from scipy.stats import spearmanr

sys.path.append(os.getcwd())
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from scripts.transformer.pretrain_contrastive_v20 import valid_decision_timestamps, load_panel

P = 'data/transformer_panel_v20'
L1, L2 = 30, 60
EMBARGO = 30
COST_BPS = 6.0
K_PNC = 3

def get_walk_forward_splits(valid_t, fold_idx, embargo=30):
    n = len(valid_t)
    if fold_idx == 0:
        tr_end = int(n * 0.70)
        val_end = int(n * 0.85)
    elif fold_idx == 1:
        tr_end = int(n * 0.75)
        val_end = int(n * 0.875)
    else:
        tr_end = int(n * 0.80)
        val_end = int(n * 0.90)
        
    train_t = valid_t[:tr_end]
    val_t = valid_t[tr_end + embargo:val_end]
    test_t = valid_t[val_end + embargo:]
    return train_t, val_t, test_t

def graded_relevance(r, cost=6e-4):
    B3 = 0.0010
    B2 = 0.0005
    B1 = 0.0000
    B0 = -0.0005
    net = r - cost
    rel = np.zeros_like(r, dtype=np.int32)
    rel[net >= B0] = 1
    rel[net >= B1] = 2
    rel[net >= B2] = 3
    rel[net >= B3] = 4
    return rel

def prep_ranker_data(X_feats, Y_ret, valid, t_indices):
    X_list, Y_list, group_list = [], [], []
    for t in t_indices:
        v = valid[t]
        cnt = int(v.sum())
        if cnt < 2:
            continue
        X_list.append(X_feats[t, v])
        Y_list.append(Y_ret[t, v])
        group_list.append(cnt)
    if len(X_list) == 0:
        return None, None, None
    X_out = np.concatenate(X_list, axis=0)
    Y_out = np.concatenate(Y_list, axis=0)
    Y_out = graded_relevance(Y_out, cost=COST_BPS / 1e4)
    return X_out, Y_out, np.array(group_list)

def evaluate_ranker(model, X_feats, Y_ret, valid, t_indices, feature_indices=None):
    """
    Evaluates ranker predictions:
    1. Spearman Rank-IC (rho) per timestamp
    2. Top-3 LONG net PnL (6 bps cost)
    3. Sharpe Ratio of portfolio returns
    """
    rhos = []
    portfolio_rets = []
    all_picks_raw_rets = []
    
    for t in t_indices:
        v = valid[t]
        if v.sum() < K_PNC + 1:
            continue
            
        x = X_feats[t, v]
        if feature_indices is not None:
            x = x[:, feature_indices]
            
        y = Y_ret[t, v]
        
        # Predict scores
        pred = model.predict(x)
        
        # Compute Spearman correlation
        if np.std(pred) > 0 and np.std(y) > 0:
            rho = spearmanr(pred, y).correlation
            if np.isfinite(rho):
                rhos.append(rho)
                
        # Top-K LONG picks
        order = np.argsort(-pred)
        picks = order[:K_PNC]
        lr = y[picks]
        
        all_picks_raw_rets.extend(lr.tolist())
        
        # Net return of the selected basket
        net_ret = lr.mean() - COST_BPS / 1e4
        portfolio_rets.append(net_ret)
        
    mean_rho = np.mean(rhos) if rhos else 0.0
    
    picks_raw = np.array(all_picks_raw_rets)
    if len(picks_raw) > 0:
        raw_bps = np.mean(picks_raw) * 1e4
        net_bps = (np.mean(picks_raw) - COST_BPS / 1e4) * 1e4
        raw_winrate = np.mean(picks_raw > 0.0) * 100.0
        net_winrate = np.mean(picks_raw > COST_BPS / 1e4) * 100.0
    else:
        raw_bps, net_bps, raw_winrate, net_winrate = 0.0, 0.0, 0.0, 0.0
        
    # Compute Sharpe Ratio (annualized, 18 decision steps per day)
    if len(portfolio_rets) > 5 and np.std(portfolio_rets) > 0:
        sharpe = (np.mean(portfolio_rets) / np.std(portfolio_rets)) * np.sqrt(252 * 18)
    else:
        sharpe = 0.0
        
    return {
        'rho': mean_rho,
        'raw_bps': raw_bps,
        'net_bps': net_bps,
        'raw_winrate': raw_winrate,
        'net_winrate': net_winrate,
        'sharpe': sharpe
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=0, help='Which fold to train (0, 1, or 2). Use -1 for all folds.')
    args = parser.parse_args()
    
    print("Loading panel and retrieval features...")
    d = load_panel()
    X1 = d['X_1h'] # (T, N, F_hand)
    Y_ret = d['Y_ret'] # (T, N)
    
    # Load retrieved features
    ret_stats_path = f'{P}/retrieved_stats_v20.npy'
    if not os.path.exists(ret_stats_path):
        print(f"Error: Retrieval features {ret_stats_path} not found. Please run build_filtered_faiss_index_v20.py first.")
        sys.exit(1)
        
    retrieved_stats = np.load(ret_stats_path) # (T, N, 12)
    
    # Concatenate features
    # shape: (T, N, F_hand + 12)
    X_both = np.concatenate([X1, retrieved_stats], axis=2)
    
    valid_t = valid_decision_timestamps(d)
    
    # Prepare valid mask
    T, N = Y_ret.shape
    valid_mask = np.zeros((T, N), dtype=bool)
    for t in valid_t:
        present = np.isfinite(X1[t, :, 0])
        valid_mask[t] = present & np.isfinite(Y_ret[t])
        
    # Feature selections
    F_hand = X1.shape[2]
    hand_feat_indices = list(range(F_hand))
    ret_feat_indices = list(range(F_hand, F_hand + 12))
    both_feat_indices = list(range(F_hand + 12))
    
    folds_to_run = [args.fold] if args.fold != -1 else [0, 1, 2]
    
    for fold in folds_to_run:
        print("\n" + "=" * 70)
        print(f"RUNNING WALK-FORWARD FOLD {fold}")
        print("=" * 70)
        
        train_t, val_t, test_t = get_walk_forward_splits(valid_t, fold)
        print(f"Train samples: {len(train_t)}, Val samples: {len(val_t)}, Test samples: {len(test_t)}")
        
        # 1. Model A: Hand-crafted features only (Baseline)
        print("\n--- Training Model A: Hand-crafted Features Only ---")
        X_tr, y_tr, g_tr = prep_ranker_data(X_both[:, :, hand_feat_indices], Y_ret, valid_mask, train_t)
        X_va, y_va, g_va = prep_ranker_data(X_both[:, :, hand_feat_indices], Y_ret, valid_mask, val_t)
        
        model_a = lgb.LGBMRanker(
            max_depth=6,
            learning_rate=0.05,
            n_estimators=1500,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        model_a.fit(
            X_tr, y_tr, group=g_tr,
            eval_set=[(X_va, y_va)], eval_group=[g_va],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        # 2. Model B: Retrieval features only (Ablation)
        print("\n--- Training Model B: Retrieval Features Only ---")
        X_tr_ret, y_tr_ret, g_tr_ret = prep_ranker_data(X_both[:, :, ret_feat_indices], Y_ret, valid_mask, train_t)
        X_va_ret, y_va_ret, g_va_ret = prep_ranker_data(X_both[:, :, ret_feat_indices], Y_ret, valid_mask, val_t)
        
        model_b = lgb.LGBMRanker(
            max_depth=6,
            learning_rate=0.05,
            n_estimators=1500,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        model_b.fit(
            X_tr_ret, y_tr_ret, group=g_tr_ret,
            eval_set=[(X_va_ret, y_va_ret)], eval_group=[g_va_ret],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        # 3. Model C: Hand-crafted + Retrieval features (Hybrid)
        print("\n--- Training Model C: Hand-crafted + Retrieval Features ---")
        X_tr_both, y_tr_both, g_tr_both = prep_ranker_data(X_both, Y_ret, valid_mask, train_t)
        X_va_both, y_va_both, g_va_both = prep_ranker_data(X_both, Y_ret, valid_mask, val_t)
        
        model_c = lgb.LGBMRanker(
            max_depth=6,
            learning_rate=0.05,
            n_estimators=1500,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        model_c.fit(
            X_tr_both, y_tr_both, group=g_tr_both,
            eval_set=[(X_va_both, y_va_both)], eval_group=[g_va_both],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        # Evaluate on Test Split
        print("\n" + "-" * 50)
        print("EVALUATION RESULTS ON TEST SPLIT")
        print("-" * 50)
        
        res_a = evaluate_ranker(model_a, X_both, Y_ret, valid_mask, test_t, hand_feat_indices)
        res_b = evaluate_ranker(model_b, X_both, Y_ret, valid_mask, test_t, ret_feat_indices)
        res_c = evaluate_ranker(model_c, X_both, Y_ret, valid_mask, test_t, both_feat_indices)
        
        print(f"Model A (Hand-crafted only):   Rho = {res_a['rho']:+.4f} | Raw Bps = {res_a['raw_bps']:+.2f} | Net Bps = {res_a['net_bps']:+.2f} | Raw WR = {res_a['raw_winrate']:.1f}% | Net WR = {res_a['net_winrate']:.1f}% | Sharpe = {res_a['sharpe']:.2f}")
        print(f"Model B (Retrieval only):      Rho = {res_b['rho']:+.4f} | Raw Bps = {res_b['raw_bps']:+.2f} | Net Bps = {res_b['net_bps']:+.2f} | Raw WR = {res_b['raw_winrate']:.1f}% | Net WR = {res_b['net_winrate']:.1f}% | Sharpe = {res_b['sharpe']:.2f}")
        print(f"Model C (Hand-crafted + Ret):  Rho = {res_c['rho']:+.4f} | Raw Bps = {res_c['raw_bps']:+.2f} | Net Bps = {res_c['net_bps']:+.2f} | Raw WR = {res_c['raw_winrate']:.1f}% | Net WR = {res_c['net_winrate']:.1f}% | Sharpe = {res_c['sharpe']:.2f}")
        print(f"Uplift (Model C vs Model A):    Delta Rho = {res_c['rho']-res_a['rho']:+.4f} | Delta Net Bps = {res_c['net_bps']-res_a['net_bps']:+.2f} | Delta Sharpe = {res_c['sharpe']-res_a['sharpe']:+.2f}")
        
    print("\n" + "=" * 70)
    print("ALL RUNS COMPLETE")
    print("=" * 70)

if __name__ == '__main__':
    main()

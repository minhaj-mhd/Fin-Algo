"""
Utility functions for Phase-1 Refined Retrieval-Based Hybrid pipeline.
Contains:
1. Quantile assignment helpers (lookahead-safe)
2. Contrastive positive/negative mask builders
3. InfoNCE and smoothness loss functions
4. Linear probe evaluation functions
5. Distance-vs-outcome diagnostic regression
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr, linregress
from sklearn.linear_model import Ridge

def fit_assign_quantiles(train_data, full_data, num_quantiles=4):
    """
    Fit quantile thresholds on train_data and assign bins to full_data.
    Returns:
        bins: array of shape full_data.shape with values in 0..(num_quantiles-1)
        thresholds: the quantile edges
    """
    valid_train = train_data[np.isfinite(train_data)]
    if len(valid_train) == 0:
        # Fallback if no valid data
        bins = np.zeros_like(full_data, dtype=np.int32)
        return bins, np.array([-np.inf, np.inf])
        
    q = np.nanpercentile(valid_train, np.linspace(0, 100, num_quantiles + 1))
    q[0] = -np.inf
    q[-1] = np.inf
    
    # Clip/replace NaNs in full_data with median or handle separately
    clean_full = np.nan_to_num(full_data, nan=np.nanmedian(valid_train))
    bins = np.digitize(clean_full, q) - 1
    # Clamp to 0..num_quantiles-1
    bins = np.clip(bins, 0, num_quantiles - 1).astype(np.int32)
    return bins, q

def build_contrastive_masks(ts, tickers, sectors, regimes, corr_matrix, ts_to_grid_idx, device):
    """
    Build PyTorch masks for contrastive positive and negative pairs in the batch.
    
    Args:
        ts: tensor (M,) - timestamps in nanoseconds
        tickers: tensor (M,) - ticker index (0..N-1)
        sectors: tensor (M,) - sector ID
        regimes: tensor (M,) - regime label
        corr_matrix: tensor (N_tickers, N_tickers) - stock correlation matrix
        ts_to_grid_idx: tensor (M,) - chronological index of the timestamp in ts_1h
        device: torch device
        
    Returns:
        pos_mask: (M, M) boolean tensor
        neg_mask: (M, M) boolean tensor
    """
    M = ts.shape[0]
    if M == 0:
        return torch.zeros((0, 0), dtype=torch.bool, device=device), torch.zeros((0, 0), dtype=torch.bool, device=device)
        
    # Expand tensors for pairwise comparisons
    ts_col = ts.unsqueeze(1)
    ts_row = ts.unsqueeze(0)
    
    tickers_col = tickers.unsqueeze(1)
    tickers_row = tickers.unsqueeze(0)
    
    sectors_col = sectors.unsqueeze(1)
    sectors_row = sectors.unsqueeze(0)
    
    regimes_col = regimes.unsqueeze(1)
    regimes_row = regimes.unsqueeze(0)
    
    grid_idx_col = ts_to_grid_idx.unsqueeze(1)
    grid_idx_row = ts_to_grid_idx.unsqueeze(0)
    
    # 1. Same stock at t+1: s[i] == s[j] and consecutive timestamps
    same_stock = (tickers_col == tickers_row)
    consecutive_t = (torch.abs(grid_idx_col - grid_idx_row) == 1)
    pos_same_stock_t1 = same_stock & consecutive_t
    
    # 2. Stocks in same sector with correlation >= 0.8
    same_sector = (sectors_col == sectors_row)
    pairwise_corr = corr_matrix[tickers_col, tickers_row]
    pos_corr = same_sector & (pairwise_corr >= 0.8)
    
    # 3. Same regime
    same_regime = (regimes_col == regimes_row)
    
    # Combine positives (excluding self-loops)
    pos_mask = pos_same_stock_t1 | pos_corr | same_regime
    eye_mask = torch.eye(M, dtype=torch.bool, device=device)
    pos_mask = pos_mask & (~eye_mask)
    
    # Negatives: different regimes AND timestamps >= 1 day apart AND unrelated sectors (corr <= 0.2)
    diff_regime = (regimes_col != regimes_row)
    one_day_ns = 24 * 3600 * 1_000_000_000
    timediff_day = (torch.abs(ts_col - ts_row) >= one_day_ns)
    neg_corr = (pairwise_corr <= 0.2)
    
    neg_mask = diff_regime & timediff_day & neg_corr
    
    return pos_mask, neg_mask

def masked_infonce_loss(embeddings, pos_mask, neg_mask, temp=0.07):
    """
    Computes masked InfoNCE contrastive loss.
    Each anchor can have multiple positives and multiple negatives in the batch.
    """
    M = embeddings.shape[0]
    if M < 2:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
    # Normalize to unit sphere
    z = nn.functional.normalize(embeddings, p=2, dim=1)
    sim = torch.matmul(z, z.t()) / temp # (M, M)
    
    # Subtract max for stability
    sim_max, _ = torch.max(sim, dim=1, keepdim=True)
    sim_stable = sim - sim_max.detach()
    exp_sim = torch.exp(sim_stable)
    
    # Compute denominator for each anchor: sum of exp_sim over all valid negatives
    # plus the positive similarity.
    neg_sums = (exp_sim * neg_mask.float()).sum(dim=1, keepdim=True) # (M, 1)
    
    # For each anchor, we sum log-probability over all its positives
    # loss_i = -1/|P_i| * sum_{p in P_i} log( exp(s_ip) / (sum_n exp(s_in) + exp(s_ip)) )
    # Let's vectorize:
    denom_p = neg_sums + exp_sim # (M, M) - cell (i, p) contains sum_n exp(s_in) + exp(s_ip)
    log_prob = sim_stable - torch.log(denom_p + 1e-8) # (M, M)
    
    # We only sum over positives
    pos_log_probs = log_prob * pos_mask.float()
    pos_counts = pos_mask.float().sum(dim=1)
    
    # Avoid division by zero
    valid_anchors = pos_counts > 0
    if not valid_anchors.any():
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
    loss_per_anchor = -pos_log_probs.sum(dim=1)[valid_anchors] / pos_counts[valid_anchors]
    return loss_per_anchor.mean()

def compute_smoothness_loss(embeddings, pos_same_stock_t1):
    """
    Smoothness loss L_smooth = mean( ||z_i - z_j||^2 ) for all consecutive same-stock pairs.
    """
    indices = torch.nonzero(pos_same_stock_t1, as_tuple=True)
    if len(indices[0]) == 0:
        return torch.tensor(0.0, device=embeddings.device)
    z_i = embeddings[indices[0]]
    z_j = embeddings[indices[1]]
    sq_dist = torch.sum((z_i - z_j) ** 2, dim=-1)
    return torch.mean(sq_dist)

def train_eval_linear_probes(X_1h, embeddings, Y_ret, valid, train_idx, val_idx, ridge_alpha=1.0):
    """
    Train Ridge regression models on train_idx and evaluate cross-sectional rank correlation on val_idx.
    Probe A: hand-crafted only
    Probe B: embedding only
    Probe C: hand-crafted + embedding
    """
    # 1. Flatten training data
    tr_hand_list, tr_emb_list, tr_y_list = [], [], []
    for t in train_idx:
        v = valid[t]
        if v.sum() == 0:
            continue
        tr_hand_list.append(X_1h[t, v])
        tr_emb_list.append(embeddings[t, v])
        tr_y_list.append(Y_ret[t, v])
        
    if len(tr_y_list) == 0:
        print("Warning: No valid training data for linear probes.")
        return 0.0, 0.0, 0.0
        
    X_tr_hand = np.concatenate(tr_hand_list, axis=0)
    X_tr_emb = np.concatenate(tr_emb_list, axis=0)
    Y_tr = np.concatenate(tr_y_list, axis=0)
    
    X_tr_both = np.concatenate([X_tr_hand, X_tr_emb], axis=1)
    
    # Train Ridge regressions
    model_a = Ridge(alpha=ridge_alpha).fit(X_tr_hand, Y_tr)
    model_b = Ridge(alpha=ridge_alpha).fit(X_tr_emb, Y_tr)
    model_c = Ridge(alpha=ridge_alpha).fit(X_tr_both, Y_tr)
    
    # 2. Evaluate cross-sectionally on validation timestamps
    rhos_a, rhos_b, rhos_c = [], [], []
    
    for t in val_idx:
        v = valid[t]
        if v.sum() < 2:
            continue
            
        x_hand = X_1h[t, v]
        x_emb = embeddings[t, v]
        y_true = Y_ret[t, v]
        
        # Predict
        pred_a = model_a.predict(x_hand)
        pred_b = model_b.predict(x_emb)
        pred_c = model_c.predict(np.concatenate([x_hand, x_emb], axis=1))
        
        # Compute Spearman correlations
        if np.std(pred_a) > 0 and np.std(y_true) > 0:
            rho_a = spearmanr(pred_a, y_true).correlation
            if np.isfinite(rho_a): rhos_a.append(rho_a)
            
        if np.std(pred_b) > 0 and np.std(y_true) > 0:
            rho_b = spearmanr(pred_b, y_true).correlation
            if np.isfinite(rho_b): rhos_b.append(rho_b)
            
        if np.std(pred_c) > 0 and np.std(y_true) > 0:
            rho_c = spearmanr(pred_c, y_true).correlation
            if np.isfinite(rho_c): rhos_c.append(rho_c)
            
    mean_rho_a = np.mean(rhos_a) if rhos_a else 0.0
    mean_rho_b = np.mean(rhos_b) if rhos_b else 0.0
    mean_rho_c = np.mean(rhos_c) if rhos_c else 0.0
    
    return mean_rho_a, mean_rho_b, mean_rho_c

def distance_vs_outcome_diagnostic(embeddings, Y_ret, valid, train_idx, val_idx, sample_size=1000):
    """
    Computes nearest-neighbor distance vs outcome return differences.
    Fits a linear regression and returns the slope and p-value.
    """
    # 1. Gather all training embeddings and future returns
    tr_emb_list, tr_y_list = [], []
    for t in train_idx:
        v = valid[t]
        if v.sum() == 0:
            continue
        tr_emb_list.append(embeddings[t, v])
        tr_y_list.append(Y_ret[t, v])
        
    if len(tr_emb_list) == 0:
        return 0.0, 1.0
        
    X_tr = np.concatenate(tr_emb_list, axis=0)
    Y_tr = np.concatenate(tr_y_list, axis=0)
    
    # 2. Gather validation embeddings and future returns
    val_emb_list, val_y_list = [], []
    for t in val_idx:
        v = valid[t]
        if v.sum() == 0:
            continue
        val_emb_list.append(embeddings[t, v])
        val_y_list.append(Y_ret[t, v])
        
    if len(val_emb_list) == 0:
        return 0.0, 1.0
        
    X_val = np.concatenate(val_emb_list, axis=0)
    Y_val = np.concatenate(val_y_list, axis=0)
    
    # Subsample validation to speed up distance computation if needed
    if len(X_val) > sample_size:
        idx = np.random.choice(len(X_val), sample_size, replace=False)
        X_val = X_val[idx]
        Y_val = Y_val[idx]
        
    # Normalize embeddings for cosine distance or use raw L2
    X_tr_norm = X_tr / np.linalg.norm(X_tr, axis=1, keepdims=True).clip(min=1e-8)
    X_val_norm = X_val / np.linalg.norm(X_val, axis=1, keepdims=True).clip(min=1e-8)
    
    distances = []
    outcome_diffs = []
    
    # Vectorized search chunk by chunk
    chunk_size = 250
    for i in range(0, len(X_val_norm), chunk_size):
        chunk_val = X_val_norm[i:i+chunk_size]
        chunk_y = Y_val[i:i+chunk_size]
        
        # Compute cosine similarity
        sims = np.dot(chunk_val, X_tr_norm.T) # (chunk_size, N_tr)
        best_idx = np.argmax(sims, axis=1)
        best_sim = sims[np.arange(len(chunk_val)), best_idx]
        
        # Euclidean distance of unit vectors is sqrt(2 - 2*sim)
        dist = np.sqrt(np.clip(2 - 2 * best_sim, 0, None))
        
        y_neighbour = Y_tr[best_idx]
        y_diff = np.abs(chunk_y - y_neighbour)
        
        distances.extend(dist)
        outcome_diffs.extend(y_diff)
        
    distances = np.array(distances)
    outcome_diffs = np.array(outcome_diffs)
    
    # Fit linear regression
    slope, intercept, r_value, p_value, std_err = linregress(distances, outcome_diffs)
    
    return slope, p_value

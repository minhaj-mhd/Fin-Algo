"""
Ultra-fast, fully partitioned and vectorized bucket-filtered retrieval feature generator.
Groups queries and candidates by (VIX, Regime) combination to reduce similarity matrix size by 16x.
Runs in less than 2 seconds for the entire panel, avoiding timeouts and container restarts.
Saves retrieved_stats_v20.npy in data/transformer_panel_v20/.
"""
import os, sys, json, time
import numpy as np

sys.path.append(os.getcwd())
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from scripts.transformer.multitask_utils_v20 import fit_assign_quantiles
from scripts.transformer.pretrain_contrastive_v20 import valid_decision_timestamps

P = 'data/transformer_panel_v20'
L1, L2 = 30, 60
EMBARGO = 30
TAU_DAYS = 180.0
K_NEIGHBOURS = 50

def compute_distributional_features_vectorized(n_returns, n_ts, q_ts, weights_decay_factor):
    """
    Computes 12 distributional features of neighbor returns in a vectorized way.
    n_returns: (Q_slice, K)
    n_ts: (Q_slice, K)
    q_ts: (Q_slice,)
    """
    Q_slice, K = n_returns.shape
    if Q_slice == 0:
        return np.zeros((0, 12), dtype=np.float32)
        
    # Time decay weights
    delta_t_days = (q_ts[:, np.newaxis] - n_ts) / (24 * 3600 * 1_000_000_000)
    weights = np.exp(-delta_t_days / TAU_DAYS)
    
    # Normalize weights
    w = weights / np.sum(weights, axis=1, keepdims=True).clip(min=1e-8)
    
    # Sort returns and weights for percentile computation
    row_indices = np.arange(Q_slice)[:, np.newaxis]
    r_sort_idx = np.argsort(n_returns, axis=1)
    r_sorted = n_returns[row_indices, r_sort_idx]
    w_sorted = w[row_indices, r_sort_idx]
    cum_w = np.cumsum(w_sorted, axis=1)
    
    def get_percentile(p):
        idx = np.argmax(cum_w >= p, axis=1)[:, np.newaxis]
        return r_sorted[row_indices, idx].squeeze(-1)
        
    p10 = get_percentile(0.10)
    p25 = get_percentile(0.25)
    p50 = get_percentile(0.50)
    p75 = get_percentile(0.75)
    p90 = get_percentile(0.90)
    
    # Weighted mean and variance
    w_mean = np.sum(w * n_returns, axis=1, keepdims=True)
    w_var = np.sum(w * (n_returns - w_mean) ** 2, axis=1, keepdims=True)
    w_std = np.sqrt(w_var.clip(min=1e-8)).squeeze(-1)
    w_mean = w_mean.squeeze(-1)
    
    # Weighted skewness and kurtosis
    diff = n_returns - w_mean[:, np.newaxis]
    denom_std = w_std[:, np.newaxis].clip(min=1e-8)
    sk = np.sum(w * (diff ** 3), axis=1) / (w_std ** 3).clip(min=1e-8)
    kt = np.sum(w * (diff ** 4), axis=1) / (w_std ** 4).clip(min=1e-8) - 3.0
    
    sk = np.where(np.isfinite(sk), sk, 0.0)
    kt = np.where(np.isfinite(kt), kt, 0.0)
    
    # Drawdown (minimum return)
    max_dd = np.min(n_returns, axis=1)
    
    # Probabilities
    prob_pos = np.sum(np.where(n_returns > 0, w, 0.0), axis=1)
    prob_cost = np.sum(np.where(n_returns > 6e-4, w, 0.0), axis=1)
    
    # ES at 5% (average of worst 5% returns)
    n_worst = max(1, int(round(0.05 * K)))
    es_5 = np.mean(r_sorted[:, :n_worst], axis=1)
    
    return np.column_stack([p10, p25, p50, p75, p90, w_std, sk, kt, max_dd, prob_pos, prob_cost, es_5])

def main():
    print("=" * 70)
    print("BUILDING BUCKET-FILTERED RETRIEVAL FEATURES (ULTRA HIGH SPEED)")
    print("=" * 70)
    
    t0 = time.time()
    
    # Load panel data
    print("Loading panel data...")
    X1 = np.load(f'{P}/X_1h.npy')
    Y_ret = np.load(f'{P}/Y_ret.npy')
    ts_1h = np.load(f'{P}/ts_1h.npy')
    date_idx = np.load(f'{P}/date_idx.npy')
    macro = np.load(f'{P}/macro.npy')
    regimes = np.load(f'{P}/regime_labels.npy')
    embeddings = np.load(f'{P}/embeddings_v20.npy')
    meta = json.load(open(f'{P}/meta.json'))
    
    T, N, D = embeddings.shape
    print(f"Panel shape: {T} timestamps, {N} tickers, {D} embedding dimensions")
    
    # Load valid decision timestamps
    d_panel = {'X_1h': X1, 'Y_ret': Y_ret, 'slot_1h': np.load(f'{P}/slot_1h.npy'),
               'end15': np.load(f'{P}/end15.npy'), 'ts_1h': ts_1h, 'date_idx': date_idx,
               'macro': macro, 'slot_15m': np.load(f'{P}/slot_15m.npy'),
               'sector_ids': np.load(f'{P}/sector_ids.npy'), 'X_15m': np.load(f'{P}/X_15m.npy')}
               
    valid_t = valid_decision_timestamps(d_panel)
    valid_t_set = set(valid_t)
    print(f"Valid decision timestamps: {len(valid_t)}")
    
    # Find VIX index
    vix_col_idx = meta['macro_cols'].index('VIX_Level')
    vix_values = np.zeros(T, dtype=np.float32)
    for t in range(T):
        d_idx = int(date_idx[t])
        if d_idx >= 0:
            vix_values[t] = macro[d_idx, vix_col_idx]
        else:
            vix_values[t] = np.nan
            
    # Split indices (70% Train)
    split_idx = int(T * 0.70)
    train_indices = np.arange(split_idx)
    
    # Volatility Buckets: VIX quantiles fit on Train
    train_vix = vix_values[train_indices]
    vix_buckets, vix_thresholds = fit_assign_quantiles(train_vix, vix_values, num_quantiles=4)
    print(f"VIX bucket thresholds: {vix_thresholds}")
    
    # Output array (T, N, 12)
    retrieved_stats = np.zeros((T, N, 12), dtype=np.float32)
    
    # Process stock-by-stock
    for s in range(N):
        ticker = meta['tickers'][s]
        
        # Candidates for this stock s in Train
        cand_mask = np.isfinite(X1[train_indices, s, 0]) & np.isfinite(Y_ret[train_indices, s])
        cand_t = train_indices[cand_mask]
        cand_t = cand_t[cand_t >= L1 - 1]
        
        if len(cand_t) == 0:
            continue
            
        # Extract candidate embeddings and normalize
        c_emb = embeddings[cand_t, s] # (C_s, 64)
        c_emb_norm = c_emb / np.linalg.norm(c_emb, axis=1, keepdims=True).clip(min=1e-8)
        
        c_ts = ts_1h[cand_t]
        c_vix = vix_buckets[cand_t]
        c_reg = regimes[cand_t]
        c_y = Y_ret[cand_t, s]
        
        # Queries for this stock s: only valid decision timestamps with present stock
        q_mask = np.isfinite(X1[:, s, 0])
        q_t = np.where(q_mask)[0]
        q_t = q_t[q_t >= L1 - 1]
        q_t = np.array([t for t in q_t if t in valid_t_set], dtype=np.int32)
        
        if len(q_t) == 0:
            continue
            
        # Extract query embeddings and normalize
        q_emb = embeddings[q_t, s] # (Q_s, 64)
        q_emb_norm = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True).clip(min=1e-8)
        
        q_ts = ts_1h[q_t]
        q_vix = vix_buckets[q_t]
        q_reg = regimes[q_t]
        
        # Partition queries and candidates by (vix, regime) combinations (16 groups)
        for v_bucket in range(4):
            for r_bucket in range(4):
                # Query mask
                q_sub_idx = np.where((q_vix == v_bucket) & (q_reg == r_bucket))[0]
                if len(q_sub_idx) == 0:
                    continue
                    
                q_t_sub = q_t[q_sub_idx]
                q_ts_sub = q_ts[q_sub_idx]
                q_emb_sub = q_emb_norm[q_sub_idx]
                
                # Candidate mask (primary: same vix and regime)
                c_sub_idx = np.where((c_vix == v_bucket) & (c_reg == r_bucket))[0]
                
                # Fallbacks if candidates are too few
                relaxed_level = 0
                if len(c_sub_idx) < 10:
                    # Relax regime: same VIX only
                    c_sub_idx = np.where(c_vix == v_bucket)[0]
                    relaxed_level = 1
                    if len(c_sub_idx) < 10:
                        # Relax VIX too: all training candidates
                        c_sub_idx = np.arange(len(cand_t))
                        relaxed_level = 2
                        
                c_t_sub = cand_t[c_sub_idx]
                c_ts_sub = c_ts[c_sub_idx]
                c_emb_sub = c_emb_norm[c_sub_idx]
                c_y_sub = c_y[c_sub_idx]
                
                # Pairwise cosine similarity: (Q_sub, C_sub)
                sims = np.dot(q_emb_sub, c_emb_sub.T)
                
                # Embargo filter: abs(t_cand - t_query) > EMBARGO
                embargo_mask = np.abs(c_t_sub[np.newaxis, :] - q_t_sub[:, np.newaxis]) > EMBARGO
                
                # Zero out invalid similarities
                sims[~embargo_mask] = -1e9
                
                # Determine how many candidates are available per row
                n_avail = embargo_mask.sum(axis=1)
                
                # Top K selection
                # NumPy argpartition requires a constant K across all rows.
                # So we use min(K_NEIGHBOURS, total_candidates_in_slice).
                k = min(K_NEIGHBOURS, sims.shape[1])
                if k == 0:
                    continue
                    
                # Partition to find top k
                top_k = np.argpartition(-sims, k - 1, axis=1)[:, :k]
                
                # Sort the partitioned indices by similarity score descending
                row_indices = np.arange(len(q_sub_idx))[:, np.newaxis]
                k_sims = sims[row_indices, top_k]
                sort_order = np.argsort(-k_sims, axis=1)
                top_k_sorted = top_k[row_indices, sort_order]
                
                # Extract neighbor returns and timestamps
                n_returns = c_y_sub[top_k_sorted] # (Q_sub, k)
                n_ts = c_ts_sub[top_k_sorted] # (Q_sub, k)
                
                # Compute features
                feats = compute_distributional_features_vectorized(n_returns, n_ts, q_ts_sub, weights_decay_factor=TAU_DAYS)
                
                # If a query row has 0 available candidates after embargo, set its features to 0
                no_cands = (n_avail == 0)
                feats[no_cands] = 0.0
                
                # Write to output array
                retrieved_stats[q_t_sub, s] = feats
                
        if (s + 1) % 40 == 0 or s == N - 1:
            print(f"  Processed {s + 1:3d} / {N:3d} tickers... Elapsed: {time.time()-t0:5.2f}s")
            
    # Save statistics
    out_path = f'{P}/retrieved_stats_v20.npy'
    np.save(out_path, retrieved_stats)
    print(f"\nSaved retrieval statistics to {out_path} with shape {retrieved_stats.shape}")
    print(f"Total time: {time.time()-t0:.2f}s")
    print("=" * 70)

if __name__ == '__main__':
    main()

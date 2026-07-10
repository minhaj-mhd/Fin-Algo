"""
Offline script to generate market regime labels using clustering on daily/hourly descriptors.
Compares K-means (K=4, 6, 8) and GMM (K=6) using Silhouette score and ANOVA p-value on subsequent returns.
Saves the best labels as regime_labels.npy in data/transformer_panel_v20/.
"""
import os, sys, json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy.stats import f_oneway

def main():
    print("=" * 70)
    print("GENERATING REGIME LABELS VIA UNSUPERVISED CLUSTERING")
    print("=" * 70)
    
    panel_dir = 'data/transformer_panel_v20'
    X1 = np.load(f'{panel_dir}/X_1h.npy')
    Y_ret = np.load(f'{panel_dir}/Y_ret.npy')
    sector_ids = np.load(f'{panel_dir}/sector_ids.npy')
    meta = json.load(open(f'{panel_dir}/meta.json'))
    
    features = meta['features']
    ret_idx = features.index('Return')
    ret_lag1_idx = features.index('Return_lag1')
    intraday_ret_idx = features.index('Intraday_Return')
    mkt_vol_idx = features.index('Market_Mean_Volatility')
    
    T = X1.shape[0]
    N = X1.shape[1]
    
    descriptors = np.zeros((T, 7), dtype=np.float32)
    
    print("Computing descriptors for all timestamps...")
    # Pre-calculate rolling correlations to speed up
    # We use a rolling window of 10 hours
    for t in range(T):
        if t < 29: # Need history for correlation or features
            continue
            
        r_t = X1[t, :, ret_idx]
        r_lag = X1[t, :, ret_lag1_idx]
        
        # 1. Realized Volatility
        realized_vol = np.nanmean(X1[t, :, mkt_vol_idx])
        if not np.isfinite(realized_vol):
            realized_vol = 0.0
            
        # 2. Median CS Correlation (over last 10 hours)
        w_ret = X1[t-10+1:t+1, :, ret_idx]
        w_ret_clean = np.nan_to_num(w_ret, nan=0.0)
        corr = np.corrcoef(w_ret_clean.T)
        corr = np.nan_to_num(corr, nan=0.0)
        corr_flat = corr[np.triu_indices_from(corr, k=1)]
        median_corr = np.median(corr_flat)
        
        # 3. Cross-Sectional Dispersion
        dispersion = np.nanstd(r_t)
        if not np.isfinite(dispersion):
            dispersion = 0.0
            
        # 4. Breadth
        breadth = np.nanmean(r_t > 0)
        if not np.isfinite(breadth):
            breadth = 0.5
            
        # 5. Sector-Entropy (Sector dispersion)
        sec_means = []
        for s_id in range(16):
            sec_means.append(np.nanmean(r_t[sector_ids == s_id]))
        sec_means = np.array(sec_means)
        sec_means = np.nan_to_num(sec_means, nan=0.0)
        sector_std = np.std(sec_means)
        
        # 6. Trend-Persistence
        valid_mask = np.isfinite(r_t) & np.isfinite(r_lag)
        if valid_mask.sum() > 5:
            trend_persistence = np.corrcoef(r_t[valid_mask], r_lag[valid_mask])[0, 1]
            if not np.isfinite(trend_persistence):
                trend_persistence = 0.0
        else:
            trend_persistence = 0.0
            
        # 7. Overnight-Gap Delta
        gap = X1[t, :, intraday_ret_idx] - r_t
        overnight_gap = np.nanmean(gap)
        if not np.isfinite(overnight_gap):
            overnight_gap = 0.0
            
        descriptors[t] = [realized_vol, median_corr, dispersion, breadth, sector_std, trend_persistence, overnight_gap]
        
    # Forward-fill descriptors for the initial warm-up period
    for t in range(29):
        descriptors[t] = descriptors[29]
        
    # Clean descriptors (impute NaNs/infs just in case)
    descriptors = np.nan_to_num(descriptors, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Scale descriptors
    scaler = StandardScaler()
    scaled_descriptors = scaler.fit_transform(descriptors)
    
    # Run clustering options
    # Standard split: we fit the clustering model on the first 70% of data (train split)
    # to maintain lookahead-safety.
    split_idx = int(T * 0.70)
    train_descriptors = scaled_descriptors[:split_idx]
    
    clusterings = {
        'KMeans_K4': KMeans(n_clusters=4, random_state=42, n_init=10),
        'KMeans_K6': KMeans(n_clusters=6, random_state=42, n_init=10),
        'KMeans_K8': KMeans(n_clusters=8, random_state=42, n_init=10),
        'GMM_K6': GaussianMixture(n_components=6, random_state=42, n_init=1)
    }
    
    best_name = None
    best_labels = None
    best_score = -2.0
    
    # Evaluate each
    for name, model in clusterings.items():
        print(f"\nEvaluating {name}...")
        model.fit(train_descriptors)
        
        if 'KMeans' in name:
            train_labels = model.labels_
            full_labels = model.predict(scaled_descriptors)
        else:
            train_labels = model.predict(train_descriptors)
            full_labels = model.predict(scaled_descriptors)
            
        # Silhouette Score on train set (subsample to speed up if too large)
        sub_indices = np.random.choice(len(train_descriptors), min(3000, len(train_descriptors)), replace=False)
        sil = silhouette_score(train_descriptors[sub_indices], train_labels[sub_indices])
        
        # ANOVA test: check if different regimes have statistically distinct next-hour returns
        # Group Y_ret[t, s] by full_labels[t]
        groups = []
        for c in range(model.n_clusters if 'KMeans' in name else model.n_components):
            # Find all timestamps in this regime
            ts_in_regime = np.where(full_labels == c)[0]
            # Get all valid returns in these timestamps
            valid_returns = Y_ret[ts_in_regime]
            valid_returns = valid_returns[np.isfinite(valid_returns)]
            if len(valid_returns) > 0:
                groups.append(valid_returns)
                
        if len(groups) >= 2:
            f_stat, p_val = f_oneway(*groups)
        else:
            f_stat, p_val = 0.0, 1.0
            
        print(f"  Silhouette Score: {sil:.4f}")
        print(f"  ANOVA F-statistic: {f_stat:.4f}, p-value: {p_val:.4e}")
        
        # Decide if this is the best one
        # Goal: p_val < 0.01 and maximize Silhouette
        if p_val < 0.01:
            if sil > best_score:
                best_score = sil
                best_name = name
                best_labels = full_labels
                
    # Fallback to KMeans_K6 if nothing has p_val < 0.01
    if best_labels is None:
        print("\n[Warning] No clustering met ANOVA p < 0.01. Falling back to KMeans_K6.")
        best_name = 'KMeans_K6'
        model = clusterings['KMeans_K6']
        model.fit(train_descriptors)
        best_labels = model.predict(scaled_descriptors)
        best_score = silhouette_score(train_descriptors, model.labels_)
        
    print(f"\nSelected Best Clustering: {best_name} (Silhouette: {best_score:.4f})")
    
    # Print label distribution
    unique, counts = np.unique(best_labels, return_counts=True)
    dist = dict(zip(unique, counts))
    print(f"Regime label distribution: {dist}")
    
    # Save labels
    out_path = f'{panel_dir}/regime_labels.npy'
    np.save(out_path, best_labels)
    print(f"Saved regime labels to {out_path}")
    print("=" * 70)

if __name__ == '__main__':
    main()

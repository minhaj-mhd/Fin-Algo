import sqlite3
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Setup paths and configurations
DB_PATH = 'data/vanguard_trades.db'
LEVERAGE = 5
SLIPPAGE = 0.06  # 0.06% per trade

def load_and_preprocess_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM trades ORDER BY timestamp ASC", conn)
    conn.close()
    
    df['final_profit_pct'] = pd.to_numeric(df['final_profit_pct'], errors='coerce')
    # Filter out trades with null final_profit_pct
    df = df.dropna(subset=['final_profit_pct']).reset_index(drop=True)
    
    features = ['tech_score', 'nlp_sentiment', 'tv_sentiment', 'one_hour_prob']
    tv_map = {'STRONG_SELL': -2, 'SELL': -1, 'NEUTRAL': 0, 'BUY': 1, 'STRONG_BUY': 2}
    
    if 'tv_sentiment' in df.columns:
        df['tv_sentiment'] = df['tv_sentiment'].replace(tv_map)
        
    for f in features:
        # Clean numeric percentages if they are strings
        df[f] = pd.to_numeric(df[f].astype(str).str.replace('%', '').str.strip(), errors='coerce').fillna(0.0)
        
    return df, features

def run_validation():
    print("=" * 60)
    print("STARTING OUT-OF-SAMPLE VALIDATION SIMULATION")
    print("=" * 60)
    
    df, features = load_and_preprocess_data()
    total_len = len(df)
    
    # 1. Chronological Split
    train_size = 1000
    df_train = df.iloc[:train_size].copy()
    df_val = df.iloc[train_size:].copy()
    val_size = len(df_val)
    
    print(f"Total clean trades: {total_len}")
    print(f"Training set (chronological first): {train_size} trades ({df_train['timestamp'].min()} to {df_train['timestamp'].max()})")
    print(f"Validation set (chronological last): {val_size} trades ({df_val['timestamp'].min()} to {df_val['timestamp'].max()})")
    
    # 2. Extract Training Winners for Model Fitting
    df_train_winners = df_train[df_train['final_profit_pct'] > 0.5].copy()
    print(f"High-return winners in training set (>0.5% PnL): {len(df_train_winners)}")
    
    # 3. Fit scaler and primary KMeans on training winners
    scaler = StandardScaler()
    X_train_winners = df_train_winners[features].values
    X_train_winners_scaled = scaler.fit_transform(X_train_winners)
    
    kmeans_primary = KMeans(n_clusters=4, init='k-means++', n_init=20, random_state=42)
    df_train_winners['primary_cluster'] = kmeans_primary.fit_predict(X_train_winners_scaled)
    
    # Scale entire train and validation sets
    X_train_all_scaled = scaler.transform(df_train[features].values)
    X_val_all_scaled = scaler.transform(df_val[features].values)
    
    # 4. Train sub-clustering models for each primary cluster on the training set
    sub_kmeans_models = {}
    print("\n--- Training Sub-Clustering Models on Training Winners ---")
    for c in range(4):
        c_winners = df_train_winners[df_train_winners['primary_cluster'] == c].copy()
        if len(c_winners) < 3:
            print(f"Primary Cluster {c}: Too few winners ({len(c_winners)}) to sub-cluster.")
            continue
            
        sub_k = 5 if c == 3 else 3
        sub_kmeans = KMeans(n_clusters=sub_k, init='k-means++', n_init=20, random_state=42)
        X_c_winners_scaled = scaler.transform(c_winners[features].values)
        c_winners['sub_cluster'] = sub_kmeans.fit_predict(X_c_winners_scaled)
        
        sub_kmeans_models[c] = {
            'model': sub_kmeans,
            'k': sub_k,
            'winners_df': c_winners
        }
        print(f"Primary Cluster {c} (k={sub_k}): clustered {len(c_winners)} winners.")

    # Let's perform two evaluations:
    # 1. FIXED RESEARCHED SWEET SPOTS (rebuilt on training set)
    # 2. DYNAMICALLY DISCOVERED SWEET SPOTS (found via training quality gates)

    # ----------------------------------------------------
    # PATH 1: Rebuilding fixed researched sweet spots on training set
    # ----------------------------------------------------
    print("\n" + "=" * 50)
    print("PATH 1: EVALUATING FIXED RESEARCHED SWEET SPOTS")
    print("=" * 50)
    
    # Let's define the fixed sweet spots indices we researched
    # format: (primary_cluster, sub_cluster, threshold, name)
    fixed_configs = [
        (0, 1, 0.75, "Cluster 0 Sub 1 (K=3, dist < 0.75)"),
        (1, 1, 0.50, "Cluster 1 Sub 1 (K=3, dist < 0.50)"),
        (2, 0, 0.50, "Cluster 2 Sub 0 (K=3, dist < 0.50)"),
        (2, 1, 1.00, "Cluster 2 Sub 1 (K=3, dist < 1.00)"),
        (3, 0, 0.75, "Cluster 3 Sub 0 (K=5, dist < 0.75)")
    ]
    
    fixed_sweet_spots = []
    for pc, sc, threshold, name in fixed_configs:
        if pc in sub_kmeans_models:
            sub_model_info = sub_kmeans_models[pc]
            sub_kmeans = sub_model_info['model']
            # Rebuilt centroid on training winners
            centroid = sub_kmeans.cluster_centers_[sc]
            fixed_sweet_spots.append({
                'name': name,
                'primary_cluster': pc,
                'sub_cluster': sc,
                'centroid': centroid,
                'threshold': threshold
            })
            
    # Evaluate Fixed on In-Sample (Train)
    train_fixed_matches = get_matches(df_train, X_train_all_scaled, fixed_sweet_spots)
    print_metrics_summary("In-Sample (Train) - Fixed Sweet Spots", train_fixed_matches, df_train)
    
    # Evaluate Fixed on Out-of-Sample (Validation)
    val_fixed_matches = get_matches(df_val, X_val_all_scaled, fixed_sweet_spots)
    print_metrics_summary("Out-of-Sample (Val) - Fixed Sweet Spots", val_fixed_matches, df_val)

    # ----------------------------------------------------
    # PATH 2: Dynamically discovering sweet spots on training set
    # ----------------------------------------------------
    print("\n" + "=" * 50)
    print("PATH 2: DYNAMICALLY DISCOVERING SWEET SPOTS ON TRAINING SET")
    print("=" * 50)
    
    discovered_sweet_spots = []
    
    for pc, sub_model_info in sub_kmeans_models.items():
        sub_kmeans = sub_model_info['model']
        sub_k = sub_model_info['k']
        
        for sc in range(sub_k):
            centroid = sub_kmeans.cluster_centers_[sc]
            # Calculate distances for all training set trades to this sub-centroid
            distances = np.linalg.norm(X_train_all_scaled - centroid, axis=1)
            
            best_threshold = None
            best_score = -999.0
            best_metrics = None
            
            # Sweep thresholds
            for t in [0.50, 0.75, 1.00, 1.50]:
                matched_indices = np.where(distances < t)[0]
                matched_trades = df_train.iloc[matched_indices]
                cnt = len(matched_trades)
                
                if cnt >= 5:
                    wr = (matched_trades['final_profit_pct'] > 0).mean() * 100
                    pnl = matched_trades['final_profit_pct'].mean()
                    
                    # Quality gates: WR >= 60% and Avg PnL >= 0.10%
                    if wr >= 60.0 and pnl >= 0.10:
                        # Choose the threshold that maximizes total yield: cnt * pnl
                        score = cnt * pnl
                        if score > best_score:
                            best_score = score
                            best_threshold = t
                            best_metrics = (cnt, wr, pnl)
                            
            if best_threshold is not None:
                name = f"Cluster {pc} Sub {sc} (K={sub_k}, dist < {best_threshold:.2f})"
                discovered_sweet_spots.append({
                    'name': name,
                    'primary_cluster': pc,
                    'sub_cluster': sc,
                    'centroid': centroid,
                    'threshold': best_threshold,
                    'train_cnt': best_metrics[0],
                    'train_wr': best_metrics[1],
                    'train_pnl': best_metrics[2]
                })
                print(f"Discovered Sweet Spot: {name} | Train Caught: {best_metrics[0]:3d} | WR: {best_metrics[1]:5.1f}% | Avg PnL: {best_metrics[2]:.4f}%")

    print(f"\nTotal Discovered Sweet Spots: {len(discovered_sweet_spots)}")
    
    # Evaluate Discovered on In-Sample (Train)
    train_disc_matches = get_matches(df_train, X_train_all_scaled, discovered_sweet_spots)
    print_metrics_summary("In-Sample (Train) - Discovered Sweet Spots", train_disc_matches, df_train)
    
    # Evaluate Discovered on Out-of-Sample (Validation)
    val_disc_matches = get_matches(df_val, X_val_all_scaled, discovered_sweet_spots)
    print_metrics_summary("Out-of-Sample (Val) - Discovered Sweet Spots", val_disc_matches, df_val)

    # 5. Generate beautiful markdown comparison report
    generate_report(df_train, df_val, fixed_sweet_spots, discovered_sweet_spots, train_fixed_matches, val_fixed_matches, train_disc_matches, val_disc_matches)

def get_matches(df_dataset, X_dataset_scaled, sweet_spots):
    # Match indices for each sweet spot
    matched_by_spot = {}
    all_matched_indices = set()
    
    for spot in sweet_spots:
        centroid = spot['centroid']
        threshold = spot['threshold']
        name = spot['name']
        
        distances = np.linalg.norm(X_dataset_scaled - centroid, axis=1)
        matched_indices = np.where(distances < threshold)[0]
        
        matched_by_spot[name] = matched_indices
        for idx in matched_indices:
            all_matched_indices.add(idx)
            
    # Return dictionary of individual matches and list of deduplicated match indices
    return {
        'individual': matched_by_spot,
        'dedup_indices': sorted(list(all_matched_indices))
    }

def print_metrics_summary(title, matches, df_dataset):
    dedup_idx = matches['dedup_indices']
    df_matched = df_dataset.iloc[dedup_idx]
    
    total = len(df_matched)
    if total == 0:
        print(f"\n[{title}]")
        print("No trades matched!")
        return
        
    wr = (df_matched['final_profit_pct'] > 0).mean() * 100
    avg_pnl = df_matched['final_profit_pct'].mean()
    
    # Calculate returns with 5x leverage and 0.06% slippage
    # Net PnL per trade = 5 * PnL - 0.06
    net_pnl_trades = 5 * df_matched['final_profit_pct'] - SLIPPAGE
    total_leverage_return = net_pnl_trades.sum()
    
    # Baseline (all trades in this dataset)
    base_wr = (df_dataset['final_profit_pct'] > 0).mean() * 100
    base_pnl = df_dataset['final_profit_pct'].mean()
    
    print(f"\n[{title}]")
    print(f"Matched Trades: {total} / {len(df_dataset)} ({total/len(df_dataset)*100:.1f}%)")
    print(f"Matched Win Rate: {wr:.2f}% (Baseline: {base_wr:.2f}%)")
    print(f"Matched Avg PnL:  {avg_pnl:.4f}% (Baseline: {base_pnl:.4f}%)")
    print(f"Total Leveraged Net Return (5x leverage, 0.06% slippage): {total_leverage_return:.2f}%")

def generate_report(df_train, df_val, fixed_sweet_spots, discovered_sweet_spots, train_fixed_m, val_fixed_m, train_disc_m, val_disc_m):
    # Helper to calculate stats
    def calc_stats(df_matched, total_size):
        if len(df_matched) == 0:
            return 0, 0.0, 0.0, 0.0, 0.0
        wr = (df_matched['final_profit_pct'] > 0).mean() * 100
        avg_pnl = df_matched['final_profit_pct'].mean()
        net_pnls = 5 * df_matched['final_profit_pct'] - SLIPPAGE
        cum_return = net_pnls.sum()
        pct_caught = len(df_matched) / total_size * 100
        return len(df_matched), pct_caught, wr, avg_pnl, cum_return

    tf_cnt, tf_pct, tf_wr, tf_pnl, tf_ret = calc_stats(df_train.iloc[train_fixed_m['dedup_indices']], len(df_train))
    vf_cnt, vf_pct, vf_wr, vf_pnl, vf_ret = calc_stats(df_val.iloc[val_fixed_m['dedup_indices']], len(df_val))
    
    td_cnt, td_pct, td_wr, td_pnl, td_ret = calc_stats(df_train.iloc[train_disc_m['dedup_indices']], len(df_train))
    vd_cnt, vd_pct, vd_wr, vd_pnl, vd_ret = calc_stats(df_val.iloc[val_disc_m['dedup_indices']], len(df_val))
    
    base_tr_wr = (df_train['final_profit_pct'] > 0).mean() * 100
    base_tr_pnl = df_train['final_profit_pct'].mean()
    base_val_wr = (df_val['final_profit_pct'] > 0).mean() * 100
    base_val_pnl = df_val['final_profit_pct'].mean()

    report = f"""# Out-of-Sample Validation Report: Veto Override Clustering

This report summarizes the performance of the veto override clustering strategy evaluated under a strict chronological train/test split. 
- **Training Set (In-Sample)**: First 1,000 trades (from {df_train['timestamp'].min()} to {df_train['timestamp'].max()})
- **Validation Set (Out-of-Sample)**: Last {len(df_val)} trades (from {df_val['timestamp'].min()} to {df_val['timestamp'].max()})
- **Leverage**: 5x | **Slippage**: 0.06% per trade

---

## 1. High-Level Comparison (In-Sample vs. Out-of-Sample)

| Strategy Configuration | Dataset | Matched Trades | Catch Rate (%) | Win Rate (%) | Avg PnL (%) | Cumulative Net Return (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (All Trades)** | Train (In-Sample) | 1,000 | 100.0% | {base_tr_wr:.2f}% | {base_tr_pnl:.4f}% | N/A |
| | Val (Out-of-Sample) | {len(df_val)} | 100.0% | {base_val_wr:.2f}% | {base_val_pnl:.4f}% | N/A |
| **Path 1: Fixed Researched Sweet Spots** | Train (In-Sample) | {tf_cnt} | {tf_pct:.1f}% | {tf_wr:.2f}% | {tf_pnl:.4f}% | {tf_ret:.2f}% |
| | Val (Out-of-Sample) | {vf_cnt} | {vf_pct:.1f}% | {vf_wr:.2f}% | {vf_pnl:.4f}% | {vf_ret:.2f}% |
| **Path 2: Dynamically Discovered Sweet Spots** | Train (In-Sample) | {td_cnt} | {td_pct:.1f}% | {td_wr:.2f}% | {td_pnl:.4f}% | {td_ret:.2f}% |
| | Val (Out-of-Sample) | {vd_cnt} | {vd_pct:.1f}% | {vd_wr:.2f}% | {vd_pnl:.4f}% | {vd_ret:.2f}% |

---

## 2. Individual Sweet Spot Breakdown (Out-of-Sample Performance)

### Path 1: Fixed Researched Sweet Spots

| Sweet Spot Name | Threshold | Train Caught | Train WR | Train PnL | Val Caught | Val WR | Val PnL | Val Net Return | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
"""
    
    # Calculate stats for individual fixed sweet spots
    for spot in fixed_sweet_spots:
        name = spot['name']
        pc = spot['primary_cluster']
        sc = spot['sub_cluster']
        t = spot['threshold']
        
        train_idx = train_fixed_m['individual'][name]
        val_idx = val_fixed_m['individual'][name]
        
        tr_cnt = len(train_idx)
        tr_wr = (df_train.iloc[train_idx]['final_profit_pct'] > 0).mean() * 100 if tr_cnt > 0 else 0.0
        tr_pnl = df_train.iloc[train_idx]['final_profit_pct'].mean() if tr_cnt > 0 else 0.0
        
        v_cnt = len(val_idx)
        v_wr = (df_val.iloc[val_idx]['final_profit_pct'] > 0).mean() * 100 if v_cnt > 0 else 0.0
        v_pnl = df_val.iloc[val_idx]['final_profit_pct'].mean() if v_cnt > 0 else 0.0
        v_ret = (5 * df_val.iloc[val_idx]['final_profit_pct'] - SLIPPAGE).sum() if v_cnt > 0 else 0.0
        
        # Status indicator
        status = "🟢 Robust" if v_wr >= 55.0 and v_pnl > 0 else "🔴 Overfit" if v_cnt > 0 else "⚪ No Match"
        
        report += f"| {name} | {t:.2f} | {tr_cnt} | {tr_wr:.1f}% | {tr_pnl:.4f}% | {v_cnt} | {v_wr:.1f}% | {v_pnl:.4f}% | {v_ret:+.2f}% | {status} |\n"

    report += """
### Path 2: Dynamically Discovered Sweet Spots

| Discovered Sweet Spot | Threshold | Train Caught | Train WR | Train PnL | Val Caught | Val WR | Val PnL | Val Net Return | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
"""

    # Calculate stats for individual discovered sweet spots
    for spot in discovered_sweet_spots:
        name = spot['name']
        t = spot['threshold']
        
        train_idx = train_disc_m['individual'][name]
        val_idx = val_disc_m['individual'][name]
        
        tr_cnt = len(train_idx)
        tr_wr = (df_train.iloc[train_idx]['final_profit_pct'] > 0).mean() * 100 if tr_cnt > 0 else 0.0
        tr_pnl = df_train.iloc[train_idx]['final_profit_pct'].mean() if tr_cnt > 0 else 0.0
        
        v_cnt = len(val_idx)
        v_wr = (df_val.iloc[val_idx]['final_profit_pct'] > 0).mean() * 100 if v_cnt > 0 else 0.0
        v_pnl = df_val.iloc[val_idx]['final_profit_pct'].mean() if v_cnt > 0 else 0.0
        v_ret = (5 * df_val.iloc[val_idx]['final_profit_pct'] - SLIPPAGE).sum() if v_cnt > 0 else 0.0
        
        status = "🟢 Robust" if v_wr >= 55.0 and v_pnl > 0 else "🔴 Overfit" if v_cnt > 0 else "⚪ No Match"
        
        report += f"| {name} | {t:.2f} | {tr_cnt} | {tr_wr:.1f}% | {tr_pnl:.4f}% | {v_cnt} | {v_wr:.1f}% | {v_pnl:.4f}% | {v_ret:+.2f}% | {status} |\n"

    # Conclusion block
    report += "\n## 3. Conclusions and Next Steps\n\n"
    
    # Logic to evaluate overall validation success
    success_f = vf_wr >= 55.0 and vf_pnl > 0
    success_d = vd_wr >= 55.0 and vd_pnl > 0
    
    if success_f or success_d:
        report += "> [!TIP]\n"
        report += f"> **Validation SUCCESSFUL**: The veto override strategy shows positive out-of-sample metrics on unseen validation data.\n"
        if success_f:
            report += f"> - Fixed configuration out-of-sample win rate is **{vf_wr:.2f}%** with average PnL **{vf_pnl:.4f}%**.\n"
        if success_d:
            report += f"> - Dynamically discovered configuration out-of-sample win rate is **{vd_wr:.2f}%** with average PnL **{vd_pnl:.4f}%**.\n"
        report += f"> The models are generalizing correctly and can proceed to live implementation.\n"
    else:
        report += "> [!CAUTION]\n"
        report += f"> **Validation FAILED (Overfitting Warning)**: The out-of-sample win rate collapsed or PnL became negative.\n"
        report += f"> - Fixed configuration out-of-sample win rate: **{vf_wr:.2f}%** (Avg PnL: **{vf_pnl:.4f}%**)\n"
        report += f"> - Dynamic configuration out-of-sample win rate: **{vd_wr:.2f}%** (Avg PnL: **{vd_pnl:.4f}%**)\n"
        report += f"> This strongly indicates that the K-Means clustering configurations are overfitting to historical noise. **We recommend NOT using this strategy in live trading without modifying features or tightening quality gates.**\n"

    report_path = r'c:\Users\loq\Desktop\Trading\finalgo\finalgo-memory-layer\finalgo\07. Cluster Research\Out_of_Sample_Validation_Report.md'
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
        
    print(f"\nMarkdown report written to: {report_path}")

if __name__ == "__main__":
    run_validation()

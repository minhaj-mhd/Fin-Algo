import sqlite3
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

DB_PATH = 'data/vanguard_trades.db'
LEVERAGE = 5
SLIPPAGE = 0.06  # 0.06% per trade
WINDOW_DAYS = 8  # 8 trading days for training (~850-950 trades)

def load_and_preprocess_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM trades ORDER BY timestamp ASC", conn)
    conn.close()
    
    df['final_profit_pct'] = pd.to_numeric(df['final_profit_pct'], errors='coerce')
    df = df.dropna(subset=['final_profit_pct']).reset_index(drop=True)
    
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    
    features = ['tech_score', 'nlp_sentiment', 'tv_sentiment', 'one_hour_prob']
    tv_map = {'STRONG_SELL': -2, 'SELL': -1, 'NEUTRAL': 0, 'BUY': 1, 'STRONG_BUY': 2}
    
    if 'tv_sentiment' in df.columns:
        df['tv_sentiment'] = df['tv_sentiment'].replace(tv_map)
        
    for f in features:
        df[f] = pd.to_numeric(df[f].astype(str).str.replace('%', '').str.strip(), errors='coerce').fillna(0.0)
        
    return df, features

def get_matches_for_set(df_dataset, X_dataset_scaled, sweet_spots):
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
            
    return {
        'individual': matched_by_spot,
        'dedup_indices': sorted(list(all_matched_indices))
    }

def run_walk_forward():
    print("=" * 60)
    print("STARTING ROLLING WALK-FORWARD VALIDATION SIMULATION")
    print("=" * 60)
    
    df, features = load_and_preprocess_data()
    unique_dates = sorted(df['date'].unique())
    total_days = len(unique_dates)
    
    print(f"Total clean trades: {len(df)}")
    print(f"Total unique trading days: {total_days}")
    print(f"Rolling Window Configuration: Train = {WINDOW_DAYS} days, Test = 1 day")
    
    if total_days <= WINDOW_DAYS:
        print("Error: Not enough trading days to run walk-forward validation.")
        return
        
    all_test_baselines = []
    all_fixed_test_matches = []
    all_disc_test_matches = []
    
    split_details = []
    
    # Fixed configs definition
    fixed_configs = [
        (0, 1, 0.75, "Cluster 0 Sub 1 (K=3, dist < 0.75)"),
        (1, 1, 0.50, "Cluster 1 Sub 1 (K=3, dist < 0.50)"),
        (2, 0, 0.50, "Cluster 2 Sub 0 (K=3, dist < 0.50)"),
        (2, 1, 1.00, "Cluster 2 Sub 1 (K=3, dist < 1.00)"),
        (3, 0, 0.75, "Cluster 3 Sub 0 (K=5, dist < 0.75)")
    ]
    
    for i in range(WINDOW_DAYS, total_days):
        train_dates = unique_dates[i - WINDOW_DAYS : i]
        test_date = unique_dates[i]
        
        df_train = df[df['date'].isin(train_dates)].copy()
        df_test = df[df['date'] == test_date].copy()
        
        print(f"\n--- Split {i - WINDOW_DAYS + 1} ---")
        print(f"Train Window: {train_dates[0]} to {train_dates[-1]} ({len(df_train)} trades)")
        print(f"Test Day:     {test_date} ({len(df_test)} trades)")
        
        # Save baseline test trades
        all_test_baselines.append(df_test)
        
        # Extract Training Winners
        df_train_winners = df_train[df_train['final_profit_pct'] > 0.5].copy()
        if len(df_train_winners) < 10:
            print(f"Warning: Too few training winners ({len(df_train_winners)}). Skipping this split.")
            continue
            
        # Fit Scaler and Primary KMeans on training winners
        scaler = StandardScaler()
        X_train_winners = df_train_winners[features].values
        X_train_winners_scaled = scaler.fit_transform(X_train_winners)
        
        kmeans_primary = KMeans(n_clusters=4, init='k-means++', n_init=20, random_state=42)
        df_train_winners['primary_cluster'] = kmeans_primary.fit_predict(X_train_winners_scaled)
        
        # Scale train and test sets
        X_train_all_scaled = scaler.transform(df_train[features].values)
        X_test_all_scaled = scaler.transform(df_test[features].values)
        
        # Train sub-clustering models
        sub_kmeans_models = {}
        for c in range(4):
            c_winners = df_train_winners[df_train_winners['primary_cluster'] == c].copy()
            if len(c_winners) < 3:
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
            
        # Rebuild Fixed Sweet Spots for this window
        fixed_sweet_spots = []
        for pc, sc, threshold, name in fixed_configs:
            if pc in sub_kmeans_models:
                sub_model_info = sub_kmeans_models[pc]
                sub_kmeans = sub_model_info['model']
                if sc < sub_model_info['k']:
                    centroid = sub_kmeans.cluster_centers_[sc]
                    fixed_sweet_spots.append({
                        'name': name,
                        'centroid': centroid,
                        'threshold': threshold
                    })
                    
        # Dynamically discover sweet spots for this window
        discovered_sweet_spots = []
        for pc, sub_model_info in sub_kmeans_models.items():
            sub_kmeans = sub_model_info['model']
            sub_k = sub_model_info['k']
            
            for sc in range(sub_k):
                centroid = sub_kmeans.cluster_centers_[sc]
                distances = np.linalg.norm(X_train_all_scaled - centroid, axis=1)
                
                best_threshold = None
                best_score = -999.0
                best_metrics = None
                
                for t in [0.50, 0.75, 1.00, 1.50]:
                    matched_indices = np.where(distances < t)[0]
                    matched_trades = df_train.iloc[matched_indices]
                    cnt = len(matched_trades)
                    
                    if cnt >= 5:
                        wr = (matched_trades['final_profit_pct'] > 0).mean() * 100
                        pnl = matched_trades['final_profit_pct'].mean()
                        
                        if wr >= 60.0 and pnl >= 0.10:
                            score = cnt * pnl
                            if score > best_score:
                                best_score = score
                                best_threshold = t
                                best_metrics = (cnt, wr, pnl)
                                
                if best_threshold is not None:
                    discovered_sweet_spots.append({
                        'name': f"Cluster {pc} Sub {sc} (K={sub_k}, dist < {best_threshold:.2f})",
                        'centroid': centroid,
                        'threshold': best_threshold
                    })
                    
        # Apply to Test Day
        fixed_test_res = get_matches_for_set(df_test, X_test_all_scaled, fixed_sweet_spots)
        disc_test_res = get_matches_for_set(df_test, X_test_all_scaled, discovered_sweet_spots)
        
        df_test_fixed_matched = df_test.iloc[fixed_test_res['dedup_indices']].copy()
        df_test_disc_matched = df_test.iloc[disc_test_res['dedup_indices']].copy()
        
        # Save matches
        all_fixed_test_matches.append(df_test_fixed_matched)
        all_disc_test_matches.append(df_test_disc_matched)
        
        # Calculate split stats
        base_wr = (df_test['final_profit_pct'] > 0).mean() * 100
        base_pnl = df_test['final_profit_pct'].mean()
        
        fixed_cnt = len(df_test_fixed_matched)
        fixed_wr = (df_test_fixed_matched['final_profit_pct'] > 0).mean() * 100 if fixed_cnt > 0 else 0.0
        fixed_pnl = df_test_fixed_matched['final_profit_pct'].mean() if fixed_cnt > 0 else 0.0
        fixed_net = (5 * df_test_fixed_matched['final_profit_pct'] - SLIPPAGE).sum() if fixed_cnt > 0 else 0.0
        
        disc_cnt = len(df_test_disc_matched)
        disc_wr = (df_test_disc_matched['final_profit_pct'] > 0).mean() * 100 if disc_cnt > 0 else 0.0
        disc_pnl = df_test_disc_matched['final_profit_pct'].mean() if disc_cnt > 0 else 0.0
        disc_net = (5 * df_test_disc_matched['final_profit_pct'] - SLIPPAGE).sum() if disc_cnt > 0 else 0.0
        
        split_details.append({
            'Split': i - WINDOW_DAYS + 1,
            'TestDate': str(test_date),
            'TestSize': len(df_test),
            'BaseWR': f"{base_wr:.1f}%",
            'BasePnL': f"{base_pnl:.4f}%",
            'FixedMatches': fixed_cnt,
            'FixedWR': f"{fixed_wr:.1f}%" if fixed_cnt > 0 else "N/A",
            'FixedPnL': f"{fixed_pnl:.4f}%" if fixed_cnt > 0 else "N/A",
            'FixedNetRet': f"{fixed_net:+.2f}%" if fixed_cnt > 0 else "0.00%",
            'DiscMatches': disc_cnt,
            'DiscWR': f"{disc_wr:.1f}%" if disc_cnt > 0 else "N/A",
            'DiscPnL': f"{disc_pnl:.4f}%" if disc_cnt > 0 else "N/A",
            'DiscNetRet': f"{disc_net:+.2f}%" if disc_cnt > 0 else "0.00%"
        })
        
        print(f"Fixed Overrides: {fixed_cnt} trades | WR: {fixed_wr:.1f}% | Avg PnL: {fixed_pnl:.4f}% | Net Ret: {fixed_net:+.2f}%")
        print(f"Dynamic Overrides: {disc_cnt} trades | WR: {disc_wr:.1f}% | Avg PnL: {disc_pnl:.4f}% | Net Ret: {disc_net:+.2f}%")

    # Aggregate performance across ALL splits
    df_all_baseline = pd.concat(all_test_baselines).reset_index(drop=True)
    df_all_fixed = pd.concat(all_fixed_test_matches).reset_index(drop=True)
    df_all_disc = pd.concat(all_disc_test_matches).reset_index(drop=True)
    
    def get_summary_stats(df_matched, df_total):
        cnt = len(df_matched)
        if cnt == 0:
            return 0, 0.0, 0.0, 0.0, 0.0
        pct = cnt / len(df_total) * 100
        wr = (df_matched['final_profit_pct'] > 0).mean() * 100
        pnl = df_matched['final_profit_pct'].mean()
        net_pnls = 5 * df_matched['final_profit_pct'] - SLIPPAGE
        cum_ret = net_pnls.sum()
        return cnt, pct, wr, pnl, cum_ret
        
    f_cnt, f_pct, f_wr, f_pnl, f_ret = get_summary_stats(df_all_fixed, df_all_baseline)
    d_cnt, d_pct, d_wr, d_pnl, d_ret = get_summary_stats(df_all_disc, df_all_baseline)
    
    base_wr = (df_all_baseline['final_profit_pct'] > 0).mean() * 100
    base_pnl = df_all_baseline['final_profit_pct'].mean()
    
    print("\n" + "=" * 60)
    print("WALK-FORWARD VALIDATION SUMMARY METRICS")
    print("=" * 60)
    print(f"Baseline (All Test Trades): WR = {base_wr:.2f}%, Avg PnL = {base_pnl:.4f}%")
    print(f"Fixed Sweet Spots: Matches = {f_cnt} ({f_pct:.1f}%), WR = {f_wr:.2f}%, Avg PnL = {f_pnl:.4f}%, Net Ret = {f_ret:+.2f}%")
    print(f"Dynamically Discovered: Matches = {d_cnt} ({d_pct:.1f}%), WR = {d_wr:.2f}%, Avg PnL = {d_pnl:.4f}%, Net Ret = {d_ret:+.2f}%")
    
    # Create the markdown report
    generate_walk_forward_report(split_details, base_wr, base_pnl, f_cnt, f_pct, f_wr, f_pnl, f_ret, d_cnt, d_pct, d_wr, d_pnl, d_ret, df_all_baseline)

def generate_walk_forward_report(splits, b_wr, b_pnl, f_cnt, f_pct, f_wr, f_pnl, f_ret, d_cnt, d_pct, d_wr, d_pnl, d_ret, df_all_baseline):
    report = f"""# Walk-Forward Validation Report: Veto Override Clustering

This report presents a rigorous **rolling walk-forward backtest** of the veto override clustering strategy. 
To represent a realistic production deployment with daily retraining, we evaluate our clustering strategy using a **8-day rolling training window** and test on the **next 1 day**.

- **Total Trading Days**: 15
- **Rolling Splits**: 7 (Testing on days 9 through 15)
- **Leverage**: 5x | **Slippage**: 0.06% per trade

---

## 1. Walk-Forward Cumulative Results (Out-of-Sample)

| Strategy Configuration | Dataset | Matched Trades | Catch Rate (%) | Win Rate (%) | Avg PnL (%) | Cumulative Net Return (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (All Test Trades)** | Walk-Forward | {len(df_all_baseline)} | 100.0% | {b_wr:.2f}% | {b_pnl:.4f}% | N/A |
| **Path 1: Fixed Researched Sweet Spots** | Walk-Forward | {f_cnt} | {f_pct:.1f}% | {f_wr:.2f}% | {f_pnl:.4f}% | {f_ret:+.2f}% |
| **Path 2: Dynamically Discovered Sweet Spots** | Walk-Forward | {d_cnt} | {d_pct:.1f}% | {d_wr:.2f}% | {d_pnl:.4f}% | {d_ret:+.2f}% |

---

## 2. Daily Walk-Forward Split Breakdown

| Split | Test Date | Test Size | Baseline WR | Baseline PnL | Fixed Overrides | Fixed WR | Fixed PnL | Fixed Net Return | Disc. Overrides | Disc. WR | Disc. PnL | Disc. Net Return |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for s in splits:
        report += f"| {s['Split']} | {s['TestDate']} | {s['TestSize']} | {s['BaseWR']} | {s['BasePnL']} | {s['FixedMatches']} | {s['FixedWR']} | {s['FixedPnL']} | {s['FixedNetRet']} | {s['DiscMatches']} | {s['DiscWR']} | {s['DiscPnL']} | {s['DiscNetRet']} |\n"
        
    report += "\n## 3. Findings and Key Insights\n\n"
    
    # Add status checks
    success_f = f_wr >= 55.0 and f_pnl > 0
    success_d = d_wr >= 55.0 and d_pnl > 0
    
    if success_f or success_d:
        report += "> [!NOTE]\n"
        report += f"> **Walk-Forward Performance Summary**:\n"
        if success_f:
            report += f"> - Fixed sweet spots achieved **{f_wr:.2f}% Win Rate** and **{f_pnl:.4f}% Average PnL** across the rolling test days, yielding **{f_ret:+.2f}% net return**.\n"
        if success_d:
            report += f"> - Dynamically discovered sweet spots achieved **{d_wr:.2f}% Win Rate** and **{d_pnl:.4f}% Average PnL**, yielding **{d_ret:+.2f}% net return**.\n"
    else:
        report += "> [!WARNING]\n"
        report += f"> **Walk-Forward FAILED (Overfitting Confirmed)**:\n"
        report += f"> Under rolling walk-forward conditions, the strategy's win rates and returns collapse:\n"
        report += f"> - Fixed Sweet Spots WR: **{f_wr:.2f}%** (Avg PnL: **{f_pnl:.4f}%** | Net Return: **{f_ret:+.2f}%**)\n"
        report += f"> - Dynamically Discovered WR: **{d_wr:.2f}%** (Avg PnL: **{d_pnl:.4f}%** | Net Return: **{d_ret:+.2f}%**)\n"
        report += f"> This confirms that the current 4-feature clustering configuration does not generalize well across sliding chronological windows.\n"
        
    report += """
## 4. Next Steps: Feature Expansion (Path 4)
Because the current 4 features do not provide robust walk-forward performance, we must proceed to Feature Expansion (Path 4). Refer to [Feature_Set_Analysis.md](file:///c:/Users/loq/Desktop/Trading/finalgo/finalgo-memory-layer/finalgo/07.%20Cluster%20Research/Feature_Set_Analysis.md) for candidates.
"""
    
    report_path = r'c:\Users\loq\Desktop\Trading\finalgo\finalgo-memory-layer\finalgo\07. Cluster Research\Walk_Forward_Validation_Report.md'
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
        
    print(f"\nWalk-forward report written to: {report_path}")

if __name__ == "__main__":
    run_walk_forward()

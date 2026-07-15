import pandas as pd
import numpy as np
import xgboost as xgb
import json

def get_period(ym):
    if ym in ['2025-08', '2025-09', '2025-10', '2025-11', '2025-12', '2026-01']:
        return "1. Testing (6m: Aug'25 - Jan'26)"
    elif ym in ['2026-02', '2026-03', '2026-04']:
        return "2. Proxy OOS (3m: Feb'26 - Apr'26)"
    elif ym >= '2026-05':
        return "3. True OOS (3m: May'26 - Jul'26)"
    return "Other"

def main():
    print("Loading data...")
    df = pd.read_parquet('data/research/v20_rolling_1h/panel.parquet')
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df['YearMonth'] = df['DateTime'].dt.strftime('%Y-%m')
    
    unique_months = sorted(df['YearMonth'].unique())
    split_idx = int(len(unique_months) * 0.8)
    test_months = unique_months[split_idx-1:] 
    
    df_test = df[df['YearMonth'].isin(test_months)].copy()
    df_test['Period'] = df_test['YearMonth'].apply(get_period)
    df_test = df_test[df_test['Period'] != "Other"]

    print("Loading v23 pairwise ranker...")
    with open('models/research/v23_rolling_1h/metadata.json') as f:
        v23_meta = json.load(f)
    v23_feats_long = v23_meta.get('features_long', v23_meta['features'])
    v23_feats_short = v23_meta.get('features_short', v23_meta['features'])
    v23_long = xgb.Booster()
    v23_long.load_model('models/research/v23_rolling_1h/xgb_long_model.json')
    v23_short = xgb.Booster()
    v23_short.load_model('models/research/v23_rolling_1h/xgb_short_model.json')

    print("Loading v25 fat tail binary logistic...")
    with open('models/research/v25_fat_tail_1h/metadata.json') as f:
        v25_meta = json.load(f)
    v25_feats_long = v25_meta.get('features_long', v25_meta['features'])
    v25_feats_short = v25_meta.get('features_short', v25_meta['features'])
    v25_long = xgb.Booster()
    v25_long.load_model('models/research/v25_fat_tail_1h/xgb_long_model.json')
    v25_short = xgb.Booster()
    v25_short.load_model('models/research/v25_fat_tail_1h/xgb_short_model.json')

    print("Scoring test set...")
    d23_l = xgb.DMatrix(df_test[v23_feats_long].values)
    d23_s = xgb.DMatrix(df_test[v23_feats_short].values)
    df_test['v23_long_score'] = v23_long.predict(d23_l)
    df_test['v23_short_score'] = v23_short.predict(d23_s)

    d25_l = xgb.DMatrix(df_test[v25_feats_long].values)
    d25_s = xgb.DMatrix(df_test[v25_feats_short].values)
    df_test['v25_long_prob'] = v25_long.predict(d25_l)
    df_test['v25_short_prob'] = v25_short.predict(d25_s)

    # 1. First, we need to extract the "Top 1" predictions for every 15-minute cross-section.
    print("Extracting Top 1 picks per cross-section...")
    top_longs = []
    top_shorts = []
    
    for qid, group in df_test.groupby('Query_ID'):
        if len(group) == 0: continue
        
        top_l = group.loc[group['v23_long_score'].idxmax()]
        top_longs.append(top_l)
        
        top_s = group.loc[group['v23_short_score'].idxmax()]
        top_shorts.append(top_s)
        
    df_longs = pd.DataFrame(top_longs)
    df_shorts = pd.DataFrame(top_shorts)
    
    # 2. Setup the dynamic rolling windows (2 weeks = 14 days)
    # We will iterate by dates
    unique_dates = sorted(df_longs['DateTime'].dt.date.unique())
    
    cost_bps = 0.0006 # 6 bps
    results = []

    print("Simulating dynamic thresholding...")
    
    target_trades_per_2w = 12 # 24 per month per side = ~50 total
    
    for i in range(14, len(unique_dates), 14):
        # Lookback window (previous 14 days)
        lookback_start = unique_dates[i-14]
        lookback_end = unique_dates[i-1]
        
        # Execution window (next 14 days, or whatever remains)
        exec_start = unique_dates[i]
        exec_end = unique_dates[min(i+13, len(unique_dates)-1)]
        
        # Calibrate Long Threshold
        calib_longs = df_longs[(df_longs['DateTime'].dt.date >= lookback_start) & (df_longs['DateTime'].dt.date <= lookback_end)]
        if len(calib_longs) > target_trades_per_2w:
            sorted_probs = sorted(calib_longs['v25_long_prob'].values, reverse=True)
            thresh_l = max(sorted_probs[target_trades_per_2w - 1], 0.65)
        else:
            thresh_l = 0.65
            
        # Calibrate Short Threshold
        calib_shorts = df_shorts[(df_shorts['DateTime'].dt.date >= lookback_start) & (df_shorts['DateTime'].dt.date <= lookback_end)]
        if len(calib_shorts) > target_trades_per_2w:
            sorted_probs = sorted(calib_shorts['v25_short_prob'].values, reverse=True)
            thresh_s = max(sorted_probs[target_trades_per_2w - 1], 0.65)
        else:
            thresh_s = 0.65
            
        # Execute Trades
        exec_longs = df_longs[(df_longs['DateTime'].dt.date >= exec_start) & (df_longs['DateTime'].dt.date <= exec_end)]
        for _, row in exec_longs.iterrows():
            if row['v25_long_prob'] > thresh_l:
                results.append({'Period': row['Period'], 'Side': 'Long', 'Return': row['Next_Hour_Return']})
                
        exec_shorts = df_shorts[(df_shorts['DateTime'].dt.date >= exec_start) & (df_shorts['DateTime'].dt.date <= exec_end)]
        for _, row in exec_shorts.iterrows():
            if row['v25_short_prob'] > thresh_s:
                results.append({'Period': row['Period'], 'Side': 'Short', 'Return': -row['Next_Hour_Return']})

    res_df = pd.DataFrame(results)
    
    def print_stats(name, df_subset):
        if len(df_subset) == 0:
            print(f"  {name: <35} |   0 trades")
            return
        trades = df_subset['Return'].values
        count = len(trades)
        gross_edge = np.mean(trades)
        net_edge = gross_edge - cost_bps
        net_wr = np.mean(trades > cost_bps)
        gross_wr = np.mean(trades > 0)
        
        print(f"  {name: <35} | {count: >3} trades | Gross: {gross_edge*10000:+.2f} bps | Net: {net_edge*10000:+.2f} bps | Net WR: {net_wr*100:.1f}%")

    print("\n" + "=" * 80)
    print("HYBRID MODEL RESULTS - DYNAMIC THRESHOLD (Target: ~50 trades/month | Cost 6 bps)")
    print("=" * 80)
    
    if res_df.empty:
        print("No trades taken.")
        return

    for period in sorted(df_test['Period'].unique()):
        print(f"\n[{period}]")
        period_df = res_df[res_df['Period'] == period] if not res_df.empty else pd.DataFrame()
        if period_df.empty:
            print("  No trades taken in this period.")
            continue
        print_stats("Longs", period_df[period_df['Side'] == 'Long'])
        print_stats("Shorts", period_df[period_df['Side'] == 'Short'])
        print_stats("Combined", period_df)

if __name__ == '__main__':
    main()

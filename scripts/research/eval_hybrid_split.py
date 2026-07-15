import pandas as pd
import numpy as np
import xgboost as xgb
import json

def get_period(ym):
    # Mapping based on user request: 6m testing, 3m proxy oos, 3m true oos
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
    df['YearMonth'] = pd.to_datetime(df['DateTime']).dt.strftime('%Y-%m')
    
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

    cost_bps = 0.0006 # 6 bps
    
    results = []

    print("Simulating selection...")
    for qid, group in df_test.groupby('Query_ID'):
        if len(group) == 0: continue
        period = group['Period'].iloc[0]
        
        # Long Top 1
        top_l = group.loc[group['v23_long_score'].idxmax()]
        if top_l['v25_long_prob'] > 0.75:
            results.append({'Period': period, 'Side': 'Long', 'Return': top_l['Next_Hour_Return']})
            
        # Short Top 1
        top_s = group.loc[group['v23_short_score'].idxmax()]
        if top_s['v25_short_prob'] > 0.75:
            results.append({'Period': period, 'Side': 'Short', 'Return': -top_s['Next_Hour_Return']})

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
    print("HYBRID MODEL RESULTS - SPLIT PERIODS (Long > 0.535 | Short > 0.55 | Cost 6 bps)")
    print("=" * 80)
    
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

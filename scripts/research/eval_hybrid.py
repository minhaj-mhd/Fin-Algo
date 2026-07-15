import pandas as pd
import numpy as np
import xgboost as xgb
import json

def main():
    print("Loading data...")
    df = pd.read_parquet('data/research/v20_rolling_1h/panel.parquet')
    df['YearMonth'] = pd.to_datetime(df['DateTime']).dt.strftime('%Y-%m')
    unique_months = sorted(df['YearMonth'].unique())
    split_idx = int(len(unique_months) * 0.8)
    test_months = unique_months[split_idx-1:] # Include the test months (the last 20%)
    print(f"Test months ({len(test_months)}): {test_months[0]} to {test_months[-1]}")
    
    df_test = df[df['YearMonth'].isin(test_months)].copy()
    
    # Load v23 (Ranker - pairwise)
    print("Loading v23 pairwise ranker...")
    with open('models/research/v23_rolling_1h/metadata.json') as f:
        v23_meta = json.load(f)
    v23_feats_long = v23_meta.get('features_long', v23_meta['features'])
    v23_feats_short = v23_meta.get('features_short', v23_meta['features'])
    v23_long = xgb.Booster()
    v23_long.load_model('models/research/v23_rolling_1h/xgb_long_model.json')
    v23_short = xgb.Booster()
    v23_short.load_model('models/research/v23_rolling_1h/xgb_short_model.json')

    # Load v24 Top20 (Binary logistic)
    print("Loading v24 binary logistic...")
    with open('models/research/v24_binary_1h_top20/metadata.json') as f:
        v24_meta = json.load(f)
    v24_feats_long = v24_meta.get('features_long', v24_meta['features'])
    v24_feats_short = v24_meta.get('features_short', v24_meta['features'])
    v24_long = xgb.Booster()
    v24_long.load_model('models/research/v24_binary_1h_top20/xgb_long_model.json')
    v24_short = xgb.Booster()
    v24_short.load_model('models/research/v24_binary_1h_top20/xgb_short_model.json')

    print("Scoring test set...")
    # Score v23
    d23_l = xgb.DMatrix(df_test[v23_feats_long].values)
    d23_s = xgb.DMatrix(df_test[v23_feats_short].values)
    df_test['v23_long_score'] = v23_long.predict(d23_l)
    df_test['v23_short_score'] = v23_short.predict(d23_s)

    # Score v24
    d24_l = xgb.DMatrix(df_test[v24_feats_long].values)
    d24_s = xgb.DMatrix(df_test[v24_feats_short].values)
    df_test['v24_long_prob'] = v24_long.predict(d24_l)
    df_test['v24_short_prob'] = v24_short.predict(d24_s)

    cost_bps = 0.0006 # 6 bps
    
    long_trades = []
    short_trades = []
    
    total_long_tops = 0
    total_short_tops = 0

    print("Simulating selection...")
    # Group by Query_ID (each 15-min cross section)
    for qid, group in df_test.groupby('Query_ID'):
        if len(group) == 0: continue
        
        # Long Top 1
        top_l = group.loc[group['v23_long_score'].idxmax()]
        total_long_tops += 1
        if top_l['v24_long_prob'] > 0.535:
            long_trades.append(top_l['Next_Hour_Return'])
            
        # Short Top 1 (lowest score since it's trained to rank negative returns higher)
        # Wait, for shorts, the ranker was trained with label=get_integer_ranks(invert=True).
        # This means higher score = higher predicted short edge (more negative return).
        # So we should pick the highest score!
        top_s = group.loc[group['v23_short_score'].idxmax()]
        total_short_tops += 1
        if top_s['v24_short_prob'] > 0.55:
            short_trades.append(-top_s['Next_Hour_Return'])

    long_trades = np.array(long_trades)
    short_trades = np.array(short_trades)
    
    def print_stats(name, trades, total_candidates):
        count = len(trades)
        if count == 0:
            print(f"{name}: 0 trades out of {total_candidates} candidates")
            return
        gross_edge = np.mean(trades)
        net_edge = gross_edge - cost_bps
        net_wr = np.mean(trades > cost_bps)
        gross_wr = np.mean(trades > 0)
        
        print(f"{name}: {count} trades taken out of {total_candidates} candidates ({count/total_candidates*100:.1f}% acceptance rate)")
        print(f"  Gross Edge: {gross_edge*10000:+.2f} bps | Gross WR: {gross_wr*100:.2f}%")
        print(f"  Net Edge:   {net_edge*10000:+.2f} bps | Net WR:   {net_wr*100:.2f}%")

    print("\n" + "=" * 50)
    print("HYBRID MODEL RESULTS (V23 Top 1 gated by V24 > 0.55)")
    print("Cost assumption: 6 bps")
    print("=" * 50)
    print_stats("Longs", long_trades, total_long_tops)
    print("-" * 50)
    print_stats("Shorts", short_trades, total_short_tops)
    print("=" * 50)

if __name__ == '__main__':
    main()

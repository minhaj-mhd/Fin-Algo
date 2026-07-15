import pandas as pd
import numpy as np
import xgboost as xgb
import json
import matplotlib.pyplot as plt
import os

def get_period(ym):
    year = ym.split('-')[0]
    return f"Year {year}"

def main():
    print("Loading data...")
    df = pd.read_parquet('data/research/v20_rolling_1h/panel.parquet')
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df['YearMonth'] = df['DateTime'].dt.strftime('%Y-%m')
    
    unique_months = sorted(df['YearMonth'].unique())
    split_idx = int(len(unique_months) * 0.8)
    dev_months = unique_months[:split_idx-1] 
    
    df_dev = df[df['YearMonth'].isin(dev_months)].copy()
    df_dev['Period'] = df_dev['YearMonth'].apply(get_period)

    print("Loading models...")
    with open('models/research/v23_rolling_1h/metadata.json') as f: v23_meta = json.load(f)
    v23_l = xgb.Booster(); v23_l.load_model('models/research/v23_rolling_1h/xgb_long_model.json')
    v23_s = xgb.Booster(); v23_s.load_model('models/research/v23_rolling_1h/xgb_short_model.json')

    with open('models/research/v25_fat_tail_1h/metadata.json') as f: v25_meta = json.load(f)
    v25_l = xgb.Booster(); v25_l.load_model('models/research/v25_fat_tail_1h/xgb_long_model.json')
    v25_s = xgb.Booster(); v25_s.load_model('models/research/v25_fat_tail_1h/xgb_short_model.json')

    d23_l = xgb.DMatrix(df_dev[v23_meta.get('features_long', v23_meta['features'])].values)
    d23_s = xgb.DMatrix(df_dev[v23_meta.get('features_short', v23_meta['features'])].values)
    df_dev['v23_long_score'] = v23_l.predict(d23_l)
    df_dev['v23_short_score'] = v23_s.predict(d23_s)

    d25_l = xgb.DMatrix(df_dev[v25_meta.get('features_long', v25_meta['features'])].values)
    d25_s = xgb.DMatrix(df_dev[v25_meta.get('features_short', v25_meta['features'])].values)
    df_dev['v25_long_prob'] = v25_l.predict(d25_l)
    df_dev['v25_short_prob'] = v25_s.predict(d25_s)

    print("Extracting Top 1 picks per cross-section...")
    top_longs, top_shorts = [], []
    for qid, group in df_dev.groupby('Query_ID'):
        if len(group) == 0: continue
        top_longs.append(group.loc[group['v23_long_score'].idxmax()])
        top_shorts.append(group.loc[group['v23_short_score'].idxmax()])
        
    df_longs, df_shorts = pd.DataFrame(top_longs), pd.DataFrame(top_shorts)
    unique_dates = sorted(df_longs['DateTime'].dt.date.unique())
    
    cost_bps = 0.0006
    results = []
    target_trades_per_2w = 12
    
    print("Simulating dynamic thresholding on Dev Set...")
    for i in range(14, len(unique_dates), 14):
        lookback_start = unique_dates[i-14]
        lookback_end = unique_dates[i-1]
        exec_start = unique_dates[i]
        exec_end = unique_dates[min(i+13, len(unique_dates)-1)]
        
        c_l = df_longs[(df_longs['DateTime'].dt.date >= lookback_start) & (df_longs['DateTime'].dt.date <= lookback_end)]
        thresh_l = max(sorted(c_l['v25_long_prob'].values, reverse=True)[target_trades_per_2w - 1], 0.65) if len(c_l) > target_trades_per_2w else 0.65
            
        c_s = df_shorts[(df_shorts['DateTime'].dt.date >= lookback_start) & (df_shorts['DateTime'].dt.date <= lookback_end)]
        thresh_s = max(sorted(c_s['v25_short_prob'].values, reverse=True)[target_trades_per_2w - 1], 0.65) if len(c_s) > target_trades_per_2w else 0.65
            
        e_l = df_longs[(df_longs['DateTime'].dt.date >= exec_start) & (df_longs['DateTime'].dt.date <= exec_end)]
        for _, row in e_l.iterrows():
            if row['v25_long_prob'] > thresh_l:
                results.append({'DateTime': row['DateTime'], 'Period': row['Period'], 'Side': 'Long', 'Net_Return': row['Next_Hour_Return'] - cost_bps})
                
        e_s = df_shorts[(df_shorts['DateTime'].dt.date >= exec_start) & (df_shorts['DateTime'].dt.date <= exec_end)]
        for _, row in e_s.iterrows():
            if row['v25_short_prob'] > thresh_s:
                results.append({'DateTime': row['DateTime'], 'Period': row['Period'], 'Side': 'Short', 'Net_Return': -row['Next_Hour_Return'] - cost_bps})

    res_df = pd.DataFrame(results).sort_values('DateTime')
    
    def print_stats(name, df_subset):
        if len(df_subset) == 0:
            print(f"  {name: <35} |   0 trades")
            return
        trades = df_subset['Net_Return'].values
        count = len(trades)
        net_edge = np.mean(trades)
        gross_edge = net_edge + cost_bps
        net_wr = np.mean(trades > 0)
        
        print(f"  {name: <35} | {count: >3} trades | Gross: {gross_edge*10000:+.2f} bps | Net: {net_edge*10000:+.2f} bps | Net WR: {net_wr*100:.1f}%")

    print("\n" + "=" * 80)
    print("HYBRID MODEL RESULTS - DYNAMIC THRESHOLD ON DEV SET (Target: ~50 trades/month | Cost 6 bps)")
    print("=" * 80)
    
    if res_df.empty:
        print("No trades taken.")
        return

    for period in sorted(df_dev['Period'].unique()):
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

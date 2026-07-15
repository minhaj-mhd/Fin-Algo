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
    print("Loading test data...")
    df = pd.read_parquet('data/research/v20_rolling_1h/panel.parquet')
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df['YearMonth'] = df['DateTime'].dt.strftime('%Y-%m')
    
    unique_months = sorted(df['YearMonth'].unique())
    split_idx = int(len(unique_months) * 0.8)
    test_months = unique_months[split_idx-1:] 
    
    df_test = df[df['YearMonth'].isin(test_months)].copy()
    df_test['Period'] = df_test['YearMonth'].apply(get_period)
    df_test = df_test[df_test['Period'] != "Other"]

    print("Loading Nifty 500 Regime data...")
    nifty = pd.read_csv('data/raw_index_cache/nifty500_1d.csv')
    nifty['timestamp'] = pd.to_datetime(nifty['timestamp'])
    nifty['date'] = nifty['timestamp'].dt.date
    nifty = nifty.sort_values('date')
    nifty['nifty_100dma'] = nifty['close'].rolling(window=100).mean()
    
    # E0.1 Fix: Shift by 1 day to prevent lookahead bias
    nifty['prev_close'] = nifty['close'].shift(1)
    nifty['prev_100dma'] = nifty['nifty_100dma'].shift(1)
    nifty['routing_timestamp'] = nifty['timestamp'].shift(1)
    
    df_test['date'] = df_test['DateTime'].dt.date
    nifty_subset = nifty[['date', 'routing_timestamp', 'prev_close', 'prev_100dma']].rename(
        columns={'prev_close': 'nifty_close', 'prev_100dma': 'nifty_100dma'}
    )
    df_test = df_test.merge(nifty_subset, on='date', how='left')
    
    # E0.1 Audit Assertion: Ensure routing data is from BEFORE the current trade's time
    routing_ts_naive = pd.to_datetime(df_test['routing_timestamp']).dt.tz_localize(None)
    datetime_naive = pd.to_datetime(df_test['DateTime']).dt.tz_localize(None)
    
    # Drop NaNs before assertion
    valid_mask = routing_ts_naive.notna() & datetime_naive.notna()
    
    mask = routing_ts_naive[valid_mask] > datetime_naive[valid_mask]
    if mask.any():
        print(f"Failing assertion! {mask.sum()} rows failed.")
        print(df_test[['date', 'routing_timestamp', 'DateTime']][valid_mask][mask].head())
    assert (routing_ts_naive[valid_mask] <= datetime_naive[valid_mask]).all(), "Lookahead bias detected in routing key!"
    
    df_test['Regime_Bull'] = df_test['nifty_close'] > df_test['nifty_100dma']
    df_test['Dist_100DMA'] = np.abs((df_test['nifty_close'] - df_test['nifty_100dma']) / df_test['nifty_100dma'])

    print("Loading v23 pairwise ranker...")
    with open('models/research/v23_rolling_1h/metadata.json') as f: v23_meta = json.load(f)
    v23_long = xgb.Booster(); v23_long.load_model('models/research/v23_rolling_1h/xgb_long_model.json')
    v23_short = xgb.Booster(); v23_short.load_model('models/research/v23_rolling_1h/xgb_short_model.json')

    print("Loading v25 BULL regime models...")
    with open('models/research/v25_fat_tail_1h_bull_100dma/metadata.json') as f: bull_meta = json.load(f)
    bull_long = xgb.Booster(); bull_long.load_model('models/research/v25_fat_tail_1h_bull_100dma/xgb_long_model.json')
    bull_short = xgb.Booster(); bull_short.load_model('models/research/v25_fat_tail_1h_bull_100dma/xgb_short_model.json')

    print("Loading v25 BEAR regime models...")
    with open('models/research/v25_fat_tail_1h_bear_100dma/metadata.json') as f: bear_meta = json.load(f)
    bear_long = xgb.Booster(); bear_long.load_model('models/research/v25_fat_tail_1h_bear_100dma/xgb_long_model.json')
    bear_short = xgb.Booster(); bear_short.load_model('models/research/v25_fat_tail_1h_bear_100dma/xgb_short_model.json')

    print("Scoring v23 rankers...")
    d23_l = xgb.DMatrix(df_test[v23_meta.get('features_long', v23_meta['features'])].values)
    d23_s = xgb.DMatrix(df_test[v23_meta.get('features_short', v23_meta['features'])].values)
    df_test['v23_long_score'] = v23_long.predict(d23_l)
    df_test['v23_short_score'] = v23_short.predict(d23_s)

    print("Extracting Top 1 picks per cross-section...")
    top_longs, top_shorts = [], []
    for qid, group in df_test.groupby('Query_ID'):
        if len(group) == 0: continue
        top_longs.append(group.loc[group['v23_long_score'].idxmax()])
        top_shorts.append(group.loc[group['v23_short_score'].idxmax()])
        
    df_longs = pd.DataFrame(top_longs)
    df_shorts = pd.DataFrame(top_shorts)

    print("Routing Top 1 picks through Regime Models...")
    # Bull Predictions
    d_bull_l = xgb.DMatrix(df_longs[bull_meta.get('features_long', bull_meta['features'])].values)
    d_bull_s = xgb.DMatrix(df_shorts[bull_meta.get('features_short', bull_meta['features'])].values)
    df_longs['bull_prob'] = bull_long.predict(d_bull_l)
    df_shorts['bull_prob'] = bull_short.predict(d_bull_s)
    
    # Bear Predictions
    d_bear_l = xgb.DMatrix(df_longs[bear_meta.get('features_long', bear_meta['features'])].values)
    d_bear_s = xgb.DMatrix(df_shorts[bear_meta.get('features_short', bear_meta['features'])].values)
    df_longs['bear_prob'] = bear_long.predict(d_bear_l)
    df_shorts['bear_prob'] = bear_short.predict(d_bear_s)

    # Route based on Nifty Regime
    df_longs['final_prob'] = np.where(df_longs['Regime_Bull'], df_longs['bull_prob'], df_longs['bear_prob'])
    df_shorts['final_prob'] = np.where(df_shorts['Regime_Bull'], df_shorts['bull_prob'], df_shorts['bear_prob'])

    # NO TRADE ZONE: Drop all days where Nifty is within 1.5% of its 100-DMA (whipsaw zone)
    df_longs = df_longs[df_longs['Dist_100DMA'] >= 0.015].copy()
    df_shorts = df_shorts[df_shorts['Dist_100DMA'] >= 0.015].copy()

    unique_dates = sorted(df_longs['DateTime'].dt.date.unique())
    cost_bps = 0.0006
    results = []
    target_trades_per_2w = 12 
    
    print("Simulating dynamic thresholding (with Regime Router)...")
    for i in range(14, len(unique_dates), 14):
        lookback_start = unique_dates[i-14]
        lookback_end = unique_dates[i-1]
        exec_start = unique_dates[i]
        exec_end = unique_dates[min(i+13, len(unique_dates)-1)]
        
        c_l = df_longs[(df_longs['DateTime'].dt.date >= lookback_start) & (df_longs['DateTime'].dt.date <= lookback_end)]
        thresh_l = 0.50
        
        c_s = df_shorts[(df_shorts['DateTime'].dt.date >= lookback_start) & (df_shorts['DateTime'].dt.date <= lookback_end)]
        thresh_s = 0.50
            
        e_l = df_longs[(df_longs['DateTime'].dt.date >= exec_start) & (df_longs['DateTime'].dt.date <= exec_end)]
        for _, row in e_l.iterrows():
            if row['final_prob'] > thresh_l:
                regime_str = 'BULL' if row['Regime_Bull'] else 'BEAR'
                results.append({'Period': row['Period'], 'Side': 'Long', 'Regime': regime_str, 'Return': row['Next_Hour_Return']})
                
        e_s = df_shorts[(df_shorts['DateTime'].dt.date >= exec_start) & (df_shorts['DateTime'].dt.date <= exec_end)]
        for _, row in e_s.iterrows():
            if row['final_prob'] > thresh_s:
                regime_str = 'BULL' if row['Regime_Bull'] else 'BEAR'
                results.append({'Period': row['Period'], 'Side': 'Short', 'Regime': regime_str, 'Return': -row['Next_Hour_Return']})

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
        
        print(f"  {name: <35} | {count: >3} trades | Gross: {gross_edge*10000:+.2f} bps | Net: {net_edge*10000:+.2f} bps | Net WR: {net_wr*100:.1f}%")

    print("\n" + "=" * 80)
    print("REGIME ROUTER (BULL/BEAR) RESULTS - OUT OF SAMPLE (Target: ~50 trades/mo)")
    print("=" * 80)
    
    if res_df.empty: return

    for period in sorted(df_test['Period'].unique()):
        print(f"\n[{period}]")
        period_df = res_df[res_df['Period'] == period] if not res_df.empty else pd.DataFrame()
        if period_df.empty:
            print("  No trades taken in this period.")
            continue
        print_stats("Longs", period_df[period_df['Side'] == 'Long'])
        print_stats("Shorts", period_df[period_df['Side'] == 'Short'])
        print_stats("Combined", period_df)

    if not res_df.empty:
        print("\n" + "=" * 80)
        print("PERFORMANCE BY MARKET REGIME (Across all valid OOS periods)")
        print("=" * 80)
        print_stats("BULL Regime Trades (>100 DMA)", res_df[res_df['Regime'] == 'BULL'])
        print_stats("BEAR Regime Trades (<100 DMA)", res_df[res_df['Regime'] == 'BEAR'])

if __name__ == '__main__':
    main()

import pandas as pd
import numpy as np
import xgboost as xgb
import json
import matplotlib.pyplot as plt
import os

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

    print("Loading models...")
    with open('models/research/v23_rolling_1h/metadata.json') as f: v23_meta = json.load(f)
    v23_s = xgb.Booster(); v23_s.load_model('models/research/v23_rolling_1h/xgb_short_model.json')

    with open('models/research/v25_fat_tail_1h/metadata.json') as f: v25_meta = json.load(f)
    v25_s = xgb.Booster(); v25_s.load_model('models/research/v25_fat_tail_1h/xgb_short_model.json')

    d23_s = xgb.DMatrix(df_test[v23_meta.get('features_short', v23_meta['features'])].values)
    df_test['v23_short_score'] = v23_s.predict(d23_s)

    d25_s = xgb.DMatrix(df_test[v25_meta.get('features_short', v25_meta['features'])].values)
    df_test['v25_short_prob'] = v25_s.predict(d25_s)

    top_shorts = []
    for qid, group in df_test.groupby('Query_ID'):
        if len(group) == 0: continue
        top_shorts.append(group.loc[group['v23_short_score'].idxmax()])
        
    df_shorts = pd.DataFrame(top_shorts)
    unique_dates = sorted(df_shorts['DateTime'].dt.date.unique())
    
    cost_bps = 0.0006
    results = []
    target_trades_per_2w = 12
    
    for i in range(14, len(unique_dates), 14):
        lookback_start = unique_dates[i-14]
        lookback_end = unique_dates[i-1]
        exec_start = unique_dates[i]
        exec_end = unique_dates[min(i+13, len(unique_dates)-1)]
        
        c_s = df_shorts[(df_shorts['DateTime'].dt.date >= lookback_start) & (df_shorts['DateTime'].dt.date <= lookback_end)]
        thresh_s = max(sorted(c_s['v25_short_prob'].values, reverse=True)[target_trades_per_2w - 1], 0.65) if len(c_s) > target_trades_per_2w else 0.65
            
        e_s = df_shorts[(df_shorts['DateTime'].dt.date >= exec_start) & (df_shorts['DateTime'].dt.date <= exec_end)]
        for _, row in e_s.iterrows():
            if row['v25_short_prob'] > thresh_s:
                results.append({'DateTime': row['DateTime'], 'Side': 'Short', 'Net_Return': -row['Next_Hour_Return'] - cost_bps})

    res_df = pd.DataFrame(results).sort_values('DateTime')
    if res_df.empty: return
    
    # Financial Simulation Parameters
    capital = 100000 # 1 Lakh
    leverage = 5
    position_size = capital * leverage # 5 Lakhs per trade
    
    res_df['PnL_INR'] = res_df['Net_Return'] * position_size
    short_df = res_df[res_df['Side'] == 'Short'].copy()
    
    # Load Real Nifty 500 Index Data
    nifty = pd.read_csv('data/raw_index_cache/nifty500_1h.csv')
    nifty['timestamp'] = pd.to_datetime(nifty['timestamp'])
    # Filter Nifty to test period
    nifty = nifty[(nifty['timestamp'] >= short_df['DateTime'].min()) & (nifty['timestamp'] <= short_df['DateTime'].max())].copy()
    
    # Compute Nifty returns
    nifty['Nifty_Return'] = nifty['close'].pct_change().fillna(0)
    
    # Merge Benchmark
    short_df = pd.merge_asof(short_df, nifty[['timestamp', 'Nifty_Return']].rename(columns={'timestamp': 'DateTime'}), on='DateTime')
    
    short_df['Cum_PnL'] = short_df['PnL_INR'].cumsum() + capital
    short_df['High_Water_Mark'] = short_df['Cum_PnL'].cummax()
    short_df['Drawdown'] = (short_df['Cum_PnL'] - short_df['High_Water_Mark']) / short_df['High_Water_Mark'] * 100
    
    # Benchmark PnL (Unlevered 1x, buy and hold proxy)
    short_df['Bench_Cum_PnL'] = (short_df['Nifty_Return'] * capital).cumsum() + capital

    # Plot
    plt.figure(figsize=(14, 12))
    
    # Cumulative PnL
    plt.subplot(3, 1, 1)
    if not short_df.empty:
        plt.plot(short_df['DateTime'], short_df['Cum_PnL'], label='Short Strategy (5x Lev, Net)', color='red', linewidth=2.5)
    
    plt.axhline(capital, color='black', linestyle='--', alpha=0.5, label='Starting Capital (1L)')
    plt.title(f'Short-Only Strategy Cumulative Equity (1L Start)')
    plt.ylabel('Equity (INR)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # NIFTY Benchmark
    plt.subplot(3, 1, 2)
    if not short_df.empty:
        plt.plot(short_df['DateTime'], short_df['Bench_Cum_PnL'], label='NIFTY 500 Benchmark (1x Lev)', color='gray', linestyle='-', alpha=0.8, linewidth=1.5)
    
    plt.axhline(capital, color='black', linestyle='--', alpha=0.5, label='Starting Capital (1L)')
    plt.title(f'Market Benchmark (NIFTY 500)')
    plt.ylabel('Equity (INR)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Drawdowns
    plt.subplot(3, 1, 3)
    if not short_df.empty:
        plt.fill_between(short_df['DateTime'], short_df['Drawdown'], 0, color='red', alpha=0.3, label='Strategy Drawdown (%)')
        
    plt.ylabel('Drawdown (%)')
    plt.xlabel('Date')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    out_dir = r"C:\Users\loq\.gemini\antigravity\brain\7cd0f0c7-604e-4cbe-966b-9062e287040f\scratch"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "financial_chart_benchmark.png")
    plt.savefig(out_path, dpi=300)
    print(f"Chart saved to {out_path}")
    
if __name__ == '__main__':
    main()

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
    v23_l = xgb.Booster(); v23_l.load_model('models/research/v23_rolling_1h/xgb_long_model.json')
    v23_s = xgb.Booster(); v23_s.load_model('models/research/v23_rolling_1h/xgb_short_model.json')

    with open('models/research/v25_fat_tail_1h/metadata.json') as f: v25_meta = json.load(f)
    v25_l = xgb.Booster(); v25_l.load_model('models/research/v25_fat_tail_1h/xgb_long_model.json')
    v25_s = xgb.Booster(); v25_s.load_model('models/research/v25_fat_tail_1h/xgb_short_model.json')

    d23_l = xgb.DMatrix(df_test[v23_meta.get('features_long', v23_meta['features'])].values)
    d23_s = xgb.DMatrix(df_test[v23_meta.get('features_short', v23_meta['features'])].values)
    df_test['v23_long_score'] = v23_l.predict(d23_l)
    df_test['v23_short_score'] = v23_s.predict(d23_s)

    d25_l = xgb.DMatrix(df_test[v25_meta.get('features_long', v25_meta['features'])].values)
    d25_s = xgb.DMatrix(df_test[v25_meta.get('features_short', v25_meta['features'])].values)
    df_test['v25_long_prob'] = v25_l.predict(d25_l)
    df_test['v25_short_prob'] = v25_s.predict(d25_s)

    top_longs, top_shorts = [], []
    for qid, group in df_test.groupby('Query_ID'):
        if len(group) == 0: continue
        top_longs.append(group.loc[group['v23_long_score'].idxmax()])
        top_shorts.append(group.loc[group['v23_short_score'].idxmax()])
        
    df_longs, df_shorts = pd.DataFrame(top_longs), pd.DataFrame(top_shorts)
    unique_dates = sorted(df_longs['DateTime'].dt.date.unique())
    
    cost_bps = 0.0006
    results = []
    target_trades_per_2w = 12
    
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
                results.append({'DateTime': row['DateTime'], 'Side': 'Long', 'Net_Return': row['Next_Hour_Return'] - cost_bps})
                
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
    
    long_df = res_df[res_df['Side'] == 'Long'].copy()
    long_df['Cum_PnL'] = long_df['PnL_INR'].cumsum() + capital
    long_df['High_Water_Mark'] = long_df['Cum_PnL'].cummax()
    long_df['Drawdown'] = (long_df['Cum_PnL'] - long_df['High_Water_Mark']) / long_df['High_Water_Mark'] * 100
    
    short_df = res_df[res_df['Side'] == 'Short'].copy()
    short_df['Cum_PnL'] = short_df['PnL_INR'].cumsum() + capital
    short_df['High_Water_Mark'] = short_df['Cum_PnL'].cummax()
    short_df['Drawdown'] = (short_df['Cum_PnL'] - short_df['High_Water_Mark']) / short_df['High_Water_Mark'] * 100

    # Plot
    plt.figure(figsize=(14, 10))
    
    # Cumulative PnL (Long vs Short)
    plt.subplot(2, 1, 1)
    if not long_df.empty:
        plt.plot(long_df['DateTime'], long_df['Cum_PnL'], label='Long Strategy', color='green')
    if not short_df.empty:
        plt.plot(short_df['DateTime'], short_df['Cum_PnL'], label='Short Strategy', color='red')
    
    plt.axhline(capital, color='black', linestyle='--', alpha=0.5, label='Starting Capital (1L)')
    plt.title(f'Cumulative Account Equity (1L Start | 5x Leverage = 5L Position Size)')
    plt.ylabel('Equity (INR)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Drawdowns
    plt.subplot(2, 1, 2)
    if not long_df.empty:
        plt.fill_between(long_df['DateTime'], long_df['Drawdown'], 0, color='green', alpha=0.2, label='Long Drawdown (%)')
    if not short_df.empty:
        plt.fill_between(short_df['DateTime'], short_df['Drawdown'], 0, color='red', alpha=0.2, label='Short Drawdown (%)')
        
    plt.ylabel('Drawdown (%)')
    plt.xlabel('Date')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    out_dir = r"C:\Users\loq\.gemini\antigravity\brain\7cd0f0c7-604e-4cbe-966b-9062e287040f\scratch"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "financial_chart.png")
    plt.savefig(out_path, dpi=300)
    print(f"Chart saved to {out_path}")
    
if __name__ == '__main__':
    main()

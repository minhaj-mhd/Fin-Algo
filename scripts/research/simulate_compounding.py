import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import xgboost as xgb

COST_BPS = 6.0
STARTING_CAPITAL = 100000
LEVERAGE = 5
OUTPUT_DIR = r"C:\Users\loq\.gemini\antigravity\brain\942602bc-be26-4372-950e-372d89b01374"

def simulate_side(trades_df, title, image_name):
    # Simulate 1-Slot Queue
    tdf = trades_df.sort_values('DateTime')
    active_until = pd.Timestamp.min
    executed = []
    capital = STARTING_CAPITAL
    
    # Track Peak for Drawdown
    peak_capital = capital
    max_dd = 0.0
    
    for _, trade in tdf.iterrows():
        trade_start = trade['DateTime']
        if trade_start < active_until:
            continue
            
        trade_end = trade_start + pd.Timedelta(hours=1)
        active_until = trade_end
        
        bps_earned = trade['net_bps']
        pnl_pct = bps_earned / 10000.0
        
        trade_pnl = (capital * LEVERAGE) * pnl_pct
        capital += trade_pnl
        
        if capital > peak_capital:
            peak_capital = capital
            
        dd = (peak_capital - capital) / peak_capital * 100
        if dd > max_dd:
            max_dd = dd
            
        executed.append({
            'DateTime': trade_start,
            'Side': trade['side'],
            'BPS': bps_earned,
            'Trade_PnL': trade_pnl,
            'Capital': capital,
            'Drawdown': dd
        })
        
    edf = pd.DataFrame(executed)
    if len(edf) == 0:
        return {'Final': STARTING_CAPITAL, 'MaxDD': 0.0, 'Trades': 0, 'WinRate': 0.0, 'edf': edf}
        
    win_rate = (edf['BPS'] > 0).mean() * 100
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    ax1.plot(edf['DateTime'], edf['Capital'], label=title, color='blue', linewidth=2)
    ax1.set_title(f"{title} Equity Curve (5x Leverage)")
    ax1.set_ylabel("Capital (Rs.)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.fill_between(edf['DateTime'], 0, -edf['Drawdown'], color='red', alpha=0.3)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_ylim(-max_dd - 2, 0)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, image_name))
    plt.close()
    
    return {
        'Final': capital, 
        'MaxDD': max_dd, 
        'Trades': len(edf), 
        'WinRate': win_rate, 
        'edf': edf
    }

def main():
    df = pd.read_parquet('data/research/v20_rolling_1h/panel_backfilled.parquet')
    df['DateTime'] = pd.to_datetime(df['DateTime']).dt.tz_localize(None)
    
    macro_files = {
        'brent': 'data/raw_global_daily/BRENT.parquet',
        'sp500': 'data/raw_global_daily/SP500.parquet'
    }
    macro_df = None
    for name, path in macro_files.items():
        mdf = pd.read_parquet(path).sort_values('timestamp')
        mdf[f'{name}_ret_prev'] = mdf['close'].pct_change().shift(1)
        mdf['date'] = mdf['timestamp'].dt.date
        mdf = mdf[['date', f'{name}_ret_prev']].dropna()
        if macro_df is None:
            macro_df = mdf
        else:
            macro_df = pd.merge(macro_df, mdf, on='date', how='outer')
            
    nifty = pd.read_csv('data/raw_index_cache/nifty50_15m.csv')
    nifty['ts'] = pd.to_datetime(nifty['ts']).dt.tz_localize(None)
    nifty = nifty.sort_values('ts')
    nifty['nifty_ret_2h'] = nifty['close'] / nifty['close'].shift(8) - 1
    nifty['date'] = nifty['ts'].dt.date
    daily_open = nifty.groupby('date')['open'].first().reset_index().rename(columns={'open': 'daily_open'})
    nifty = pd.merge(nifty, daily_open, on='date', how='left')
    nifty['nifty_intraday'] = nifty['close'] / nifty['daily_open'] - 1
    
    n_2h_map = dict(zip(nifty['ts'], nifty['nifty_ret_2h']))
    n_in_map = dict(zip(nifty['ts'], nifty['nifty_intraday']))
    
    df['date'] = df['DateTime'].dt.date
    df = pd.merge(df, macro_df, on='date', how='left')
    df['nifty_2h'] = df['DateTime'].map(n_2h_map)
    df['nifty_in'] = df['DateTime'].map(n_in_map)
    df = df.dropna(subset=['nifty_2h', 'nifty_in', 'brent_ret_prev', 'sp500_ret_prev'])
    
    v20_feats = json.load(open('models/research/v20_rolling_1h/metadata.json'))['features']
    df = df.dropna(subset=v20_feats + ['Next_Hour_Return'])
    
    X = xgb.DMatrix(np.nan_to_num(df[v20_feats].values.astype(np.float32)), feature_names=v20_feats)
    bs = xgb.Booster(); bs.load_model('models/research/v20_rolling_1h/xgb_short_model.json')
    bl = xgb.Booster(); bl.load_model('models/research/v20_rolling_1h/xgb_long_model.json')
    
    df['ss'] = bs.predict(X)
    df['ls'] = bl.predict(X)
    ss_mean = df.groupby('DateTime')['ss'].transform('mean')
    ls_mean = df.groupby('DateTime')['ls'].transform('mean')
    df['short_conv'] = (df['ss'] - ss_mean) - (df['ls'] - ls_mean)
    df['long_conv'] = (df['ls'] - ls_mean) - (df['ss'] - ss_mean)
    
    time_mask = (df['DateTime'].dt.time >= pd.to_datetime('10:15').time()) & (df['DateTime'].dt.time <= pd.to_datetime('14:15').time())
    df = df[time_mask]
    
    all_trades = []
    
    for ts, g in df.groupby('DateTime'):
        t_time = ts.time()
        n2h = g['nifty_2h'].iloc[0]
        nin = g['nifty_in'].iloc[0]
        # Filtered Long Gate
        sp500 = g['sp500_ret_prev'].iloc[0]
        if (n2h > 0.0025 and nin > 0.0020):
            c = g[g['ls'] > 0.035].sort_values('long_conv', ascending=False)
            if len(c)>0:
                trade = c.iloc[0].copy()
                trade['net_bps'] = trade['Next_Hour_Return']*10000 - COST_BPS
                trade['side'] = 'LONG'
                
                # Apply Global Overlay Filter
                valid_long = False
                if sp500 < -0.005:
                    pass # Veto Risk Off completely
                elif sp500 > 0.005:
                    if n2h > 0.0070: valid_long = True # Require Extreme Rally
                else: # Neutral
                    if n2h < 0.0040 or n2h > 0.0070: valid_long = True # Veto Strong Rally
                    
                if valid_long:
                    all_trades.append(trade)

        # Multivariate Short Gate
        brent = g['brent_ret_prev'].iloc[0]
        sp500 = g['sp500_ret_prev'].iloc[0]
        if (n2h <= 0.0025 or nin > 0.0036):
            if (t_time < pd.to_datetime('11:30').time() or t_time > pd.to_datetime('13:00').time()):
                dyn_prob_multi = 0.082
                global_risk_on = sp500 > 0.005
                local_weak = n2h < -0.0010
                
                if global_risk_on and not local_weak:
                    dyn_prob_multi = 0.110
                
                c_multi = g[g['ss'] > dyn_prob_multi].sort_values('short_conv', ascending=False)
                if len(c_multi)>0:
                    trade = c_multi.iloc[0].copy()
                    trade['net_bps'] = -trade['Next_Hour_Return']*10000 - COST_BPS
                    trade['side'] = 'SHORT'
                    all_trades.append(trade)

    tdf = pd.DataFrame(all_trades).sort_values(['DateTime', 'side'], ascending=[True, False])
    
    long_only = tdf[tdf['side'] == 'LONG']
    short_only = tdf[tdf['side'] == 'SHORT']
    
    print("Simulating Long Only...")
    res_l = simulate_side(long_only, "Long Only", "long_equity.png")
    
    print("Simulating Short Only...")
    res_s = simulate_side(short_only, "Short Only", "short_equity.png")
    
    print("Simulating Combined...")
    res_c = simulate_side(tdf, "Combined Multivariate", "combined_equity.png")
    
    # Create Overlay Plot
    plt.figure(figsize=(12, 6))
    if len(res_c['edf']) > 0:
        plt.plot(res_c['edf']['DateTime'], res_c['edf']['Capital'], label=f"Combined (Final: Rs.{res_c['Final']/100000:.1f}L)", color='purple', linewidth=2.5)
    if len(res_s['edf']) > 0:
        plt.plot(res_s['edf']['DateTime'], res_s['edf']['Capital'], label=f"Shorts Only (Final: Rs.{res_s['Final']/100000:.1f}L)", color='red', linewidth=1.5, alpha=0.8, linestyle='--')
    if len(res_l['edf']) > 0:
        plt.plot(res_l['edf']['DateTime'], res_l['edf']['Capital'], label=f"Longs Only (Final: Rs.{res_l['Final']/100000:.1f}L)", color='green', linewidth=1.5, alpha=0.8, linestyle='--')
        
    plt.title("Equity Curve Overlay: Combined vs Individual Legs (5x Leverage)")
    plt.ylabel("Capital (Rs.)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "overlay_equity.png"))
    plt.close()
    
    # Write to a markdown file
    output_path = os.path.join(OUTPUT_DIR, 'compounding_visualization.md')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# 🚀 1-Slot Engine: Graphical Visualization\n\n")
        f.write(f"**Starting Capital:** ₹{STARTING_CAPITAL:,.2f}\n")
        f.write(f"**Margin Leverage:** {LEVERAGE}x\n\n")
        
        f.write("## 🌟 Master Equity Overlay\n")
        f.write("![Overlay Equity Curve](C:/Users/loq/.gemini/antigravity/brain/942602bc-be26-4372-950e-372d89b01374/overlay_equity.png)\n\n")
        
        f.write("## 1. Combined (Longs + Multivariate Shorts)\n")
        f.write(f"- **Final Capital**: ₹{res_c['Final']:,.2f} (+{((res_c['Final']/STARTING_CAPITAL)-1)*100:.2f}%)\n")
        f.write(f"- **Max Drawdown**: {res_c['MaxDD']:.2f}%\n")
        f.write(f"- **Trades**: {res_c['Trades']} | **Win Rate**: {res_c['WinRate']:.1f}%\n\n")
        f.write("![Combined Equity Curve](C:/Users/loq/.gemini/antigravity/brain/942602bc-be26-4372-950e-372d89b01374/combined_equity.png)\n\n")
        
        f.write("## 2. Short Only (Multivariate Dynamic)\n")
        f.write(f"- **Final Capital**: ₹{res_s['Final']:,.2f} (+{((res_s['Final']/STARTING_CAPITAL)-1)*100:.2f}%)\n")
        f.write(f"- **Max Drawdown**: {res_s['MaxDD']:.2f}%\n")
        f.write(f"- **Trades**: {res_s['Trades']} | **Win Rate**: {res_s['WinRate']:.1f}%\n\n")
        f.write("![Short Equity Curve](C:/Users/loq/.gemini/antigravity/brain/942602bc-be26-4372-950e-372d89b01374/short_equity.png)\n\n")
        
        f.write("## 3. Long Only (Static)\n")
        f.write(f"- **Final Capital**: ₹{res_l['Final']:,.2f} (+{((res_l['Final']/STARTING_CAPITAL)-1)*100:.2f}%)\n")
        f.write(f"- **Max Drawdown**: {res_l['MaxDD']:.2f}%\n")
        f.write(f"- **Trades**: {res_l['Trades']} | **Win Rate**: {res_l['WinRate']:.1f}%\n\n")
        f.write("![Long Equity Curve](C:/Users/loq/.gemini/antigravity/brain/942602bc-be26-4372-950e-372d89b01374/long_equity.png)\n\n")

if __name__ == '__main__':
    main()

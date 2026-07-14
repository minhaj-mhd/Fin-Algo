import os
import sys
import json
import warnings
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import time, date

warnings.filterwarnings("ignore")
sys.path.insert(0, os.getcwd())

COST_BPS = 6.0

def load_global_macro():
    macro_files = {
        'brent': 'data/raw_global_daily/BRENT.parquet',
        'sp500': 'data/raw_global_daily/SP500.parquet'
    }
    
    macro_df = None
    for name, path in macro_files.items():
        if os.path.exists(path):
            df = pd.read_parquet(path).sort_values('timestamp')
            df[f'{name}_ret_prev'] = df['close'].pct_change().shift(1)
            df['date'] = df['timestamp'].dt.date
            df = df[['date', f'{name}_ret_prev']].dropna()
            
            if macro_df is None: macro_df = df
            else: macro_df = pd.merge(macro_df, df, on='date', how='outer')
                
    return macro_df

def main():
    macro_df = load_global_macro()
    if macro_df is None: return

    print("Loading Nifty data...")
    nifty = pd.read_csv('data/raw_index_cache/nifty50_15m.csv')
    nifty['ts'] = pd.to_datetime(nifty['ts']).dt.tz_localize(None)
    nifty = nifty.sort_values('ts').reset_index(drop=True)
    
    nifty['nifty_ret_2h'] = nifty['close'] / nifty['close'].shift(8) - 1
    nifty['date'] = nifty['ts'].dt.date
    daily_open = nifty.groupby('date')['open'].first().reset_index().rename(columns={'open': 'daily_open'})
    nifty = pd.merge(nifty, daily_open, on='date', how='left')
    nifty['nifty_intraday'] = nifty['close'] / nifty['daily_open'] - 1
    
    nifty_map = dict(zip(nifty['ts'], nifty['nifty_ret_2h']))
    nifty_intra_map = dict(zip(nifty['ts'], nifty['nifty_intraday']))

    print("Loading panel backfilled data...")
    df = pd.read_parquet('data/research/v20_rolling_1h/panel_backfilled.parquet')
    df['DateTime'] = pd.to_datetime(df['DateTime']).dt.tz_localize(None)
        
    time_mask = (df['DateTime'].dt.time >= time(10, 15)) & (df['DateTime'].dt.time <= time(14, 15))
    df = df[time_mask]
    
    df['date'] = df['DateTime'].dt.date
    df = pd.merge(df, macro_df, on='date', how='left')
    
    df['nifty_ret_2h'] = df['DateTime'].map(nifty_map)
    df['nifty_intraday'] = df['DateTime'].map(nifty_intra_map)
    df = df.dropna(subset=['nifty_ret_2h', 'nifty_intraday', 'brent_ret_prev', 'sp500_ret_prev'])

    print("Loading XGB models...")
    v20_feats = json.load(open('models/research/v20_rolling_1h/metadata.json'))['features']
    df = df.dropna(subset=v20_feats + ['Next_Hour_Return'])
    
    X = xgb.DMatrix(np.nan_to_num(df[v20_feats].values.astype(np.float32)), feature_names=v20_feats)
    bl = xgb.Booster(); bl.load_model('models/research/v20_rolling_1h/xgb_long_model.json')
    bs = xgb.Booster(); bs.load_model('models/research/v20_rolling_1h/xgb_short_model.json')

    df['ls'] = bl.predict(X)
    df['ss'] = bs.predict(X)
    ls_mean = df.groupby('DateTime')['ls'].transform('mean')
    ss_mean = df.groupby('DateTime')['ss'].transform('mean')
    df['long_conv'] = (df['ls'] - ls_mean) - (df['ss'] - ss_mean)

    cands = []
    for ts, g in df.groupby('DateTime'):
        t_time = ts.time()
        n2h = g['nifty_ret_2h'].iloc[0]
        nin = g['nifty_intraday'].iloc[0]
        
        # MUST pass LONG structural gate
        if (n2h > 0.0025 and nin > 0.0020):
            # Do NOT sort and take top 1 yet, because we need to sweep probabilities first!
            # We will just append all rows that pass the structural gate, and we'll filter top 1 in the sweep.
            # Actually, to simulate exactly, we should just sweep over probabilities and THEN take the top 1.
            pass
            
    # Simpler way to get the exact matrix:
    print("\n" + "="*80)
    print("MULTIVARIATE PROBABILITY MATRIX FOR LONGS: GLOBAL x LOCAL")
    print("="*80)
    
    global_cats = ['Risk Off (SP500 < -0.5%)', 'Neutral', 'Risk On (SP500 > 0.5%)']
    local_cats = ['Base Rally (N2H < 40bps)', 'Strong Rally (N2H > 40bps)', 'Extreme Rally (N2H > 70bps)']
    prob_tiers = [0.030, 0.040, 0.050, 0.060, 0.070, 0.075]
    
    # Pre-calculate regimes for the whole dataframe
    def get_local_regime(n2h):
        if n2h > 0.0070: return 'Extreme Rally (N2H > 70bps)'
        elif n2h > 0.0040: return 'Strong Rally (N2H > 40bps)'
        else: return 'Base Rally (N2H < 40bps)'
    def get_global_regime(sp500):
        if sp500 > 0.005: return 'Risk On (SP500 > 0.5%)'
        elif sp500 < -0.005: return 'Risk Off (SP500 < -0.5%)'
        else: return 'Neutral'
        
    df['local_regime'] = df['nifty_ret_2h'].apply(get_local_regime)
    df['global_regime'] = df['sp500_ret_prev'].apply(get_global_regime)
    
    for g_reg in global_cats:
        for l_reg in local_cats:
            print(f"\n[GLOBAL: {g_reg}] + [LOCAL: {l_reg}]")
            
            for p in prob_tiers:
                p_trades = []
                for ts, g in df.groupby('DateTime'):
                    n2h = g['nifty_ret_2h'].iloc[0]
                    nin = g['nifty_intraday'].iloc[0]
                    g_reg_val = g['global_regime'].iloc[0]
                    l_reg_val = g['local_regime'].iloc[0]
                    
                    if (n2h > 0.0025 and nin > 0.0020) and g_reg_val == g_reg and l_reg_val == l_reg:
                        c = g[g['ls'] > p].sort_values('long_conv', ascending=False)
                        if len(c) > 0:
                            trade = c.iloc[0].copy()
                            trade['net_bps'] = trade['Next_Hour_Return'] * 10000 - COST_BPS
                            trade['is_win'] = trade['net_bps'] > 0
                            p_trades.append(trade)
                            
                p_sub = pd.DataFrame(p_trades)
                if len(p_sub) == 0: continue
                print(f"  prob > {p:<5.3f} | Trades: {len(p_sub):3d} | Win: {p_sub['is_win'].mean():.1%} | Net BPS: {p_sub['net_bps'].mean():>6.2f} | Total BPS: {p_sub['net_bps'].sum():>7.2f}")

if __name__ == '__main__':
    main()

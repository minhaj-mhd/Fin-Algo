import pandas as pd
import numpy as np

COST_BPS = 6.0

def main():
    print("Loading macro and trade data...")
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
    
    import json
    import xgboost as xgb
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
    
    time_mask = (df['DateTime'].dt.time >= pd.to_datetime('10:15').time()) & (df['DateTime'].dt.time <= pd.to_datetime('14:15').time())
    df = df[time_mask]
    
    st_shorts = []
    dy_shorts_defensive = []
    dy_shorts_multi = []
    
    for ts, g in df.groupby('DateTime'):
        t_time = ts.time()
        n2h = g['nifty_2h'].iloc[0]
        nin = g['nifty_in'].iloc[0]
        brent = g['brent_ret_prev'].iloc[0]
        sp500 = g['sp500_ret_prev'].iloc[0]
        
        # MUST pass Nifty Structural Gate
        if (n2h <= 0.0025 or nin > 0.0036):
            if (t_time < pd.to_datetime('11:30').time() or t_time > pd.to_datetime('13:00').time()):
                
                # --- STATIC BASELINE ---
                c = g[g['ss'] > 0.082].sort_values('short_conv', ascending=False)
                if len(c)>0:
                    trade = c.iloc[0].copy()
                    trade['net_bps'] = -trade['Next_Hour_Return']*10000 - COST_BPS
                    st_shorts.append(trade)
                    
                # --- DYNAMIC PROBABILITY (DEFENSIVE - GLOBAL ONLY) ---
                dyn_prob_def = 0.082
                if sp500 > 0.005: dyn_prob_def = 0.110
                elif sp500 < -0.005: dyn_prob_def = 0.090
                if brent > 0.02: dyn_prob_def = max(dyn_prob_def, 0.095)
                
                c_def = g[g['ss'] > dyn_prob_def].sort_values('short_conv', ascending=False)
                if len(c_def)>0:
                    trade = c_def.iloc[0].copy()
                    trade['net_bps'] = -trade['Next_Hour_Return']*10000 - COST_BPS
                    dy_shorts_defensive.append(trade)
                    
                # --- DYNAMIC PROBABILITY (MULTIVARIATE - GLOBAL x LOCAL) ---
                dyn_prob_multi = 0.082
                
                global_risk_on = sp500 > 0.005
                local_weak = n2h < -0.0010
                
                if global_risk_on and not local_weak:
                    # Global rally, and India is NOT decoupled weak. Shorting is toxic.
                    dyn_prob_multi = 0.110
                
                c_multi = g[g['ss'] > dyn_prob_multi].sort_values('short_conv', ascending=False)
                if len(c_multi)>0:
                    trade = c_multi.iloc[0].copy()
                    trade['net_bps'] = -trade['Next_Hour_Return']*10000 - COST_BPS
                    dy_shorts_multi.append(trade)

    s1 = pd.DataFrame(st_shorts)
    s2 = pd.DataFrame(dy_shorts_defensive)
    s3 = pd.DataFrame(dy_shorts_multi)
    
    def rpt(df, name):
        if len(df) == 0: return f"{name}: 0 trades"
        return f"{name:25s} | Trades: {len(df):3d} | Win Rate: {(df['net_bps']>0).mean():.1%} | Avg Net BPS: {df['net_bps'].mean():.2f} | Total Net BPS: {df['net_bps'].sum():.2f}"
        
    print("\n--- RESULTS COMPARISON (11-MONTH OOS) ---")
    print(rpt(s1, "STATIC SHORTS (0.082)"))
    print(rpt(s2, "DYNAMIC (GLOBAL ONLY)"))
    print(rpt(s3, "DYNAMIC (MULTIVARIATE)"))
    
if __name__ == '__main__':
    main()

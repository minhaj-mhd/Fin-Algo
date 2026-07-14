import pandas as pd
import numpy as np
import json
import xgboost as xgb

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
        
        # Long Gate exactly as in simulate
        if (n2h > 0.0025 and nin > 0.0020):
            c = g[g['ls'] > 0.075].sort_values('long_conv', ascending=False)
            if len(c)>0:
                trade = c.iloc[0].copy()
                trade['net_bps'] = trade['Next_Hour_Return']*10000 - 6.0
                trade['sp500'] = g['sp500_ret_prev'].iloc[0]
                trade['n2h'] = n2h
                all_trades.append(trade)

    print(f"Total Long Trades Extracted: {len(all_trades)}")
    for t in all_trades:
        print(f"{t['DateTime']} | LS: {t['ls']:.4f} | Conv: {t['long_conv']:.4f} | SP500: {t['sp500']:>7.2%} | N2H: {t['n2h']:>7.2%} | BPS: {t['net_bps']:.1f}")

if __name__ == '__main__':
    main()

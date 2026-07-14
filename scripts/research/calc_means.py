import pandas as pd
import json, xgboost as xgb, numpy as np
import os

COST_BPS = 6.0

def main():
    # Load macro
    macro_files = {'brent': 'data/raw_global_daily/BRENT.parquet', 'sp500': 'data/raw_global_daily/SP500.parquet'}
    m_df = None
    for n, p in macro_files.items():
        d = pd.read_parquet(p)
        d[f'{n}_ret_prev'] = d['close'].pct_change().shift(1)
        d['date'] = d['timestamp'].dt.date
        d = d[['date', f'{n}_ret_prev']].dropna()
        m_df = d if m_df is None else pd.merge(m_df, d, on='date', how='outer')

    # Load nifty
    nifty = pd.read_csv('data/raw_index_cache/nifty50_15m.csv')
    nifty['ts'] = pd.to_datetime(nifty['ts']).dt.tz_localize(None)
    nifty['nifty_ret_2h'] = nifty['close'] / nifty['close'].shift(8) - 1
    nifty['date'] = nifty['ts'].dt.date
    do = nifty.groupby('date')['open'].first().reset_index().rename(columns={'open': 'd_open'})
    nifty = pd.merge(nifty, do, on='date')
    nifty['nifty_intraday'] = nifty['close'] / nifty['d_open'] - 1
    n2_map = dict(zip(nifty['ts'], nifty['nifty_ret_2h']))
    ni_map = dict(zip(nifty['ts'], nifty['nifty_intraday']))

    # Load panel
    df = pd.read_parquet('data/research/v20_rolling_1h/panel_backfilled.parquet')
    df['DateTime'] = pd.to_datetime(df['DateTime']).dt.tz_localize(None)
    df['date'] = df['DateTime'].dt.date
    df = pd.merge(df, m_df, on='date', how='left')
    df['nifty_2h'] = df['DateTime'].map(n2_map)
    df['nifty_in'] = df['DateTime'].map(ni_map)
    df = df.dropna(subset=['nifty_2h','nifty_in','sp500_ret_prev', 'brent_ret_prev'])

    time_mask = (df['DateTime'].dt.time >= pd.to_datetime('10:15').time()) & (df['DateTime'].dt.time <= pd.to_datetime('14:15').time())
    df = df[time_mask]

    feats = json.load(open('models/research/v20_rolling_1h/metadata.json'))['features']
    df = df.dropna(subset=feats + ['Next_Hour_Return'])
    X = xgb.DMatrix(np.nan_to_num(df[feats].values.astype(np.float32)), feature_names=feats)
    bl = xgb.Booster(); bl.load_model('models/research/v20_rolling_1h/xgb_long_model.json')
    bs = xgb.Booster(); bs.load_model('models/research/v20_rolling_1h/xgb_short_model.json')
    df['ls'] = bl.predict(X)
    df['ss'] = bs.predict(X)
    df['long_conv'] = (df['ls'] - df.groupby('DateTime')['ls'].transform('mean')) - (df['ss'] - df.groupby('DateTime')['ss'].transform('mean'))
    df['short_conv'] = (df['ss'] - df.groupby('DateTime')['ss'].transform('mean')) - (df['ls'] - df.groupby('DateTime')['ls'].transform('mean'))

    all_trades = []

    for ts, g in df.groupby('DateTime'):
        t_time = ts.time()
        n2h = g['nifty_2h'].iloc[0]
        nin = g['nifty_in'].iloc[0]
        sp500 = g['sp500_ret_prev'].iloc[0]
        
        # Long Gate
        if (n2h > 0.0025 and nin > 0.0020):
            c = g[g['ls'] > 0.035].sort_values('long_conv', ascending=False)
            if len(c)>0:
                trade = c.iloc[0].copy()
                trade['net_bps'] = trade['Next_Hour_Return']*10000 - COST_BPS
                trade['side'] = 'LONG'
                
                valid_long = False
                if sp500 < -0.005:
                    pass 
                elif sp500 > 0.005:
                    if n2h > 0.0070: valid_long = True
                else:
                    if n2h < 0.0040 or n2h > 0.0070: valid_long = True
                    
                if valid_long:
                    all_trades.append(trade)

        # Short Gate
        brent = g['brent_ret_prev'].iloc[0]
        dyn_prob = 0.082
        if sp500 > 0.005:
            if n2h > 0.0040:
                dyn_prob = 0.110
        if t_time < pd.to_datetime('11:30').time() or t_time > pd.to_datetime('13:00').time():
            c = g[g['ss'] > dyn_prob].sort_values('short_conv', ascending=False)
            if len(c)>0:
                trade = c.iloc[0].copy()
                trade['net_bps'] = -trade['Next_Hour_Return']*10000 - COST_BPS
                trade['side'] = 'SHORT'
                all_trades.append(trade)
                
    tdf = pd.DataFrame(all_trades).sort_values('DateTime')
    
    def calc_queue(trades):
        active_until = pd.NaT
        executed = []
        for i, row in trades.iterrows():
            if pd.isna(active_until) or row['DateTime'] >= active_until:
                executed.append(row)
                active_until = row['DateTime'] + pd.Timedelta(minutes=55)
        return pd.DataFrame(executed)
    
    longs = calc_queue(tdf[tdf['side'] == 'LONG'])
    shorts = calc_queue(tdf[tdf['side'] == 'SHORT'])
    
    print("--- LONGS ---")
    l_win = longs[longs['net_bps'] > 0]
    l_loss = longs[longs['net_bps'] <= 0]
    print(f"Wins: {len(l_win)} | Mean BPS: {l_win['net_bps'].mean():.2f}")
    print(f"Loss: {len(l_loss)} | Mean BPS: {l_loss['net_bps'].mean():.2f}")
    
    # Get biggest gainer and loser
    max_long = longs.loc[longs['net_bps'].idxmax()]
    min_long = longs.loc[longs['net_bps'].idxmin()]
    print(f"Biggest Winner: +{max_long['net_bps']:.2f} BPS (on {max_long['DateTime']})")
    print(f"Biggest Loser: {min_long['net_bps']:.2f} BPS (on {min_long['DateTime']})")
    
    print("\n--- SHORTS ---")
    s_win = shorts[shorts['net_bps'] > 0]
    s_loss = shorts[shorts['net_bps'] <= 0]
    print(f"Wins: {len(s_win)} | Mean BPS: {s_win['net_bps'].mean():.2f}")
    print(f"Loss: {len(s_loss)} | Mean BPS: {s_loss['net_bps'].mean():.2f}")
    
    max_short = shorts.loc[shorts['net_bps'].idxmax()]
    min_short = shorts.loc[shorts['net_bps'].idxmin()]
    print(f"Biggest Winner: +{max_short['net_bps']:.2f} BPS (on {max_short['DateTime']})")
    print(f"Biggest Loser: {min_short['net_bps']:.2f} BPS (on {min_short['DateTime']})")

if __name__ == '__main__':
    main()

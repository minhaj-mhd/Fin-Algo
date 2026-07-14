import pandas as pd
import json, xgboost as xgb, numpy as np

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
        brent = g['brent_ret_prev'].iloc[0]
        
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
        if (n2h <= 0.0025 or nin > 0.0036):
            if (t_time < pd.to_datetime('11:30').time() or t_time > pd.to_datetime('13:00').time()):
                dyn_prob_multi = 0.082
                if sp500 > 0.005 and n2h >= -0.0010:
                    dyn_prob_multi = 0.110
                
                c_multi = g[g['ss'] > dyn_prob_multi].sort_values('short_conv', ascending=False)
                if len(c_multi)>0:
                    trade = c_multi.iloc[0].copy()
                    trade['net_bps'] = -trade['Next_Hour_Return']*10000 - COST_BPS
                    trade['side'] = 'SHORT'
                    all_trades.append(trade)
                
    tdf = pd.DataFrame(all_trades).sort_values(['DateTime', 'side'], ascending=[True, False])
    
    # Sim Combined Queue
    active_until = pd.NaT
    executed = []
    for i, row in tdf.iterrows():
        if pd.isna(active_until) or row['DateTime'] >= active_until:
            executed.append(row)
            active_until = row['DateTime'] + pd.Timedelta(minutes=55)
    
    edf = pd.DataFrame(executed)
    edf['Month'] = edf['DateTime'].dt.to_period('M')
    edf['Week'] = edf['DateTime'].dt.to_period('W')
    
    print("==============================")
    print("      MONTHLY STATISTICS      ")
    print("==============================")
    monthly = edf.groupby('Month').agg(
        Trades=('net_bps', 'count'),
        Win_Rate=('net_bps', lambda x: (x>0).mean()*100),
        Total_BPS=('net_bps', 'sum'),
        Avg_BPS=('net_bps', 'mean')
    )
    print(monthly.round(2).to_string())
    
    print("\n==============================")
    print("   WEEKLY STATS (SUMMARY)     ")
    print("==============================")
    weekly = edf.groupby('Week').agg(
        Trades=('net_bps', 'count'),
        Total_BPS=('net_bps', 'sum'),
    )
    print(f"Total Weeks Traded: {len(weekly)}")
    print(f"Positive Weeks: {len(weekly[weekly['Total_BPS'] > 0])} ({len(weekly[weekly['Total_BPS'] > 0])/len(weekly)*100:.1f}%)")
    print(f"Negative Weeks: {len(weekly[weekly['Total_BPS'] <= 0])} ({len(weekly[weekly['Total_BPS'] <= 0])/len(weekly)*100:.1f}%)")
    print(f"Best Week: {weekly['Total_BPS'].max():.2f} BPS")
    print(f"Worst Week: {weekly['Total_BPS'].min():.2f} BPS")
    print(f"Average Weekly Trades: {weekly['Trades'].mean():.1f}")
    print(f"Average Weekly BPS: {weekly['Total_BPS'].mean():.2f}")

if __name__ == '__main__':
    main()

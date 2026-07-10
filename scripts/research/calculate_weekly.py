import pandas as pd
import json
import xgboost as xgb
import numpy as np
from datetime import time, date

COST_BPS = 6.0

def main():
    print("Loading data...")
    nifty = pd.read_csv('data/raw_index_cache/nifty50_15m.csv')
    nifty['ts'] = pd.to_datetime(nifty['ts'])
    if nifty['ts'].dt.tz is not None:
        nifty['ts'] = nifty['ts'].dt.tz_localize(None)
    
    nifty = nifty.sort_values('ts').reset_index(drop=True)
    nifty['nifty_ret_2h'] = nifty['close'] / nifty['close'].shift(8) - 1
    
    # Calculate Intraday Return
    nifty['date'] = nifty['ts'].dt.date
    daily_open = nifty.groupby('date')['open'].first().reset_index()
    daily_open.rename(columns={'open': 'daily_open'}, inplace=True)
    nifty = pd.merge(nifty, daily_open, on='date', how='left')
    nifty['nifty_intraday'] = nifty['close'] / nifty['daily_open'] - 1
    
    nifty_map = dict(zip(nifty['ts'], nifty['nifty_ret_2h']))
    nifty_intra_map = dict(zip(nifty['ts'], nifty['nifty_intraday']))

    df = pd.read_parquet('data/research/v20_rolling_1h/panel_backfilled.parquet')
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    if df['DateTime'].dt.tz is not None:
        df['DateTime'] = df['DateTime'].dt.tz_localize(None)
    df = df[df['DateTime'].dt.date >= date(2025, 8, 1)]
    
    time_mask = (df['DateTime'].dt.time >= time(10, 15)) & (df['DateTime'].dt.time <= time(14, 15))
    df = df[time_mask]
    
    df['nifty_ret_2h'] = df['DateTime'].map(nifty_map)
    df['nifty_intraday'] = df['DateTime'].map(nifty_intra_map)
    df = df.dropna(subset=['nifty_ret_2h', 'nifty_intraday'])

    v20_feats = json.load(open('models/research/v20_rolling_1h/metadata.json'))['features']
    df = df.dropna(subset=v20_feats + ['Next_Hour_Return'])
    X = xgb.DMatrix(np.nan_to_num(df[v20_feats].values.astype(np.float32)), feature_names=v20_feats)

    bs = xgb.Booster()
    bs.load_model('models/research/v20_rolling_1h/xgb_short_model.json')
    bl = xgb.Booster()
    bl.load_model('models/research/v20_rolling_1h/xgb_long_model.json')

    df['ss'] = bs.predict(X)
    df['ls'] = bl.predict(X)

    ss_mean = df.groupby('DateTime')['ss'].transform('mean')
    ls_mean = df.groupby('DateTime')['ls'].transform('mean')
    df['short_conviction'] = (df['ss'] - ss_mean) - (df['ls'] - ls_mean)
    df['long_conviction'] = (df['ls'] - ls_mean) - (df['ss'] - ss_mean)

    trades = []
    for ts, g in df.groupby('DateTime'):
        nifty_2h = g['nifty_ret_2h'].iloc[0]
        nifty_intraday = g['nifty_intraday'].iloc[0]
        
        if nifty_2h <= 0.0025 or nifty_intraday > 0.0036:
            t = ts.time()
            if t < time(11, 30) or t > time(13, 0):
                cands = g[g['ss'] > 0.082].sort_values('short_conviction', ascending=False)
                if len(cands) > 0:
                    p = cands.iloc[0]
                    trades.append((ts, 'SHORT', p['Ticker'], -p['Next_Hour_Return'] * 10000))
            
        if nifty_2h > 0.0025 and nifty_intraday > 0.0020:
            cands = g.sort_values('long_conviction', ascending=False)
            if len(cands) > 0:
                p = cands.iloc[0]
                trades.append((ts, 'LONG', p['Ticker'], p['Next_Hour_Return'] * 10000))
                
    td = pd.DataFrame(trades, columns=['DateTime', 'Side', 'Ticker', 'Gross_BPS'])
    td['Net_BPS'] = td['Gross_BPS'] - COST_BPS
    td['PnL'] = td['Net_BPS'] * 50  # 5L notional * 5x leverage / 10000 = 50 per BPS approx? Wait, previous code used:
    # `subset['net_bps'].sum() * 50` so 1 BPS = 50 Rs.
    
    td['Week'] = td['DateTime'].dt.strftime('%Y-W%V')
    
    weekly = td.groupby('Week').agg({
        'DateTime': 'count',
        'Net_BPS': 'sum',
        'PnL': 'sum'
    }).rename(columns={'DateTime': 'Trades'})
    
    print("\n=== Weekly Returns ===")
    print(weekly.to_string())
    
    print(f"\nTotal Weeks: {len(weekly)}")
    print(f"Profitable Weeks: {(weekly['PnL'] > 0).sum()} ({(weekly['PnL'] > 0).mean():.1%})")
    print(f"Max Weekly Drawdown: Rs. {weekly['PnL'].min():,.0f}")
    print(f"Max Weekly Profit: Rs. {weekly['PnL'].max():,.0f}")
    print(f"Average Weekly PnL: Rs. {weekly['PnL'].mean():,.0f}")

if __name__ == '__main__':
    main()

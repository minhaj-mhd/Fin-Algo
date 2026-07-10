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
    
    # Calculate Nifty features
    nifty['nifty_ret_2h'] = nifty['close'] / nifty['close'].shift(8) - 1
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
        
    # df = df[(df['DateTime'].dt.date >= date(2026, 4, 1)) & (df['DateTime'].dt.date <= date(2026, 5, 31))]
    
    time_mask = (df['DateTime'].dt.time >= time(10, 15)) & (df['DateTime'].dt.time <= time(14, 15))
    df = df[time_mask]
    
    df['nifty_ret_2h'] = df['DateTime'].map(nifty_map)
    df['nifty_intraday'] = df['DateTime'].map(nifty_intra_map)
    df = df.dropna(subset=['nifty_ret_2h', 'nifty_intraday'])

    print("Loading models...")
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

    print("Extracting short trades for April/May under current regime gates...")
    short_cands = []
    
    for ts, g in df.groupby('DateTime'):
        nifty_2h = g['nifty_ret_2h'].iloc[0]
        nifty_intraday = g['nifty_intraday'].iloc[0]
        
        # Apply the current short gate
        if nifty_2h <= 0.0025 or nifty_intraday > 0.0036:
            cands = g[g['ss'] > 0.082].sort_values('short_conviction', ascending=False)
            if len(cands) > 0:
                p = cands.iloc[0].copy()
                p['net_bps'] = -p['Next_Hour_Return'] * 10000 - COST_BPS
                p['is_win'] = p['net_bps'] > 0
                short_cands.append(p)
            
    sdf = pd.DataFrame(short_cands)
    print(f"Total Short Trades in Apr/May: {len(sdf)} | Win: {sdf['is_win'].mean():.1%} | BPS: {sdf['net_bps'].mean():.2f}")
    
    # Feature comparison for Wins vs Losses in April/May
    print("\nFeature means for April/May Short Trades (Wins vs Losses):")
    win_means = sdf[sdf.is_win][v20_feats + ['nifty_ret_2h', 'nifty_intraday']].mean()
    loss_means = sdf[~sdf.is_win][v20_feats + ['nifty_ret_2h', 'nifty_intraday']].mean()
    
    diff = (win_means - loss_means).abs()
    diff = diff.sort_values(ascending=False)
    
    for feat in diff.index[:20]:
        print(f"{feat:30s} | Win: {win_means[feat]:>8.4f} | Loss: {loss_means[feat]:>8.4f} | Diff: {diff[feat]:>8.4f}")

    print("\nTime of Day Analysis:")
    sdf['time'] = sdf['DateTime'].dt.time
    for t, group in sdf.groupby('time'):
        print(f"{t}: {len(group):3d} trades, Win: {group['is_win'].mean():.1%}, BPS: {group['net_bps'].mean():.2f}")


if __name__ == '__main__':
    main()

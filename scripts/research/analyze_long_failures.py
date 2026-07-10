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

    print("Loading models...")
    v20_feats = json.load(open('models/research/v20_rolling_1h/metadata.json'))['features']
    df = df.dropna(subset=v20_feats + ['Next_Hour_Return'])
    X = xgb.DMatrix(np.nan_to_num(df[v20_feats].values.astype(np.float32)), feature_names=v20_feats)

    bl = xgb.Booster()
    bl.load_model('models/research/v20_rolling_1h/xgb_long_model.json')
    bs = xgb.Booster()
    bs.load_model('models/research/v20_rolling_1h/xgb_short_model.json')
    
    df['ls'] = bl.predict(X)
    df['ss'] = bs.predict(X)

    ls_mean = df.groupby('DateTime')['ls'].transform('mean')
    ss_mean = df.groupby('DateTime')['ss'].transform('mean')
    df['long_conviction'] = (df['ls'] - ls_mean) - (df['ss'] - ss_mean)

    print("Extracting current Long trades...")
    long_trades = []
    
    for ts, g in df.groupby('DateTime'):
        nifty_2h = g['nifty_ret_2h'].iloc[0]
        nifty_intraday = g['nifty_intraday'].iloc[0]
        
        # Apply the current long gate
        if nifty_2h > 0.0025:
            cands = g.sort_values('long_conviction', ascending=False)
            if len(cands) > 0:
                p = cands.iloc[0].copy()
                p['net_bps'] = p['Next_Hour_Return'] * 10000 - COST_BPS
                p['is_win'] = p['net_bps'] > 0
                long_trades.append(p)
            
    ldf = pd.DataFrame(long_trades)
    print(f"Total Long Trades: {len(ldf)} | Win: {ldf['is_win'].mean():.1%} | BPS: {ldf['net_bps'].mean():.2f}")
    
    print("\n--- 1. Absolute Long Score (ls) Threshold ---")
    for thresh in [0.0, 0.02, 0.04, 0.06, 0.08, 0.082, 0.1]:
        subset = ldf[ldf['ls'] > thresh]
        if len(subset) > 0:
            print(f"If ls > {thresh:.3f}: {len(subset):3d} trades | Win: {subset['is_win'].mean():.1%} | BPS: {subset['net_bps'].mean():.2f}")

    print("\n--- 2. Long Conviction Score Threshold ---")
    for q in [0, 0.25, 0.5, 0.75, 0.9]:
        thresh = ldf['long_conviction'].quantile(q)
        subset = ldf[ldf['long_conviction'] > thresh]
        print(f"Top {100-q*100:.0f}% (lc > {thresh:.3f}): {len(subset):3d} trades | Win: {subset['is_win'].mean():.1%} | BPS: {subset['net_bps'].mean():.2f}")

    print("\n--- 3. Time of Day Analysis ---")
    ldf['time'] = ldf['DateTime'].dt.time
    for t, group in ldf.groupby('time'):
        print(f"{t}: {len(group):3d} trades, Win: {group['is_win'].mean():.1%}, BPS: {group['net_bps'].mean():.2f}")

    print("\n--- 4. Intraday Nifty Extension (`nifty_intraday`) ---")
    try:
        ldf['intra_bin'] = pd.qcut(ldf['nifty_intraday'], q=5)
        for name, group in ldf.groupby('intra_bin'):
            print(f"Nifty Intraday {name}: {len(group):3d} trades | Win: {group['is_win'].mean():.1%} | BPS: {group['net_bps'].mean():.2f}")
    except Exception as e:
        print("Binning failed:", e)

    # Feature comparison for Wins vs Losses
    print("\n--- 5. Feature means for Long Trades (Wins vs Losses) ---")
    win_means = ldf[ldf.is_win][v20_feats + ['nifty_ret_2h', 'nifty_intraday', 'ls', 'long_conviction']].mean()
    loss_means = ldf[~ldf.is_win][v20_feats + ['nifty_ret_2h', 'nifty_intraday', 'ls', 'long_conviction']].mean()
    
    diff = (win_means - loss_means).abs()
    diff = diff.sort_values(ascending=False)
    
    for feat in diff.index[:15]:
        print(f"{feat:30s} | Win: {win_means[feat]:>8.4f} | Loss: {loss_means[feat]:>8.4f} | Diff: {diff[feat]:>8.4f}")


if __name__ == '__main__':
    main()

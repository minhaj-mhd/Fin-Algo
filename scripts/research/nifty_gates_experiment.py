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
    nifty['nifty_ret_1d'] = nifty['close'] / nifty['close'].shift(25) - 1  # roughly 1 day of 15m bars
    nifty['nifty_sma_20'] = nifty['close'].rolling(20).mean()
    nifty['nifty_dist_sma20'] = nifty['close'] / nifty['nifty_sma_20'] - 1
    nifty['nifty_vol_20'] = nifty['close'].pct_change().rolling(20).std()
    
    # Calculate intraday trend (from today's open)
    nifty['date'] = nifty['ts'].dt.date
    daily_open = nifty.groupby('date')['open'].first().reset_index()
    daily_open.rename(columns={'open': 'daily_open'}, inplace=True)
    nifty = pd.merge(nifty, daily_open, on='date', how='left')
    nifty['nifty_intraday'] = nifty['close'] / nifty['daily_open'] - 1
    
    # Create maps
    feats = ['nifty_ret_2h', 'nifty_ret_1d', 'nifty_dist_sma20', 'nifty_vol_20', 'nifty_intraday']
    maps = {f: dict(zip(nifty['ts'], nifty[f])) for f in feats}

    df = pd.read_parquet('data/research/v20_rolling_1h/panel_backfilled.parquet')
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    if df['DateTime'].dt.tz is not None:
        df['DateTime'] = df['DateTime'].dt.tz_localize(None)
    df = df[df['DateTime'].dt.date >= date(2025, 8, 1)]
    
    time_mask = (df['DateTime'].dt.time >= time(10, 15)) & (df['DateTime'].dt.time <= time(14, 15))
    df = df[time_mask]
    
    for f in feats:
        df[f] = df['DateTime'].map(maps[f])
    
    df = df.dropna(subset=feats)

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
    df['long_conviction'] = (df['ls'] - ls_mean) - (df['ss'] - ss_mean)

    print("Extracting ALL short and long candidates...")
    short_cands = []
    long_cands = []
    
    for ts, g in df.groupby('DateTime'):
        # Just grab the top short and top long regardless of gate
        sc = g[g['ss'] > 0.082].sort_values('short_conviction', ascending=False)
        if len(sc) > 0:
            p = sc.iloc[0].copy()
            p['net_bps'] = -p['Next_Hour_Return'] * 10000 - COST_BPS
            short_cands.append(p)
            
        lc = g.sort_values('long_conviction', ascending=False)
        if len(lc) > 0: # Note: Long currently doesn't have an ls>x threshold in the combined book except Nifty, but let's just evaluate the top 1
            p = lc.iloc[0].copy()
            p['net_bps'] = p['Next_Hour_Return'] * 10000 - COST_BPS
            long_cands.append(p)
            
    sdf = pd.DataFrame(short_cands)
    ldf = pd.DataFrame(long_cands)
    
    print(f"Total Short Cands: {len(sdf)} | Win: {(sdf['net_bps']>0).mean():.1%} | BPS: {sdf['net_bps'].mean():.2f}")
    print(f"Total Long Cands: {len(ldf)} | Win: {(ldf['net_bps']>0).mean():.1%} | BPS: {ldf['net_bps'].mean():.2f}")
    
    print("\n--- SHORT REGIME GATES ---")
    for feat in feats:
        print(f"\nAnalyzing Short Edge across: {feat}")
        try:
            sdf['bin'] = pd.qcut(sdf[feat], q=5)
            for name, group in sdf.groupby('bin'):
                print(f"Bin {name}: {len(group):3d} trades, Win: {(group['net_bps']>0).mean():.1%}, BPS: {group['net_bps'].mean():.2f}")
        except:
            print("Failed to bin.")

    print("\n--- LONG REGIME GATES ---")
    for feat in feats:
        print(f"\nAnalyzing Long Edge across: {feat}")
        try:
            ldf['bin'] = pd.qcut(ldf[feat], q=5)
            for name, group in ldf.groupby('bin'):
                print(f"Bin {name}: {len(group):3d} trades, Win: {(group['net_bps']>0).mean():.1%}, BPS: {group['net_bps'].mean():.2f}")
        except:
            print("Failed to bin.")

if __name__ == '__main__':
    main()

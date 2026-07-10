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
    nifty_map = dict(zip(nifty['ts'], nifty['nifty_ret_2h']))

    df = pd.read_parquet('data/research/v20_rolling_1h/panel_backfilled.parquet')
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    if df['DateTime'].dt.tz is not None:
        df['DateTime'] = df['DateTime'].dt.tz_localize(None)
    df = df[df['DateTime'].dt.date >= date(2025, 8, 1)]
    
    # Filter only 10:15 - 14:15
    time_mask = (df['DateTime'].dt.time >= time(10, 15)) & (df['DateTime'].dt.time <= time(14, 15))
    df = df[time_mask]
    df['nifty_ret_2h'] = df['DateTime'].map(nifty_map)
    df = df.dropna(subset=['nifty_ret_2h'])

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

    print("Extracting short trades...")
    trades_df = []
    for ts, g in df.groupby('DateTime'):
        cands = g[g['ss'] > 0.082].sort_values('short_conviction', ascending=False)
        if len(cands) > 0:
            p = cands.iloc[0]
            # Capture features for analysis
            trade_data = p[v20_feats].to_dict()
            trade_data['ts'] = ts
            trade_data['Ticker'] = p['Ticker']
            trade_data['net_bps'] = -p['Next_Hour_Return'] * 10000 - COST_BPS
            trade_data['nifty_ret_2h'] = p['nifty_ret_2h']
            trades_df.append(trade_data)
            
    td = pd.DataFrame(trades_df)
    td['is_win'] = td['net_bps'] > 0
    td['post_feb'] = td['ts'].dt.date >= date(2026, 2, 1)

    print(f"Total short trades: {len(td)}")
    print(f"Pre-Feb trades: {len(td[~td.post_feb])}, Win rate: {td[~td.post_feb]['is_win'].mean():.1%}, Avg BPS: {td[~td.post_feb]['net_bps'].mean():.2f}")
    print(f"Post-Feb trades: {len(td[td.post_feb])}, Win rate: {td[td.post_feb]['is_win'].mean():.1%}, Avg BPS: {td[td.post_feb]['net_bps'].mean():.2f}")

    # Analyze features for Post-Feb losing trades vs winning trades
    post_df = td[td.post_feb]
    
    print("\nFeature means for Post-Feb Short Trades (Wins vs Losses):")
    # Compare means of features
    win_means = post_df[post_df.is_win][v20_feats + ['nifty_ret_2h']].mean()
    loss_means = post_df[~post_df.is_win][v20_feats + ['nifty_ret_2h']].mean()
    
    diff = (win_means - loss_means).abs()
    diff = diff.sort_values(ascending=False)
    
    for feat in diff.index[:15]:
        print(f"{feat:30s} | Win: {win_means[feat]:>8.4f} | Loss: {loss_means[feat]:>8.4f} | Diff: {diff[feat]:>8.4f}")

    # Also let's check Nifty regime correlation
    print("\nNifty Regime analysis on ALL short trades:")
    for thresh in [-0.005, -0.0025, 0.0, 0.0025, 0.005]:
        subset = td[td['nifty_ret_2h'] > thresh]
        if len(subset) > 0:
            print(f"If Nifty_2h > {thresh:>7.4f}: {len(subset):3d} trades, Win: {subset['is_win'].mean():.1%}, BPS: {subset['net_bps'].mean():.2f}")
            
        subset2 = td[td['nifty_ret_2h'] < thresh]
        if len(subset2) > 0:
            print(f"If Nifty_2h < {thresh:>7.4f}: {len(subset2):3d} trades, Win: {subset2['is_win'].mean():.1%}, BPS: {subset2['net_bps'].mean():.2f}")

    print("\nNifty Regime analysis on PRE-Feb short trades:")
    pre_df = td[~td.post_feb]
    for thresh in [-0.005, -0.0025, 0.0, 0.0025, 0.005]:
        subset = pre_df[pre_df['nifty_ret_2h'] < thresh]
        if len(subset) > 0:
            print(f"If Nifty_2h < {thresh:>7.4f}: {len(subset):3d} trades, Win: {subset['is_win'].mean():.1%}, BPS: {subset['net_bps'].mean():.2f}")


if __name__ == '__main__':
    main()

import pandas as pd
import json
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from datetime import time, date
import os

COST_BPS = 6.0
NOTIONAL = 500_000.0

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

    print("Loading models...")
    v20_feats = json.load(open('models/research/v20_rolling_1h/metadata.json'))['features']
    df = df.dropna(subset=v20_feats + ['Next_Hour_Return'])
    X = xgb.DMatrix(np.nan_to_num(df[v20_feats].values.astype(np.float32)), feature_names=v20_feats)

    bs = xgb.Booster()
    bs.load_model('models/research/v20_rolling_1h/xgb_short_model.json')
    bl = xgb.Booster()
    bl.load_model('models/research/v20_rolling_1h/xgb_long_model.json')

    print("Predicting...")
    df['ss'] = bs.predict(X)
    df['ls'] = bl.predict(X)

    ss_mean = df.groupby('DateTime')['ss'].transform('mean')
    ls_mean = df.groupby('DateTime')['ls'].transform('mean')
    df['short_conviction'] = (df['ss'] - ss_mean) - (df['ls'] - ls_mean)
    df['long_conviction'] = (df['ls'] - ls_mean) - (df['ss'] - ss_mean)

    print("Building book...")
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
                    trades.append((ts, 'SHORT', -p['Next_Hour_Return'] * 10000))
                
        if nifty_2h > 0.0025 and nifty_intraday > 0.0020:
            cands = g.sort_values('long_conviction', ascending=False)
            if len(cands) > 0:
                p = cands.iloc[0]
                trades.append((ts, 'LONG', p['Next_Hour_Return'] * 10000))

    td = pd.DataFrame(trades, columns=['ts', 'side', 'gross_bps'])
    td['ts'] = pd.to_datetime(td['ts'])
    td['net_bps'] = td.gross_bps - COST_BPS
    td['bookRs'] = td.net_bps / 10000 * NOTIONAL
    td['date'] = td['ts'].dt.date

    td = td.sort_values('ts')
    
    # Calculate Daily PnLs for Long, Short, and Combined
    daily_combined = td.groupby('date')['bookRs'].sum().reset_index().rename(columns={'bookRs': 'pnl_combined'})
    daily_long = td[td.side == 'LONG'].groupby('date')['bookRs'].sum().reset_index().rename(columns={'bookRs': 'pnl_long'})
    daily_short = td[td.side == 'SHORT'].groupby('date')['bookRs'].sum().reset_index().rename(columns={'bookRs': 'pnl_short'})
    
    # Merge all into one daily dataframe
    all_dates = pd.DataFrame({'date': pd.to_datetime(td['date']).dt.date.unique()})
    daily_pnl = all_dates.merge(daily_combined, on='date', how='left')\
                         .merge(daily_long, on='date', how='left')\
                         .merge(daily_short, on='date', how='left').fillna(0)
                         
    daily_pnl = daily_pnl.sort_values('date')
    
    # Cumulative PnL
    daily_pnl['cum_combined'] = daily_pnl['pnl_combined'].cumsum()
    daily_pnl['cum_long'] = daily_pnl['pnl_long'].cumsum()
    daily_pnl['cum_short'] = daily_pnl['pnl_short'].cumsum()
    
    # Drawdowns
    daily_pnl['dd_combined'] = daily_pnl['cum_combined'] - daily_pnl['cum_combined'].cummax()
    daily_pnl['dd_long'] = daily_pnl['cum_long'] - daily_pnl['cum_long'].cummax()
    daily_pnl['dd_short'] = daily_pnl['cum_short'] - daily_pnl['cum_short'].cummax()

    # Group Nifty by date for plotting (using daily close)
    nifty_daily = nifty[nifty['ts'].dt.date >= date(2025, 8, 1)].groupby(nifty['ts'].dt.date)['close'].last().reset_index()
    nifty_daily.rename(columns={'ts': 'date'}, inplace=True)
    
    # Merge daily PnL with Nifty daily close
    daily_pnl = pd.merge(daily_pnl, nifty_daily, on='date', how='left')

    print("Plotting...")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 14), gridspec_kw={'height_ratios': [2.5, 1.5, 1]}, sharex=True)
    
    dates = pd.to_datetime(daily_pnl['date'])
    
    # Subplot 1: Cumulative PnL
    ax1.plot(dates, daily_pnl['cum_combined'], color='black', linewidth=2.5, label='Combined PnL')
    ax1.plot(dates, daily_pnl['cum_long'], color='green', linewidth=1.5, linestyle='--', label='Long Leg PnL')
    ax1.plot(dates, daily_pnl['cum_short'], color='red', linewidth=1.5, linestyle='--', label='Short Leg PnL')
    ax1.fill_between(dates, daily_pnl['cum_combined'], color='black', alpha=0.05)
    ax1.set_title('Cumulative PnL Breakdown vs NIFTY 50', fontsize=16)
    ax1.set_ylabel('Cumulative PnL (Rs)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper left', fontsize=11)

    # Subplot 2: Drawdown
    ax2.fill_between(dates, daily_pnl['dd_combined'], color='black', alpha=0.2, label='Combined Drawdown')
    ax2.plot(dates, daily_pnl['dd_combined'], color='black', linewidth=1.5)
    ax2.plot(dates, daily_pnl['dd_long'], color='green', linewidth=1, alpha=0.6, label='Long Drawdown')
    ax2.plot(dates, daily_pnl['dd_short'], color='red', linewidth=1, alpha=0.6, label='Short Drawdown')
    ax2.set_ylabel('Drawdown (Rs)', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='lower left', fontsize=11)
    
    # Subplot 3: NIFTY 50
    ax3.plot(dates, daily_pnl['close'], color='blue', linewidth=1.5, label='NIFTY 50 Close')
    ax3.set_ylabel('NIFTY 50', fontsize=12)
    ax3.set_xlabel('Date', fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend(loc='upper left', fontsize=11)
    
    plt.tight_layout()
    output_path = r"C:\Users\loq\.gemini\antigravity\brain\5da8b7a2-8ac8-435c-9e17-cf5e9031f4bc\drawdown_nifty_plot.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")

if __name__ == '__main__':
    main()

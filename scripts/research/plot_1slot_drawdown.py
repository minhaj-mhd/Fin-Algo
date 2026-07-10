import pandas as pd
import json
import xgboost as xgb
import numpy as np
from datetime import time, date
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

COST_BPS = 6.0
NOTIONAL = 500_000.0
STARTING_CAPITAL = 100_000.0

def main():
    print("Loading data for drawdown analysis...")
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
    slot_free_at = pd.Timestamp.min
    
    for ts, g in df.groupby('DateTime'):
        if ts < slot_free_at: continue
            
        nifty_2h = g['nifty_ret_2h'].iloc[0]
        nifty_intraday = g['nifty_intraday'].iloc[0]
        t = ts.time()
        
        valid_short = False
        valid_long = False
        best_short = None
        best_long = None
        
        if nifty_2h <= 0.0025 or nifty_intraday > 0.0036:
            if t < time(11, 30) or t > time(13, 0):
                cands = g[g['ss'] > 0.082].sort_values('short_conviction', ascending=False)
                if len(cands) > 0:
                    best_short = cands.iloc[0]
                    valid_short = True
                    
        if nifty_2h > 0.0025 and nifty_intraday > 0.0020:
            cands = g.sort_values('long_conviction', ascending=False)
            if len(cands) > 0:
                best_long = cands.iloc[0]
                valid_long = True

        trade_taken = False
        if valid_short and valid_long:
            p = best_short
            trades.append((ts, 'SHORT', p['Ticker'], -p['Next_Hour_Return'] * 10000))
            trade_taken = True
        elif valid_short:
            p = best_short
            trades.append((ts, 'SHORT', p['Ticker'], -p['Next_Hour_Return'] * 10000))
            trade_taken = True
        elif valid_long:
            p = best_long
            trades.append((ts, 'LONG', p['Ticker'], p['Next_Hour_Return'] * 10000))
            trade_taken = True
            
        if trade_taken:
            slot_free_at = ts + pd.Timedelta(hours=1)

    td = pd.DataFrame(trades, columns=['ts', 'side', 'tk', 'gross_bps'])
    td['ts'] = pd.to_datetime(td['ts'])
    td = td.sort_values('ts').reset_index(drop=True)
    td['net_bps'] = td.gross_bps - COST_BPS
    td['pnl_rs'] = td.net_bps / 10000 * NOTIONAL
    
    # Calculate account equity and drawdown
    td['cumulative_pnl'] = td['pnl_rs'].cumsum()
    td['equity'] = STARTING_CAPITAL + td['cumulative_pnl']
    
    td['peak_equity'] = td['equity'].cummax()
    td['drawdown_rs'] = td['equity'] - td['peak_equity']
    td['drawdown_pct'] = (td['drawdown_rs'] / td['peak_equity']) * 100

    max_dd_rs = td['drawdown_rs'].min()
    max_dd_pct = td['drawdown_pct'].min()
    final_equity = td['equity'].iloc[-1]
    total_ret_pct = (final_equity - STARTING_CAPITAL) / STARTING_CAPITAL * 100
    
    print(f"Final Equity: {final_equity:,.0f} | Return: {total_ret_pct:.1f}%")
    print(f"Max Drawdown: {max_dd_rs:,.0f} Rs ({max_dd_pct:.1f}%)")
    
    # PLOTTING
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    ax1.plot(td['ts'], td['equity'], color='#1f77b4', linewidth=2, label='Portfolio Equity')
    ax1.fill_between(td['ts'], STARTING_CAPITAL, td['equity'], where=(td['equity'] >= STARTING_CAPITAL), interpolate=True, color='#1f77b4', alpha=0.1)
    ax1.fill_between(td['ts'], STARTING_CAPITAL, td['equity'], where=(td['equity'] < STARTING_CAPITAL), interpolate=True, color='red', alpha=0.1)
    ax1.axhline(STARTING_CAPITAL, color='black', linestyle='--', alpha=0.5)
    ax1.set_title(f'1-Slot Capital Growth (Starting Capital: ₹1L, Margin: 5x)', fontsize=14, pad=15)
    ax1.set_ylabel('Account Equity (₹)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Highlight max drawdown point
    mdd_idx = td['drawdown_rs'].idxmin()
    mdd_date = td['ts'].iloc[mdd_idx]
    ax1.scatter(mdd_date, td['equity'].iloc[mdd_idx], color='red', s=50, zorder=5)
    ax1.annotate(f'Max DD: ₹{max_dd_rs:,.0f}\n({max_dd_pct:.1f}%)',
                 xy=(mdd_date, td['equity'].iloc[mdd_idx]),
                 xytext=(10, -40), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', color='red'), color='red')
                 
    # Drawdown plot
    ax2.fill_between(td['ts'], 0, td['drawdown_pct'], color='red', alpha=0.3)
    ax2.plot(td['ts'], td['drawdown_pct'], color='red', linewidth=1)
    ax2.set_title('Drawdown (%)', fontsize=12)
    ax2.set_ylabel('Drawdown %', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis dates
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax.get_xticklabels(), rotation=45)
        
    plt.tight_layout()
    save_path = r'C:\Users\loq\.gemini\antigravity\brain\5da8b7a2-8ac8-435c-9e17-cf5e9031f4bc\1slot_realistic_drawdown.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    
    # Save statistics for AI to report
    stats = {
        'total_trades': len(td),
        'start_cap': STARTING_CAPITAL,
        'final_cap': final_equity,
        'ret_pct': total_ret_pct,
        'max_dd_rs': max_dd_rs,
        'max_dd_pct': max_dd_pct,
        'ret_to_dd': abs(total_ret_pct / max_dd_pct) if max_dd_pct != 0 else float('inf')
    }
    with open(r'C:\Users\loq\.gemini\antigravity\brain\5da8b7a2-8ac8-435c-9e17-cf5e9031f4bc\1slot_stats.json', 'w') as f:
        json.dump(stats, f)

if __name__ == '__main__':
    main()

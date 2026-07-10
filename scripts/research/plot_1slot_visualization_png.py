import pandas as pd
import json
import xgboost as xgb
import numpy as np
from datetime import time, date, datetime
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

COST_BPS = 6.0
NOTIONAL = 500_000.0
STARTING_CAPITAL = 100_000.0

def main():
    print("Loading data for PNG visualization...")
    nifty = pd.read_csv('data/raw_index_cache/nifty50_15m.csv')
    nifty['ts'] = pd.to_datetime(nifty['ts'])
    if nifty['ts'].dt.tz is not None: nifty['ts'] = nifty['ts'].dt.tz_localize(None)
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
    if df['DateTime'].dt.tz is not None: df['DateTime'] = df['DateTime'].dt.tz_localize(None)
    df = df[df['DateTime'].dt.date >= date(2025, 8, 1)]
    time_mask = (df['DateTime'].dt.time >= time(10, 15)) & (df['DateTime'].dt.time <= time(14, 15))
    df = df[time_mask]
    
    df['nifty_ret_2h'] = df['DateTime'].map(nifty_map)
    df['nifty_intraday'] = df['DateTime'].map(nifty_intra_map)
    df = df.dropna(subset=['nifty_ret_2h', 'nifty_intraday'])

    v20_feats = json.load(open('models/research/v20_rolling_1h/metadata.json'))['features']
    df = df.dropna(subset=v20_feats + ['Next_Hour_Return'])
    X = xgb.DMatrix(np.nan_to_num(df[v20_feats].values.astype(np.float32)), feature_names=v20_feats)

    bs, bl = xgb.Booster(), xgb.Booster()
    bs.load_model('models/research/v20_rolling_1h/xgb_short_model.json')
    bl.load_model('models/research/v20_rolling_1h/xgb_long_model.json')

    df['ss'], df['ls'] = bs.predict(X), bl.predict(X)
    ss_mean = df.groupby('DateTime')['ss'].transform('mean')
    ls_mean = df.groupby('DateTime')['ls'].transform('mean')
    df['short_conviction'] = (df['ss'] - ss_mean) - (df['ls'] - ls_mean)
    df['long_conviction'] = (df['ls'] - ls_mean) - (df['ss'] - ss_mean)

    trades = []
    slot_free_at = pd.Timestamp.min
    
    for ts, g in df.groupby('DateTime'):
        if ts < slot_free_at: continue
        nifty_2h, nifty_intraday, t = g['nifty_ret_2h'].iloc[0], g['nifty_intraday'].iloc[0], ts.time()
        
        valid_short, valid_long = False, False
        best_short, best_long = None, None
        
        if nifty_2h <= 0.0025 or nifty_intraday > 0.0036:
            if t < time(11, 30) or t > time(13, 0):
                cands = g[g['ss'] > 0.082].sort_values('short_conviction', ascending=False)
                if len(cands) > 0: best_short, valid_short = cands.iloc[0], True
                    
        if nifty_2h > 0.0025 and nifty_intraday > 0.0020:
            cands = g.sort_values('long_conviction', ascending=False)
            if len(cands) > 0: best_long, valid_long = cands.iloc[0], True

        if valid_short and valid_long:
            trades.append((ts, 'SHORT', best_short['Ticker'], -best_short['Next_Hour_Return'] * 10000))
            slot_free_at = ts + pd.Timedelta(hours=1)
        elif valid_short:
            trades.append((ts, 'SHORT', best_short['Ticker'], -best_short['Next_Hour_Return'] * 10000))
            slot_free_at = ts + pd.Timedelta(hours=1)
        elif valid_long:
            trades.append((ts, 'LONG', best_long['Ticker'], best_long['Next_Hour_Return'] * 10000))
            slot_free_at = ts + pd.Timedelta(hours=1)

    td = pd.DataFrame(trades, columns=['ts', 'side', 'tk', 'gross_bps'])
    td['ts'] = pd.to_datetime(td['ts'])
    td = td.sort_values('ts').reset_index(drop=True)
    td['net_bps'] = td.gross_bps - COST_BPS
    td['pnl_rs'] = td.net_bps / 10000 * NOTIONAL

    all_dates = pd.date_range(td['ts'].min().date(), td['ts'].max().date(), freq='D')
    plot_df = pd.DataFrame({'date': all_dates})
    
    td['date'] = td['ts'].dt.date
    daily_pnl = td.groupby(['date', 'side'])['pnl_rs'].sum().unstack(fill_value=0).reset_index()
    daily_pnl['date'] = pd.to_datetime(daily_pnl['date'])
    
    plot_df = pd.merge(plot_df, daily_pnl, on='date', how='left').fillna(0)
    
    if 'SHORT' not in plot_df.columns: plot_df['SHORT'] = 0
    if 'LONG' not in plot_df.columns: plot_df['LONG'] = 0
    
    plot_df['Combined'] = plot_df['SHORT'] + plot_df['LONG']
    
    plot_df['cum_short'] = plot_df['SHORT'].cumsum()
    plot_df['cum_long'] = plot_df['LONG'].cumsum()
    plot_df['cum_combined'] = plot_df['Combined'].cumsum()
    
    def calc_dd(series):
        peak = series.cummax()
        return series - peak
        
    plot_df['dd_short'] = calc_dd(plot_df['cum_short'])
    plot_df['dd_long'] = calc_dd(plot_df['cum_long'])
    plot_df['dd_combined'] = calc_dd(plot_df['cum_combined'])

    # Final Returns for text
    ret_c = plot_df['cum_combined'].iloc[-1]
    ret_s = plot_df['cum_short'].iloc[-1]
    ret_l = plot_df['cum_long'].iloc[-1]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})
    fig.patch.set_facecolor('#121212')
    
    for ax in [ax1, ax2]:
        ax.set_facecolor('#1e1e1e')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#333333')
        ax.grid(True, color='#333333', alpha=0.5)

    colors = {'Combined': '#00E676', 'Short': '#FF5252', 'Long': '#29B6F6'}
    
    # 1. Equity
    ax1.plot(plot_df['date'], STARTING_CAPITAL + plot_df['cum_combined'], color=colors['Combined'], linewidth=2.5, label=f'Combined (+₹{ret_c:,.0f})')
    ax1.plot(plot_df['date'], STARTING_CAPITAL + plot_df['cum_short'], color=colors['Short'], linewidth=1.5, label=f'Short Leg (+₹{ret_s:,.0f})')
    ax1.plot(plot_df['date'], STARTING_CAPITAL + plot_df['cum_long'], color=colors['Long'], linewidth=1.5, label=f'Long Leg (+₹{ret_l:,.0f})')
    ax1.axhline(STARTING_CAPITAL, color='white', linestyle='--', alpha=0.3)
    
    ax1.set_title("1-Slot Multi-Leg Portfolio Performance (Starting Cap: ₹1L, 5x Margin)", color='white', fontsize=16, pad=20)
    ax1.set_ylabel("Account Equity (₹)", color='white', fontsize=12)
    ax1.legend(loc='upper left', facecolor='#1e1e1e', edgecolor='white', labelcolor='white')

    # Shaded slice (Feb 20 - Mar 24)
    gap_start = pd.to_datetime('2026-02-20')
    gap_end = pd.to_datetime('2026-03-24')
    ax1.axvspan(gap_start, gap_end, color='white', alpha=0.08, label='Backfilled Gap')
    ax2.axvspan(gap_start, gap_end, color='white', alpha=0.08)

    # 2. Drawdown
    ax2.plot(plot_df['date'], plot_df['dd_combined'], color=colors['Combined'], linewidth=2, label='Combined DD')
    ax2.fill_between(plot_df['date'], 0, plot_df['dd_combined'], color=colors['Combined'], alpha=0.1)
    
    ax2.plot(plot_df['date'], plot_df['dd_short'], color=colors['Short'], linewidth=1, label='Short DD')
    ax2.plot(plot_df['date'], plot_df['dd_long'], color=colors['Long'], linewidth=1, label='Long DD')

    # Max DD markers
    mdd_c_idx = plot_df['dd_combined'].idxmin()
    mdd_s_idx = plot_df['dd_short'].idxmin()
    mdd_l_idx = plot_df['dd_long'].idxmin()

    for mdd_idx, col, c_name in [(mdd_c_idx, 'dd_combined', 'Combined'), (mdd_s_idx, 'dd_short', 'Short'), (mdd_l_idx, 'dd_long', 'Long')]:
        val = plot_df[col].iloc[mdd_idx]
        dt_val = plot_df['date'].iloc[mdd_idx]
        ax2.scatter(dt_val, val, color=colors[c_name], s=50, zorder=5)
        ax2.annotate(f'₹{val:,.0f}',
                     xy=(dt_val, val), xytext=(0, -15), textcoords='offset points',
                     ha='center', color=colors[c_name], fontsize=10)

    ax2.set_ylabel("Drawdown (₹)", color='white', fontsize=12)
    
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax.get_xticklabels(), rotation=45, color='white')

    plt.tight_layout()
    out_path = r'C:\Users\loq\.gemini\antigravity\brain\5da8b7a2-8ac8-435c-9e17-cf5e9031f4bc\1slot_multi_leg.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"PNG generated at {out_path}")

if __name__ == '__main__':
    main()

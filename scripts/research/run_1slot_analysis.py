import pandas as pd
import json
import xgboost as xgb
import numpy as np
from datetime import time, date

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

    print(f"Dataset size after matching NIFTY: {len(df)}")
    if len(df) == 0: return

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

    print("Computing metrics with 1-slot limit...")
    trades = []
    slot_free_at = pd.Timestamp.min
    
    for ts, g in df.groupby('DateTime'):
        if ts < slot_free_at:
            continue
            
        nifty_2h = g['nifty_ret_2h'].iloc[0]
        nifty_intraday = g['nifty_intraday'].iloc[0]
        t = ts.time()
        
        valid_short = False
        valid_long = False
        best_short = None
        best_long = None
        
        # Check Short
        if nifty_2h <= 0.0025 or nifty_intraday > 0.0036:
            if t < time(11, 30) or t > time(13, 0):
                cands = g[g['ss'] > 0.082].sort_values('short_conviction', ascending=False)
                if len(cands) > 0:
                    best_short = cands.iloc[0]
                    valid_short = True
                    
        # Check Long
        if nifty_2h > 0.0025 and nifty_intraday > 0.0020:
            cands = g.sort_values('long_conviction', ascending=False)
            if len(cands) > 0:
                best_long = cands.iloc[0]
                valid_long = True

        trade_taken = False
        if valid_short and valid_long:
            # Priority to short
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
    td['net_bps'] = td.gross_bps - COST_BPS
    td['bookRs'] = td.net_bps / 10000 * NOTIONAL
    td['date'] = td['ts'].dt.date
    td['month'] = td['ts'].dt.to_period('M')

    md = "# 1-Slot Combined Results (11-Month Backtest)\n\n"
    md += f"## Combined Book Summary (1-Slot Limit, Strict Gates)\n"
    md += f"- **Total Trades:** {len(td)} (Shorts: {len(td[td.side=='SHORT'])}, Longs: {len(td[td.side=='LONG'])})\n"
    md += f"- **Win Rate:** {(td.net_bps > 0).mean():.1%}\n"
    md += f"- **Avg Net BPS:** {td.net_bps.mean():.2f}\n"
    md += f"- **Total Profit:** Rs. {td.bookRs.sum():+,.0f}\n\n"

    shorts = td[td.side=='SHORT']
    longs = td[td.side=='LONG']
    if len(shorts) > 0:
        md += f"### Shorts\n"
        md += f"- Trades: {len(shorts)} | Win: {(shorts.net_bps > 0).mean():.1%} | Avg Net: {shorts.net_bps.mean():.2f} BPS\n"
    if len(longs) > 0:
        md += f"### Longs\n"
        md += f"- Trades: {len(longs)} | Win: {(longs.net_bps > 0).mean():.1%} | Avg Net: {longs.net_bps.mean():.2f} BPS\n\n"

    md += "### Monthly Breakdown (Walk-Forward / Out-of-Sample mapping)\n"
    for m in sorted(td['month'].unique()):
        m_tr = td[td['month'] == m]
        md += f"- **{m}**: Trades: {len(m_tr):3d} | Net BPS: {m_tr.net_bps.mean():+5.2f} | PnL: Rs. {m_tr.bookRs.sum():+9,.0f}\n"

    with open("data/research/v20_rolling_1h/1slot_results.md", "w") as f:
        f.write(md)
    print("Done. 1slot_results.md generated.")

if __name__ == '__main__':
    main()

import pandas as pd
import json
import xgboost as xgb
import numpy as np
from datetime import time, date

# --- Configuration ---
COST_BPS = 6.0
NOTIONAL = 500_000.0  # 5x Leverage on 1 Lakh base capital
SHORT_THRESH = 0.082
LONG_NIFTY_THRESH = 0.0025

# 1. Load Nifty 50 15m
nifty = pd.read_csv('data/raw_index_cache/nifty50_15m.csv')
nifty['ts'] = pd.to_datetime(nifty['ts'])
nifty = nifty.sort_values('ts').reset_index(drop=True)
nifty['nifty_ret_2h'] = nifty['close'] / nifty['close'].shift(8) - 1
nifty_map = dict(zip(nifty['ts'], nifty['nifty_ret_2h']))

# 2. Load Data
df = pd.read_parquet('data/research/v20_rolling_1h/panel.parquet')
df['DateTime'] = pd.to_datetime(df['DateTime'])
df = df[df['DateTime'].dt.date >= date(2025, 8, 1)]

time_mask = (df['DateTime'].dt.time >= time(10, 15)) & (df['DateTime'].dt.time <= time(14, 15))
df = df[time_mask]

# Apply Nifty return
df['nifty_ret_2h'] = df['DateTime'].map(nifty_map)
df = df.dropna(subset=['nifty_ret_2h'])

# 3. Load Models & Metadata
v20_feats = json.load(open('models/research/v20_rolling_1h/metadata.json'))['features']
bs = xgb.Booster()
bs.load_model('models/research/v20_rolling_1h/xgb_short_model.json')
bl = xgb.Booster()
bl.load_model('models/research/v20_rolling_1h/xgb_long_model.json')

# 4. Predict & Score
df = df.dropna(subset=v20_feats + ['Next_Hour_Return'])
X = xgb.DMatrix(np.nan_to_num(df[v20_feats].values.astype(np.float32)), feature_names=v20_feats)
df['ss'] = bs.predict(X)
df['ls'] = bl.predict(X)

ss_mean = df.groupby('DateTime')['ss'].transform('mean')
ls_mean = df.groupby('DateTime')['ls'].transform('mean')
df['short_conviction'] = (df['ss'] - ss_mean) - (df['ls'] - ls_mean)
df['long_conviction'] = (df['ls'] - ls_mean) - (df['ss'] - ss_mean)

trades = []
for ts, g in df.groupby('DateTime'):
    nifty_2h = g['nifty_ret_2h'].iloc[0]
    
    # Check Short
    short_cands = g[g['ss'] > SHORT_THRESH].sort_values('short_conviction', ascending=False)
    if len(short_cands) > 0:
        p = short_cands.iloc[0]
        # Short PnL = negative of Next Hour Return
        gross_bps = -p['Next_Hour_Return'] * 10000
        trades.append((ts, 'SHORT', p['Ticker'], gross_bps))
        
    # Check Long
    if nifty_2h > LONG_NIFTY_THRESH:
        long_cands = g.sort_values('long_conviction', ascending=False)
        if len(long_cands) > 0:
            p = long_cands.iloc[0]
            # Long PnL = positive of Next Hour Return
            gross_bps = p['Next_Hour_Return'] * 10000
            trades.append((ts, 'LONG', p['Ticker'], gross_bps))

td = pd.DataFrame(trades, columns=['ts', 'side', 'tk', 'gross_bps'])

if len(td) == 0:
    print("No trades found.")
    import sys
    sys.exit()

td['net_bps'] = td.gross_bps - COST_BPS
td['bookRs'] = td.net_bps / 10000 * NOTIONAL
td['date'] = td['ts'].dt.date
td['month'] = td['ts'].dt.to_period('M')

# Calculate Drawdown on cumulative PnL
td = td.sort_values('ts')
daily_pnl = td.groupby('date')['bookRs'].sum().reset_index()
daily_pnl['cum_pnl'] = daily_pnl['bookRs'].cumsum()
daily_pnl['peak'] = daily_pnl['cum_pnl'].cummax()
daily_pnl['drawdown'] = daily_pnl['cum_pnl'] - daily_pnl['peak']
max_dd = daily_pnl['drawdown'].min()

# 5. Summary
print("="*60)
print(f" COMBINED STRATEGY 11-MONTH TEST (5x Leverage, Rs. 5L Notional)")
print("="*60)
print(f"Total Trades : {len(td)} (Shorts: {len(td[td.side=='SHORT'])}, Longs: {len(td[td.side=='LONG'])})")
print(f"Win Rate     : {(td.net_bps > 0).mean():.1%}")
print(f"Avg Net BPS  : {td.net_bps.mean():.2f}")
print(f"Total Profit : Rs. {td.bookRs.sum():+,.0f}")
print(f"Max Drawdown : Rs. {max_dd:+,.0f}")
print(f"Return/MDD   : {abs(td.bookRs.sum() / max_dd) if max_dd != 0 else 0:.2f}")
print("-"*60)
print("\nMonthly Breakdown:")
for m in sorted(td['month'].unique()):
    m_tr = td[td['month'] == m]
    shorts = len(m_tr[m_tr.side == 'SHORT'])
    longs = len(m_tr[m_tr.side == 'LONG'])
    win = (m_tr.net_bps > 0).mean()
    net = m_tr.net_bps.mean()
    rs = m_tr.bookRs.sum()
    print(f"{m} | Trades: {len(m_tr):3d} (S:{shorts:<2d} L:{longs:<2d}) | Win: {win:.1%} | BPS: {net:>+6.2f} | PnL: Rs. {rs:>+9,.0f}")

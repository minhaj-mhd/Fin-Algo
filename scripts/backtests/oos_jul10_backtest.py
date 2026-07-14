import pandas as pd
import json
import xgboost as xgb
import numpy as np
from datetime import time, date
import datetime as dt

# --- Configuration ---
COST_BPS = 6.0
NOTIONAL = 200_000.0  # 5x Leverage on 40K base capital
SHORT_THRESH = 0.082
LONG_NIFTY_THRESH = 0.0025

# 1. Load Nifty 50 15m
nifty = pd.read_csv('data/raw_index_cache/nifty50_15m.csv')
ist_times = []
for t_str in nifty['ts']:
    t = pd.to_datetime(t_str)
    # The old data (before June 2026) was incorrectly saved with +0000 but was actually IST
    # The newly fetched data (June 4 2026 onwards) is correctly in UTC
    if t.date() < date(2026, 6, 1):
        # Old data: it's already IST, just strip the timezone
        ist_times.append(t.tz_localize(None))
    else:
        # New data: it's true UTC, so convert to IST then strip
        ist_times.append(t.tz_convert('Asia/Kolkata').tz_localize(None))
nifty['ts'] = ist_times
nifty = nifty.sort_values('ts').reset_index(drop=True)
nifty['nifty_ret_2h'] = nifty['close'] / nifty['close'].shift(8) - 1
nifty_map = dict(zip(nifty['ts'], nifty['nifty_ret_2h']))

# 1.5 Load S&P 500
import yfinance as yf
sp500 = yf.download('^GSPC', start='2025-07-01', end='2026-07-30', progress=False)
if isinstance(sp500.columns, pd.MultiIndex):
    sp500.columns = sp500.columns.get_level_values(0)
sp500 = sp500.reset_index()
sp500['Date'] = pd.to_datetime(sp500['Date']).dt.date
sp500['sp500_ret'] = sp500['Close'].pct_change()
sp500_ret_dict = {r['Date']: r['sp500_ret'] for _, r in sp500.iterrows()}

prev_sp500_cache = {}
def get_prev_sp500_ret(curr_date):
    if curr_date in prev_sp500_cache: return prev_sp500_cache[curr_date]
    prev_dates = [d for d in sp500_ret_dict.keys() if d < curr_date]
    ret = sp500_ret_dict[max(prev_dates)] if prev_dates else 0
    prev_sp500_cache[curr_date] = ret
    return ret

# 2. Load Data
df = pd.read_parquet('data/research/v20_rolling_1h/panel.parquet')
df['DateTime'] = pd.to_datetime(df['DateTime'])
df = df[df['DateTime'].dt.date >= date(2026, 6, 4)]

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
locked_until = None

for ts in sorted(df['DateTime'].unique()):
    if locked_until is not None and ts < locked_until:
        continue
        
    g = df[df['DateTime'] == ts]
    sp500_ret = get_prev_sp500_ret(ts.date())
    nifty_2h = g['nifty_ret_2h'].iloc[0]
    
    t_time = pd.to_datetime(ts).time()
    if dt.time(11, 30) <= t_time <= dt.time(13, 0):
        continue

    # Dynamic Short Threshold
    dynamic_short_thresh = SHORT_THRESH
    if sp500_ret > 0.005 and nifty_2h >= -0.0010:
        dynamic_short_thresh = 0.110

    short_cands = g[g['ss'] > dynamic_short_thresh].sort_values('short_conviction', ascending=False)
    
    longs_allowed = (sp500_ret >= -0.005)
    long_cands = pd.DataFrame()
    if longs_allowed and nifty_2h > LONG_NIFTY_THRESH:
        long_cands = g.sort_values('long_conviction', ascending=False)

    trade_taken = False
    
    # Priority: Short if valid, else Long
    if len(short_cands) > 0:
        p = short_cands.iloc[0]
        gross_bps = -p['Next_Hour_Return'] * 10000
        trades.append((ts, 'SHORT', p['Ticker'], gross_bps))
        trade_taken = True
    elif len(long_cands) > 0:
        p = long_cands.iloc[0]
        gross_bps = p['Next_Hour_Return'] * 10000
        trades.append((ts, 'LONG', p['Ticker'], gross_bps))
        trade_taken = True
        
    if trade_taken:
        # Lock for 1 hour to prevent overlapping trades
        locked_until = ts + pd.Timedelta(hours=1)

td = pd.DataFrame(trades, columns=['ts', 'side', 'tk', 'gross_bps'])

if len(td) == 0:
    print("No trades found.")
    import sys
    sys.exit()

td['net_bps'] = td.gross_bps - COST_BPS
td['date'] = td['ts'].dt.date
td['month'] = td['ts'].dt.to_period('M')

# Geometric Compounding
import datetime as dt
td = td.sort_values('ts').reset_index(drop=True)
base_capital = 40_000.0
leverage = 5.0
bookRs_list = []
capital_list = []

for i, row in td.iterrows():
    trade_notional = base_capital * leverage
    profit_rs = (row['net_bps'] / 10000.0) * trade_notional
    base_capital += profit_rs
    bookRs_list.append(profit_rs)
    capital_list.append(base_capital)

td['bookRs'] = bookRs_list
td['capital'] = capital_list
td['short_pnl'] = np.where(td['side'] == 'SHORT', td['bookRs'], 0)
td['long_pnl'] = np.where(td['side'] == 'LONG', td['bookRs'], 0)

# Calculate Drawdown on cumulative PnL
daily_pnl = td.groupby('date').agg({
    'bookRs': 'sum', 
    'capital': 'last',
    'short_pnl': 'sum',
    'long_pnl': 'sum'
}).reset_index()
daily_pnl['cum_pnl'] = daily_pnl['capital'] - 40_000.0
daily_pnl['cum_short_pnl'] = daily_pnl['short_pnl'].cumsum()
daily_pnl['cum_long_pnl'] = daily_pnl['long_pnl'].cumsum()
daily_pnl['peak'] = daily_pnl['cum_pnl'].cummax()
daily_pnl['drawdown'] = daily_pnl['cum_pnl'] - daily_pnl['peak']
max_dd = daily_pnl['drawdown'].min()

daily_pnl['peak_capital'] = daily_pnl['capital'].cummax()
daily_pnl['drawdown_pct'] = (daily_pnl['capital'] - daily_pnl['peak_capital']) / daily_pnl['peak_capital']
max_dd_pct = daily_pnl['drawdown_pct'].min()

# 5. Summary
print("="*60)
print(f" TRUE 1-SLOT STRATEGY OOS TEST (Geometric 5x, Rs. 40K Base)")
print("="*60)
print(f"Total Trades : {len(td)} (Shorts: {len(td[td.side=='SHORT'])}, Longs: {len(td[td.side=='LONG'])})")
print(f"Win Rate     : {(td.net_bps > 0).mean():.1%}")
print(f"Avg Net BPS  : {td.net_bps.mean():.2f}")
print(f"Total Profit : Rs. {td.bookRs.sum():+,.0f}")
print(f"Max Drawdown : Rs. {max_dd:+,.0f} ({max_dd_pct:.2%})")
print(f"Return/MDD   : {abs(td.bookRs.sum() / max_dd) if max_dd != 0 else 0:.2f}")
print("-"*60)
for side in ['SHORT', 'LONG']:
    s_td = td[td.side == side]
    wins = s_td[s_td.net_bps > 0]['net_bps']
    losses = s_td[s_td.net_bps <= 0]['net_bps']
    
    print(f"{side} METRICS:")
    print(f"  Win Rate  : {(s_td.net_bps > 0).mean():.1%}")
    print(f"  Avg BPS   : {s_td.net_bps.mean():+6.2f} BPS")
    print(f"  Mean Win  : {wins.mean():+6.2f} BPS")
    print(f"  Mean Loss : {losses.mean():+6.2f} BPS")
    print(f"  Max Win   : {wins.max() if len(wins)>0 else 0:+6.2f} BPS")
    print(f"  Max Loss  : {losses.min() if len(losses)>0 else 0:+6.2f} BPS")
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

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

ax1.plot(daily_pnl['date'], daily_pnl['capital'], label='Total Geometric 5x Capital', color='blue', linewidth=2)
ax1.plot(daily_pnl['date'], 40000 + daily_pnl['cum_short_pnl'], label='Short Contribution', color='purple', linestyle='--', linewidth=1.5)
ax1.plot(daily_pnl['date'], 40000 + daily_pnl['cum_long_pnl'], label='Long Contribution', color='green', linestyle='--', linewidth=1.5)
ax1.axhline(y=40000, color='red', linestyle='--', label='Base Capital (Rs. 40K)', alpha=0.7)
ax1.set_title('OOS Equity Curve & Drawdown (Strict 1-Slot + Geometric 5x)', fontsize=14)
ax1.set_ylabel('Capital (Rs.)', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=12)
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

ax2.fill_between(daily_pnl['date'], daily_pnl['drawdown'], 0, color='red', alpha=0.3, label='Drawdown (Rs.)')
ax2.plot(daily_pnl['date'], daily_pnl['drawdown'], color='red', linewidth=1)
ax2.set_ylabel('Drawdown', fontsize=12)
ax2.set_xlabel('Date', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=12)
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

plt.tight_layout()
out_path = r"C:\Users\loq\.gemini\antigravity\brain\38a08ecd-5d90-4032-ae18-2524e430c6da\equity_curve.png"
plt.savefig(out_path)
print(f"\nPlot saved to {out_path}")

h1_mask = td['date'] <= dt.date(2026, 1, 31)
h2_mask = td['date'] > dt.date(2026, 1, 31)

h1_short_pnl = td[h1_mask & (td['side'] == 'SHORT')]['bookRs'].sum()
h1_long_pnl = td[h1_mask & (td['side'] == 'LONG')]['bookRs'].sum()
h1_start_cap = 40_000.0
h1_end_cap = daily_pnl[daily_pnl['date'] <= dt.date(2026, 1, 31)]['capital'].iloc[-1] if not daily_pnl[daily_pnl['date'] <= dt.date(2026, 1, 31)].empty else 40_000.0

h2_short_pnl = td[h2_mask & (td['side'] == 'SHORT')]['bookRs'].sum()
h2_long_pnl = td[h2_mask & (td['side'] == 'LONG')]['bookRs'].sum()
h2_start_cap = h1_end_cap

print("-" * 60)
print(f"H1 (Aug-Jan) Start Capital: Rs. {h1_start_cap:,.0f}")
print(f"  -> Short Contribution : {h1_short_pnl / h1_start_cap * 100:+7.2f}% (Rs. {h1_short_pnl:,.0f})")
print(f"  -> Long Contribution  : {h1_long_pnl / h1_start_cap * 100:+7.2f}% (Rs. {h1_long_pnl:,.0f})")

print(f"H2 (Feb-Jun) Start Capital: Rs. {h2_start_cap:,.0f}")
print(f"  -> Short Contribution : {h2_short_pnl / h2_start_cap * 100:+7.2f}% (Rs. {h2_short_pnl:,.0f})")
print(f"  -> Long Contribution  : {h2_long_pnl / h2_start_cap * 100:+7.2f}% (Rs. {h2_long_pnl:,.0f})")
print("-" * 60)

print("\n=== MONTHLY RETURNS (%) ===")
daily_pnl['month_str'] = daily_pnl['date'].apply(lambda x: x.strftime('%Y-%m'))
monthly_caps = daily_pnl.groupby('month_str')['capital'].last()
prev_cap = 40_000.0
for m, cap in monthly_caps.items():
    ret_pct = (cap - prev_cap) / prev_cap * 100
    print(f"{m} : {ret_pct:+7.2f}% (End Cap: Rs. {cap:,.0f})")
    prev_cap = cap

print("\n=== WEEKLY RETURNS (%) ===")
daily_pnl['week_str'] = daily_pnl['date'].apply(lambda x: f"{x.isocalendar()[0]}-W{x.isocalendar()[1]:02d}")
weekly_caps = daily_pnl.groupby('week_str', sort=True)['capital'].last()
prev_cap = 40_000.0
for w, cap in weekly_caps.items():
    ret_pct = (cap - prev_cap) / prev_cap * 100
    print(f"{w} : {ret_pct:+7.2f}% (End Cap: Rs. {cap:,.0f})")
    prev_cap = cap

print("\n=== SHORT ONLY MONTHLY RETURNS (%) ===")
prev_cap = 40_000.0
for m, cap in monthly_caps.items():
    m_mask = td['date'].apply(lambda x: x.strftime('%Y-%m')) == m
    short_pnl = td[m_mask & (td['side'] == 'SHORT')]['bookRs'].sum()
    ret_pct = (short_pnl / prev_cap) * 100
    print(f"{m} : {ret_pct:+7.2f}% (Short PnL: Rs. {short_pnl:,.0f})")
    prev_cap = cap

print("\n=== SHORT ONLY WEEKLY RETURNS (%) ===")
prev_cap = 40_000.0
for w, cap in weekly_caps.items():
    w_mask = td['date'].apply(lambda x: f"{x.isocalendar()[0]}-W{x.isocalendar()[1]:02d}") == w
    short_pnl = td[w_mask & (td['side'] == 'SHORT')]['bookRs'].sum()
    ret_pct = (short_pnl / prev_cap) * 100
    print(f"{w} : {ret_pct:+7.2f}% (Short PnL: Rs. {short_pnl:,.0f})")
    prev_cap = cap

print("\n=== LONG ONLY MONTHLY RETURNS (%) ===")
prev_cap = 40_000.0
for m, cap in monthly_caps.items():
    m_mask = td['date'].apply(lambda x: x.strftime('%Y-%m')) == m
    long_pnl = td[m_mask & (td['side'] == 'LONG')]['bookRs'].sum()
    ret_pct = (long_pnl / prev_cap) * 100
    print(f"{m} : {ret_pct:+7.2f}% (Long PnL: Rs. {long_pnl:,.0f})")
    prev_cap = cap

print("\n=== LONG ONLY WEEKLY RETURNS (%) ===")
prev_cap = 40_000.0
for w, cap in weekly_caps.items():
    w_mask = td['date'].apply(lambda x: f"{x.isocalendar()[0]}-W{x.isocalendar()[1]:02d}") == w
    long_pnl = td[w_mask & (td['side'] == 'LONG')]['bookRs'].sum()
    ret_pct = (long_pnl / prev_cap) * 100
    print(f"{w} : {ret_pct:+7.2f}% (Long PnL: Rs. {long_pnl:,.0f})")
    prev_cap = cap

print("\n=== WEEKLY WIN RATES (%) ===")
td['week_str'] = td['date'].apply(lambda x: f"{x.isocalendar()[0]}-W{x.isocalendar()[1]:02d}")
for w in sorted(td['week_str'].unique()):
    w_tr = td[td['week_str'] == w]
    
    s_tr = w_tr[w_tr['side'] == 'SHORT']
    s_win = (s_tr['net_bps'] > 0).mean() * 100 if len(s_tr) > 0 else float('nan')
    s_trades = len(s_tr)
    
    l_tr = w_tr[w_tr['side'] == 'LONG']
    l_win = (l_tr['net_bps'] > 0).mean() * 100 if len(l_tr) > 0 else float('nan')
    l_trades = len(l_tr)
    
    s_win_str = f"{s_win:5.1f}%" if not pd.isna(s_win) else "  N/A "
    l_win_str = f"{l_win:5.1f}%" if not pd.isna(l_win) else "  N/A "
    print(f"{w} | Shorts: {s_win_str} ({s_trades:2d} trades) | Longs: {l_win_str} ({l_trades:2d} trades)")

print('=== ALL TRADES ===')
print(td[['ts', 'side', 'tk', 'net_bps', 'bookRs']].to_string(index=False))


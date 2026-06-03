"""
DEEP AUDIT: Is 20% monthly return real or an illusion?

The "unslotted return" sums up net_return across ALL trades, where each
net_return is a % of the CAPITAL DEPLOYED IN THAT TRADE. But if you can't
deploy 100% of your account into every trade (because trades overlap),
the actual account return is much lower.

This script computes the REAL equity curve with proper position sizing.
"""
import json
import os
import pandas as pd
from collections import defaultdict

# Load all premium trades
files = [
    'data/strategy_25x_results.json',
    'data/strategy_10_new_results.json',
    'data/strategy_15_final_results.json'
]

premium_legs_config = [
    (2, 'SHORT'), (8, 'SHORT'), (10, 'LONG'), (18, 'SHORT'),
    (19, 'LONG'), (35, 'LONG'), (36, 'LONG'), (39, 'LONG'), (42, 'SHORT')
]

all_trades = []
for fpath in files:
    if not os.path.exists(fpath): continue
    with open(fpath, 'r') as f:
        data = json.load(f)
        for k, v in data.get('strategies', {}).items():
            if k.startswith('trades_'):
                sid = int(k.split('_')[1])
                matching_sides = [side for s, side in premium_legs_config if s == sid]
                if matching_sides:
                    side = matching_sides[0]
                    for t in v:
                        if t['side'] == side:
                            t['strategy_id'] = sid
                            all_trades.append(t)

df = pd.DataFrame(all_trades)
df['entry_dt'] = pd.to_datetime(df['entry_time'])
df['exit_dt'] = pd.to_datetime(df['exit_time'])
df = df.sort_values('entry_dt').reset_index(drop=True)

print(f"Total premium trades: {len(df)}")
print(f"Total raw sum of net_return: {df['net_return'].sum()*100:+.2f}%")
print(f"  (This is the misleading 'unslotted' number)\n")

# ====================================================================
# METHOD 1: Fixed position sizing (1/N of capital per trade)
# ====================================================================
print("=" * 70)
print("METHOD 1: FIXED POSITION SIZING")
print("=" * 70)

for N in [3, 4, 6, 8]:
    frac = 1.0 / N
    # Each trade's contribution to ACCOUNT return = net_return * frac
    account_return = df['net_return'].sum() * frac
    print(f"  If each trade = 1/{N} of capital ({frac*100:.1f}%):")
    print(f"    Account Return = {account_return*100:+.2f}%")
    print(f"    Max capital at risk (if {N} concurrent) = {N * frac * 100:.0f}%")

# ====================================================================
# METHOD 2: ACTUAL EQUITY CURVE SIMULATION
# ====================================================================
print(f"\n{'=' * 70}")
print("METHOD 2: ACTUAL EQUITY CURVE SIMULATION")
print("=" * 70)

# For each trade, we need to know how many OTHER trades are active at entry
# Then size the trade as 1/max_concurrent_at_any_point

# First: find max concurrent trades at any point
events = []
for _, row in df.iterrows():
    events.append((row['entry_dt'], 'ENTER', row.name))
    events.append((row['exit_dt'], 'EXIT', row.name))

events.sort(key=lambda x: (x[0], 0 if x[1] == 'EXIT' else 1))  # Exits before entries at same time

active_count = 0
max_concurrent = 0
for dt, action, idx in events:
    if action == 'ENTER':
        active_count += 1
        max_concurrent = max(max_concurrent, active_count)
    else:
        active_count -= 1

print(f"  Max concurrent trades: {max_concurrent}")

# Simulate with FIXED sizing = 1/max_concurrent
frac = 1.0 / max_concurrent
starting_capital = 1000000  # 10 Lakh
capital = starting_capital
daily_equity = {}

# Process trades in chronological order by exit time
for _, trade in df.sort_values('exit_dt').iterrows():
    # Capital deployed in this trade = frac * capital_at_entry
    # For simplicity, use starting capital (no compounding within month)
    deployed = starting_capital * frac
    pnl = deployed * trade['net_return']
    capital += pnl
    
    date_str = trade['date']
    daily_equity[date_str] = capital

account_return_pct = (capital - starting_capital) / starting_capital * 100

print(f"\n  Position size per trade: 1/{max_concurrent} = {frac*100:.1f}% of capital")
print(f"  Starting Capital: Rs. {starting_capital:,.0f}")
print(f"  Ending Capital:   Rs. {capital:,.0f}")
print(f"  Absolute P&L:     Rs. {capital - starting_capital:,.0f}")
print(f"  REAL Account Return: {account_return_pct:+.2f}%")

# ====================================================================
# METHOD 3: DYNAMIC SIZING (adapt to actual concurrency)
# ====================================================================
print(f"\n{'=' * 70}")
print("METHOD 3: DYNAMIC SIZING (size = 1/active_count at entry)")
print("=" * 70)

# For each trade, determine how many trades are active at entry time
# Size that trade as 1/max(active_at_entry, 1)
active_set = set()
trade_sizes = {}

events2 = []
for _, row in df.iterrows():
    events2.append((row['entry_dt'], 'ENTER', row.name))
    events2.append((row['exit_dt'], 'EXIT', row.name))
events2.sort(key=lambda x: (x[0], 0 if x[1] == 'EXIT' else 1))

for dt, action, idx in events2:
    if action == 'ENTER':
        active_set.add(idx)
        # Size this trade based on how many are now active
        concurrent = len(active_set)
        trade_sizes[idx] = 1.0 / concurrent
    else:
        active_set.discard(idx)

capital_dyn = starting_capital
for _, trade in df.sort_values('exit_dt').iterrows():
    frac_dyn = trade_sizes[trade.name]
    deployed = starting_capital * frac_dyn  # Use starting capital (no compounding)
    pnl = deployed * trade['net_return']
    capital_dyn += pnl

dyn_return = (capital_dyn - starting_capital) / starting_capital * 100
print(f"  Starting Capital: Rs. {starting_capital:,.0f}")
print(f"  Ending Capital:   Rs. {capital_dyn:,.0f}")
print(f"  Absolute P&L:     Rs. {capital_dyn - starting_capital:,.0f}")
print(f"  REAL Account Return (Dynamic): {dyn_return:+.2f}%")

# ====================================================================
# METHOD 4: WITH INTRADAY LEVERAGE (common in Indian markets)
# ====================================================================
print(f"\n{'=' * 70}")
print("METHOD 4: WITH INTRADAY LEVERAGE")
print("=" * 70)

for leverage in [1, 2, 5]:
    effective_frac = min(1.0, leverage / max_concurrent)
    account_ret = df['net_return'].sum() * effective_frac * 100
    print(f"  {leverage}x leverage, {max_concurrent} max concurrent:")
    print(f"    Per-trade allocation: {effective_frac*100:.1f}% of base capital")
    print(f"    Account Return: {account_ret:+.2f}%")
    print(f"    Max exposure: {effective_frac * max_concurrent * 100:.0f}% of base capital")

# ====================================================================
# SUMMARY
# ====================================================================
print(f"\n{'=' * 70}")
print("AUDIT SUMMARY")
print("=" * 70)
print(f"""
The "unslotted return" of +21.10% is the SUM of individual trade P&Ls,
each expressed as a % of the capital deployed in that specific trade.

But since trades OVERLAP (max {max_concurrent} concurrent), you CANNOT deploy
100% of your capital into every trade. You must divide your capital.

REAL account returns (no leverage):
  - Conservative (1/{max_concurrent} fixed):  {(df['net_return'].sum() / max_concurrent)*100:+.2f}%
  - Dynamic sizing:             {dyn_return:+.2f}%
  
With common intraday leverage (5x):
  - Account Return:             {df['net_return'].sum() * min(1.0, 5/max_concurrent) * 100:+.2f}%

The previously reported "slotted" return of +4.12% was actually closer
to reality than the +21.10% unslotted figure, because the slot mechanism
was implicitly modeling capital constraints.
""")

# Per-trade stats
print("PER-TRADE STATISTICS:")
print(f"  Avg trade net return: {df['net_return'].mean()*100:+.4f}%")
print(f"  Median trade return:  {df['net_return'].median()*100:+.4f}%")
print(f"  Best trade:           {df['net_return'].max()*100:+.4f}%")
print(f"  Worst trade:          {df['net_return'].min()*100:+.4f}%")
print(f"  Std dev:              {df['net_return'].std()*100:.4f}%")

wins = df[df['net_return'] > 0]
losses = df[df['net_return'] <= 0]
print(f"\n  Winners: {len(wins)} ({len(wins)/len(df)*100:.1f}%)")
print(f"    Avg win:  {wins['net_return'].mean()*100:+.4f}%")
print(f"  Losers:  {len(losses)} ({len(losses)/len(df)*100:.1f}%)")
print(f"    Avg loss: {losses['net_return'].mean()*100:+.4f}%")
print(f"\n  Expectancy per trade: {df['net_return'].mean()*100:+.4f}%")
print(f"  Edge (win% × avg_win - loss% × avg_loss): {(len(wins)/len(df) * wins['net_return'].mean() + len(losses)/len(df) * losses['net_return'].mean())*100:+.4f}%")

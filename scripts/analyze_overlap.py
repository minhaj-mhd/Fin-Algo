import json
import os
import pandas as pd

files = [
    'data/strategy_25x_results.json',
    'data/strategy_10_new_results.json',
    'data/strategy_15_final_results.json'
]

# The selected premium legs
premium_legs_config = [
    (2, 'SHORT'),
    (8, 'SHORT'),
    (10, 'LONG'),
    (18, 'SHORT'),
    (19, 'LONG'),
    (35, 'LONG'),
    (36, 'LONG'),
    (39, 'LONG'),
    (42, 'SHORT')
]

all_trades = []

for fpath in files:
    if not os.path.exists(fpath): continue
    with open(fpath, 'r') as f:
        data = json.load(f)
        for k, v in data.get('strategies', {}).items():
            if k.startswith('trades_'):
                sid = int(k.split('_')[1])
                # Find matching side if any
                matching_sides = [side for s, side in premium_legs_config if s == sid]
                if matching_sides:
                    side = matching_sides[0]
                    leg_trades = [t for t in v if t['side'] == side]
                    for t in leg_trades:
                        t['strategy_id'] = sid
                        all_trades.append(t)

df = pd.DataFrame(all_trades)
print(f"Total trades parsed: {len(df)}")

if len(df) == 0:
    print("No trades found.")
    exit()

# Parse times to datetime
df['entry_dt'] = pd.to_datetime(df['entry_time'])
df['exit_dt'] = pd.to_datetime(df['exit_time'])

# Build a list of all 15-minute intervals between start and end of test month
start_time = df['entry_dt'].min()
end_time = df['exit_dt'].max()
all_times = pd.date_range(start=start_time, end=end_time, freq='15min')

concurrency = []
for t in all_times:
    # A trade is active at time 't' if entry_dt <= t < exit_dt
    active = df[(df['entry_dt'] <= t) & (t < df['exit_dt'])]
    count = len(active)
    if count > 0:
        concurrency.append({'time': t, 'count': count, 'tickers': list(active['ticker'])})

df_c = pd.DataFrame(concurrency)

if len(df_c) == 0:
    print("No concurrent trades found.")
    exit()

max_concurrent = df_c['count'].max()
avg_concurrent = df_c['count'].mean()
p90_concurrent = df_c['count'].quantile(0.90)
p95_concurrent = df_c['count'].quantile(0.95)

print("\n=== PORTFOLIO CONCURRENCY METRICS ===")
print(f"Maximum concurrent trades: {max_concurrent}")
print(f"Average concurrent trades (when active): {avg_concurrent:.2f}")
print(f"90th Percentile of concurrent trades: {p90_concurrent:.1f}")
print(f"95th Percentile of concurrent trades: {p95_concurrent:.1f}")

print("\nPeak times (Top 5 busiest intervals):")
df_c_sorted = df_c.sort_values(by='count', ascending=False)
for idx, row in df_c_sorted.head(5).iterrows():
    print(f"  {row['time']} | Count: {row['count']} | Tickers: {row['tickers']}")

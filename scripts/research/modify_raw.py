import sys

with open('scripts/raw_model_test.py', 'r') as f:
    lines = f.readlines()

start_idx = -1
end_idx = -1
for i, line in enumerate(lines):
    if '# STRATEGY 1: "Daily Macro Gatekeeper"' in line:
        start_idx = i - 1
    if '# End of simulation for this strategy' in line:
        end_idx = i - 1

new_logic = """
            # STRATEGY 1: Raw 1-Hour Model (Top 5)
            if strategy_id == 1:
                if time_str in ["09:15", "10:15", "11:15", "12:15", "13:15", "14:15"]:
                    if daily_trade_count[date_str] < limit:
                        top_1h_longs = sorted([x for x in active_15m if x['h1_long_rank'] <= 5 and x['h1_long_conv'] > 0], key=lambda x: x['h1_long_rank'])
                        top_1h_shorts = sorted([x for x in active_15m if x['h1_short_rank'] <= 5 and x['h1_short_conv'] > 0], key=lambda x: x['h1_short_rank'])
                        
                        for x in top_1h_longs:
                            ticker = x['ticker']
                            if ticker not in active_tickers:
                                active_trades.append({'ticker': ticker, 'side': 'LONG', 'entry_price': x['close'], 'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0})
                                active_tickers.add(ticker)
                                daily_trade_count[date_str] += 1
                                if daily_trade_count[date_str] >= limit: break
                                
                        for x in top_1h_shorts:
                            if daily_trade_count[date_str] >= limit: break
                            ticker = x['ticker']
                            if ticker not in active_tickers:
                                active_trades.append({'ticker': ticker, 'side': 'SHORT', 'entry_price': x['close'], 'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0})
                                active_tickers.add(ticker)
                                daily_trade_count[date_str] += 1

            # STRATEGY 2: Raw 15-Minute Model (Top 5)
            elif strategy_id == 2:
                if daily_trade_count[date_str] < limit:
                    for x in top_15m_longs[:5]:
                        if x['long_conv'] > 0:
                            ticker = x['ticker']
                            if ticker not in active_tickers:
                                active_trades.append({'ticker': ticker, 'side': 'LONG', 'entry_price': x['close'], 'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0})
                                active_tickers.add(ticker)
                                daily_trade_count[date_str] += 1
                                if daily_trade_count[date_str] >= limit: break
                                
                    for x in top_15m_shorts[:5]:
                        if daily_trade_count[date_str] >= limit: break
                        if x['short_conv'] > 0:
                            ticker = x['ticker']
                            if ticker not in active_tickers:
                                active_trades.append({'ticker': ticker, 'side': 'SHORT', 'entry_price': x['close'], 'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0})
                                active_tickers.add(ticker)
                                daily_trade_count[date_str] += 1
"""

if start_idx != -1 and end_idx != -1:
    lines[start_idx:end_idx+1] = [new_logic + '\n']

# Now replace strategies list
start_list = -1
end_list = -1
for i, line in enumerate(lines):
    if 'strategies_info = [' in line:
        start_list = i
    if start_list != -1 and ']' in line and '25, "Triple-Timeframe' in lines[i-1]:
        end_list = i
        break

if start_list != -1 and end_list != -1:
    lines[start_list:end_list+1] = [
        "    strategies_info = [\n",
        "        (1, 'Raw 1-Hour Model Top 5'),\n",
        "        (2, 'Raw 15-Min Model Top 5')\n",
        "    ]\n"
    ]

with open('scripts/raw_model_test.py', 'w') as f:
    f.writelines(lines)

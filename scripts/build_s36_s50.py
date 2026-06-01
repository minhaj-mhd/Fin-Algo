import re

def build():
    with open('scripts/strategy_10_new_backtest.py', 'r') as f:
        content = f.read()
        
    start_str = '            # =================================================================\n            # STRATEGY 26:'
    end_str = '        # End of simulation for this strategy, evaluate trades'
    
    start_idx = content.find(start_str)
    end_idx = content.find(end_str)
    
    if start_idx == -1 or end_idx == -1:
        print("Could not find replacement bounds!")
        return

    new_strategies = """
            # =================================================================
            # STRATEGY 36: The Opening Drive
            # =================================================================
            if strategy_id == 36:
                if time_str <= "10:30":
                    if daily_trade_count[date_str] < limit:
                        for x in top_15m_longs:
                            if x['gap_pct'] > 0.003 and x['daily_long_rank'] <= 10 and x['long_rank'] <= 10:
                                ticker = x['ticker']
                                if ticker not in active_tickers:
                                    active_trades.append({'ticker': ticker, 'side': 'LONG', 'entry_price': x['close'], 'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0})
                                    active_tickers.add(ticker)
                                    daily_trade_count[date_str] += 1
                                    if daily_trade_count[date_str] >= limit: break

            # =================================================================
            # STRATEGY 37: Mid-Morning Reversal
            # =================================================================
            elif strategy_id == 37:
                if "10:15" <= time_str <= "11:30":
                    if daily_trade_count[date_str] < limit:
                        for x in top_15m_shorts:
                            if x['short_rank'] <= 10 and x['ibs'] < 0.3 and x['atr_14_pct'] > 0.002:
                                ticker = x['ticker']
                                if ticker not in active_tickers:
                                    active_trades.append({'ticker': ticker, 'side': 'SHORT', 'entry_price': x['close'], 'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0})
                                    active_tickers.add(ticker)
                                    daily_trade_count[date_str] += 1
                                    if daily_trade_count[date_str] >= limit: break

            # =================================================================
            # STRATEGY 38: Dead-Cat Bounce
            # =================================================================
            elif strategy_id == 38:
                curr_vol = dict_rolling_vol.get(T, 0.0)
                if curr_vol > dict_p50_vol.get(T, 0.0):
                    if daily_trade_count[date_str] < limit:
                        for x in top_15m_longs:
                            if x['long_rank'] <= 10 and x['ibs'] > 0.7 and x['h1_long_rank'] > 20:
                                ticker = x['ticker']
                                if ticker not in active_tickers:
                                    active_trades.append({'ticker': ticker, 'side': 'SHORT', 'entry_price': x['close'], 'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0})
                                    active_tickers.add(ticker)
                                    daily_trade_count[date_str] += 1
                                    if daily_trade_count[date_str] >= limit: break

            # =================================================================
            # STRATEGY 39: The VWAP Pinch
            # =================================================================
            elif strategy_id == 39:
                if daily_trade_count[date_str] < limit:
                    for x in top_15m_longs:
                        if x['long_rank'] <= 3 and x['atr_14_pct'] < 0.002:
                            if abs(x['close'] - x['ma20']) / x['ma20'] < 0.002:
                                ticker = x['ticker']
                                if ticker not in active_tickers:
                                    active_trades.append({'ticker': ticker, 'side': 'LONG', 'entry_price': x['close'], 'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0})
                                    active_tickers.add(ticker)
                                    daily_trade_count[date_str] += 1
                                    if daily_trade_count[date_str] >= limit: break

            # =================================================================
            # STRATEGY 40: Multi-Timeframe Alignment
            # =================================================================
            elif strategy_id == 40:
                if daily_trade_count[date_str] < limit:
                    for x in top_15m_longs:
                        if x['h1_long_rank'] <= 10 and x['m30_long_rank'] <= 10 and x['long_rank'] <= 5 and x['ibs'] < 0.3:
                            ticker = x['ticker']
                            if ticker not in active_tickers:
                                active_trades.append({'ticker': ticker, 'side': 'LONG', 'entry_price': x['close'], 'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0})
                                active_tickers.add(ticker)
                                daily_trade_count[date_str] += 1
                                if daily_trade_count[date_str] >= limit: break

            # =================================================================
            # STRATEGY 41: Late-Day Liquidation
            # =================================================================
            elif strategy_id == 41:
                curr_vol = dict_rolling_vol.get(T, 0.0)
                if time_str >= "14:00" and curr_vol > dict_p50_vol.get(T, 0.0):
                    if daily_trade_count[date_str] < limit:
                        for x in top_15m_shorts:
                            if x['short_rank'] <= 10 and x['ibs'] < 0.3:
                                ticker = x['ticker']
                                if ticker not in active_tickers:
                                    active_trades.append({'ticker': ticker, 'side': 'SHORT', 'entry_price': x['close'], 'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0})
                                    active_tickers.add(ticker)
                                    daily_trade_count[date_str] += 1
                                    if daily_trade_count[date_str] >= limit: break

            # =================================================================
            # STRATEGY 42: Trend Exhaustion Trap
            # =================================================================
            elif strategy_id == 42:
                if daily_trade_count[date_str] < limit:
                    for x in top_15m_shorts:
                        if x['daily_long_rank'] <= 15 and x['short_rank'] <= 10 and x['ibs'] < 0.3:
                            ticker = x['ticker']
                            if ticker not in active_tickers:
                                active_trades.append({'ticker': ticker, 'side': 'SHORT', 'entry_price': x['close'], 'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0})
                                active_tickers.add(ticker)
                                daily_trade_count[date_str] += 1
                                if daily_trade_count[date_str] >= limit: break

            # =================================================================
            # STRATEGY 43: Relative Strength Breakout
            # =================================================================
            elif strategy_id == 43:
                if daily_trade_count[date_str] < limit:
                    for x in top_15m_longs:
                        if x['spread_zscore'] > 2.0 and x['long_rank'] <= 3 and x['vol_rank_pct'] > 0.80:
                            ticker = x['ticker']
                            if ticker not in active_tickers:
                                active_trades.append({'ticker': ticker, 'side': 'LONG', 'entry_price': x['close'], 'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0})
                                active_tickers.add(ticker)
                                daily_trade_count[date_str] += 1
                                if daily_trade_count[date_str] >= limit: break

            # =================================================================
            # STRATEGY 44: Mean Reversion Grind
            # =================================================================
            elif strategy_id == 44:
                if daily_trade_count[date_str] < limit:
                    for x in top_15m_longs:
                        if x['spread_zscore'] < -1.5 and x['long_rank'] <= 10:
                            ticker = x['ticker']
                            if ticker not in active_tickers:
                                active_trades.append({'ticker': ticker, 'side': 'LONG', 'entry_price': x['close'], 'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0})
                                active_tickers.add(ticker)
                                daily_trade_count[date_str] += 1
                                if daily_trade_count[date_str] >= limit: break

            # =================================================================
            # STRATEGY 45: Volatility Contraction Short
            # =================================================================
            elif strategy_id == 45:
                if daily_trade_count[date_str] < limit:
                    for x in top_15m_shorts:
                        if x['short_rank'] <= 5 and x['atr_14_pct'] < 0.002:
                            ticker = x['ticker']
                            if ticker not in active_tickers:
                                active_trades.append({'ticker': ticker, 'side': 'SHORT', 'entry_price': x['close'], 'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0})
                                active_tickers.add(ticker)
                                daily_trade_count[date_str] += 1
                                if daily_trade_count[date_str] >= limit: break

            # =================================================================
            # STRATEGY 46: 1H Pullback Buy
            # =================================================================
            elif strategy_id == 46:
                if daily_trade_count[date_str] < limit:
                    for x in active_15m:
                        if x['h1_long_rank'] <= 5 and x['long_rank'] > 15 and x['ibs'] < 0.3:
                            ticker = x['ticker']
                            if ticker not in active_tickers:
                                active_trades.append({'ticker': ticker, 'side': 'LONG', 'entry_price': x['close'], 'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0})
                                active_tickers.add(ticker)
                                daily_trade_count[date_str] += 1
                                if daily_trade_count[date_str] >= limit: break

            # =================================================================
            # STRATEGY 47: EOD Squeeze Long
            # =================================================================
            elif strategy_id == 47:
                if time_str >= "14:30":
                    if daily_trade_count[date_str] < limit:
                        for x in top_15m_longs:
                            if x['long_rank'] <= 10 and x['ibs'] > 0.7 and x['atr_14_pct'] < 0.005:
                                ticker = x['ticker']
                                if ticker not in active_tickers:
                                    active_trades.append({'ticker': ticker, 'side': 'LONG', 'entry_price': x['close'], 'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0})
                                    active_tickers.add(ticker)
                                    daily_trade_count[date_str] += 1
                                    if daily_trade_count[date_str] >= limit: break

            # =================================================================
            # STRATEGY 48: The Gap and Crap
            # =================================================================
            elif strategy_id == 48:
                if time_str <= "10:00":
                    if daily_trade_count[date_str] < limit:
                        for x in top_15m_shorts:
                            if x['gap_pct'] > 0.015 and x['short_rank'] <= 5:
                                ticker = x['ticker']
                                if ticker not in active_tickers:
                                    active_trades.append({'ticker': ticker, 'side': 'SHORT', 'entry_price': x['close'], 'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0})
                                    active_tickers.add(ticker)
                                    daily_trade_count[date_str] += 1
                                    if daily_trade_count[date_str] >= limit: break

            # =================================================================
            # STRATEGY 49: The Persistent Dip
            # =================================================================
            elif strategy_id == 49:
                if daily_trade_count[date_str] < limit:
                    for x in active_15m:
                        if x['persist_long_3bar_5'] and x['ibs'] < 0.2:
                            ticker = x['ticker']
                            if ticker not in active_tickers:
                                active_trades.append({'ticker': ticker, 'side': 'LONG', 'entry_price': x['close'], 'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0})
                                active_tickers.add(ticker)
                                daily_trade_count[date_str] += 1
                                if daily_trade_count[date_str] >= limit: break

            # =================================================================
            # STRATEGY 50: The Perfect Storm
            # =================================================================
            elif strategy_id == 50:
                curr_vol = dict_rolling_vol.get(T, 0.0)
                if curr_vol > dict_p50_vol.get(T, 0.0):
                    if daily_trade_count[date_str] < limit:
                        for x in active_15m:
                            if x['h1_long_rank'] <= 15 and x['m30_long_rank'] <= 15 and x['long_rank'] <= 15 and x['daily_long_rank'] <= 15 and x['ibs'] > 0.5:
                                ticker = x['ticker']
                                if ticker not in active_tickers:
                                    active_trades.append({'ticker': ticker, 'side': 'LONG', 'entry_price': x['close'], 'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0})
                                    active_tickers.add(ticker)
                                    daily_trade_count[date_str] += 1
                                    if daily_trade_count[date_str] >= limit: break
"""

    content = content[:start_idx] + new_strategies + "\n" + content[end_idx:]

    # Fix ma20 KeyError in the generated script
    ma20_target = """            'bb_upper': float(row['bb_upper']),
            'bb_lower': float(row['bb_lower']),"""
    ma20_replacement = """            'bb_upper': float(row['bb_upper']),
            'bb_lower': float(row['bb_lower']),
            'ma20': float(row['ma20']) if pd.notna(row['ma20']) else float(row['Close']),"""
    content = content.replace(ma20_target, ma20_replacement)
    
    # Replace strategies_info list
    list_start_str = '    strategies_info = ['
    list_end_str = '    ]\n'
    
    l_s = content.find(list_start_str)
    l_e = content.find(list_end_str, l_s)
    
    new_info = '''    strategies_info = [
        (36, "The Opening Drive"),
        (37, "Mid-Morning Reversal"),
        (38, "Dead-Cat Bounce"),
        (39, "The VWAP Pinch"),
        (40, "Multi-Timeframe Alignment"),
        (41, "Late-Day Liquidation"),
        (42, "Trend Exhaustion Trap"),
        (43, "Relative Strength Breakout"),
        (44, "Mean Reversion Grind"),
        (45, "Volatility Contraction Short"),
        (46, "1H Pullback Buy"),
        (47, "EOD Squeeze Long"),
        (48, "The Gap and Crap"),
        (49, "The Persistent Dip"),
        (50, "The Perfect Storm")
    ]
'''
    content = content[:l_s] + new_info + content[l_e + len(list_end_str):]
    
    # Also change output filename so we don't overwrite
    content = content.replace('"data/strategy_10_new_results.json"', '"data/strategy_15_final_results.json"')
    content = content.replace('BACKTEST SIMULATION SUMMARY TABLE (NEW S26-S35)', 'BACKTEST SIMULATION SUMMARY TABLE (FINAL S36-S50)')

    # Add 36-50 to target_hold max holds map just in case, wait, uniform 4-bar is applied for all! So we are safe.

    with open('scripts/strategy_15_final_backtest.py', 'w') as f:
        f.write(content)
        
    print("Successfully built scripts/strategy_15_final_backtest.py")

if __name__ == '__main__':
    build()

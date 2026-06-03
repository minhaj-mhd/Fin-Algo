import re

def build():
    with open('scripts/strategy_25x_4bar_test.py', 'r') as f:
        content = f.read()
        
    # Replace the avg_holds logic with a uniform 4-bar hold for the new strategies
    old_avg_holds = """                    # DYNAMIC AVG HOLD TEST
                    avg_holds = {
                        1: 1, 2: 4, 3: 1, 4: 1, 5: 1, 6: 3, 7: 1, 8: 3, 9: 1, 10: 1,
                        11: 4, 12: 3, 13: 2, 14: 4, 15: 1, 16: 4, 17: 4, 18: 6, 19: 4,
                        20: 2, 21: 4, 22: 3, 23: 2, 24: 4, 25: 2
                    }
                    target_hold = avg_holds.get(strategy_id, 2)"""
                    
    new_avg_holds = """                    # UNIFORM 4-BAR HOLD TEST FOR S26-S35
                    target_hold = 4"""
    
    content = content.replace(old_avg_holds, new_avg_holds)
    
    # Extract the chunk from STRATEGY 1 to end of STRATEGY 25
    start_str = '            # =================================================================\n            # STRATEGY 1: "Daily Macro Gatekeeper"\n            # ================================================================='
    
    # We will find where STRATEGY 25 ends.
    # It ends before "if not exit_reason and is_last_bar:" ? No, the strategy logic is in "EVALUATE NEW ENTRIES"
    # The start_str is under "# B. EVALUATE NEW ENTRIES"
    
    end_str = '        # End of simulation for this strategy, evaluate trades'
    
    start_idx = content.find(start_str)
    end_idx = content.find(end_str)
    
    if start_idx == -1 or end_idx == -1:
        print("Could not find replacement bounds!")
        return

    new_strategies = """
            # =================================================================
            # STRATEGY 26: The Morning Gap Reversal
            # =================================================================
            if strategy_id == 26:
                if time_str == "09:15":
                    if daily_trade_count[date_str] < limit:
                        for x in top_15m_longs:
                            if x['gap_pct'] < -0.01 and x['h1_long_rank'] <= 3 and x['daily_long_rank'] <= 3:
                                ticker = x['ticker']
                                if ticker not in active_tickers:
                                    active_trades.append({'ticker': ticker, 'side': 'LONG', 'entry_price': x['close'], 'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0})
                                    active_tickers.add(ticker)
                                    daily_trade_count[date_str] += 1
                                    if daily_trade_count[date_str] >= limit: break

            # =================================================================
            # STRATEGY 27: Consecutive Conviction Acceleration
            # =================================================================
            elif strategy_id == 27:
                if daily_trade_count[date_str] < limit:
                    for x in top_15m_longs:
                        if x['long_rank'] <= 10 and x['long_conv'] > x['long_conv_lag1'] and x['long_conv_lag1'] > x['long_conv_lag2'] and x['long_conv_lag2'] > 0:
                            ticker = x['ticker']
                            if ticker not in active_tickers:
                                active_trades.append({'ticker': ticker, 'side': 'LONG', 'entry_price': x['close'], 'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0})
                                active_tickers.add(ticker)
                                daily_trade_count[date_str] += 1
                                if daily_trade_count[date_str] >= limit: break
                    for x in top_15m_shorts:
                        if daily_trade_count[date_str] >= limit: break
                        if x['short_rank'] <= 10 and x['short_conv'] > x['short_conv_lag1'] and x['short_conv_lag1'] > x['short_conv_lag2'] and x['short_conv_lag2'] > 0:
                            ticker = x['ticker']
                            if ticker not in active_tickers:
                                active_trades.append({'ticker': ticker, 'side': 'SHORT', 'entry_price': x['close'], 'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0})
                                active_tickers.add(ticker)
                                daily_trade_count[date_str] += 1

            # =================================================================
            # STRATEGY 28: Midday Volatility Squeeze
            # =================================================================
            elif strategy_id == 28:
                curr_vol = dict_rolling_vol.get(T, 0.0)
                if curr_vol < dict_p30_vol.get(T, 0.0): # LOW VOL
                    if time_str in ["11:30", "11:45", "12:00", "12:15", "12:30", "12:45", "13:00"]:
                        if daily_trade_count[date_str] < limit:
                            for x in top_15m_longs:
                                if x['long_rank'] <= 3 and 0.4 <= x['ibs'] <= 0.6:
                                    ticker = x['ticker']
                                    if ticker not in active_tickers:
                                        active_trades.append({'ticker': ticker, 'side': 'LONG', 'entry_price': x['close'], 'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0})
                                        active_tickers.add(ticker)
                                        daily_trade_count[date_str] += 1
                                        if daily_trade_count[date_str] >= limit: break

            # =================================================================
            # STRATEGY 29: Contrarian High-Vol Fade
            # =================================================================
            elif strategy_id == 29:
                curr_vol = dict_rolling_vol.get(T, 0.0)
                if curr_vol > dict_p70_vol.get(T, 0.0): # HIGH VOL
                    if daily_trade_count[date_str] < limit:
                        for x in top_15m_shorts:
                            if x['short_rank'] <= 5 and x['close'] >= x['bb_upper']:
                                ticker = x['ticker']
                                if ticker not in active_tickers:
                                    active_trades.append({'ticker': ticker, 'side': 'SHORT', 'entry_price': x['close'], 'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0})
                                    active_tickers.add(ticker)
                                    daily_trade_count[date_str] += 1
                                    if daily_trade_count[date_str] >= limit: break

            # =================================================================
            # STRATEGY 30: Macro Alignment Scalper
            # =================================================================
            elif strategy_id == 30:
                if daily_trade_count[date_str] < limit:
                    for x in top_15m_longs:
                        if x['long_rank'] <= 5 and x['h1_short_rank'] <= 5 and x['daily_short_rank'] <= 5:
                            # 15M says Buy, but macro says massive Short. We short it.
                            ticker = x['ticker']
                            if ticker not in active_tickers:
                                active_trades.append({'ticker': ticker, 'side': 'SHORT', 'entry_price': x['close'], 'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0})
                                active_tickers.add(ticker)
                                daily_trade_count[date_str] += 1
                                if daily_trade_count[date_str] >= limit: break

            # =================================================================
            # STRATEGY 31: Extreme Z-Score Reversion
            # =================================================================
            elif strategy_id == 31:
                if daily_trade_count[date_str] < limit:
                    for x in active_15m:
                        if x['spread_zscore'] > 2.5: # Extremely overvalued conviction
                            ticker = x['ticker']
                            if ticker not in active_tickers:
                                active_trades.append({'ticker': ticker, 'side': 'SHORT', 'entry_price': x['close'], 'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0})
                                active_tickers.add(ticker)
                                daily_trade_count[date_str] += 1
                                if daily_trade_count[date_str] >= limit: break
                        elif x['spread_zscore'] < -2.5: # Extremely undervalued conviction
                            ticker = x['ticker']
                            if ticker not in active_tickers:
                                active_trades.append({'ticker': ticker, 'side': 'LONG', 'entry_price': x['close'], 'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0})
                                active_tickers.add(ticker)
                                daily_trade_count[date_str] += 1
                                if daily_trade_count[date_str] >= limit: break

            # =================================================================
            # STRATEGY 32: The Persistent Anchor
            # =================================================================
            elif strategy_id == 32:
                if time_str == "13:00":
                    if daily_trade_count[date_str] < limit:
                        for x in active_15m:
                            if x['persist_long_4bar_3']:
                                ticker = x['ticker']
                                if ticker not in active_tickers:
                                    active_trades.append({'ticker': ticker, 'side': 'LONG', 'entry_price': x['close'], 'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0})
                                    active_tickers.add(ticker)
                                    daily_trade_count[date_str] += 1
                                    if daily_trade_count[date_str] >= limit: break

            # =================================================================
            # STRATEGY 33: EOD Retail Liquidity Trap
            # =================================================================
            elif strategy_id == 33:
                if time_str >= "14:15":
                    if daily_trade_count[date_str] < limit:
                        for x in top_15m_shorts:
                            if x['short_rank'] <= 5 and x['ibs'] > 0.85 and x['short_conv'] > x['long_conv']:
                                ticker = x['ticker']
                                if ticker not in active_tickers:
                                    active_trades.append({'ticker': ticker, 'side': 'SHORT', 'entry_price': x['close'], 'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0})
                                    active_tickers.add(ticker)
                                    daily_trade_count[date_str] += 1
                                    if daily_trade_count[date_str] >= limit: break

            # =================================================================
            # STRATEGY 34: Triple-Timeframe Laggard
            # =================================================================
            elif strategy_id == 34:
                if daily_trade_count[date_str] < limit:
                    for x in active_15m:
                        if x['h1_long_rank'] <= 3 and x['m30_long_rank'] <= 3 and x['long_rank'] > 20:
                            ticker = x['ticker']
                            if ticker not in active_tickers:
                                active_trades.append({'ticker': ticker, 'side': 'LONG', 'entry_price': x['close'], 'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0})
                                active_tickers.add(ticker)
                                daily_trade_count[date_str] += 1
                                if daily_trade_count[date_str] >= limit: break

            # =================================================================
            # STRATEGY 35: Volatility Contraction Breakout
            # =================================================================
            elif strategy_id == 35:
                if daily_trade_count[date_str] < limit:
                    for x in active_15m:
                        if x['atr_14_pct'] < 0.002 and x['blend_long_rank'] <= 5:
                            ticker = x['ticker']
                            if ticker not in active_tickers:
                                active_trades.append({'ticker': ticker, 'side': 'LONG', 'entry_price': x['close'], 'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0})
                                active_tickers.add(ticker)
                                daily_trade_count[date_str] += 1
                                if daily_trade_count[date_str] >= limit: break

"""

    content = content[:start_idx] + new_strategies + "\n" + content[end_idx:]
    
    # Replace strategies_info list
    list_start_str = '    strategies_info = ['
    list_end_str = '    ]\n'
    
    l_s = content.find(list_start_str)
    l_e = content.find(list_end_str, l_s)
    
    new_info = '''    strategies_info = [
        (26, "The Morning Gap Reversal"),
        (27, "Consecutive Conviction Acceleration"),
        (28, "Midday Volatility Squeeze"),
        (29, "Contrarian High-Vol Fade"),
        (30, "Macro Alignment Scalper"),
        (31, "Extreme Z-Score Reversion"),
        (32, "The Persistent Anchor"),
        (33, "EOD Retail Liquidity Trap"),
        (34, "Triple-Timeframe Laggard"),
        (35, "Volatility Contraction Breakout")
    ]
'''
    content = content[:l_s] + new_info + content[l_e + len(list_end_str):]
    
    # Also change output filename so we don't overwrite
    content = content.replace('"data/strategy_25x_results.json"', '"data/strategy_10_new_results.json"')
    content = content.replace('BACKTEST SIMULATION SUMMARY TABLE (25 STRATEGIES)', 'BACKTEST SIMULATION SUMMARY TABLE (NEW S26-S35)')

    with open('scripts/strategy_10_new_backtest.py', 'w') as f:
        f.write(content)
        
    print("Successfully built scripts/strategy_10_new_backtest.py")

if __name__ == '__main__':
    build()

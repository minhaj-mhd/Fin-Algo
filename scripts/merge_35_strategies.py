import re

def merge():
    with open('scripts/strategy_25x_4bar_test.py', 'r') as f:
        content = f.read()
        
    # We need to insert S26-S35 after S25.
    # Find the end of S25 block.
    # The end of S25 block is right before:
    #         # End of simulation for this strategy, evaluate trades
    
    end_s25_str = '        # End of simulation for this strategy, evaluate trades'
    end_idx = content.find(end_s25_str)
    
    if end_idx == -1:
        print("Could not find end of S25 block!")
        return

    new_strategies = """
            # =================================================================
            # STRATEGY 26: The Morning Gap Reversal
            # =================================================================
            elif strategy_id == 26:
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

    content = content[:end_idx] + new_strategies + content[end_idx:]
    
    # Update max limits dictionary
    old_max_daily = """        max_daily = {
            1: 6, 2: 4, 3: 6, 4: 8, 5: 4, 6: 6, 8: 6, 9: 6, 10: 4,
            11: 6, 12: 6, 13: 6, 14: 6, 15: 4, 16: 4, 17: 6, 18: 4,
            19: 6, 20: 6, 21: 6, 22: 4, 23: 4, 24: 6, 25: 6
        }"""
        
    new_max_daily = """        max_daily = {
            1: 6, 2: 4, 3: 6, 4: 8, 5: 4, 6: 6, 8: 6, 9: 6, 10: 4,
            11: 6, 12: 6, 13: 6, 14: 6, 15: 4, 16: 4, 17: 6, 18: 4,
            19: 6, 20: 6, 21: 6, 22: 4, 23: 4, 24: 6, 25: 6,
            26: 4, 27: 6, 28: 4, 29: 6, 30: 4, 31: 6, 32: 6, 33: 6, 34: 6, 35: 4
        }"""
        
    content = content.replace(old_max_daily, new_max_daily)
    
    # Replace avg_holds
    old_avg_holds = """                    avg_holds = {
                        1: 1, 2: 4, 3: 1, 4: 1, 5: 1, 6: 3, 7: 1, 8: 3, 9: 1, 10: 1,
                        11: 4, 12: 3, 13: 2, 14: 4, 15: 1, 16: 4, 17: 4, 18: 6, 19: 4,
                        20: 2, 21: 4, 22: 3, 23: 2, 24: 4, 25: 2
                    }"""
    new_avg_holds = """                    avg_holds = {
                        1: 1, 2: 4, 3: 1, 4: 1, 5: 1, 6: 3, 7: 1, 8: 3, 9: 1, 10: 1,
                        11: 4, 12: 3, 13: 2, 14: 4, 15: 1, 16: 4, 17: 4, 18: 6, 19: 4,
                        20: 2, 21: 4, 22: 3, 23: 2, 24: 4, 25: 2,
                        26: 4, 27: 4, 28: 4, 29: 3, 30: 4, 31: 4, 32: 4, 33: 4, 34: 4, 35: 4
                    }"""
    content = content.replace(old_avg_holds, new_avg_holds)
    
    # Replace strategies list
    list_end_str = '        (25, "Triple-Timeframe Momentum Grid")\n    ]'
    new_list_end = '''        (25, "Triple-Timeframe Momentum Grid"),
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
    ]'''
    
    content = content.replace(list_end_str, new_list_end)
    
    # File outputs
    content = content.replace('"data/strategy_25x_results.json"', '"data/strategy_35x_results.json"')
    content = content.replace('BACKTEST SIMULATION SUMMARY TABLE (25 STRATEGIES)', 'BACKTEST SIMULATION SUMMARY TABLE (35 STRATEGIES)')

    with open('scripts/strategy_35x_backtest.py', 'w') as f:
        f.write(content)
        
    print("Successfully built scripts/strategy_35x_backtest.py")

if __name__ == '__main__':
    merge()

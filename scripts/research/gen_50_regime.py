import os
import re

def build_50_regime():
    with open('scripts/strategy_35x_backtest.py', 'r') as f:
        content = f.read()
        
    # We will inject the GatekeeperList class at the top of the file
    gk_class = """
class GatekeeperList(list):
    def __init__(self, apply_gatekeeper, d_daily, dict_daily, daily_trade_count, date_str, limit):
        super().__init__()
        self.apply_gatekeeper = apply_gatekeeper
        self.d_daily = d_daily
        self.dict_daily = dict_daily
        self.daily_trade_count = daily_trade_count
        self.date_str = date_str
        self.limit = limit
        
    def append(self, trade):
        # Prevent appending if daily limit is reached
        if self.daily_trade_count[self.date_str] >= self.limit:
            return
            
        if self.apply_gatekeeper and self.d_daily and self.d_daily in self.dict_daily:
            ticker = trade.get('ticker')
            if not ticker and 'long_ticker' in trade:
                # Pair trade S6 logic
                pass # Pairs are market neutral, usually we skip gating, or we gate both legs. S6 is market neutral so let's allow.
            elif ticker:
                d_pred = self.dict_daily[self.d_daily].get(ticker)
                if d_pred:
                    if trade['side'] == 'LONG' and d_pred['long_rank'] > 0.40 * d_pred['count']:
                        return # Blocked by gatekeeper
                    if trade['side'] == 'SHORT' and d_pred['short_rank'] > 0.40 * d_pred['count']:
                        return # Blocked by gatekeeper
        super().append(trade)
"""
    content = content.replace("import pandas as pd", "import pandas as pd\n" + gk_class)

    # Change simulate_strategy signature
    content = content.replace("def simulate_strategy(strategy_id, name):", "def simulate_strategy(strategy_id, name, apply_gatekeeper=False):")
    
    # Replace active_trades = [] with GatekeeperList
    content = content.replace("active_trades = [] # list of active trade dicts", 
                              "active_trades = [] # Changed further down")
    
    # Inject the actual instantiation inside the bar loop so it has access to d_daily?
    # No, active_trades tracks ACTIVE trades across bars. We shouldn't drop them.
    # The Gatekeeper only blocks entries!
    # If we replace active_trades.append with a function call `try_entry(trade)` it would be safer.
    # Let's replace `active_trades.append` with a custom block:
    # Actually, if we just redefine `active_trades = GatekeeperList(...)` it will persist trades, and `.append()` will just filter entries.
    
    # Better approach: string replace `active_trades.append(` with `append_trade(active_trades, `
    # And define `append_trade(active_trades_list, trade_dict)` inside simulate_strategy.
    
    append_func = """        def append_trade(lst, trade):
            if apply_gatekeeper and d_daily and d_daily in dict_daily:
                ticker = trade.get('ticker')
                if ticker:
                    d_pred = dict_daily[d_daily].get(ticker)
                    if d_pred:
                        if trade['side'] == 'LONG' and d_pred['long_rank'] > 0.40 * d_pred['count']:
                            return # Blocked
                        if trade['side'] == 'SHORT' and d_pred['short_rank'] > 0.40 * d_pred['count']:
                            return # Blocked
            lst.append(trade)
"""
    content = content.replace("        active_trades = [] # Changed further down",
                              "        active_trades = [] # Changed further down\n" + append_func)
                              
    content = content.replace("active_trades.append(", "append_trade(active_trades, ")
    
    # Now we need to append S36-S50 from build_s36_s50.py
    with open('scripts/build_s36_s50.py', 'r') as f:
        s36_content = f.read()
        
    start_s36 = s36_content.find('# STRATEGY 36:')
    end_s36 = s36_content.find('"""\n\n    content = content[:start_idx]')
    s36_block = s36_content[start_s36 - 80:end_s36] # Include the separator
    
    # Also we need to change S36's active_trades.append in this block
    s36_block = s36_block.replace("active_trades.append(", "append_trade(active_trades, ")
    
    # Find the end of S35 in content
    end_s35_str = '        # End of simulation for this strategy, evaluate trades'
    end_idx = content.find(end_s35_str)
    content = content[:end_idx] + s36_block + "\n" + content[end_idx:]
    
    # Fix ma20 KeyError for S39
    ma20_target = """            'bb_upper': float(row['bb_upper']),
            'bb_lower': float(row['bb_lower']),"""
    ma20_replacement = """            'bb_upper': float(row['bb_upper']),
            'bb_lower': float(row['bb_lower']),
            'ma20': float(row['ma20']) if pd.notna(row['ma20']) else float(row['Close']),"""
    content = content.replace(ma20_target, ma20_replacement)
    
    # Update max_daily limits map
    old_max_daily = """        max_daily = {
            1: 6, 2: 4, 3: 6, 4: 8, 5: 4, 6: 6, 8: 6, 9: 6, 10: 4,
            11: 6, 12: 6, 13: 6, 14: 6, 15: 4, 16: 4, 17: 6, 18: 4,
            19: 6, 20: 6, 21: 6, 22: 4, 23: 4, 24: 6, 25: 6,
            26: 4, 27: 6, 28: 4, 29: 6, 30: 4, 31: 6, 32: 6, 33: 6, 34: 6, 35: 4
        }"""
    new_max_daily = """        max_daily = {
            1: 6, 2: 4, 3: 6, 4: 8, 5: 4, 6: 6, 8: 6, 9: 6, 10: 4,
            11: 6, 12: 6, 13: 6, 14: 6, 15: 4, 16: 4, 17: 6, 18: 4,
            19: 6, 20: 6, 21: 6, 22: 4, 23: 4, 24: 6, 25: 6,
            26: 4, 27: 6, 28: 4, 29: 6, 30: 4, 31: 6, 32: 6, 33: 6, 34: 6, 35: 4,
            36: 4, 37: 4, 38: 4, 39: 4, 40: 4, 41: 4, 42: 4, 43: 4, 44: 4, 45: 4,
            46: 4, 47: 4, 48: 4, 49: 4, 50: 4
        }"""
    content = content.replace(old_max_daily, new_max_daily)
    
    # Update avg_holds
    old_avg_holds = """                    avg_holds = {
                        1: 1, 2: 4, 3: 1, 4: 1, 5: 1, 6: 3, 7: 1, 8: 3, 9: 1, 10: 1,
                        11: 4, 12: 3, 13: 2, 14: 4, 15: 1, 16: 4, 17: 4, 18: 6, 19: 4,
                        20: 2, 21: 4, 22: 3, 23: 2, 24: 4, 25: 2,
                        26: 4, 27: 4, 28: 4, 29: 3, 30: 4, 31: 4, 32: 4, 33: 4, 34: 4, 35: 4
                    }"""
    new_avg_holds = """                    avg_holds = {
                        1: 1, 2: 4, 3: 1, 4: 1, 5: 1, 6: 3, 7: 1, 8: 3, 9: 1, 10: 1,
                        11: 4, 12: 3, 13: 2, 14: 4, 15: 1, 16: 4, 17: 4, 18: 6, 19: 4,
                        20: 2, 21: 4, 22: 3, 23: 2, 24: 4, 25: 2,
                        26: 4, 27: 4, 28: 4, 29: 3, 30: 4, 31: 4, 32: 4, 33: 4, 34: 4, 35: 4,
                        36: 4, 37: 4, 38: 4, 39: 4, 40: 4, 41: 4, 42: 4, 43: 4, 44: 4, 45: 4,
                        46: 4, 47: 4, 48: 4, 49: 4, 50: 4
                    }"""
    content = content.replace(old_avg_holds, new_avg_holds)
    
    # Now replace the loop where simulate_strategy is called!
    # Instead of just calling simulate_strategy(s_id, s_name), we need to call it twice.
    
    old_loop = """    for s_id, s_name in strategies_info:
        simulate_strategy(s_id, s_name)"""
        
    new_loop = """
    # 50-Strategy Regime-Aware Gated vs Ungated
    regime_results = {}
    for s_id, s_name in strategies_info:
        res_ungated = simulate_strategy(s_id, s_name, apply_gatekeeper=False)
        res_gated = simulate_strategy(s_id, s_name, apply_gatekeeper=True)
        regime_results[f"strategy_{s_id}_ungated"] = res_ungated
        regime_results[f"strategy_{s_id}_gated"] = res_gated
"""
    content = content.replace(old_loop, new_loop)
    
    # Update the strategies list
    # We need to find the `strategies_info = [` block
    s_info_start = content.find("    strategies_info = [")
    s_info_end = content.find("    ]\n", s_info_start)
    
    full_strategies = '''    strategies_info = [
        (1, "Daily Macro Gatekeeper"),
        (2, "Short-Side Specialist"),
        (3, "Timeframe Divergence Fade"),
        (4, "Score Momentum Scalper"),
        (5, "Power Hour Sniper"),
        (6, "Market-Neutral Pairs"),
        (7, "Volatility Regime Switcher"),
        (8, "Opening Range Breakout (ORB) + Confirmation"),
        (9, "Conviction Spread Z-Score"),
        (10, "Quad-Timeframe Unanimous"),
        (11, "Hourly Trend Rider"),
        (12, "Pre-Noon Reversal Scalp"),
        (13, "Microstructure Exhaustion Fade"),
        (14, "Volume Shock Breakout"),
        (15, "Triple Moving Conviction Fade"),
        (16, "Overnight Gap Fade"),
        (17, "Micro-Volatility Breakout"),
        (18, "Regime-Aware Trend Follower"),
        (19, "Low-Vol Squeeze Scalper"),
        (20, "High-Vol Knife Catcher"),
        (21, "Intraday Trend Exhaustion"),
        (22, "Conviction Peak Divergence"),
        (23, "Opening Range Trend Runner"),
        (24, "Rolling Z-Score Momentum"),
        (25, "Triple-Timeframe Momentum Grid"),
        (26, "The Morning Gap Reversal"),
        (27, "Consecutive Conviction Acceleration"),
        (28, "Midday Volatility Squeeze"),
        (29, "Contrarian High-Vol Fade"),
        (30, "Macro Alignment Scalper"),
        (31, "Extreme Z-Score Reversion"),
        (32, "The Persistent Anchor"),
        (33, "EOD Retail Liquidity Trap"),
        (34, "Triple-Timeframe Laggard"),
        (35, "Volatility Contraction Breakout"),
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
    ]'''
    content = content[:s_info_start] + full_strategies + content[s_info_end + 6:]
    
    # Update saving logic
    old_save = """    # Save the output results json
    output_path = "data/strategy_35x_results.json"
    with open(output_path, "w") as f:
        json.dump({
            'holdout_month': TEST_MONTH,
            'tickers_universe_size': len(common_tickers),
            'strategies': results,
            'backtested_at': datetime.now().isoformat()
        }, f, indent=2)"""
        
    new_save = """    # Save the output results json
    output_path = "data/strategy_50_regime_results.json"
    with open(output_path, "w") as f:
        json.dump({
            'holdout_month': TEST_MONTH,
            'tickers_universe_size': len(common_tickers),
            'strategies': regime_results,
            'backtested_at': datetime.now().isoformat()
        }, f, indent=2)"""
    content = content.replace(old_save, new_save)
    
    # Replace the summary print block at the end
    old_summary_start = content.find("    # 5. PRINT SUMMARY ASCII MARKDOWN TABLE")
    new_summary = """    print("\\n" + "=" * 120)
    print("REGIME-AWARE COMPARATIVE BACKTEST SUMMARY (GATED VS UNGATED) - ALL 50 STRATEGIES")
    print("=" * 120)
    headers = ["ID", "Strategy Name", "Gated WR", "Ungated WR", "Gated Net %", "Ungated Net %", "Verdict"]
    print(f"| {' | '.join(headers)} |")
    print(f"|{'-|-'.join(['-' * len(h) for h in headers])}|")
    
    for s_id, s_name in strategies_info:
        r_ungated = regime_results[f"strategy_{s_id}_ungated"]
        r_gated = regime_results[f"strategy_{s_id}_gated"]
        
        s_name_trunc = s_name[:35]
        
        g_wr_str = f"{r_gated['win_rate']*100:.1f}% ({r_gated['total_trades']}t)"
        u_wr_str = f"{r_ungated['win_rate']*100:.1f}% ({r_ungated['total_trades']}t)"
        
        g_pnl = r_gated['total_return']*100
        u_pnl = r_ungated['total_return']*100
        g_pnl_str = f"{g_pnl:+.2f}%"
        u_pnl_str = f"{u_pnl:+.2f}%"
        
        if g_pnl > u_pnl + 0.5:
            verdict = "TREND: Gatekeeper Required"
        elif u_pnl > g_pnl + 0.5:
            verdict = "TACTICAL: Run Ungated"
        elif g_pnl > 0 and u_pnl > 0:
            verdict = "ROBUST: Profitable in Both"
        else:
            verdict = "FAIL: Unprofitable"
            
        print(f"| {s_id:<2} | {s_name_trunc:<35} | {g_wr_str:<15} | {u_wr_str:<15} | {g_pnl_str:<11} | {u_pnl_str:<13} | {verdict:<25} |")
    print("=" * 120)

if __name__ == "__main__":
    main()
"""
    content = content[:old_summary_start] + new_summary
    
    with open('scripts/strategy_50_regime_backtest.py', 'w') as f:
        f.write(content)
        
    print("Successfully built scripts/strategy_50_regime_backtest.py")

if __name__ == '__main__':
    build_50_regime()

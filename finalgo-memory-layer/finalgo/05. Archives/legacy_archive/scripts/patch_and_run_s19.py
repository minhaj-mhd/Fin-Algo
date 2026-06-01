import sys
import os
import re

with open("scripts/strategy_35x_backtest.py", "r", encoding="utf-8") as f:
    source = f.read()

# Modify the strategies list
source = re.sub(r'strategies_info = \[\s*\(1, "Daily Macro Gatekeeper"\),.*?\]', r'strategies_info = [(19, "Low-Volatility Grind (IBS Reversion)")]', source, flags=re.DOTALL)

# Add print statements to evaluate_strategy_trades
old_eval = "    return {\n        'strategy': name,"
new_eval = """    print(f"\\n--- TRADES FOR {name} ---")
    if trades:
        import pandas as _pd
        _df = _pd.DataFrame(trades)
        _pd.set_option('display.max_columns', None)
        _pd.set_option('display.max_rows', None)
        _pd.set_option('display.width', 1000)
        print(_df.to_string())
    else:
        print("NO TRADES")
    print("---------------------------\\n")
    return {
        'strategy': name,"""
source = source.replace(old_eval, new_eval)

with open("scripts/run_s19_temp.py", "w", encoding="utf-8") as f:
    f.write(source)

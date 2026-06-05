from scripts.vanguard.trade_state import TradeStateManager
import sqlite3

conn = sqlite3.connect('data/vanguard_trades.db')
conn.row_factory = sqlite3.Row
c = conn.cursor()
c.execute("SELECT * FROM trades WHERE trade_id = 'TRADE-BAJAJ-AUTO.NS-LONG-260604123154'")
row = c.fetchone()
trade = dict(row)
print("Before state evaluation:")
print("breakeven_locked:", trade.get("breakeven_locked"))
print("peak_profit_pct:", trade.get("peak_profit_pct"))
print("stop_loss_pct:", trade.get("stop_loss_pct"))

from datetime import datetime
pnl = float(trade["peak_profit_pct"])  # simulate it hitting its peak
should_exit, exit_status, exit_note = TradeStateManager.evaluate_open_trade_exit(trade, trade["entry_price"], pnl, datetime.now())

print("\nAfter simulating hit peak PnL:")
print("breakeven_locked:", trade.get("breakeven_locked"))

pnl = -0.0583  # simulate falling back to negative
should_exit, exit_status, exit_note = TradeStateManager.evaluate_open_trade_exit(trade, trade["entry_price"], pnl, datetime.now())

print("\nAfter simulating falling back:")
print("is_breakeven_exit triggered?", should_exit, exit_status)

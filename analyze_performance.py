import sqlite3
import pandas as pd

def analyze_performance():
    conn = sqlite3.connect('data/vanguard_trades.db')
    
    # Get all trades for today
    # Timestamp starts with '2026-06-04'
    query = """
    SELECT ticker, side, status, final_profit_pct, peak_profit_pct, net_pnl_amt, strategy_id, comment, exit_time
    FROM trades 
    WHERE timestamp LIKE '2026-06-04%'
    """
    
    df = pd.read_sql_query(query, conn)
    
    if df.empty:
        print("No trades found for today (2026-06-04).")
        return
        
    print(f"Total Trades Today: {len(df)}")
    
    closed_trades = df[df['status'].isin(['CLOSED', 'TAKE_PROFIT', 'STOP_LOSS'])]
    print(f"Closed Trades: {len(closed_trades)}")
    
    open_trades = df[df['status'] == 'OPEN']
    print(f"Open Trades: {len(open_trades)}")
    
    if not closed_trades.empty:
        total_pnl_amt = closed_trades['net_pnl_amt'].sum()
        winners = closed_trades[closed_trades['final_profit_pct'] > 0]
        losers = closed_trades[closed_trades['final_profit_pct'] < 0]
        breakevens = closed_trades[closed_trades['final_profit_pct'] == 0]
        
        print(f"\\n--- Realized Performance ---")
        print(f"Total Realized P&L Amount: Rs{total_pnl_amt:.2f}")
        print(f"Win Rate: {(len(winners)/len(closed_trades))*100:.1f}% ({len(winners)}W / {len(losers)}L / {len(breakevens)}BE)")
        
        print(f"\\n--- Closed Trades Details ---")
        for _, row in closed_trades.iterrows():
            reason = "Conviction Flip" if "Conviction Flip" in row['comment'] else "Regular Exit"
            if "Breakeven" in row['comment']: reason = "Breakeven Exit"
            print(f"{row['ticker']} ({row['side']}): {row['final_profit_pct']:.2f}% (Peak: {row['peak_profit_pct']:.2f}%) [{reason}]")
            
    if not open_trades.empty:
        print(f"\\n--- Active Open Trades ---")
        for _, row in open_trades.iterrows():
            print(f"{row['ticker']} ({row['side']}): Current PnL {row['final_profit_pct']:.2f}% (Peak: {row['peak_profit_pct']:.2f}%)")

if __name__ == '__main__':
    analyze_performance()

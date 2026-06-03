import pandas as pd
import numpy as np

trades_file = 'data/strategy_1030/backtest_trades_best.csv'
df = pd.read_csv(trades_file)

print(f"Total Trades: {len(df)}")
print(f"Win Rate: {(df['Net_Return'] > 0).mean():.2%}")
print(f"Average Win: {df[df['Net_Return'] > 0]['Net_Return'].mean():.4f}")
print(f"Average Loss: {df[df['Net_Return'] <= 0]['Net_Return'].mean():.4f}")

# Sort by return to see if it's outlier driven
df = df.sort_values('Net_Return', ascending=False)
print("\n--- Top 5 Best Trades ---")
print(df[['Date', 'Ticker', 'Direction', 'Net_Return']].head(5))

print("\n--- Top 5 Worst Trades ---")
print(df[['Date', 'Ticker', 'Direction', 'Net_Return']].tail(5))

# Calculate contribution of top X trades
total_profit_all = df['Net_Return'].sum()
top_10_profit = df['Net_Return'].head(10).sum()
print(f"\nTotal Net Return (Sum of all trades): {total_profit_all:.4f}")
print(f"Profit from top 10 trades: {top_10_profit:.4f}")
print(f"Top 10 trades account for {top_10_profit / total_profit_all:.1%} of total returns.")

# Check the long/short breakdown
print("\n--- By Direction ---")
print(df.groupby('Direction')['Net_Return'].agg(['count', 'mean', 'sum']))

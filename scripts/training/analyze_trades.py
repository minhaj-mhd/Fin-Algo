import pandas as pd

print("=" * 64)
print("TRADE ANALYSIS: Logic A1 Longs (Last 3 Months: 2026-04 to 2026-06)")
print("=" * 64)

df = pd.read_csv('data/A1_Long_Trades_Last12M.csv')
# Filter for last 3 months
df['YearMonth'] = df['DateTime'].str[:7]
df_3m = df[df['YearMonth'] >= '2026-04'].copy()

print(f"Total trades in last 3 months: {len(df_3m)}")

# Group by Ticker
print("\nTop Winning Tickers (by frequency & average return):")
ticker_stats = df_3m.groupby('Ticker').agg(
    Trades=('Return', 'count'),
    Avg_Raw_Return_bps=('Return', lambda x: x.mean() * 10000),
    Win_Rate=('Return', lambda x: (x > 0).mean() * 100)
).sort_values(by='Trades', ascending=False)

print(ticker_stats[ticker_stats['Trades'] >= 2])

print("\nBest Individual Trades (Top 10):")
best_trades = df_3m.sort_values(by='Return', ascending=False).head(10)
for _, row in best_trades.iterrows():
    print(f"  {row['DateTime']} | {row['Ticker']:<10} | V10_Rank: {row['V10_Rank_Score']:.4f} | V18_Prob: {row['V18_Prob']:.2%} | Ret: {row['Return']*10000:+.1f} bps")

print("=" * 64)

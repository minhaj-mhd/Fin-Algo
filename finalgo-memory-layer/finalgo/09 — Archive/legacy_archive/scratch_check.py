import pandas as pd
import numpy as np

# Load small chunks to inspect datetime formats and unique values
daily = pd.read_csv("data/ranking_data_upstox_daily_5y.csv", nrows=100)
m15 = pd.read_csv("data/ranking_data_upstox_15min_1y.csv", nrows=100)

print("Daily columns:", daily.columns.tolist())
print("Daily head:")
print(daily[['DateTime', 'Ticker', 'Close']].head())

print("\n15Min columns:", m15.columns.tolist())
print("15Min head:")
print(m15[['DateTime', 'Ticker', 'Close']].head())

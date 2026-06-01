import pandas as pd

# Load first 100 rows of 1H and 30M data
h1 = pd.read_csv("data/ranking_data_upstox_3y.csv", nrows=100)
m30 = pd.read_csv("data/ranking_data_upstox_30min_1y.csv", nrows=100)

print("1H Unique Time Parts in DateTime (first 100 rows):")
print(h1['DateTime'].str[11:].unique())

print("\n30M Unique Time Parts in DateTime (first 100 rows):")
print(m30['DateTime'].str[11:].unique())

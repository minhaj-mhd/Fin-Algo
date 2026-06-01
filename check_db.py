import sqlite3
import pandas as pd
conn = sqlite3.connect('data/vanguard_trades.db')
print(pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn))
try:
    df = pd.read_sql("SELECT * FROM trades LIMIT 1;", conn)
    print("Columns:", df.columns.tolist())
except Exception as e:
    print(e)

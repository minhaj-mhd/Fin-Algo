import sqlite3
conn = sqlite3.connect('data/vanguard_trades.db')
c = conn.cursor()
c.execute("SELECT COUNT(*) FROM trades WHERE ticker = 'BAJAJ-AUTO.NS' AND status = 'OPEN'")
print("Number of OPEN BAJAJ-AUTO trades in DB:", c.fetchone()[0])

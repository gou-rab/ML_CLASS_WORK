import sqlite3
import pandas as pd

conn = sqlite3.connect('sample-data.sql')  # ← Only this change
df = pd.read_sql_query("SELECT * FROM users", conn)
print(df.head())
conn.close()

import sqlite3

con = sqlite3.connect("exp3.db")
cur = con.cursor()
cur.execute("""SELECT * FROM exp3""")
for row in cur.fetchall():
  print(row)

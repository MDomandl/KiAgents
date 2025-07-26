
import pandas as pd
#import mysql.connector
import pymysql

# 🔧 Verbindung zur MySQL-Datenbank
conn = pymysql.connect(
    host="127.0.0.1",
    user="chatuser",
    password="chatpass",
    database="gptchats",
    charset="utf8mb4",
    cursorclass=pymysql.cursors.DictCursor
)
cursor = conn.cursor()

# 📥 Kategorien aus Excel einlesen
df_kategorien = pd.read_excel("chat_kategorien.xlsx")

# 🚀 Kategorien einfügen
for name in df_kategorien.iloc[:, 0].dropna().unique():
    cursor.execute(
        "INSERT IGNORE INTO kategorien (name) VALUES (%s)", (name,)
    )

conn.commit()
print("✅ Kategorien erfolgreich importiert.")
cursor.close()
conn.close()

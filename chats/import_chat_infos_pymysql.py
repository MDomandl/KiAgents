
import pandas as pd
import pymysql

# 📥 Excel-Datei mit Zuordnung (Suchbegriff → Kategorie) laden
df_infos = pd.read_excel("chat_infos.xlsx")

# 🔧 Verbindung zur MySQL-Datenbank über PyMySQL
conn = pymysql.connect(
    host="127.0.0.1",
    user="chatuser",
    password="chatpass",
    database="gptchats",
    charset="utf8mb4",
    cursorclass=pymysql.cursors.DictCursor
)
cursor = conn.cursor()

# Alle Chats holen
cursor.execute("SELECT id, titel FROM chats")
chat_records = cursor.fetchall()
chat_map = {titel.lower(): chat_id for chat_id, titel in chat_records}

# Alle Kategorien holen
cursor.execute("SELECT id, name FROM kategorien")
kat_records = cursor.fetchall()
kat_map = {name.lower(): kat_id for kat_id, name in kat_records}

# Kategorien vorschlagen anhand von Teil-Treffern im Titel
zuordnungen = []
for index, row in df_infos.iterrows():
    suchbegriff = str(row[0]).strip().lower()
    kategoriename = str(row[1]).strip().lower()

    for titel, chat_id in chat_map.items():
        if suchbegriff in titel:
            kategorie_id = kat_map.get(kategoriename)
            if kategorie_id:
                zuordnungen.append((chat_id, kategorie_id, 3))  # Relevanz = 3 (Standard)

# Duplikate entfernen
zuordnungen = list(set(zuordnungen))

# In chat_kategorien eintragen
for chat_id, kategorie_id, relevanz in zuordnungen:
    cursor.execute(
        "INSERT IGNORE INTO chat_kategorien (chat_id, kategorie_id, relevanz) VALUES (%s, %s, %s)",
        (chat_id, kategorie_id, relevanz)
    )

conn.commit()
print("✅ Kategorievorschläge erfolgreich eingetragen (mit PyMySQL).")
cursor.close()
conn.close()

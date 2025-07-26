
import json
import pymysql
from datetime import datetime

# 🔧 MySQL-Verbindung konfigurieren
conn = pymysql.connect(
    host="127.0.0.1",
    user="chatuser",
    password="chatpass",
    database="gptchats",
    charset="utf8mb4",
    cursorclass=pymysql.cursors.DictCursor
)
cursor = conn.cursor()

# 📥 conversations.json laden
with open("conversations.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Jeder Chat-Eintrag im JSON ist ein Element in der Liste
for chat in data:
    chat_id = chat.get("id")
    titel = chat.get("title", "")[:255]
    create_time = chat.get("create_time", 0)
    erstellt_am = datetime.fromtimestamp(create_time)

    # Letzte Nachricht finden
    messages = chat.get("mapping", {}).values()
    message_times = []
    for msg in messages:
        if "create_time" in msg and isinstance(msg["create_time"], (int, float)):
            message_times.append(msg["create_time"])
    letzte_aenderung = datetime.fromtimestamp(max(message_times)) if message_times else erstellt_am

    message_count = len(messages)
    chat_link = f"https://chat.openai.com/c/{chat_id}"

    # Einfügen, wenn noch nicht vorhanden
    cursor.execute(
        "INSERT IGNORE INTO chats (chat_id, titel, erstellt_am, letzte_aenderung, message_count, chat_link, status) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s)",
        (chat_id, titel, erstellt_am, letzte_aenderung, message_count, chat_link, 'neu')
    )

conn.commit()
print("✅ Chats erfolgreich in MySQL importiert.")
cursor.close()
conn.close()

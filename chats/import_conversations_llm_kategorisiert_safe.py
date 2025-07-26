
import json
import subprocess
import pymysql
from datetime import datetime
import pandas as pd

def lade_json(pfad):
    with open(pfad, "r", encoding="utf-8") as f:
        return json.load(f)

def lade_excel_chat_infos(pfad):
    df = pd.read_excel("chat_infos.xlsx")
    return [str(row[0]).strip().lower() for _, row in df.iterrows() if pd.notna(row[0])]

def verbinde_mit_datenbank():
    return pymysql.connect(
         host="127.0.0.1",
        user="chatuser",
        password="chatpass",
        database="gptchats",
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor
    )

def hole_kategorien(cursor):
    cursor.execute("SELECT id, name FROM kategorien")
    return {row["name"].lower(): row["id"] for row in cursor.fetchall()}

def bereinige_vorschlag(text, kategorien_liste):
    clean = text.lower().strip().replace('"', '').replace("'", "")
    for k in kategorien_liste:
        if k in clean:
            return k
    return None

def generiere_kategorievorschlag(text, kategorien_liste):
    kategorien_str = ", ".join(kategorien_liste)
    prompt = (
        f"Weise dem folgenden Inhalt genau eine dieser Kategorien zu:/n"
        f"{kategorien_str}/n/n"
        f"Inhalt:/n{text}/n/n"
        f"Nur die Kategorie als einzelnes Wort zurückgeben."
    )
    try:
        result = subprocess.run(
            ["ollama", "run", "llama3", prompt],
            capture_output=True,
            text=True,
            timeout=30,
            encoding="utf-8",
            errors="ignore"
        )
        return result.stdout.strip()
    except Exception as e:
        return f"[Fehlgeschlagen: {e}]"

def verarbeite_chat(chat, kategorien, stichwoerter, cursor):
    chat_id = chat.get("id")
    titel = chat.get("title", "")[:255]
    messages = list(chat.get("mapping", {}).values())
    erstellt_am = datetime.fromtimestamp(chat.get("create_time", 0))
    message_count = len(messages)
    letzte_aenderung = erstellt_am
    chat_link = f"https://chat.openai.com/c/{chat_id}"

    # Sicherer Zugriff auf chat_text
    chat_text = "\n\n".join(
        str(m["message"]["content"]["parts"][0])
        for m in messages
        if (
            "message" in m and isinstance(m["message"], dict) and
            "content" in m["message"] and isinstance(m["message"]["content"], dict) and
            "parts" in m["message"]["content"] and
            isinstance(m["message"]["content"]["parts"], list) and
            m["message"]["content"]["parts"]
        )
    )[:1000]

    inhalt = f"{titel}\n\n{chat_text}"
    raw_vorschlag = generiere_kategorievorschlag(inhalt, list(kategorien.keys()))
    bereinigt = bereinige_vorschlag(raw_vorschlag, kategorien.keys())

    print(f"📝 Chat: {titel}")
    print(f"📥 Rohvorschlag: {raw_vorschlag}")
    print(f"✅ Bereinigt: {bereinigt}")

    if bereinigt and bereinigt in kategorien:
        cursor.execute(
            "INSERT INTO chats (chat_id, titel, erstellt_am, letzte_aenderung, message_count, chat_link, status, zusammenfassung) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
            "ON DUPLICATE KEY UPDATE letzte_aenderung=VALUES(letzte_aenderung), message_count=VALUES(message_count), status='überschreiben'",
            (chat_id, titel, erstellt_am, letzte_aenderung, message_count, chat_link, 'neu', '[Zusammenfassung über LLM-Kategorisierung]')
        )
        cursor.execute(
            "SELECT id FROM chats WHERE chat_id=%s", (chat_id,)
        )
        result = cursor.fetchone()
        if result:
            chat_db_id = result["id"]
            cursor.execute(
                "INSERT IGNORE INTO chat_kategorien (chat_id, kategorie_id, relevanz) VALUES (%s, %s, %s)",
                (chat_db_id, kategorien[bereinigt], 3)
            )

def main():
    daten = lade_json("conversations.json")
    stichwoerter = lade_excel_chat_infos("chat_infos.xlsx")
    conn = verbinde_mit_datenbank()
    cursor = conn.cursor()
    kategorien = hole_kategorien(cursor)

    for i, chat in enumerate(daten[:5]):
        print(f"🔄 Chat {i+1}/{len(daten)}")
        verarbeite_chat(chat, kategorien, stichwoerter, cursor)

    conn.commit()
    cursor.close()
    conn.close()
    print("✅ LLM-Kategorisierung abgeschlossen.")

if __name__ == "__main__":
    main()

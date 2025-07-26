
import json
import pymysql
from datetime import datetime
import subprocess

def lade_json(pfad):
    with open(pfad, "r", encoding="utf-8") as f:
        return json.load(f)

def verbinde_mit_datenbank():
    return pymysql.connect(
        host="127.0.0.1",
        user="robouser",
        password="dein_passwort",
        database="chatprojekt",
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor
    )

def hole_importierte_chats(cursor):
    cursor.execute("SELECT chat_id, message_count FROM chats")
    return {row['chat_id']: row['message_count'] for row in cursor.fetchall()}

def berechne_letzte_aenderung(messages):
    zeiten = []
    for msg in messages:
        if "create_time" in msg and isinstance(msg["create_time"], (int, float)):
            zeiten.append(msg["create_time"])
    return datetime.fromtimestamp(max(zeiten)) if zeiten else None

def generiere_zusammenfassung_ollama(text):
    try:
        result = subprocess.run(
            ["ollama", "run", "llama3", f"Fasse diesen Chatinhalt kurz zusammen:

{text}"],
            capture_output=True, text=True, timeout=30
        )
        return result.stdout.strip()
    except Exception as e:
        return f"[Zusammenfassung fehlgeschlagen: {e}]"

def verarbeite_chat(chat, bekannte_chats, cursor):
    chat_id = chat.get("id")
    if not chat_id:
        return

    titel = chat.get("title", "")[:255]
    erstellt_am = datetime.fromtimestamp(chat.get("create_time", 0))
    messages = list(chat.get("mapping", {}).values())
    message_count = len(messages)
    letzte_aenderung = berechne_letzte_aenderung(messages) or erstellt_am
    chat_link = f"https://chat.openai.com/c/{chat_id}"

    # Zusammenfassung vorbereiten
    chat_text = "\n\n".join(
        msg["message"]["content"]["parts"][0]
        for msg in messages if "message" in msg and msg["message"].get("content")
    )[:4000]  # Kürzen für Ollama

    zusammenfassung = generiere_zusammenfassung_ollama(chat_text)

    if chat_id in bekannte_chats:
        if bekannte_chats[chat_id] != message_count:
            cursor.execute(
                "UPDATE chats SET titel=%s, erstellt_am=%s, letzte_aenderung=%s, "
                "message_count=%s, chat_link=%s, status=%s, zusammenfassung=%s WHERE chat_id=%s",
                (titel, erstellt_am, letzte_aenderung, message_count, chat_link, 'überschreiben', zusammenfassung, chat_id)
            )
            print(f"🔁 Aktualisiert: {titel}")
        else:
            print(f"➡️  Unverändert: {titel}")
    else:
        cursor.execute(
            "INSERT INTO chats (chat_id, titel, erstellt_am, letzte_aenderung, message_count, chat_link, status, zusammenfassung) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            (chat_id, titel, erstellt_am, letzte_aenderung, message_count, chat_link, 'neu', zusammenfassung)
        )
        print(f"➕ Eingefügt: {titel}")

def main():
    daten = lade_json("conversations.json")
    conn = verbinde_mit_datenbank()
    cursor = conn.cursor()
    bekannte_chats = hole_importierte_chats(cursor)

    for chat in daten:
        verarbeite_chat(chat, bekannte_chats, cursor)

    conn.commit()
    cursor.close()
    conn.close()
    print("✅ Import abgeschlossen.")

if __name__ == "__main__":
    main()


import json
import pymysql
from datetime import datetime
import subprocess
import pandas as pd

UNGUELTIGE_ZUSAMMENFASSUNGEN = [
    "turn ended.",
    "no further action will be taken.",
    "*no response*",
    "kein text mehr.",
    ""
]

UNGUELTIGE_ANFAENGE = [
    "hier ist der chatinhalt",
    "hier ist eine kurze zusammenfassung",
    "here's a brief summary",
    "this is the summary",
    "hier ist der inhalt in 3 kurzen sätzen",
    "hier sind die 3 kurzen sätze",
    "hier sind die drei kurzen sätze",
     "here's a brief summary of the chat content:",
    "hier ist eine kurze zusammenfassung des chatinhalts:",
    "Hier ist der Chatinhalt in 3 kurzen Sätzen zusammengefasst:",
    "Hier sind die drei kurzen Sätze:",
    "Hier ist die Zusammenfassung:",
    "Hier ist die Zusammenfassung in 3 Sätzen:",
    "Here are the 3 short sentences summarizing the chat content:",
    "Hier ist eine Zusammenfassung des Chats in 3 kurzen Sätzen:",
    "Here is a summary of the chat content in 3 short sentences:"
]

def lade_json(pfad):
    with open(pfad, "r", encoding="utf-8") as f:
        return json.load(f)

def lade_excel_chat_infos(pfad):
    df = pd.read_excel(pfad)
    stichwoerter = []
    for index, row in df.iterrows():
        if pd.notna(row.iloc[0]):
            stichwoerter.append(str(row.iloc[0]).strip().lower())
    print(f"🔎 Geladene Stichwörter aus chat_infos.xlsx: {len(stichwoerter)}")
    return stichwoerter

def verbinde_mit_datenbank():
    return pymysql.connect(
        host="127.0.0.1",
        user="chatuser",
        password="chatpass",
        database="gptchats",
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor
    )

def hole_importierte_chats(cursor):
    cursor.execute("SELECT chat_id, message_count FROM chats")
    return {row['chat_id']: row['message_count'] for row in cursor.fetchall()}

def hole_kategorien(cursor):
    cursor.execute("SELECT id, name FROM kategorien")
    return {row["name"].lower(): row["id"] for row in cursor.fetchall()}

def berechne_letzte_aenderung(messages):
    zeiten = []
    for msg in messages:
        if "create_time" in msg and isinstance(msg["create_time"], (int, float)):
            zeiten.append(msg["create_time"])
    return datetime.fromtimestamp(max(zeiten)) if zeiten else None

def generiere_zusammenfassung_ollama(text):
    prompt = (
        f"Fasse den folgenden Chatinhalt in 3 kurzen Sätzen zusammen. "
        f"Keine Einleitung oder Floskeln, keine Meta-Kommentare. Nur Inhalt:/n/n{text}"
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
        raw_output = result.stdout.strip()
        cleaned = raw_output.lower().strip()
        for start in UNGUELTIGE_ANFAENGE:
            if cleaned.startswith(start) or cleaned in UNGUELTIGE_ZUSAMMENFASSUNGEN:
                return "[Zusammenfassung nicht erzeugbar]"
        return raw_output
    except Exception as e:
        return f"[Zusammenfassung fehlgeschlagen: {e}]"

def schlage_kategorien_vor(titel, zusammenfassung, kategorien):
    text = (titel + " " + zusammenfassung).lower()
    zuordnung = []
    for name, kat_id in kategorien.items():
        if name in text:
            zuordnung.append((kat_id, 3))
            print(f"✅ KI-Kategorievorschlag: {name}")
    return list(set(zuordnung))

def verarbeite_chat(chat, bekannte_chats, kategorien, stichwoerter, cursor):
    chat_id = chat.get("id")
    if not chat_id:
        return

    titel = chat.get("title", "")[:255]
    erstellt_am = datetime.fromtimestamp(chat.get("create_time", 0))
    messages = list(chat.get("mapping", {}).values())
    message_count = len(messages)
    letzte_aenderung = berechne_letzte_aenderung(messages) or erstellt_am
    chat_link = f"https://chat.openai.com/c/{chat_id}"

    chat_text = "\n\n".join(
        str(msg["message"]["content"]["parts"][0])
        for msg in messages
        if (
            "message" in msg and
            isinstance(msg["message"], dict) and
            "content" in msg["message"] and
            isinstance(msg["message"]["content"], dict) and
            "parts" in msg["message"]["content"] and
            isinstance(msg["message"]["content"]["parts"], list) and
            msg["message"]["content"]["parts"]
        )
    )[:4000]

    zusammenfassung = generiere_zusammenfassung_ollama(chat_text)
    print(f"📝 Titel: {titel}")
    print(f"🧠 Zusammenfassung: {zusammenfassung}")

    if chat_id in bekannte_chats:
        if bekannte_chats[chat_id] != message_count:
            cursor.execute(
                "UPDATE chats SET titel=%s, erstellt_am=%s, letzte_aenderung=%s, "
                "message_count=%s, chat_link=%s, status=%s, zusammenfassung=%s WHERE chat_id=%s",
                (titel, erstellt_am, letzte_aenderung, message_count, chat_link, 'überschreiben', zusammenfassung, chat_id)
            )
        else:
            print(f"➡️  Unverändert: {titel}")
            return
    else:
        cursor.execute(
            "INSERT INTO chats (chat_id, titel, erstellt_am, letzte_aenderung, message_count, chat_link, status, zusammenfassung) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            (chat_id, titel, erstellt_am, letzte_aenderung, message_count, chat_link, 'neu', zusammenfassung)
        )

    # Kategoriezuordnung nach Manfreds Logik
    zuordnungen = []
    if any(stichwort in titel.lower() for stichwort in stichwoerter):
        if "sonstige infos" in kategorien:
            zuordnungen = [(kategorien["sonstige infos"], 3)]
            print(f"🔍 Titel enthält Stichwort – Kategorie 'Sonstige Infos' zugewiesen.")
        else:
            print("⚠️ Kategorie 'Sonstige Infos' nicht gefunden!")
    else:
        zuordnungen = schlage_kategorien_vor(titel, zusammenfassung, kategorien)

    if not zuordnungen:
        print(f"❌ Keine Kategorie zugewiesen für: {titel}")

    cursor.execute("SELECT id FROM chats WHERE chat_id=%s", (chat_id,))
    result = cursor.fetchone()
    if result:
        db_id = result['id']
        for kat_id, relevanz in zuordnungen:
            cursor.execute(
                "INSERT IGNORE INTO chat_kategorien (chat_id, kategorie_id, relevanz) VALUES (%s, %s, %s)",
                (db_id, kat_id, relevanz)
            )
            print(f"➡️  Kategorie-ID {kat_id} mit Relevanz {relevanz} zugeordnet.")

def main():
    daten = lade_json("conversations.json")
    stichwoerter = lade_excel_chat_infos("chat_infos.xlsx")

    conn = verbinde_mit_datenbank()
    cursor = conn.cursor()

    bekannte_chats = hole_importierte_chats(cursor)
    kategorien = hole_kategorien(cursor)

    for chat in daten:
        verarbeite_chat(chat, bekannte_chats, kategorien, stichwoerter, cursor)

    conn.commit()
    cursor.close()
    conn.close()
    print("✅ Import mit finaler Kategorisierungslogik abgeschlossen.")

if __name__ == "__main__":
    main()

import json
import subprocess
import pymysql
from datetime import datetime
import pandas as pd
import re

def lade_json(pfad):
    with open(pfad, "r", encoding="utf-8") as f:
        return json.load(f)

def lade_excel_chat_infos(pfad):
    df = pd.read_excel(pfad)
    return {str(row.iloc[0]).strip().lower(): str(row.iloc[1]).strip().lower()
            for _, row in df.iterrows() if pd.notna(row.iloc[0]) and pd.notna(row.iloc[1])}

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

def extrahiere_kategorien_und_relevanz(text, kategorien_liste):
    kategorien_gefunden = {}
    lines = text.lower().splitlines()

    for line in lines:
        for k in kategorien_liste:
            if re.search(rf'\b{k}\b', line):
                if k not in kategorien_gefunden:
                    kategorien_gefunden[k] = 50

    for line in lines:
        for k in kategorien_gefunden:
            if k in line and "/" in line:
                match = re.search(r"(\d+)\s*/\s*(\d+)", line)
                if match:
                    zahl1 = int(match.group(1))
                    zahl2 = int(match.group(2))
                    if zahl2 > 0:
                        kategorien_gefunden[k] = int((zahl1 / zahl2) * 100)

    return list(kategorien_gefunden.items())

def generiere_kategorievorschlag(text, kategorien_liste):
    kategorien_str = ", ".join(kategorien_liste)
    prompt = (
        f"Weise dem folgenden Inhalt eine oder mehrere passende Kategorien zu:\n"
        f"{kategorien_str}\n\n"
        f"Inhalt:\n{text}\n\n"
        f"Gib die Antwort im Format:\n"
        f"* Kategorie: X\n* Kategorie: Y\n...\n\n"
        f"Und gib für jede Kategorie eine Relevanz an wie: Relevanz: 4/5"
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

def verarbeite_chat(chat, kategorien, stichwort_mapping, cursor):
    chat_id = chat.get("id")
    titel = chat.get("title", "")[:255].strip()
    messages = list(chat.get("mapping", {}).values())
    erstellt_am = datetime.fromtimestamp(chat.get("create_time", 0))
    message_count = len(messages)
    letzte_aenderung = erstellt_am
    chat_link = f"https://chat.openai.com/c/{chat_id}"

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
    kategorien_gesamt = {}

    for suchwort, manuelle_kategorie in stichwort_mapping.items():
        if suchwort.lower() in titel.lower():
            kategorien_gesamt[manuelle_kategorie] = 100
            print(f"🔎 Manuelle Kategorie erkannt: {manuelle_kategorie}")

    raw_vorschlag = generiere_kategorievorschlag(inhalt, list(kategorien.keys()))
    print(f"📥 LLM-Rohvorschlag: {raw_vorschlag}")
    kategorien_llm = extrahiere_kategorien_und_relevanz(raw_vorschlag, kategorien.keys())

    for kat, rel in kategorien_llm:
        if kat not in kategorien_gesamt:
            kategorien_gesamt[kat] = rel

    print(f"✅ Erkannt: {list(kategorien_gesamt.items())}")

    if kategorien_gesamt:
        cursor.execute(
            "INSERT INTO chats (chat_id, titel, erstellt_am, letzte_aenderung, message_count, chat_link, status, zusammenfassung) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
            "ON DUPLICATE KEY UPDATE letzte_aenderung=VALUES(letzte_aenderung), message_count=VALUES(message_count), status='überschreiben'",
            (chat_id, titel, erstellt_am, letzte_aenderung, message_count, chat_link, 'neu', '[Zusammenfassung folgt]')
        )
        cursor.execute("SELECT id FROM chats WHERE chat_id=%s", (chat_id,))
        result = cursor.fetchone()
        if result:
            chat_db_id = result["id"]
            for kat, rel in kategorien_gesamt.items():
                if kat in kategorien:
                    cursor.execute(
                        "INSERT IGNORE INTO chat_kategorien (chat_id, kategorie_id, relevanz) VALUES (%s, %s, %s)",
                        (chat_db_id, kategorien[kat], rel)
                    )
    else:
        print("⚠️ Keine passende Kategorie gefunden.")

def main():
    daten = lade_json("conversations.json")
    stichwort_mapping = lade_excel_chat_infos("chat_infos.xlsx")
    conn = verbinde_mit_datenbank()
    cursor = conn.cursor()
    kategorien = hole_kategorien(cursor)

    for i, chat in enumerate(daten):
        print(f"\n\n🔄 Verarbeite Chat {i+1}/{len(daten)}: {chat.get('title', '')}")
        verarbeite_chat(chat, kategorien, stichwort_mapping, cursor)

    conn.commit()
    cursor.close()
    conn.close()
    print("✅ LLM-Kategorisierung (V4.2) abgeschlossen.")

if __name__ == "__main__":
    main()
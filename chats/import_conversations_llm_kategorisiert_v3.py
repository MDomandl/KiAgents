import json
import subprocess
import pymysql
from datetime import datetime
import pandas as pd

def lade_json(pfad):
    with open(pfad, "r", encoding="utf-8") as f:
        return json.load(f)

def lade_excel_chat_infos(pfad):
    df = pd.read_excel(pfad)
    return {str(row.iloc[0]).strip().lower(): str(row.iloc[1]).strip() for _, row in df.iterrows() if pd.notna(row.iloc[0]) and pd.notna(row.iloc[1])}

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
    kategorie = None
    relevanz = 50
    for teil in clean.split(","):
        if "relevanz" in teil:
            relevanz = int("".join(filter(str.isdigit, teil)))
        else:
            for k in kategorien_liste:
                if k in teil:
                    kategorie = k
                    break
    return kategorie, relevanz

def generiere_kategorievorschlag(text, kategorien_liste):
    kategorien_str = ", ".join(kategorien_liste)
    prompt = (
        f"Weise dem folgenden Inhalt genau eine dieser Kategorien zu:/n"
        f"{kategorien_str}/n/n"
        f"Inhalt:/n{text}/n/n"
        f"Gib die Antwort im Format: Kategorie: XYZ, Relevanz: 85"
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
    kategorie = None
    relevanz = 100

    # Schritt 1: manuelles Mapping aus chat_infos.xlsx
    for suchwort, kategorie_manuell in stichwort_mapping.items():
        if suchwort.lower() in titel.lower():
            kategorie = kategorie_manuell.lower()
            print(f"🔎 Manuelle Kategorie erkannt: {kategorie}")
            break

    # Schritt 2: LLM-Vorschlag, falls keine manuelle gefunden wurde
    if not kategorie:
        raw_vorschlag = generiere_kategorievorschlag(inhalt, list(kategorien.keys()))
        print(f"📥 LLM-Rohvorschlag: {raw_vorschlag}")
        kategorie, relevanz = bereinige_vorschlag(raw_vorschlag, kategorien.keys())
        print(f"✅ LLM-Bereinigt: {kategorie}, Relevanz: {relevanz}")

    # Schritt 3: Speicherung, falls gültige Kategorie gefunden wurde
    if kategorie and kategorie in kategorien:
        cursor.execute(
            "INSERT INTO chats (chat_id, titel, erstellt_am, letzte_aenderung, message_count, chat_link, status, zusammenfassung) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
            "ON DUPLICATE KEY UPDATE letzte_aenderung=VALUES(letzte_aenderung), message_count=VALUES(message_count), status='überschreiben'",
            (chat_id, titel, erstellt_am, letzte_aenderung, message_count, chat_link, 'neu', '[Zusammenfassung über LLM-Kategorisierung]')
        )
        cursor.execute("SELECT id FROM chats WHERE chat_id=%s", (chat_id,))
        result = cursor.fetchone()
        if result:
            chat_db_id = result["id"]
            cursor.execute(
                "INSERT IGNORE INTO chat_kategorien (chat_id, kategorie_id, relevanz) VALUES (%s, %s, %s)",
                (chat_db_id, kategorien[kategorie], relevanz)
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
        print(f"/n/n🔄 Verarbeite Chat {i+1}/{len(daten)}: {chat.get('title', '')[:60]}")
        verarbeite_chat(chat, kategorien, stichwort_mapping, cursor)

    conn.commit()
    cursor.close()
    conn.close()
    print("✅ LLM-Kategorisierung (V3) abgeschlossen.")

if __name__ == "__main__":
    main()
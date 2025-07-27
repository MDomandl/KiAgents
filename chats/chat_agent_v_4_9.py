# chat_agent_v_4_7_mit_manuell_insert.py
# Weiterentwickelte Version basierend auf V4.7 mit intelligenterer Behandlung von manuellen und LLM-Kategorien

import json
import subprocess
import pymysql
from datetime import datetime
import pandas as pd
import re
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import os

CHROMA_PATH = r"D:\Users\doman\Documents\OneDrive\Dokumente\Programmierung\Projekte\AiAgents\chats"
SKIP_UNVERÄNDERTE_CHATS = False

vectordb = None


def init_chroma():
    model_name = "intfloat/e5-large-v2"
    embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={"device": "cpu"},  # oder "cuda" bei GPU
    encode_kwargs={"normalize_embeddings": True}
)
    vectordb = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    return vectordb

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
    blocks = re.split(r'\*+\s*Kategorie:|\*', text, flags=re.IGNORECASE)
    for block in blocks[1:]:
        zeilen = block.strip().splitlines()
        if not zeilen:
            continue
        erste_zeile = zeilen[0]
        match_kat = re.match(r"[*\-]?\s*([a-zA-ZäöüßÄÖÜ0-9_\- ]+):(?:\s*(?:Relevance|Relevanz):)?\s*([0-5])/5", erste_zeile.strip(), re.IGNORECASE)
        if not match_kat:
            continue
        katname = match_kat.group(1).strip().lower()
        katname = re.sub(r"[^\wäöüß\-]", "", katname)
        relevanz = 50
        for line in zeilen:
            match = re.search(r"(\d+)\s*/\s*(\d+)", line)
            if match:
                zahl1 = int(match.group(1))
                zahl2 = int(match.group(2))
                if zahl2 > 0:
                    relevanz = int((zahl1 / zahl2) * 100)
                break
        if relevanz >= 40:
            kategorien_gefunden[katname] = relevanz
    return list(kategorien_gefunden.items())

def generiere_kategorievorschlag(text, kategorien_liste):
    kategorien_str = ", ".join(kategorien_liste)
    prompt = (
        f"Weise dem folgenden Inhalt eine oder mehrere passende Kategorien zu:\n"
        f"{kategorien_str}\n\n"
        f"Inhalt:\n{text}\n\n"
        f"Gib die Antwort bitte ausschließlich(!) im folgenden Format zurück ohne Erklärungen oder mehrere Kategorien pro Zeile:\n"
        f"* Verkehr: 4/5"
        f"\nUnd gib für jede Kategorie eine Relevanz an wie: 4/5"
        f"\nBeispiel: \n* Bewerbung: 5/5"
        f"\nBitte keine Zeile mit N/A ausgeben"
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

def generiere_zusammenfassung(text):
    prompt = f"Fasse den folgenden Chat knapp zusammen (max. 3 Sätze):\n\n{text[:2000]}"
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
        return "[Zusammenfassung fehlgeschlagen]"

def braucht_llm_kategorisierung(chat_id, cursor):
    cursor.execute(
        "SELECT COUNT(*) as anzahl FROM chat_kategorien WHERE chat_id = %s AND quelle IN ('llama3', 'gpt4')",
        (chat_id,)
    )
    result = cursor.fetchone()
    return result["anzahl"] == 0

def verarbeite_chat(chat, kategorien, stichwort_mapping, cursor):
    chat_id = chat.get("id")
    titel = chat.get("title", "")[:255].strip()
    messages = list(chat.get("mapping", {}).values())
    erstellt_am = datetime.fromtimestamp(chat.get("create_time", 0))
    message_count = len(messages)
    letzte_aenderung = datetime.fromtimestamp(chat.get("update_time", 0))
    chat_link = f"https://chat.openai.com/c/{chat_id}"

    if SKIP_UNVERÄNDERTE_CHATS:
        cursor.execute("SELECT message_count FROM chats WHERE chat_id = %s", (chat_id,))
        result = cursor.fetchone()
        if result and result["message_count"] == message_count:
            print(f"⏩ Chat '{titel}' wurde nicht verändert – übersprungen.")
            return

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
    )[:3000]

    inhalt = f"{titel}\n\n{chat_text}"
    kategorien_manuell = {}

    for suchwort, manuelle_kategorie in stichwort_mapping.items():
        if suchwort.lower() in titel.lower():
            kategorien_manuell[manuelle_kategorie] = 100
            print(f"🔎 Manuelle Kategorie erkannt: {manuelle_kategorie}")

    # Insert oder Update des Chats
    zusammenfassung = "[Noch keine LLM-Zusammenfassung]"
    cursor.execute(
        "INSERT INTO chats (chat_id, titel, erstellt_am, letzte_aenderung, message_count, chat_link, status, zusammenfassung) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
        "ON DUPLICATE KEY UPDATE letzte_aenderung=VALUES(letzte_aenderung), message_count=VALUES(message_count)",
        (chat_id, titel, erstellt_am, letzte_aenderung, message_count, chat_link, 'neu', zusammenfassung)
    )
    cursor.execute("SELECT id FROM chats WHERE chat_id=%s", (chat_id,))
    result = cursor.fetchone()
    if not result:
        print(f"❌ Fehler beim Abrufen der Chat-ID für '{titel}'")
        return

    chat_db_id = result["id"]

    # LLM-Kategorisierung + Zusammenfassung nur wenn nötig
    kategorien_llm = {}
    if braucht_llm_kategorisierung(chat_db_id, cursor):
        raw_vorschlag = generiere_kategorievorschlag(inhalt, list(kategorien.keys()))
        print(f"📥 LLM-Rohvorschlag: {raw_vorschlag}")
        for kat, rel in extrahiere_kategorien_und_relevanz(raw_vorschlag, kategorien.keys()):
            if kat not in kategorien_manuell:
                kategorien_llm[kat] = rel

        zusammenfassung = generiere_zusammenfassung(inhalt)
        print(f"📝 Zusammenfassung: {zusammenfassung}")

        speichere_embedding(chat_id, titel, zusammenfassung, inhalt)


        # Zusammenfassung aktualisieren
        cursor.execute("UPDATE chats SET zusammenfassung = %s WHERE id = %s", (zusammenfassung, chat_db_id))

        # Nur LLM-Kategorien löschen
        cursor.execute("DELETE FROM chat_kategorien WHERE chat_id = %s AND quelle IN ('llama3', 'gpt4')", (chat_db_id,))
    else:
        print(f"⏭️ Chat '{titel}' wurde bereits LLM-kategorisiert – LLM-Skip.")

    # Alle Kategorien kombinieren und eintragen
    alle_kategorien = kategorien_manuell.copy()
    alle_kategorien.update(kategorien_llm)

    for kat, rel in alle_kategorien.items():
        quelle = "llama3" if kat in kategorien_llm else "manuell"
        if kat in kategorien:
            cursor.execute(
                "INSERT INTO chat_kategorien (chat_id, kategorie_id, relevanz, quelle) "
                "VALUES (%s, %s, %s, %s) "
                "ON DUPLICATE KEY UPDATE relevanz = VALUES(relevanz), quelle = VALUES(quelle)",
                (chat_db_id, kategorien[kat], rel, quelle)
            )

def speichere_embedding(chat_id, title, summary, content, overwrite=True):    
    text = f"passage: Titel: {title}\nZusammenfassung: {summary}\nInhalt: {content}"

    metadata = {
        "chat_id": chat_id,
        "title": title,
    }
    print(f"🔄 Speichere Embedding für Chat {chat_id}...")

    print(f"✅ Eingefügt in Chroma: {chat_id} – {title[:50]}...")

    # Doppelte Einträge vermeiden?
    if not overwrite:
        result = vectordb.similarity_search(text, k=3)
        for doc in result:
            if doc.metadata.get("chat_id") == chat_id:
                print(f"⏭️ Chat {chat_id} bereits in Chroma gespeichert – wird übersprungen.")
                return

    # Speichern
    vectordb.add_documents([
        Document(page_content=text, metadata=metadata)
    ])
    vectordb.persist()
    print(f"💾 Chat {chat_id} in Chroma gespeichert.")


def main():
    embeddings = OllamaEmbeddings(model="bge-m3")
    embedding = embeddings.embed_query("test")
    print("✅ Modell geladen und einsatzbereit.")
    global vectordb
    vectordb = init_chroma()
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
    print("✅ LLM-Kategorisierung (V4.7 mit manuell/llm-Merge) abgeschlossen.")

if __name__ == "__main__":
    main()

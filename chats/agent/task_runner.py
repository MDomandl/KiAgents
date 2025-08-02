from agent.chat_loader import lade_json, lade_excel_chat_infos
from agent.vectorstore import init_chroma, speichere_embedding
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from agent.kategorisieren import generiere_kategorievorschlag, extrahiere_kategorien_und_relevanz, hole_kategorien, braucht_llm_kategorisierung
from agent.zusammenfassen import generiere_zusammenfassung, get_chat_text
from agent.db_writer import verbinde_mit_datenbank, insert_kategorien, insert_update_chats
from datetime import datetime
from agent.nutzerfreigabe import frage_benutzer

CHROMA_PATH = r"D:\Users\doman\Documents\OneDrive\Dokumente\Programmierung\Projekte\AiAgents\chats"
SKIP_UNVERÄNDERTE_CHATS = False

vectordb = None

def fuehre_tasks_aus():
    print("🔄 Starte Agentenaufgaben...")
    daten = lade_json("conversations.json")
    stichwort_mapping = lade_excel_chat_infos("chat_infos.xlsx")
    conn = verbinde_mit_datenbank()
    cursor = conn.cursor()
    db_kategorien = hole_kategorien(cursor)
    vectordb = init_chroma()
    for i, chat in enumerate(daten):
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

        chat_text = get_chat_text(messages)

        inhalt = f"{titel}\n\n{chat_text}"
        kategorien_manuell = {}

        for suchwort, manuelle_kategorie in stichwort_mapping.items():
            if suchwort.lower() in titel.lower():
                kategorien_manuell[manuelle_kategorie] = 100
                print(f"🔎 Manuelle Kategorie erkannt: {manuelle_kategorie}")

        zusammenfassung = "[Noch keine LLM-Zusammenfassung]"
        insert_update_chats(chat_id, titel, erstellt_am, letzte_aenderung, message_count, chat_link, zusammenfassung, cursor)
        cursor.execute("SELECT id FROM chats WHERE chat_id=%s", (chat_id,))
        result = cursor.fetchone()
        if not result:
            print(f"❌ Fehler beim Abrufen der Chat-ID für '{titel}'")
            return

        chat_db_id = result["id"]
        kategorien_llm = {}
        if braucht_llm_kategorisierung(chat_db_id, cursor):
            raw_vorschlag = generiere_kategorievorschlag(inhalt, list(db_kategorien.keys()))
            print(f"📥 LLM-Rohvorschlag: {raw_vorschlag}")
            for kat, rel in extrahiere_kategorien_und_relevanz(raw_vorschlag, db_kategorien.keys()):
                if kat not in kategorien_manuell:
                    kategorien_llm[kat] = rel

            zusammenfassung = generiere_zusammenfassung(inhalt)
           
            print(f"📝 Zusammenfassung: {zusammenfassung}\n📦 Kategorien: {raw_vorschlag}")           
            speichere_embedding(chat_id, titel, zusammenfassung, inhalt, vectordb)

            # Zusammenfassung aktualisieren
            cursor.execute("UPDATE chats SET zusammenfassung = %s WHERE id = %s", (zusammenfassung, chat_db_id))

            # Nur LLM-Kategorien löschen
            cursor.execute("DELETE FROM chat_kategorien WHERE chat_id = %s AND quelle IN ('llama3', 'gpt4')", (chat_db_id,))
        else:
            print(f"⏭️ Chat '{titel}' wurde bereits LLM-kategorisiert – LLM-Skip.")
        
        # Alle Kategorien kombinieren und eintragen: also die manuellen und die vom llm
        # db_kategorien: die aus der Datenbank
        alle_kategorien = kategorien_manuell.copy()
        alle_kategorien.update(kategorien_llm)
        insert_kategorien(alle_kategorien, db_kategorien, kategorien_llm, chat_db_id, cursor)
        

    conn.commit()
    cursor.close()
    conn.close()
    print("✅ LLM-Kategorisierung (V5.0 mit manuell/llm-Merge) abgeschlossen.")
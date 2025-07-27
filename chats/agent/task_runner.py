from agent.chat_loader import lade_json, lade_excel_chat_infos
from agent.vectorstore import init_chroma, speichere_embedding
from agent.kategorisieren import generiere_kategorievorschlag, extrahiere_kategorien_und_relevanz
from agent.zusammenfassen import generiere_zusammenfassung
from agent.nutzerfreigabe import frage_benutzer

def fuehre_tasks_aus():
    print("🔄 Starte Agentenaufgaben...")
    daten = lade_json("conversations.json")
    stichwoerter = lade_excel_chat_infos("chat_infos.xlsx")
    vectordb = init_chroma()
    for i, chat in enumerate(daten):
        print(f"📄 Chat {i+1}/{len(daten)}: {chat.get('title', '')}")
        titel = chat.get("title", "")[:255]
        text = f"{titel}\n\n" + str(chat)[:2000]
        zusammenfassung = generiere_zusammenfassung(text)
        kategorien = generiere_kategorievorschlag(text, stichwoerter.values())
        print(f"📝 Zusammenfassung: {zusammenfassung}\n📦 Kategorien: {kategorien}")
        speichere_embedding(chat.get("id", f"chat_{i}"), titel, zusammenfassung, text, vectordb)

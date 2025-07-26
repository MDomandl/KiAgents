# search_test.py
import chromadb
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from ollama import Client
import os
import pprint

# === 1. Datenbankpfad setzen ===
CHROMA_DB_PATH = r"D:\Users\doman\Documents\OneDrive\Dokumente\Programmierung\Projekte\AiAgents\chats"

# === 2. Embedding-Funktion auswählen ===
# Achtung: OpenAI braucht API-Key in ENV: OPENAI_API_KEY
# ollama_model = "mistral" oder "llama3" (je nachdem was du nutzt)
embedding_function = OllamaEmbeddings(model="bge-m3")
vectordb = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_function)
# === 3. Client und Collection laden ===
#client = chromadb.PersistentClient(path=CHROMA_DB_PATH)



#collection = client.get_or_create_collection(name="langchain", embedding_function=embedding_function)
#print("Anzahl Einträge in Chroma:", collection.count())
# === 4. Interaktive Suche ===
def suche_aehnliche_chats(query_text, anzahl=5):
    docs = vectordb.similarity_search(query_text, anzahl)
   # ergebnis = collection.query(query_texts=[query_text], n_results=anzahl)
   # pprint.pprint(ergebnis)
    print("\n🔍 Ähnliche Chats gefunden:\n")
    for i, doc in enumerate(docs, 1):
        print(f"{i}. Titel: {doc.metadata.get('title')}")
        print(f"   Inhalt (Auszug): {doc.page_content[:200]}...\n")    

if __name__ == "__main__":
    print("🔎 Semantische Suche starten...\n")
    user_input = input("➡️ Bitte gib einen Suchtext ein (z. B. 'Roboter mit Akku'): ")
    suche_aehnliche_chats(user_input)

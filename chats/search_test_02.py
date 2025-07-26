from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import os

# === 1. Konfiguration ===
CHROMA_DB_PATH = r"D:\Users\doman\Documents\OneDrive\Dokumente\Programmierung\Projekte\AiAgents\chats"

# === 2. Embedding-Funktion (gleich wie beim Import) ===
embedding_function = OllamaEmbeddings(model="bge-m3")

# === 3. Vektordatenbank laden ===
vectordb = Chroma(
    persist_directory=CHROMA_DB_PATH,
    embedding_function=embedding_function
)

# === 4. Nutzerabfrage ===
query = input("➡️ Bitte gib einen Suchtext ein (z. B. 'Roboter mit Akku'): ").strip()

if not query:
    print("❗ Kein Suchtext eingegeben.")
    exit()

# === 5. Suche durchführen ===
print("\n🔍 Ähnliche Chats gefunden:\n")
results = vectordb.similarity_search_with_score(query, k=5)

for i, (doc, score) in enumerate(results, 1):
    title = doc.metadata.get("title", "Kein Titel")
    chat_id = doc.metadata.get("chat_id", "Unbekannt")
    snippet = doc.page_content[:300].replace("\n", " ").strip()
    
    print(f"{i}. Titel: {title}")
    print(f"   Ähnlichkeit (Score): {score:.2f}")
    print(f"   Inhalt (Auszug): {snippet}\n")

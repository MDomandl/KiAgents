from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import subprocess

import os

# === 1. Konfiguration ===
CHROMA_DB_PATH = r"D:\Users\doman\Documents\OneDrive\Dokumente\Programmierung\Projekte\AiAgents\chats"

# === 2. Embedding-Funktion (gleich wie beim Import) ===
embedding_function = HuggingFaceEmbeddings(
    model_name="intfloat/e5-large-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# === 3. Vektordatenbank laden ===
vectordb = Chroma(
    persist_directory=CHROMA_DB_PATH,
    embedding_function=embedding_function
)

# === 4. Nutzerabfrage ===
user_input = input("\n➡️ Bitte gib einen Suchtext ein (z. B. 'Roboter mit Akku'): ")
query = f"query: {user_input}"

if not query:
    print("❗ Kein Suchtext eingegeben.")
    exit()

# === 5. Suche durchführen ===
print("\n🔍 Ähnliche Chats gefunden:\n")
results = vectordb.similarity_search_with_score(query, k=10)
score_threshold = 0.4
filtered_results = [(doc, score) for doc, score in results if score <= score_threshold]
if not filtered_results:
    print("😕 Keine relevanten Ergebnisse gefunden.")
else:
    for i, (doc, score) in enumerate(filtered_results, 1):
        title = doc.metadata.get("title", "Kein Titel")
        chat_id = doc.metadata.get("chat_id", "Unbekannt")
        prompt = f"Fasse den folgenden Chat knapp zusammen (max. 5 Sätze):\n\n{doc.page_content[:4000]}"
        
        result = subprocess.run(
            ["ollama", "run", "llama3", prompt],
            capture_output=True,
            text=True,
            timeout=30,
            encoding="utf-8",
            errors="ignore"
        )
        snippet = result.stdout.strip()
        
        print(f"{i}. Titel: {title}")
        print(f"   Ähnlichkeit (Score): {score:.2f}")
        print(f"   Inhalt (Auszug): {snippet}\n")

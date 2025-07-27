from langchain_chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import subprocess
import os

# === 1. Konfiguration ===
CHROMA_DB_PATH = r"D:\Users\doman\Documents\OneDrive\Dokumente\Programmierung\Projekte\AiAgents\chats"

# === 2. Embedding-Funktion ===
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

if not query.strip():
    print("❗ Kein Suchtext eingegeben.")
    exit()

# === 5. Suche durchführen ===
print("\n🔍 Ähnliche Chats gefunden (sortiert nach Relevanz):\n")
results = vectordb.similarity_search_with_score(query, k=10)

relevanz_threshold = 0.60

# Score umwandeln in Relevanz (1 - score), sortieren absteigend
relevanz_results = [
    (doc, 1 - score) for doc, score in results
    if isinstance(score, float) and (1 - score) >= relevanz_threshold
]
relevanz_results.sort(key=lambda x: x[1], reverse=True)

# === 6. Ergebnisse anzeigen ===
if not relevanz_results:
    print("😕 Keine relevanten Ergebnisse gefunden.")
else:
    for i, (doc, relevanz) in enumerate(relevanz_results, 1):
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
        print(f"   Relevanz: {relevanz:.2f}")
        print(f"   Inhalt (Auszug): {snippet}\n")

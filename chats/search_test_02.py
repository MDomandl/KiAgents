from langchain_chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import os

# === 1. Konfiguration ===
CHROMA_DB_PATH = r"D:\\Users\\doman\\Documents\\OneDrive\\Dokumente\\Programmierung\\Projekte\\AiAgents\\chats"
RELEVANZ_THRESHOLD = 0.5
KEYWORD_BONUS = 0.2
GEWICHT_EMBEDDING = 0.7
GEWICHT_KEYWORD = 0.3

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
user_input = input("\n➡️ Bitte gib einen Suchtext ein (z. B. 'Roboter mit Akku'): ").strip()
query = f"query: {user_input}"

if not user_input:
    print("❗ Kein Suchtext eingegeben.")
    exit()

suchwoerter = user_input.lower().split()

# === 5. Embedding-Suche durchführen ===
print("\n🔍 Debug-Ausgabe für kombinierte Relevanz:\n")
results = vectordb.similarity_search_with_score(query, k=15)

anzeige_liste = []

for i, (doc, score) in enumerate(results, 1):
    if not isinstance(score, float):
        continue

    embedding_relevanz = 1 - score
    title = doc.metadata.get("title", "").lower()
    inhalt = doc.page_content.lower()

    # Keyword-Bonus prüfen
    keyword_bonus = 0.0
    for wort in suchwoerter:
        if wort in title or wort in inhalt:
            keyword_bonus = KEYWORD_BONUS
            break

    # Kombinierter Score
    gesamt_relevanz = (embedding_relevanz * GEWICHT_EMBEDDING) + (keyword_bonus * GEWICHT_KEYWORD)

    print(f"{i}. Titel: {title[:80]}...")
    print(f"   🔹 Embedding-Relevanz: {embedding_relevanz:.3f}")
    print(f"   🔸 Keyword-Bonus: {keyword_bonus:.3f}")
    print(f"   ✅ Gesamt-Relevanz: {gesamt_relevanz:.3f} (Threshold: {RELEVANZ_THRESHOLD})")

    if gesamt_relevanz >= RELEVANZ_THRESHOLD:
        anzeige_liste.append((gesamt_relevanz, title))
        print("   ➕ Wird angezeigt\n")
    else:
        print("   ➖ Zu niedrig für Anzeige\n")

# === 6. Ausgabe anzeigen ===
if anzeige_liste:
    print("\n📋 Ergebnisse mit ausreichend Relevanz:")
    for score, title in sorted(anzeige_liste, reverse=True):
        print(f"🔸 {title} (Score: {score:.3f})")
else:
    print("❌ Keine Ergebnisse über dem Relevanz-Threshold.")

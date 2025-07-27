from langchain_chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import os

# === 1. Konfiguration ===
CHROMA_DB_PATH = r"D:\\Users\\doman\\Documents\\OneDrive\\Dokumente\\Programmierung\\Projekte\\AiAgents\\chats"
RELEVANZ_THRESHOLD = 0.36
KEYWORD_BONUS_MAX = 0.3
GEWICHT_EMBEDDING = 0.6
GEWICHT_KEYWORD = 0.4
TITEL_WEIGHT = 1.5  # Gewichtung für Treffer im Titel

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
anzahl_woerter = len(suchwoerter)

# === 5. Embedding-Suche durchführen ===
print("\n🔍 Debug-Ausgabe für kombinierte Relevanz (mit gewichteten Keywords):\n")
results = vectordb.similarity_search_with_score(query, k=15)

anzeige_liste = []

for i, (doc, score) in enumerate(results, 1):
    if not isinstance(score, float):
        continue

    embedding_relevanz = 1 - score
    title = doc.metadata.get("title", "").lower()
    inhalt = doc.page_content.lower()

    keyword_score = 0.0
    for wort in suchwoerter:
        if wort in title:
            keyword_score += TITEL_WEIGHT
        elif wort in inhalt:
            keyword_score += 1.0

    # Normieren auf max mögliche Punktzahl (TITEL_WEIGHT * anzahl_woerter)
    max_score = TITEL_WEIGHT * anzahl_woerter
    keyword_ratio = min(keyword_score / max_score, 1.0) if anzahl_woerter > 0 else 0
    keyword_bonus = keyword_ratio * KEYWORD_BONUS_MAX

    # Kombinierter Score
    gesamt_relevanz = (embedding_relevanz * GEWICHT_EMBEDDING) + (keyword_bonus * GEWICHT_KEYWORD)

    print(f"{i}. Titel: {title[:80]}...")
    print(f"   🔹 Embedding-Relevanz: {embedding_relevanz:.3f}")
    print(f"   🔸 Keyword-Bonus (gewichtet): {keyword_bonus:.3f}")
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

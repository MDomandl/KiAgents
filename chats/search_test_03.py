from langchain_chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import pymysql
import os

# === 1. Konfiguration ===
CHROMA_DB_PATH = r"D:\\Users\\doman\\Documents\\OneDrive\\Dokumente\\Programmierung\\Projekte\\AiAgents\\chats"
RELEVANZ_THRESHOLD = 0.36
KEYWORD_BONUS_MAX = 0.3
GEWICHT_EMBEDDING = 0.6
GEWICHT_KEYWORD = 0.4
GEWICHT_KATEGORIE = 0.25
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

def verbinde_mit_datenbank():
    return pymysql.connect(
        host="127.0.0.1",
        user="chatuser",
        password="chatpass",
        database="gptchats",
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor
    )

conn = verbinde_mit_datenbank()
cursor = conn.cursor()

def hole_zusammenfassung(chat_id, cursor):
    cursor.execute("SELECT zusammenfassung FROM chats WHERE chat_id = %s", (chat_id,))
    row = cursor.fetchone()   
    return row if row else "[keine Zusammenfassung]"

def hole_id(chat_id, cursor):
    cursor.execute("SELECT id FROM chats WHERE chat_id = %s", (chat_id,))
    row = cursor.fetchone()   
    return row if row else "[keine id]"

def hole_kategorien(chat_id, cursor):
    print(f"ich hole: {chat_id}...")
    cursor.execute("""
        SELECT k.name, ck.relevanz 
        FROM chat_kategorien ck
        JOIN kategorien k ON ck.kategorie_id = k.id
        WHERE ck.chat_id = %s
        ORDER  BY ck.relevanz
    """, (chat_id,))   
    return {row["name"].lower() for row in cursor.fetchall()}

def normalisiere(s: str) -> str:
    return s.lower().strip()

def erkenne_query_kategorien(query: str, alle_kategorien: list[str]) -> set[str]:
    q = normalisiere(query)
    tokens = set(q.split())
    # einfache Heuristik: Kategorie-Name kommt als ganzes Wort in der Query vor
    return {k for k in alle_kategorien if normalisiere(k) in tokens}

def kategorie_match_score(chat_id: str, query_kats: set[str], kat_by_chat: dict) -> float:
    paare = kat_by_chat.get(chat_id, [])
    # Relevanz normalisieren (0..1) und nur passende Kategorien berücksichtigen
    scores = [(rel / 100.0) for (name, rel) in paare if normalisiere(name) in query_kats]
    if not scores:
        return 0.0
    return max(scores)   # oder sum(scores)/len(scores)


# === 4. Nutzerabfrage ===
user_input = input("\n➡️ Bitte gib einen Suchtext ein (z. B. 'Roboter mit Akku'): ").strip()
query = f"query: {user_input}"

if not user_input:
    print("❗ Kein Suchtext eingegeben.")
    exit()

suchwoerter = user_input.lower().split()
anzahl_woerter = len(suchwoerter)

# === 5. Embedding-Suche durchführen ===
# Kategorie-Relevanz vorbereiten
cursor.execute("""
    SELECT ck.chat_id, k.name, ck.relevanz 
    FROM chat_kategorien ck
    JOIN kategorien k ON ck.kategorie_id = k.id
""")
kat_by_chat = {}
for row in cursor.fetchall():
    cid = row['chat_id']
    eintrag = (row['name'].lower(), row['relevanz'])
    kat_by_chat.setdefault(cid, []).append(eintrag)

print("\n🔍 Debug-Ausgabe für kombinierte Relevanz (mit gewichteten Keywords):\n")
results = vectordb.similarity_search_with_score(query, k=15)

anzeige_liste = []

for i, (doc, score) in enumerate(results, 1):
    if not isinstance(score, float):
        continue

    embedding_relevanz = 1 - score
    title = doc.metadata.get("title", "").lower()  
    chat_id = doc.metadata.get("chat_id", "keine chat_id").lower()      
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

    
    # Kategorie-Relevanz berechnen
    kategorie_score = kategorie_match_score(chat_id, set(suchwoerter), kat_by_chat)
    kategorie_bonus = kategorie_score * GEWICHT_KATEGORIE

    # Kombinierter Score
    
    gesamt_relevanz = (embedding_relevanz * GEWICHT_EMBEDDING) + (keyword_bonus * GEWICHT_KEYWORD) + kategorie_bonus

    print(f"{i}. Titel: {title[:80]}...")
    print(f"   🔹 Embedding-Relevanz: {embedding_relevanz:.3f}")
    print(f"   🔸 Keyword-Bonus (gewichtet): {keyword_bonus:.3f}")
    print(f"   ✅ Gesamt-Relevanz: {gesamt_relevanz:.3f} (Threshold: {RELEVANZ_THRESHOLD})")
    print(f"   📌 Kategorie-Bonus: {kategorie_bonus:.3f}")

    if gesamt_relevanz >= RELEVANZ_THRESHOLD:
        anzeige_liste.append((gesamt_relevanz, title, chat_id))
        print("   ➕ Wird angezeigt\n")
    else:
        print("   ➖ Zu niedrig für Anzeige\n")

# === 6. Ausgabe anzeigen ===
if anzeige_liste:
    print("\n📋 Ergebnisse mit ausreichend Relevanz:")
    for score, title, chat_id in sorted(anzeige_liste, reverse=True):
        zusammenfassung =  hole_zusammenfassung(chat_id, cursor)    
        id = hole_id(chat_id, cursor)['id']       
        kategorien = hole_kategorien(id, cursor)      
        print(f"🔸 id is at: {kategorien}")  
        print(f"🔸 {title} (Score: {score:.3f}) {chat_id}")
        if kategorien:
            print(f"   🏷️ Kategorien: {', '.join(kategorien)}")
        print(f"   📝 {zusammenfassung['zusammenfassung'][:300]}{'...' if len(zusammenfassung) > 300 else ''}\n")
else:
    print("❌ Keine Ergebnisse über dem Relevanz-Threshold.")

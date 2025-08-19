from embeddings import ermittle_embedding_relevanz
from kategorien_logik import ermittle_kategorien_relevanz
from datenbank import erzeuge_db_verbindung
import pymysql

def suche_chats(suchtext):
    connection = erzeuge_db_verbindung()
    cursor = connection.cursor(pymysql.cursors.DictCursor)

    cursor.execute("SELECT * FROM chats")
    chats = cursor.fetchall()

    relevanz_treffer = []
    for chat in chats:
        embedding_relevanz = ermittle_embedding_relevanz(suchtext, chat['zusammenfassung'])
        kategorien_relevanz = ermittle_kategorien_relevanz(suchtext, chat['id'], cursor)
        keyword_bonus = ermittle_keyword_bonus(suchtext, chat['zusammenfassung'], chat['titel'])

        gesamt_relevanz = 0.7 * embedding_relevanz + 0.2 * kategorien_relevanz + 0.1 * keyword_bonus

        if gesamt_relevanz >= 0.36:
            relevanz_treffer.append({
                'id': chat['id'],
                'titel': chat['titel'],
                'zusammenfassung': chat['zusammenfassung'],
                'gesamt_relevanz': round(gesamt_relevanz, 3)
            })

    relevanz_treffer.sort(key=lambda x: x['gesamt_relevanz'], reverse=True)
    return relevanz_treffer

def ermittle_keyword_bonus(suchtext, *texte):
    bonus = 0
    for text in texte:
        if suchtext.lower() in text.lower():
            bonus += 0.1
    return min(bonus, 1.0)



def lade_chat_detail(chat_id):
    connection = erzeuge_db_verbindung()

    with connection:
        with connection.cursor() as cursor:
            # Lade Chat-Grunddaten
            cursor.execute("SELECT id, titel, zusammenfassung, inhalt FROM chats WHERE id = %s", (chat_id,))
            chat = cursor.fetchone()

            if not chat:
                return {
                    "id": chat_id,
                    "titel": "Nicht gefunden",
                    "zusammenfassung": "",
                    "inhalt": "Dieser Chat konnte nicht gefunden werden.",
                    "kategorien": []
                }

            # Lade zugehörige Kategorien
            cursor.execute("""
                SELECT k.titel FROM kategorien k
                JOIN chat_kategorien ck ON k.id = ck.kategorie_id
                WHERE ck.chat_id = %s
            """, (chat_id,))
            kategorien = [row["titel"] for row in cursor.fetchall()]
            chat["kategorien"] = kategorien

    return chat


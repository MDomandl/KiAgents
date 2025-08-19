import pymysql

def verbinde_mit_datenbank():
    return pymysql.connect(
        host="127.0.0.1",
        user="chatuser",
        password="chatpass",
        database="gptchats",
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor
    )

def insert_update_chats(chat_id, titel, erstellt_am, letzte_aenderung, message_count, chat_link, zusammenfassung, cursor):
        zusammenfassung = "[Noch keine LLM-Zusammenfassung]"
        cursor.execute(
            "INSERT INTO chats (chat_id, titel, erstellt_am, letzte_aenderung, message_count, chat_link, status, zusammenfassung) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
            "ON DUPLICATE KEY UPDATE letzte_aenderung=VALUES(letzte_aenderung), message_count=VALUES(message_count)",
            (chat_id, titel, erstellt_am, letzte_aenderung, message_count, chat_link, 'neu', zusammenfassung)
        )
        return cursor.lastrowid

def  insert_kategorien(alle_kategorien, kategorien, kategorien_llm, chat_db_id, cursor):
    for kat, rel in alle_kategorien.items():
            quelle = "llama3" if kat in kategorien_llm else "manuell"
            if kat in kategorien:
                cursor.execute(
                    "INSERT INTO chat_kategorien (chat_id, kategorie_id, relevanz, quelle) "
                    "VALUES (%s, %s, %s, %s) "
                    "ON DUPLICATE KEY UPDATE relevanz = VALUES(relevanz), quelle = VALUES(quelle)",
                    (chat_db_id, kategorien[kat], rel, quelle)
                )     
            

def update_zusammenfassung(chat_id: int, zusammenfassung: str, cursor):
    cursor.execute(
        "UPDATE chats SET zusammenfassung=%s WHERE id=%s",
        (zusammenfassung, chat_id)
    )

def update_llm_kategorien(chat_id: int, kategorien: dict, cursor):
    # Alte LLM-Kategorien löschen
    cursor.execute("DELETE FROM chat_kategorien WHERE chat_id=%s AND quelle='llm'", (chat_id,))
    
    # Neue einfügen
    for kategorie, relevanz in kategorien.items():
        cursor.execute(
            "INSERT INTO chat_kategorien (chat_id, kategorie, quelle, relevanz) VALUES (%s, %s, %s, %s)",
            (chat_id, kategorie, 'llm', relevanz)
        )

def speichere_chat_nachrichten(chat_id, nachrichten, cursor):
    sql = '''
        INSERT INTO chat_messages (chat_id, rolle, text, erstellt_am, position)
        VALUES (%s, %s, %s, %s, %s)
    '''

    for i, nachricht in enumerate(nachrichten):
        cursor.execute(sql, (
            chat_id,
            nachricht.get("rolle", "unknown"),
            nachricht.get("text", ""),
            nachricht.get("erstellt_am"),
            i
        ))

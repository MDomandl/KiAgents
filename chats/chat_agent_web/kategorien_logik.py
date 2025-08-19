def ermittle_kategorien_relevanz(chat_id, query, cursor):
    # Hole die Kategorien und Relevanzwerte aus der Pivot-Tabelle
    sql = '''
        SELECT k.name, ck.relevanz
        FROM chat_kategorien ck
        JOIN kategorien k ON ck.kategorie_id = k.id
        WHERE ck.chat_id = %s
    '''
    cursor.execute(sql, (chat_id,))
    kategorien = cursor.fetchall()

    # Bonuspunkte je nach Relevanz und Query-Keyword-Übereinstimmung
    bonus = 0.0
    for eintrag in kategorien:
        name = eintrag['name'].lower()
        if name in query.lower():
            bonus += float(eintrag['relevanz']) * 0.2  # Gewichtung ggf. anpassen

    return round(bonus, 3)

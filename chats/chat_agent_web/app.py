from flask import Flask, render_template, request
import pymysql

app = Flask(__name__)

# DB-Verbindung (anpassen)
def get_db_connection():
    return pymysql.connect(
        host="127.0.0.1",
        user="chatuser",
        password="chatpass",
        database="gptchats",
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor
    )

@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    if request.method == 'POST':
        query = request.form['query']
        connection = get_db_connection()
        with connection:
            with connection.cursor() as cursor:
                cursor.execute("""
                    SELECT
                        c.id,
                        c.titel,
                        c.zusammenfassung,
                        MAX(ck.relevanz) AS relevanz,
                        GROUP_CONCAT(k.name SEPARATOR ', ') AS kategorien
                    FROM chats c
                    LEFT JOIN chat_kategorien ck ON c.id = ck.chat_id
                    LEFT JOIN kategorien k ON ck.kategorie_id = k.id
                    WHERE c.titel LIKE %s OR c.zusammenfassung LIKE %s
                    GROUP BY c.id
                    ORDER BY relevanz DESC
                    LIMIT 10;
                """, (f"%{query}%", f"%{query}%"))
                results = cursor.fetchall()
    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request
from search_logic import suche_chats

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    suchergebnisse = []
    query = ''
    if request.method == 'POST':
        query = request.form['query']
        suchergebnisse = suche_chats(query)
    return render_template('index.html', suchergebnisse=suchergebnisse, query=query)

@app.route('/chat/<string:chat_id>')
def chat_detail(chat_id):
    # Diese Funktion lädt aus der DB den vollständigen Inhalt zum Chat
    from search_logic import lade_chat_detail
    chat = lade_chat_detail(chat_id)
    return render_template('detail.html', chat=chat)

if __name__ == '__main__':
    app.run(debug=True)

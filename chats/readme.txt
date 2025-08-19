# 📌 Erweiterungs-Ideen (Stand: August 2025)
# Diese Punkte sind geplant bzw. als mögliche nächste Schritte notiert:
#
# 1. 🔍 Semantische Suche mit Chroma / FAISS
#    - Embedding-Generierung (OpenAI/Ollama)
#    - Speicherung in ChromaDB
#    - Ähnlichkeitssuche: „Zeig mir ähnliche Chats wie...“
#
# 2. 🧠 LangChain Memory
#    - Aufbau eines Langzeitgedächtnisses für Themen
#    - Gruppierung und Kontextverlauf über Zeit
#
# 3. 🔄 Interaktionsverlauf & Mini-Autonomie
#    - Agent entscheidet, ob neue Kategorien nötig sind
#    - Rückfragen bei unklarer Einordnung
#
# 4. 📦 Web-Frontend (Flask/Streamlit)
#    - Suche, Ähnlichkeitsfilter
#    - Anzeige von Titel/Zusammenfassung
#    - Manuelle Kategoriebearbeitung möglich
#
# 5. 🧪 Hybrid-Modell: Ollama + GPT
#    - Standard: Ollama (lokal)
#    - Backup: GPT-4 via API bei Bedarf
#
# 6. 🔄 Vollautomatisierung
#    - Agent als Watchdog oder Dienst
#    - Periodischer Import/Kategorisierung


# ToDo's für Punkt 4
🧾 Unsere bisherige Agenten-ToDo-Liste (Stand: ✅ = erledigt, 🟡 = teilerledigt)
Nr.	Thema	Status
1	💾 Embedding + Keyword + Relevanz-basiertes Suchsystem	✅ Erledigt
2	🧠 Automatische Zusammenfassung und Kategorisierung (Ollama)	✅ Erledigt
3	📊 Gewichtete Relevanzanzeige mit Debug-Infos	✅ Erledigt
4	🔍 Anzeige von Titel, Score, Zusammenfassung und Kategorien	✅ Erledigt
5	🏷️ Nutzung von Relevanz in chat_kategorien zur Suchgewichtung	✅ Neu!
6	🧾 Detailanzeige eines Chats bei Auswahl	🟡 Aufgeschoben – später in HTML
7	🌐 HTML-basierte Ergebnisanzeige (suche.html, detail.html)	🔜 In Planung
8	📂 HTML mit Filteroptionen für Kategorien / Zeit / Score	🔜 Geplant
9	🧪 Vergleich Ollama vs. OpenAI Embeddings	🔜 Optional
10	🗃️ Export der Suchergebnisse als PDF / CSV	🔜 Optional
11	🧱 Modul für Datenpflege / Nachbearbeitung	🔜 Optional
12	🔄 Langfristige Automatisierung via Agent / Daemon	🔜 Visionär
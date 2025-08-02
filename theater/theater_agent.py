from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents.agent_types import AgentType
import os

# 🔑 API-Key hier einsetzen
# siehe Keepass



# 🔍 Suchwerkzeug aktivieren
search = DuckDuckGoSearchRun()

# 🧠 Sprachmodell vorbereiten
llm = OpenAI(temperature=0.7)

# 🧰 Tools definieren
tools = [
    Tool(
        name="Web Search",
        func=search.run,
        description="Nützlich zur Websuche nach Theaterstücken"
    )
]

# 🤖 Agent initialisieren
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 🎯 Dein Recherche-Ziel
ziel = """
Finde mindestens 3 Theaterstücke mit schwarzem Humor, 4–6 Rollen, 2-4 weibliche und 1-2 männliche, kleiner Bühnenbedarf, ideal für Tourneetheater.
Fasse jedes Stück kurz zusammen und gib die Bezugsquelle an (z. B. theatertexte.de oder Dramatikerverband).
"""

# ▶️ Agent starten
antwort = agent.run(ziel)

# 📄 Ergebnis anzeigen
print("\n🎭 GEFUNDENE STÜCKE:\n")
print(antwort)

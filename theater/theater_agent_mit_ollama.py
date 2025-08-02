from ddgs import DDGS
from rich.console import Console
from rich.table import Table
import ollama

console = Console()

# 🎯 Deine Kriterien
suchkriterien = "Theaterstück mit schwarzer Humor 4 bis 6 Rollen kleine Bühne alter der Schauspieler ca. 60 Jahre Stücke für Erwachsene saalbühne keine Mundart site:theatertexte.de"

console.print(f"\n🔍 Suche nach: [bold green]{suchkriterien}[/bold green]")

# 🔎 Ergebnisse holen
with DDGS() as ddgs:
    ergebnisse = ddgs.text(suchkriterien, region="de-de", safesearch="Moderate", max_results=8)

# 📋 Suchergebnisse vorbereiten
zusammenfassung_text = ""
tabelle = Table(title="🎭 Gefundene Stücke", show_lines=True)
tabelle.add_column("Titel", style="cyan", no_wrap=True)
tabelle.add_column("Kurzbeschreibung", style="dim", max_width=60)
tabelle.add_column("Link", style="magenta")

for treffer in ergebnisse:
    titel = treffer.get("title", "Kein Titel")
    beschreibung = treffer.get("body", "Keine Beschreibung")
    link = treffer.get("href", "Kein Link")
    
    tabelle.add_row(titel, beschreibung, link)

    zusammenfassung_text += f"- Titel: {titel}\n  Beschreibung: {beschreibung}\n  Link: {link}\n\n"

console.print(tabelle)

# 🤖 Zusammenfassung durch Ollama (Mistral)
console.print("\n🧠 [bold yellow]KI-Zusammenfassung läuft...[/bold yellow]\n")

antwort = ollama.chat(model="mistral", messages=[
    {
        "role": "user",
        "content": f"""
Du bist ein Theaterdramaturg. Ich habe folgende Suchergebnisse für Theaterstücke mit schwarzem Humor und kleiner Besetzung:

{zusammenfassung_text}

Gib mir eine strukturierte Bewertung:
- Welche 3 Stücke wirken am besten geeignet?
- Was sind ihre Stärken?
- Was könnte für ein Tourneetheater besonders gut passen?
Bitte auf Deutsch antworten.
"""
    }
])

console.print("[bold green]📄 Zusammenfassung:[/bold green]\n")
console.print(antwort["message"]["content"])
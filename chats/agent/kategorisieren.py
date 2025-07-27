import subprocess
import re

def generiere_kategorievorschlag(text, kategorien_liste):
    kategorien_str = ", ".join(kategorien_liste)
    prompt = (
        f"Weise dem folgenden Inhalt eine oder mehrere passende Kategorien zu:\n"
        f"{kategorien_str}\n\nInhalt:\n{text}\n\n"
        f"Gib die Antwort bitte ausschließlich(!) im folgenden Format zurück:\n"
        f"* Bewerbung: 5/5"
    )
    result = subprocess.run(["ollama", "run", "llama3", prompt],
                            capture_output=True, text=True, timeout=30, encoding="utf-8", errors="replace")
    return result.stdout.strip()

def extrahiere_kategorien_und_relevanz(text, kategorien_liste):
    kategorien_gefunden = {}
    blocks = re.split(r'\*+\s*Kategorie:|\*', text, flags=re.IGNORECASE)
    for block in blocks[1:]:
        zeilen = block.strip().splitlines()
        if not zeilen:
            continue
        erste_zeile = zeilen[0]
        match_kat = re.match(r"[*\-]?\s*([a-zA-ZäöüßÄÖÜ0-9_\- ]+):(?:\s*(?:Relevance|Relevanz):)?\s*([0-5])/5", erste_zeile.strip(), re.IGNORECASE)
        if not match_kat:
            continue
        katname = match_kat.group(1).strip().lower()
        relevanz = 50
        for line in zeilen:
            match = re.search(r"(\d+)\s*/\s*(\d+)", line)
            if match:
                zahl1 = int(match.group(1))
                zahl2 = int(match.group(2))
                if zahl2 > 0:
                    relevanz = int((zahl1 / zahl2) * 100)
                break
        if relevanz >= 40:
            kategorien_gefunden[katname] = relevanz
    return list(kategorien_gefunden.items())


import json
import subprocess

UNGUELTIGE_ZUSAMMENFASSUNGEN = [
    "turn ended.",
    "no further action will be taken.",
    "*no response*",
    "kein text mehr.",
    ""
]

UNGUELTIGE_ANFAENGE = [
    "hier ist der chatinhalt",
    "hier ist eine kurze zusammenfassung",
    "here's a brief summary",
    "this is the summary",
    "hier ist der inhalt in 3 kurzen sätzen",
    "hier sind die 3 kurzen sätze",
    "hier sind die drei kurzen sätze"
]

def lade_json(pfad):
    with open(pfad, "r", encoding="utf-8") as f:
        return json.load(f)

def generiere_zusammenfassung_ollama_debug(text):
    prompt = (
        f"Fasse den folgenden Chatinhalt in 3 kurzen Sätzen zusammen. "
        f"Keine Einleitung oder Floskeln, keine Meta-Kommentare. Nur Inhalt:/n/n{text}"
    )
    print("\n📤 Prompt an Ollama:")
    print(prompt[:1000] + "..." if len(prompt) > 1000 else prompt)

    try:
        result = subprocess.run(
            ["ollama", "run", "llama3", prompt],
            capture_output=True,
            text=True,
            timeout=30,
            encoding="utf-8",
            errors="ignore"
        )
        print("\n📥 Roh-Ausgabe von Ollama (stdout):")
        print(result.stdout.strip())

        if result.stderr:
            print("\n⚠️ Ollama stderr:")
            print(result.stderr.strip())

        cleaned = result.stdout.strip().lower()
        for start in UNGUELTIGE_ANFAENGE:
            if cleaned.startswith(start):
                print("\n⛔️ Startphrase unerwünscht → Zusammenfassung blockiert.")
                return "[Zusammenfassung nicht erzeugbar]"
        if cleaned in UNGUELTIGE_ZUSAMMENFASSUNGEN:
            print("\n⛔️ Inhalt gilt als unbrauchbar → Zusammenfassung blockiert.")
            return "[Zusammenfassung nicht erzeugbar]"
        return result.stdout.strip()

    except Exception as e:
        print(f"❌ Fehler bei subprocess: {e}")
        return f"[Zusammenfassung fehlgeschlagen: {e}]"

def main():
    daten = lade_json("conversations.json")
    erster_chat = daten[0]
    titel = erster_chat.get("title", "Kein Titel")
    print(f"📝 Titel des ersten Chats: {titel}")

    messages = list(erster_chat.get("mapping", {}).values())
    message_text = "\n\n".join(
        str(msg["message"]["content"]["parts"][0])
        for msg in messages
        if (
            "message" in msg and
            isinstance(msg["message"], dict) and
            "content" in msg["message"] and
            isinstance(msg["message"]["content"], dict) and
            "parts" in msg["message"]["content"] and
            isinstance(msg["message"]["content"]["parts"], list) and
            msg["message"]["content"]["parts"]
        )
    )[:4000]

    print(f"📄 Anzahl Messages: {len(messages)}")
    print(f"🧾 Textausschnitt (4000 Zeichen): {message_text[:300]}{'...' if len(message_text) > 300 else ''}")

    zusammenfassung = generiere_zusammenfassung_ollama_debug(message_text)
    print(f"🧠 Ergebnis-Zusammenfassung:\n{zusammenfassung}")

if __name__ == "__main__":
    main()


import json
import subprocess

UNGUELTIGE_ANFAENGE = [
    "hier ist der chatinhalt",
    "hier ist eine kurze zusammenfassung",
    "here's a brief summary",
    "this is the summary",
    "hier ist der inhalt in 3 kurzen sätzen",
    "hier sind die 3 kurzen sätze",
    "hier sind die drei kurzen sätze",
     "here's a brief summary of the chat content:",
    "hier ist eine kurze zusammenfassung des chatinhalts:",
    "Hier ist der Chatinhalt in 3 kurzen Sätzen zusammengefasst:",
    "Hier sind die drei kurzen Sätze:",
    "Hier ist die Zusammenfassung:",
    "Hier ist die Zusammenfassung in 3 Sätzen:",
    "Here are the 3 short sentences summarizing the chat content:",
    "Hier ist eine Zusammenfassung des Chats in 3 kurzen Sätzen:",
    "Here is a summary of the chat content in 3 short sentences:",
    "In 3 kurzen sätzen zusammengefasst:",
    "In 3 kurzen sätzen:",
    ""
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
        raw_output = result.stdout.strip()
        print("\n📥 Roh-Ausgabe von Ollama (stdout):")
        print(raw_output)

        cleaned = raw_output.lower().strip()
        for start in UNGUELTIGE_ANFAENGE:
            if cleaned.startswith(start):
                print(f"✂️ Entferne Startphrase: '{start}'")
                raw_output = raw_output[len(start):].strip().lstrip(":.-— ").capitalize()
                break

        return raw_output

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

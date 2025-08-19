import subprocess

def generiere_zusammenfassung(text):
    prompt = f"Fasse den folgenden Chat knapp zusammen (max. 3 Sätze):\n\n{text[:2000]}"
    result = subprocess.run(["ollama", "run", "llama3", prompt],
                            capture_output=True, text=True, timeout=90, encoding="utf-8", errors="ignore")
    return result.stdout.strip()

def get_chat_text(messages):
        return "\n\n".join(
            str(m["message"]["content"]["parts"][0])
            for m in messages
            if (
                "message" in m and isinstance(m["message"], dict) and
                "content" in m["message"] and isinstance(m["message"]["content"], dict) and
                "parts" in m["message"]["content"] and
                isinstance(m["message"]["content"]["parts"], list) and
                m["message"]["content"]["parts"]
            )
        )[:3000]
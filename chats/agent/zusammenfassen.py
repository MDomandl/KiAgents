import subprocess

def generiere_zusammenfassung(text):
    prompt = f"Fasse den folgenden Chat knapp zusammen (max. 3 Sätze):\n\n{text[:2000]}"
    result = subprocess.run(["ollama", "run", "llama3", prompt],
                            capture_output=True, text=True, timeout=30, encoding="utf-8", errors="replace")
    return result.stdout.strip()

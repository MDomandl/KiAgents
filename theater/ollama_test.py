import ollama

antwort = ollama.chat(model="mistral", messages=[
    {"role": "user", "content": "Nenne mir drei Theaterstücke mit schwarzem Humor für 4–6 Personen."}
])

print(antwort["message"]["content"])
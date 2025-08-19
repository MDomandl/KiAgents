from sentence_transformers import SentenceTransformer, util

# Modell laden (achte darauf, dass du das gleiche Modell verwendest wie beim Speichern!)
model = SentenceTransformer('intfloat/e5-large-v2')

# Embedding-Relevanz berechnen
def ermittle_embedding_relevanz(query, text):
    query_embedding = model.encode(query, convert_to_tensor=True)
    text_embedding = model.encode(text, convert_to_tensor=True)
    score = util.cos_sim(query_embedding, text_embedding).item()
    return round(score, 3)

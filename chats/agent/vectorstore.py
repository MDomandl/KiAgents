from langchain_chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from agent.config import CHROMA_PATH

def init_chroma():
    embeddings = HuggingFaceEmbeddings(
        model_name='intfloat/e5-large-v2',
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

def speichere_embedding(chat_id, title, summary, content, vectordb, overwrite=True):
    text = f"passage: Titel: {title}\nZusammenfassung: {summary}\nInhalt: {content}"
    metadata = {"chat_id": chat_id, "title": title}
    print(f"🔄 Speichere Embedding für Chat {chat_id}...")
    print(f"✅ Eingefügt in Chroma: {chat_id} – {title[:50]}...")
    if not overwrite:
        result = vectordb.similarity_search(text, k=3)
        for doc in result:
            if doc.metadata.get("chat_id") == chat_id:
                print(f"⏭️ Chat {chat_id} bereits vorhanden – wird übersprungen.")
                return
    vectordb.add_documents([Document(page_content=text, metadata=metadata)])
  #  vectordb.persist()
    print(f"💾 Chat {chat_id} gespeichert.")

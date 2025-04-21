import chromadb
from chromadb.config import Settings
from app.core.config import settings


client = chromadb.Client(Settings(
    persist_directory=settings.chroma_persist_dir,
    anonymized_telemetry=False
))

collection = client.get_or_create_collection("pdf_chunks")

def store_chunk_embedding(chunk_id: str, text: str, embedding: list):
    collection.add(
        ids=[chunk_id],
        documents=[text],
        embeddings=[embedding]
    )

def query_similar_chunks(query_embedding: list, k: int = 3):
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

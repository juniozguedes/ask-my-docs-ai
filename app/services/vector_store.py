# services/vectorstore.py
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from app.core.config import settings

def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory=settings.chroma_persist_dir, embedding_function=embeddings)


# Create a temporary in-memory vectorstore just for this session
def create_vectorstore_from_chunks(chunks):
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    import tempfile

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # Use a temp directory for session-based vectorstore (non-persistent)
    persist_dir = tempfile.mkdtemp()

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )

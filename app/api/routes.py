from fastapi import APIRouter, UploadFile, File, HTTPException
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from app.core.config import settings
from app.services import llm
import tempfile
import os
import logging
import asyncio

logger = logging.getLogger(__name__)

router = APIRouter()

# Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Faster embeddings
CHROMA_PATH = settings.chroma_persist_dir

# Global state for vector store
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

async def process_pdf(file_path: str):
    """Process PDF asynchronously"""
    loop = asyncio.get_event_loop()
    loader = PyPDFLoader(file_path)
    pages = await loop.run_in_executor(None, loader.load)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return text_splitter.split_documents(pages)

@router.post("/concept")
async def concept(file: UploadFile = File(...)):
    try:
        logger.info("Processing PDF file")
        # Validate file type
        if file.content_type != "application/pdf":
            raise HTTPException(400, detail="Only PDF files are allowed")

        # Process PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        try:
            logger.info("Processing PDF file")
            # Async document processing
            chunks = await process_pdf(temp_path)
            
            # Update vector store
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: db.add_documents(chunks)
            )
            
            # Query processing
            logger.info("Querying vector store")
            query = "What is the title of the document?"
            docs = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: db.similarity_search(query, k=3)
            )
            context = "\n\n".join([doc.page_content for doc in docs])

            # Generate response
            logger.info("Generating response")
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: llm.create_chat_completion(
                    messages=[{
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {query}"
                    }],
                    max_tokens=400,
                    temperature=0.3
                )
            )
            logger.info("Response generated")
            logger.info("Response: %s", response["choices"])
            return {"answer": response["choices"][0]["message"]["content"]}

        finally:
            os.unlink(temp_path)

    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(500, detail=f"Processing error: {str(e)}")
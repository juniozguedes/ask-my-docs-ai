from fastapi import APIRouter, UploadFile, File, HTTPException
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import settings
from app.services.llm import create_response
from app.workers.pdf_processor import process_pdf
import tempfile
import os
import logging
import asyncio
from app.services.vector_store import get_vectorstore

logger = logging.getLogger(__name__)

router = APIRouter()

# Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Faster embeddings
CHROMA_PATH = settings.chroma_persist_dir

# Global state for vector store
db = get_vectorstore()


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
            query = "Who is the author of this document? does any name appear on the footer?"
            docs = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: db.similarity_search(query, k=3)
            )
            context = "\n\n".join([doc.page_content for doc in docs])

            # Generate response
            logger.info("Generating response")
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: create_response(context, query))


            logger.info("Response generated")
            logger.info("Response: %s", response["choices"])
            return {"answer": response["choices"][0]["message"]["content"]}

        finally:
            os.unlink(temp_path)

    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(500, detail=f"Processing error: {str(e)}")
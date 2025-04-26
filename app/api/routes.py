import os
import logging
import asyncio
import tempfile

from fastapi import APIRouter, UploadFile, File, HTTPException, Form

from app.core.config import settings
from app.services.llm import create_response_from_messages
from app.services.session_manager import create_session, get_session, set_vectorstore, add_chat, get_chat_history
from app.workers.pdf_processor import process_pdf
from app.services.vector_store import get_vectorstore, create_vectorstore_from_chunks

logger = logging.getLogger(__name__)

router = APIRouter()

# Configuration
CHROMA_PATH = settings.chroma_persist_dir

# Global state for vector store
db = get_vectorstore()


@router.post("/askmydocs/upload")
async def upload_pdf(file: UploadFile = File(...)):
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
            
            # Create session
            session_id = create_session()


            # Create a NEW vectorstore in memory
            vectorstore = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: create_vectorstore_from_chunks(chunks)
            )
            set_vectorstore(session_id, vectorstore)
            logger.info(f"Session {session_id} created")
            return {"session_id": session_id}

        finally:
            os.unlink(temp_path)

    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(500, detail=f"Processing error: {str(e)}")


@router.post("/askmydocs/ask")
async def ask_question(session_id: str = Form(...), query: str = Form(...)):
    try:
        logger.info(f"Received question for session {session_id}")

        session = get_session(session_id)
        if not session:
            raise HTTPException(404, detail="Session not found")

        vectorstore = session["vectorstore"]
        if not vectorstore:
            raise HTTPException(400, detail="No document uploaded yet")

        # Retrieve relevant context
        docs = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: vectorstore.similarity_search(query, k=3)
        )
        context = "\n\n".join([doc.page_content for doc in docs])

        # Build complete chat history
        chat_history = get_chat_history(session_id)
        full_messages = chat_history + [
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]

        # Generate response
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: create_response_from_messages(full_messages))

        # Save this interaction into chat history
        add_chat(session_id, "user", query)
        add_chat(session_id, "assistant", response["choices"][0]["message"]["content"])

        return {"answer": response["choices"][0]["message"]["content"]}

    except Exception as e:
        logger.error(f"Ask error: {str(e)}")
        raise HTTPException(500, detail=f"Ask error: {str(e)}")
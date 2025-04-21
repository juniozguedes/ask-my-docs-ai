from fastapi import APIRouter, UploadFile, File
from app.services.file_handler import save_and_process_pdf

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.embeddings import get_embedding
from app.services.vector_store import query_similar_chunks
from app.services.llm import generate_answer
router = APIRouter()

class AskRequest(BaseModel):
    question: str

@router.post("/ask")
async def ask_question(payload: AskRequest):
    try:
        query = payload.question
        query_embedding = get_embedding(query)
        results = query_similar_chunks(query_embedding, k=4)
        context_chunks = results['documents'][0]

        answer = generate_answer(question=query, context_chunks=context_chunks)
        return {"answer": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        return {"error": "Only PDF files are supported."}
    
    chunks = await save_and_process_pdf(file)

    for chunk in chunks:
        embedding = get_embedding(chunk)
        store_chunk_embedding(str(uuid.uuid4()), chunk, embedding)

    return {"filename": file.filename, "chunks": chunks}

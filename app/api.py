from fastapi import APIRouter, UploadFile, File
from app.file_handler import save_and_process_pdf

router = APIRouter()

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        return {"error": "Only PDF files are supported."}
    
    chunks = await save_and_process_pdf(file)
    return {"filename": file.filename, "chunks": chunks}

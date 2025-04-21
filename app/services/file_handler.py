import os
import uuid
import pdfplumber

DOCUMENTS_DIR = "documents"
os.makedirs(DOCUMENTS_DIR, exist_ok=True)

async def save_and_process_pdf(file):
    file_id = str(uuid.uuid4())
    file_path = os.path.join(DOCUMENTS_DIR, f"{file_id}_{file.filename}")
    
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Extract text
    with pdfplumber.open(file_path) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)

    # Chunking (simple split for now, later we'll improve)
    chunks = text.split("\n\n")  # crude paragraph-level split

    return [chunk.strip() for chunk in chunks if chunk.strip()]

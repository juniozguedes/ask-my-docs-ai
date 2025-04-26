# workers/pdf_processor.py
import asyncio
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


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

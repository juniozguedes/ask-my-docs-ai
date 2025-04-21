from fastapi import FastAPI
from app.api import router as api_router
import logging

logging.getLogger("pdfminer").setLevel(logging.ERROR)
app = FastAPI(title="AI Doc Chatbot")

app.include_router(api_router)

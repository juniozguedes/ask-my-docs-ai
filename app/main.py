from fastapi import FastAPI
from app.api.routes import router as api_router
from app.core.config import settings
from app.core.logging import init_logging
import logging

# Reduce pdfminer noise
logging.getLogger("pdfminer").setLevel(logging.ERROR)

# Set up app logging
init_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Doc Chatbot")

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 App starting in %s mode", settings.environment)

app.include_router(api_router)

from fastapi import FastAPI
from app.api.routes import router as api_router
from app.core.config import settings
from app.core.logging import init_logging
from app.services.llm import load_model
import logging

init_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Doc Chatbot")

@app.on_event("startup")
async def startup_event():
    logger.info("Starting in %s mode", settings.environment)
    logger.info("Loading LLaMA model")
    load_model()
    logger.info("LLaMA model loaded")

app.include_router(api_router)

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    environment: str = "dev"
    openai_api_key: str = ""
    chroma_persist_dir: str = "chroma_db"
    model_dir: str = "models"
    class Config:
        env_file = ".env"

settings = Settings()

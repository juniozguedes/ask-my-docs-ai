from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    chroma_persist_dir: str = "vector_db"
    environment: str = "dev"

    class Config:
        env_file = ".env"

settings = Settings()

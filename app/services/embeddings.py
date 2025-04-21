import os 
from openai import OpenAI
from typing import List
from app.core.config import settings

client = OpenAI(api_key=settings.openai_api_key)

def get_embedding(text: str, model="text-embedding-3-small") -> list:
    response = client.embeddings.create(
        input=[text],
        model=model
    )
    return response.data[0].embedding

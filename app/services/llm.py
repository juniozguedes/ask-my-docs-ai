from openai import OpenAI
from app.core.config import settings


client = OpenAI(api_key=settings.openai_api_key)

def generate_answer(question: str, context_chunks: list) -> str:
    context = "\n\n".join(context_chunks)
    prompt = f"""
You are an assistant that answers questions based only on the provided context.

Context:
{context}

Question: {question}
Answer:
    """.strip()

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You answer questions about uploaded documents."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()
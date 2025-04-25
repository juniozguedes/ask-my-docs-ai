from llama_cpp import Llama
import os
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)
llm = None  # Lazy-loaded model

def load_model():
    global llm
    if llm is None:
        model_path = os.path.join(settings.model_dir, "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf")
        logger.info(f"Loading LLaMA model {model_path}")
        print(f"🧠 Loading LLaMA model from: {model_path}")
        if not os.path.exists(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")
        
        llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_gpu_layers=-1,
            n_threads=6,
            verbose=False
        )


def create_chat_completion(messages, max_tokens=512, temperature=0.7):
    if llm is None:
        raise RuntimeError("LLM not loaded. Call `load_model()` first.")
    
    formatted = [{"role": msg["role"], "content": msg["content"]} for msg in messages]
    return llm.create_chat_completion(
        messages=formatted,
        max_tokens=max_tokens,
        temperature=temperature
    )


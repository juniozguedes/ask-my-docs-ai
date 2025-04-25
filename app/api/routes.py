from fastapi import APIRouter, UploadFile, File, HTTPException
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForCausalLM
from app.core.config import settings
import torch
import tempfile
import os
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Configuration
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLAMA_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
CHROMA_PATH = settings.chroma_persist_dir

# Llama-3 specific tokens
EOS_TOKEN = "<|eot_id|>"

@router.post("/concept")
async def concept(file: UploadFile = File(...)):
    try:
        logger.info("Processing PDF file")
        # Validate file type
        if file.content_type != "application/pdf":
            raise HTTPException(400, detail="Only PDF files are allowed")

        # File handling
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        # Process PDF
        loader = PyPDFLoader(temp_path)
        pages = loader.load()
        os.unlink(temp_path)  # Cleanup temp file

        # Text splitting
        logger.info("Splitting text into chunks")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = text_splitter.split_documents(pages)

        # Generate embeddings
        logger.info("Generating embeddings")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

        # Generate Chroma
        logger.info("Generating Chroma")
        db = Chroma.from_documents(
            chunks, 
            embeddings, 
            persist_directory=CHROMA_PATH
        )

        # Initialize Llama model with authentication
        logger.info("Initializing Llama model")
        tokenizer = AutoTokenizer.from_pretrained(
            LLAMA_MODEL_NAME,
            token=True  # Use HF token from login
        )

        model = AutoModelForCausalLM.from_pretrained(
            LLAMA_MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=True  # Use HF token from login
        )

        # Query processing
        logger.info("Querying Llama model")
        query = "What is the headline or title of the document?"
        logger.info("Query: %s", query)
        docs = db.similarity_search(query, k=5)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Format prompt using Llama 3's template
        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]

        # Tokenize with correct template
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        # Generation parameters
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids(EOS_TOKEN)
        ]

        # Generate response
        logger.info("Generating response")
        outputs = model.generate(
            input_ids,
            max_new_tokens=512,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )
        logger.info("Finished generating response")

        # Decode response
        logger.info("Decoding response")
        response = tokenizer.decode(
            outputs[0][input_ids.shape[-1]:], 
            skip_special_tokens=True
        )
        print(response)
        return {"answer": response}

    except Exception as e:
        raise HTTPException(500, detail=f"Processing error: {str(e)}")
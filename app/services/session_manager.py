import uuid
import logging


logger = logging.getLogger(__name__)

# Global in-memory session store (replace with Redis later)
sessions = {}

def create_session():
    logger.info("Creating new session")
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "vectorstore": None,
        "chat_history": []
    }
    logger.info(f"Session {session_id} created")
    return session_id

def get_session(session_id):
    logger.info(f"Getting session {session_id}")
    return sessions.get(session_id)

def set_vectorstore(session_id, vectorstore):
    logger.info(f"Setting vectorstore for session {session_id}")
    if session_id in sessions:
        sessions[session_id]["vectorstore"] = vectorstore
        logger.info(f"Set vectorstore for session {session_id}")
    else:
        logger.warning(f"Session {session_id} not found")

def add_chat(session_id, role, content):
    logger.info(f"Adding chat to session {session_id}")
    if session_id in sessions:
        sessions[session_id]["chat_history"].append({
            "role": role,
            "content": content
        })
        logger.info(f"Added chat to session {session_id}")
    else:
        logger.warning(f"Session {session_id} not found")

def get_chat_history(session_id):
    logger.info(f"Getting chat history for session {session_id}")
    if session_id in sessions:
        return sessions[session_id]["chat_history"]
    return []

import uuid

# Global in-memory session store (replace with Redis later)
sessions = {}

def create_session():
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "vectorstore": None,
        "chat_history": []
    }
    return session_id

def get_session(session_id):
    return sessions.get(session_id)

def set_vectorstore(session_id, vectorstore):
    if session_id in sessions:
        sessions[session_id]["vectorstore"] = vectorstore

def add_chat(session_id, role, content):
    if session_id in sessions:
        sessions[session_id]["chat_history"].append({
            "role": role,
            "content": content
        })

def get_chat_history(session_id):
    if session_id in sessions:
        return sessions[session_id]["chat_history"]
    return []

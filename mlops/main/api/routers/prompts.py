from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from typing import Annotated
from sqlalchemy.orm import Session
from pydantic import BaseModel
from ..database import SessionLocal
from .auth import get_current_user
from ..models import Prompts
from langchain_core.documents import Document
from ..ragchain import build_chain
from ..mongodb import queries_collection
from ..redisdb_client import redis_client
import json

router = APIRouter(prefix="/prompts", tags=["prompts"])

def document_to_dict(doc: Document):
    return {
        "page_content": doc.page_content,
        "metadata": doc.metadata
    }

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session, Depends(get_db)]
user_dependency = Annotated[dict, Depends(get_current_user)]


def add_chat_pair(redis_client, user_id, session_id, user_msg, assistant_msg):
    key = f"chat:session:{user_id}:{session_id}"
    pair = json.dumps({
        "question": user_msg,
        "answer": assistant_msg,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })
    redis_client.rpush(key, pair)


def add_session_to_user(redis_client, user_id, session_id, title):
    redis_client.hset(f"chat:sessions:{user_id}", session_id, title)


def get_sessions(redis_client, user_id):
    return redis_client.hgetall(f"chat:sessions:{user_id}")


def get_session_pairs(redis_client, user_id, session_id):
    raw_pairs = redis_client.lrange(f"chat:session:{user_id}:{session_id}", 0, -1)
    pairs = []
    for p in raw_pairs:
        try:
            pairs.append(json.loads(p))
        except Exception:
            continue
    return pairs


class CreatePrompt(BaseModel):
    question: str
    session_id: str

@router.get("/chat/sessions")
def list_sessions(user: user_dependency):
    sessions = get_sessions(redis_client, user["user_id"])
    return {k.decode(): v.decode() for k, v in sessions.items()}


@router.get("/chat/session/{session_id}")
def get_session(session_id: str, user: user_dependency):
    pairs = get_session_pairs(redis_client, user["user_id"], session_id)
    return pairs

@router.get("/addsession")
def add_session(db:db_dependency,user:user_dependency):
    user_prompts=db.query(Prompts).filter(user["user_id"]==Prompts.owner_id).filter(Prompts.session_id).max()
    new_session = Prompts(



    )

@router.post("/create-prompt")
async def create_prompt(create: CreatePrompt, db: db_dependency, user: user_dependency):
    if user is None:
        raise HTTPException(status_code=401, detail="Not logged in")
    user_id = user.get("user_id")
    
    qa_chain, embeddings = await build_chain(user_id)
    result = await qa_chain.ainvoke({"input": create.question})

    if not result:
        raise HTTPException(status_code=400, detail="No answer generated")

    embedding_vector = embeddings.embed_query(create.question)
    result["context"] = [document_to_dict(doc) for doc in result["context"]]
    short_title = " ".join(result["answer"].split()[:5])
    session_id = create.session_id
    new_prompt = Prompts(
        title=short_title,
        question=create.question,
        owner_id=user_id,
        session_id = int(session_id)
    )
    db.add(new_prompt)
    db.commit()
    db.refresh(new_prompt)
    prompt_id_current = new_prompt.id
    
    queries_collection.insert_one({
        "user_id": user_id,
        "question": create.question,
        "answer": result["answer"],
        "embedding": embedding_vector,
        "timestamp": datetime.now(timezone.utc),
        "session_id": int(session_id),
        "sql_prompt_id": prompt_id_current
    })
    prompt_id=str(prompt_id_current)
    redis_client.lpush(f"chat:session:{user_id}:{session_id}:{prompt_id}", json.dumps({
        "question": create.question,
        "answer": result["answer"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sql_prompt_id": prompt_id_current
    }))
    redis_client.ltrim(f"chat:session:{user_id}:{session_id}:{prompt_id}", 0, 99)

    add_session_to_user(redis_client, user_id, session_id, short_title)
    add_chat_pair(redis_client, user_id, session_id, create.question, result["answer"])

    return {"answer": result["answer"]}

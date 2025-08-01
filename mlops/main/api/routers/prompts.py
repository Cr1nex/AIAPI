from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from typing import Annotated
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
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
    sessions = redis_client.hgetall(f"chat:sessions:{user_id}")
    return {k.decode(): v.decode() for k, v in sessions.items()}

def get_session_pairs(redis_client, user_id, session_id):
    pattern = f"chat:session:{user_id}:{session_id}:*"
    keys = redis_client.keys(pattern)
    all_pairs = []
    for key in keys:
        raw_pairs = redis_client.lrange(key, 0, -1)
        for p in raw_pairs:
            try:
                all_pairs.append(json.loads(p.decode("utf-8")))
            except Exception:
                continue
    return all_pairs

class CreatePrompt(BaseModel):
    question: str
    session_id: str  = Field(default="0", min_length=1)

@router.get("/chat/sessions")
async def get_sessions(user: user_dependency):
    user_id = user["user_id"]
    redis_key = f"chat:sessions:{user_id}"
    sessions_raw = redis_client.hgetall(redis_key)

    sessions = []
    for session_id, title in sessions_raw.items():
        sessions.append({
            "session_id": session_id.decode(),
            "title": title.decode()
        })

    return sessions


@router.get("/chat/session/{session_id}")
async def get_session(session_id: str, user: user_dependency):
    if user is None:
        raise HTTPException(status_code=401, detail="Not logged in")
    return get_session_pairs(redis_client, user["user_id"], session_id)

@router.get("/addsession")
async def add_session(db: db_dependency, user: user_dependency):
    if user is None:
        raise HTTPException(status_code=401, detail="Not logged in")
    
    user_id=user["user_id"]
    

    last_prompt = db.query(Prompts).filter(Prompts.owner_id == user_id).order_by(Prompts.session_id.desc()).first()
    new_session_id = (last_prompt.session_id + 1) if last_prompt else 1
    
    new_prompt = Prompts(
        title="New session",
        question="New session",
        owner_id=user["user_id"],
        session_id=new_session_id
    )
    db.add(new_prompt)
    db.commit()
    db.refresh(new_prompt)
    
    session_id_str = str(new_prompt.session_id)
    prompt_id_str = str(new_prompt.id)
    redis_key = f"chat:session:{user_id}:{session_id_str}:{prompt_id_str}"

    
    redis_client.lpush(redis_key, json.dumps({
        "question": "__init__",
        "answer": "Session started.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sql_prompt_id": new_prompt.id
    }))
    add_session_to_user(redis_client, user_id, session_id_str, f"New Session {session_id_str}")
    return {"new_session_id": new_session_id, "prompt_id": new_prompt.id}

@router.post("/create-prompt")
async def create_prompt(create: CreatePrompt, db: db_dependency, user: user_dependency):
    if user is None:
        raise HTTPException(status_code=401, detail="Not logged in")

    user_id = user["user_id"]
    session_id_str = create.session_id

    if session_id_str == "0":
        
        last_prompt = db.query(Prompts).filter(Prompts.owner_id == user_id).order_by(Prompts.session_id.desc()).first()
        if last_prompt:
            session_id = last_prompt.session_id  
        else:
                       
            session_id = 1 

            
            new_session_prompt = Prompts(
                title="New session",
                question="New session",
                owner_id=user_id,
                session_id=session_id
            )
            db.add(new_session_prompt)
            db.commit()
            db.refresh(new_session_prompt)

           
            redis_key = f"chat:session:{user_id}:{session_id}:{new_session_prompt.id}"
            redis_client.lpush(redis_key, json.dumps({
                "question": "__init__",
                "answer": "Session started.",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "sql_prompt_id": new_session_prompt.id
            }))
            add_session_to_user(redis_client, user_id, str(session_id), "New Session 1")

        session_id_str = str(session_id)
    else:
        session_id = int(session_id_str)

    
    qa_chain, embeddings = await build_chain(user_id, session_id_str)
    result = await qa_chain.ainvoke({"input": create.question})

    if not result or "answer" not in result:
        raise HTTPException(status_code=400, detail="No answer generated")

    short_title = " ".join(result["answer"].split()[:5])

    new_prompt = Prompts(
        title=short_title,
        question=create.question,
        owner_id=user_id,
        session_id=session_id
    )
    db.add(new_prompt)
    db.commit()
    db.refresh(new_prompt)

    embedding_vector = embeddings.embed_query(create.question)
    result["context"] = [document_to_dict(doc) for doc in result["context"]]

    prompt_id = str(new_prompt.id)

    queries_collection.insert_one({
        "user_id": user_id,
        "question": create.question,
        "answer": result["answer"],
        "embedding": embedding_vector,
        "timestamp": datetime.now(timezone.utc),
        "session_id": session_id,
        "sql_prompt_id": new_prompt.id
    })

    redis_client.lpush(f"chat:session:{user_id}:{session_id_str}:{prompt_id}", json.dumps({
        "question": create.question,
        "answer": result["answer"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sql_prompt_id": prompt_id
    }))
    redis_client.ltrim(f"chat:session:{user_id}:{session_id_str}:{prompt_id}", 0, 99)
    add_session_to_user(redis_client, user_id, session_id_str, short_title)

    return {"answer": result["answer"]}
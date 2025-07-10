from datetime import datetime ,timezone
from fastapi import APIRouter,Depends,HTTPException
from typing import Annotated
from ..database import SessionLocal
from sqlalchemy.orm import Session
from pydantic import BaseModel
from .auth import get_current_user 
from ..models import Prompts
from typing_extensions import Annotated, TypedDict
import os
from langchain_core.documents import Document
from fastapi import Depends, APIRouter
from ..ragchain import build_chain
from .auth import get_current_user
from ..mongodb import queries_collection
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

router = APIRouter(prefix="/prompts",
                   tags=["prompts"])


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

db_dependency = Annotated[Session,Depends(get_db)]
user_dependency = Annotated[dict,Depends(get_current_user)]

class create_prompt(BaseModel):
    title : str
    question :str



@router.post("/create-prompt")
async def create_prompt(create:create_prompt,db:db_dependency,user:user_dependency):
    
    if user is None:
        raise HTTPException(status_code=401, detail="Not logged in")
    user_id = user.get("user_id") 
    qa_chain, embeddings = build_chain(user_id,db)
    result = await qa_chain.ainvoke({
    "input": create.question
})

    if not result:
        raise HTTPException(status_code=400, detail="No answer generated")

   
    embedding_vector = embeddings.embed_query(create.question)
    result["context"] = [document_to_dict(doc) for doc in result["context"]]
    new_prompt = Prompts(
        title=create.title,
        question=create.question,  
        owner_id=user_id,
    )
    db.add(new_prompt)
    db.commit()
    db.refresh(new_prompt)
    prompt_id_current=new_prompt.id
    queries_collection.insert_one(
        {
            "user_id": user_id,
            "question": create.question,
            "answer": result,
            "embedding": embedding_vector,
            "timestamp": datetime.now(timezone.utc),
            "sql_prompt_id": prompt_id_current
        }
    )

    
    

    return {"answer": result}
    


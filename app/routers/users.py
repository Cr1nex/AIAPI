from fastapi import APIRouter,Depends,HTTPException
from typing import Annotated
from ..database import SessionLocal
from sqlalchemy.orm import Session
from .auth import get_current_user
from ..models import Prompts


router = APIRouter(prefix="/users",
                   tags=["users"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session,Depends(get_db)]
user_dependency = Annotated[dict,Depends(get_current_user)]


@router.get("/prompts")
async def get_prompts(db:db_dependency,user:user_dependency):
    if user == None:
        raise HTTPException(status_code=401,detail="No user")

    prompts = db.query(Prompts).filter(Prompts.owner_id == user.get("user_id")).all()
    if prompts == None:
        raise HTTPException(status_code=401,detail="No prompts")
    return prompts


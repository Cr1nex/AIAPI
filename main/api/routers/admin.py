from fastapi import APIRouter,Depends,HTTPException
from typing import Annotated
from ..database import SessionLocal
from sqlalchemy.orm import Session
from ..models import Users,Prompts
from .auth import get_current_user


router = APIRouter(prefix="/admin",
                   tags=["admin"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session,Depends(get_db)]
user_dependency = Annotated[dict,Depends(get_current_user)]

@router.get("/getusers")
async def get_users(db:db_dependency,usera:user_dependency):
    if usera.get("user_role") != "admin":
        raise HTTPException(status_code=401,detail="Not Authenticated")

    user = db.query(Users).all()
    
    if user is None:
        raise HTTPException(status_code=401,detail="No users!!")
    
    return user


    
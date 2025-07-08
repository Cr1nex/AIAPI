from fastapi import Depends
from fastapi.routing import APIRouter
from ..database import SessionLocal
from typing import Annotated
from sqlalchemy.orm import Session


router = APIRouter(prefix='/users',
                   tags=['users'])

def get_user():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

    

db_dependency = Annotated[Session,Depends(get_user)]
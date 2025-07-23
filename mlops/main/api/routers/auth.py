from fastapi import APIRouter,Depends,HTTPException
from typing import Annotated
from fastapi.responses import JSONResponse
from ..database import SessionLocal
from sqlalchemy.orm import Session
from pydantic import BaseModel
from passlib.context import CryptContext
from datetime import timedelta, datetime, timezone
from jose import jwt,JWTError
from fastapi.security import OAuth2PasswordBearer,OAuth2PasswordRequestForm
from ..models import Users
from dotenv import load_dotenv
import os

load_dotenv()

SECRET_KEY = os.getenv('SECRET_KEY')
ALGORITHM = os.getenv('ALGORITHM')

ACCESS_TOKEN_EXPIRE_MINUTES = 30

router = APIRouter(prefix="/auth",
                   tags=["auth"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session,Depends(get_db)]
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_bearer = OAuth2PasswordBearer(tokenUrl='auth/token')

class Create_User_Request(BaseModel):
    username:str
    email:str 
    first_name :str
    last_name :str
    phone_number:str
    password:str

class Token_Data(BaseModel):
    username:str
    password:str

class Token(BaseModel):
    access_token: str
    token_type: str



def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(username:str,user_id:int,user_role:str,expires:timedelta):
    encode = {"sub":username,"id":user_id,"role":user_role}
    expires = datetime.now(timezone.utc) + expires
    encode.update({"exp": int(expires.timestamp())})
    return jwt.encode(encode,SECRET_KEY,algorithm = ALGORITHM)


def authenticate_user(db, username: str, password: str):
    user = db.query(Users).filter(Users.username == username).first()
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user



async def get_current_user(token: Annotated[str, Depends(oauth2_bearer)]):
    try:
        
        payload = jwt.decode(token,SECRET_KEY,algorithms=ALGORITHM)
    
        username : str  = payload.get("sub")
        user_role : str = payload.get("role")
        user_id : int = payload.get("id")
        if user_id == None:
            raise HTTPException(status_code=401 , detail="Could not validate")
    
        return {"sub":username,"user_id":user_id,"user_role":user_role}
    except JWTError:
        raise HTTPException(status_code=401 , detail="Could not validate")
        
@router.get("/me")
async def read_current_user(user: Annotated[dict, Depends(get_current_user)]):
    return {
        "username": user["sub"],
        "user_id": user["user_id"],
        "role": user["user_role"]
    }
@router.post("/create-user")
async def create_user(create :Create_User_Request,db:db_dependency):
    new_hashed_password = get_password_hash(create.password)
    create_user_model = Users(
        username = create.username,
        email = create.email,
        first_name = create.first_name,
        last_name = create.last_name,
        phone_number = create.phone_number,
        hashed_password = new_hashed_password

    )
    db.add(create_user_model)
    db.commit()
@router.post("/token",response_model=Token)
async def login(db:db_dependency,form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    
    user = authenticate_user(db,form_data.username,form_data.password)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    token = create_access_token(user.username,user.id,user.role,timedelta(minutes=20))
    return {"access_token": token , "token_type": "bearer"}




"""
@router.post("/cookie/")
def create_cookie(db:db_dependency,form_data:Token_Data):
    user = authenticate_user(db,form_data.username,form_data.password)
    if user is None:
        raise HTTPException(status_code=400, detail="Pls Login")
    token = create_access_token(user.username,user.id,user.role,timedelta(minutes=20))
    response = JSONResponse(content={"access_token": token, "token_type": "bearer"})
    response.set_cookie(key="access_token", value=token, httponly=True,secure=False,samesite="None")
    return response

"""

    



from sqlalchemy import Column,String,Integer,Boolean,ForeignKey
from .database import Base


class Users(Base):
    __tablename__ = "users"

    id = Column(Integer,primary_key=True,index=True)
    username = Column(String,unique=True)
    email = Column(String,unique=True)
    hashed_password = Column(String)
    first_name = Column(String,nullable=True)
    last_name = Column(String,nullable=True)
    phone_number = Column(String,unique=True)
    is_active = Column(Boolean,default=True)
    deleted_account = Column(Boolean,default=False) 
    role = Column(String,nullable=True)


class Prompts(Base):
    __tablename__ = "prompts"
    id = Column(Integer,primary_key=True,index=True)
    title = Column(String)
    question = Column(String)
    deleted_prompt = Column(Boolean,default=False)
    session_id = Column(Integer,default=False)
    owner_id = Column(Integer,ForeignKey("users.id"))
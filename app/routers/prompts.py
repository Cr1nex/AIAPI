from fastapi import APIRouter,Depends,HTTPException
from typing import Annotated
from ..database import SessionLocal
from sqlalchemy.orm import Session
from pydantic import BaseModel
from .auth import get_current_user 
from ..models import Prompts
from getpass import getpass
from sre_parse import State
from langchain_core.messages import SystemMessage, trim_messages ,HumanMessage, AIMessage
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from typing import Sequence
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
import os
import streamlit as st
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass
os.environ["LANGSMITH_TRACING"] = "true"
if "LANGSMITH_API_KEY" not in os.environ:
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass(
        prompt="Enter your LangSmith API key (optional): "
    )
if "LANGSMITH_PROJECT" not in os.environ:
    os.environ["LANGSMITH_PROJECT"] = getpass.getpass(
        prompt='Enter your LangSmith Project Name (default = "default"): '
    )
    if not os.environ.get("LANGSMITH_PROJECT"):
        os.environ["LANGSMITH_PROJECT"] = "default"



if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")



model = init_chat_model("gpt-4o-mini", model_provider="openai")
trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]

trimmer.invoke(messages)


workflow = StateGraph(state_schema=State)

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You talk like a pirate. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

def call_model(state: State):
    trimmed_messages = trimmer.invoke(state["messages"])
    prompt = prompt_template.invoke(
        {"messages": trimmed_messages, "language": ["language"]}
    )
    response = model.invoke(prompt)
    return {"messages": [response]}


workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "abc678"}}


router = APIRouter(prefix="/prompts",
                   tags=["prompts"])


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
    description :str


@router.post("/create-prompt")
async def create_prompt(create:create_prompt,db:db_dependency,user:user_dependency):
    
    if user == None:
        raise HTTPException(status_code=401,detail="Not logged in")
    

    query = create.description
    language = "English"

    input_messages = messages + [HumanMessage(query)]
    output = app.invoke(
        {"messages": input_messages, "language": language},
        config,
    )
    result = output["messages"][-1]
    if result == None:
        raise HTTPException(status_code=401,detail="No prompt given")
    

    created_prompt = Prompts(title= create.title,
                             description= create.description,
                             owner_id =  user.get("user_id")               
                             )    
    db.add(created_prompt)
    db.commit()
    return result.content
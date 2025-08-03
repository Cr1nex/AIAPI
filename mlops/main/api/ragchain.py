import os
import json
import redis
import numpy as np
from typing import Any, Optional , List
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_redis import RedisConfig, RedisVectorStore
from langchain_core.retrievers import BaseRetriever
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from .llm import embeddings, llm
from .mongodb import rag_collection
from .redisdb_client import redis_client
from dataclasses import dataclass
import redis.exceptions
def load_docs():
    loader = WebBaseLoader(
        web_paths=(
            "https://lilianweng.github.io/posts/2023-06-23-agent/",
            "https://reactjs.org/docs/getting-started.html",
            "https://developer.mozilla.org/en-US/docs/Web/JavaScript",
            "https://en.wikipedia.org/wiki/Machine_learning",
            "https://towardsdatascience.com/",
            "https://realpython.com/"
        )
    )
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)
import redis.exceptions

def redis_index_exists(redis_client, index_name: str) -> bool:
 
    try:
      
        redis_client.ft(index_name).info()
        return True
    except redis.exceptions.ResponseError as e:
        
        if "Unknown Index name" in str(e):
            return False
        
        raise
    except Exception as e:
        
        print(f"An unexpected error occurred while checking Redis index: {e}")
        return False
    
def build_redis_store(redis_client):
    config = RedisConfig(
        index_name="newsgroups",
        redis_url="redis://localhost:6379/1" ,
        metadata_schema=[{"name": "category", "type": "tag"}]
    )
    print("Checking index:", config.index_name)
    if redis_index_exists(redis_client, config.index_name):
        vector_store = RedisVectorStore(embeddings, config=config, redis_client=redis_client)
        print("Redis store exists. Loading.")
    else:
        print("Creating new Redis vector store.")
        splits = load_docs()
        vector_store = RedisVectorStore.from_documents(splits, embeddings, config=config, redis_client=redis_client)

        for doc in splits:
            rag_collection.update_one(
                {"content": doc.page_content},
                {"$setOnInsert": {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }},
                upsert=True
            )

    return vector_store, embeddings


def retrieve_session_messages_redis(redis_client, user_id: int, session_id: str, top_k=5):
    key_pattern = f"chat:session:{user_id}:{session_id}:*"

    
    keys = [k.decode() for k in redis_client.scan_iter(match=key_pattern)]

    
    try:
        keys = sorted(keys, key=lambda k: int(k.rsplit(":", 1)[-1]))
    except ValueError as e:
        print(f"Error sorting keys: {e}")
        return []

    
    keys = keys[-top_k:]

    docs = []
    for key in keys:
        try:
            redis_type = redis_client.type(key.encode())
            if redis_type != b'list':
                print(f"Skipping key {key}, type is {redis_type}")
                continue

            items = redis_client.lrange(key.encode(), 0, -1)
            if not items:
                print(f"Key {key} is an empty list.")
                continue

            # Take the first (and only) item
            msg = json.loads(items[0].decode())

            question = msg.get("question", "")
            answer = msg.get("answer", "")

            docs.append(Document(
                page_content=f"[PastUserQuery]\nQuestion:\n{question}\n\n[PastSystemQuery]\nAnswer:\n{answer}"
            ))

        except Exception as e:
            print(f"Error parsing item from {key}: {e}")
            continue

    return docs

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a chatbot that answers using the provided context.Don't go out of context"
        "Do not say phrases like 'Based on the provided context' or 'According to the context.' "
        "Answer like a chatbot but don't explain if the question is out of context"
        "If the answer is not close to the context, reply like a chatbot.\n\n"
        "CONTEXT:\n{context}",
    ),
    ("human", "{input}"),
])


class CombinedRetriever(BaseRetriever):
    vector_store: Any 
    embeddings: Any 
    user: int 
    session_id: Optional[str] = None
    top_k: int = 5

    def _get_relevant_documents(self, query: str) -> List[Any]:
        vector_docs = self.vector_store.similarity_search(query, k=self.top_k)
        if self.session_id:
            past_docs = retrieve_session_messages_redis(
                redis_client, self.user, self.session_id, top_k=self.top_k
            )
            return vector_docs + past_docs
        return vector_docs

    async def _aget_relevant_documents(self, query: str) -> List[Any]:
        return self._get_relevant_documents(query)
    
async def build_chain(user, session_id):
    vector_store, embeddings = build_redis_store(redis_client)
    retriever = CombinedRetriever(vector_store=vector_store , embeddings=embeddings , user=user , session_id=session_id , top_k=5)
    combine_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=combine_chain)
    return chain, embeddings

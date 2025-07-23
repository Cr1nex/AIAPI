import json
import os
import numpy as np
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

VECTOR_DB = "faiss_index"  


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


def build_redis_store():
    
    
    config = RedisConfig(
        index_name="newsgroups",
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
        metadata_schema=[{"name": "category", "type": "tag"}]
    )
    splits = load_docs()

    redis_client.hgetall(f"newsgroups::")

    vector_store = RedisVectorStore.from_documents(splits, embeddings, config=config)
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
    key = f"chat:session:{user_id}:{session_id}"
    raw_pairs = redis_client.lrange(key, -top_k, -1)
    docs = []
    for p in raw_pairs:
        try:
            pair = json.loads(p)
            question = pair["question"]
            answer = pair["answer"]
            docs.append(Document(
                page_content=f"[PastUserQuery]\nQuestion:\n{question}\n\n[PastSystemQuery]\nAnswer:\n{answer}"
            ))
        except Exception:
            continue
    return docs


prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a chatbot that answers using the provided context. "
        "Do not say phrases like 'Based on the provided context' or 'According to the context.' "
        "Answer like a chatbot try not to use the data you learned through training."
        "If the answer is not close to the context, reply like a chatbot.\n\n"
        "CONTEXT:\n{context}",
    ),
    ("human", "{input}"),
])


from typing import Any, Optional

class CombinedRetriever(BaseRetriever):
    vector_store: Any
    embeddings: Any
    user: Any
    session_id: Optional[str] = None
    top_k: int = 5

    def __init__(self, vector_store, embeddings, user, session_id=None, top_k=5):
        super().__init__(
            vector_store=vector_store,
            embeddings=embeddings,
            user=user,
            session_id=session_id,
            top_k=top_k,
        )

    def get_relevant_documents(self, query):
        vector_docs = self.vector_store.similarity_search(query, k=self.top_k)
        if self.session_id:
            past_docs = retrieve_session_messages_redis(
                redis_client, self.user, self.session_id, top_k=self.top_k
            )
            return vector_docs + past_docs
        return vector_docs

    async def aget_relevant_documents(self, query):
        return self.get_relevant_documents(query)

async def build_chain(user):
    vector_store, embeddings = build_redis_store()
    retriever = CombinedRetriever(vector_store, embeddings, user, top_k=5)
    combine_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=combine_chain)
    return chain, embeddings

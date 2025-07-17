import os
from fastapi import Depends
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever
from sqlalchemy.orm import Session

from .llm import embeddings, llm
from .mongodb import queries_collection, rag_collection
from .database import SessionLocal

from typing import Any, List

VECTOR_DB = "faiss_index"


def load_docs():
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",
            "https://reactjs.org/docs/getting-started.html",
            "https://developer.mozilla.org/en-US/docs/Web/JavaScript",
            "https://en.wikipedia.org/wiki/Machine_learning",
            "https://towardsdatascience.com/",  
            "https://realpython.com/"  )
        
    )
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)


def build_faiss_store():
    try:
        index_file_path = os.path.join(VECTOR_DB, "index.faiss")
        if os.path.exists(index_file_path):
            vector_store = FAISS.load_local(
                VECTOR_DB,
                embeddings,
                allow_dangerous_deserialization=True
            )
            return vector_store, embeddings
        else:
            raise FileNotFoundError("FAISS index file not found.")
    except Exception as e:
        print(f"[!] Failed to load FAISS vector store: {e}")
        print("[!] Rebuilding FAISS vector store from scratch...")

    splits = load_docs()
    vector_store = FAISS.from_documents(splits, embeddings)

    os.makedirs(VECTOR_DB, exist_ok=True)
    vector_store.save_local(VECTOR_DB)

    

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


def retrieve_past_queries(question: str, embeddings, db, user: int, top_k: int = 5):
    query_emb = np.array(embeddings.embed_query(question))
    
    past_queries = list(queries_collection.find({"embedding": {"$exists": True}, "user_id": user}))
    scored = []
    for q in past_queries:
        
        sql_id = q.get("sql_prompt_id")
        scored.append((sql_id, q))

    top = sorted(scored, key=lambda x: x[0], reverse=True)[:top_k]
    
    """
        def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

        scored = []
        for q in past_queries:   #Find another way
            emb = q.get("embedding")
            sql_id = q.get("sql_prompt_id")

            if not emb or sql_id is None:
                continue

            
            

            score = cosine_sim(query_emb, np.array(emb))
            scored.append((score, q))

        if not scored:
            return []

        top = sorted(scored, key=lambda x: x[0], reverse=True)[:top_k]
    """
    return [
        Document(f"[PastUserQuery]\nQuestion:\n{doc.get('question', '').strip()}\n\n"
            f"[PastSystemQuery]\nAnswer:\n{doc.get('answer', '').strip()}")
        for _, doc in top
    ]


prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are an assistant that answers using the provided CONTEXT. "
     "Do not say phrases like 'Based on the provided context' or 'According to the context.' "
     "Answer concisely and directly. "
     "If the answer is not close to the context, reply: 'I don’t have enough information to answer that.'\n\n"
     "CONTEXT:\n{context}"
    ),
    
    ("human", "{input}")
])



class CombinedRetriever(BaseRetriever):
    vector_store: Any
    embeddings: Any
    db: Any
    user: Any
    top_k: int = 5

    def __init__(self, vector_store, embeddings, db, user, top_k=5):
        super().__init__(
            vector_store=vector_store,
            embeddings=embeddings,
            db=db,
            user=user,
            top_k=top_k
        )

    def get_relevant_documents(self, query):
        faiss_docs = self.vector_store.similarity_search(query, k=self.top_k)
        past_query_docs = retrieve_past_queries(query, self.embeddings, self.db, self.user, top_k=self.top_k)
        return faiss_docs + past_query_docs

    async def aget_relevant_documents(self, query):
        return self.get_relevant_documents(query)


async def build_chain(user, db):
    vector_store, embeddings = build_faiss_store()
    retriever = CombinedRetriever(
        vector_store=vector_store,
        embeddings=embeddings,
        db=db,
        user=user,
        top_k=5
    )

    combine_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=combine_chain)

    return chain, embeddings

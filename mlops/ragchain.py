import os
import pickle
import faiss
import numpy as np
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever

from bs4 import SoupStrainer
from .llm import embeddings, llm
from .mongodb import queries_collection, rag_collection

VECTOR_DB_PATH = "faiss_index"

def load_docs():
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=SoupStrainer(class_=("post-content", "post-title", "post-header"))
        ),
    )
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)



def build_faiss_store():
    if os.path.exists(VECTOR_DB_PATH):
        with open(os.path.join(VECTOR_DB_PATH, "faiss_store.pkl"), "rb") as f:
            vector_store = pickle.load(f)
        vector_store.index = faiss.read_index(os.path.join(VECTOR_DB_PATH, "faiss.index"))
    else:
        splits = load_docs()
        vector_store = FAISS.from_documents(splits, embeddings)
        os.makedirs(VECTOR_DB_PATH, exist_ok=True)
        faiss.write_index(vector_store.index, os.path.join(VECTOR_DB_PATH, "faiss.index"))
        with open(os.path.join(VECTOR_DB_PATH, "faiss_store.pkl"), "wb") as f:
            pickle.dump(vector_store, f)

        
        for doc in splits:
            rag_collection.update_one(
                {"content": doc.page_content},
                {"$setOnInsert": {"content": doc.page_content, "metadata": doc.metadata}},
                upsert=True,
            )

    return vector_store, embeddings


# past queries from MongoDB (cosine similarity)
def retrieve_past_queries(question, embeddings, top_k=3):
    query_emb = embeddings.embed_query(question)
    past_queries = list(queries_collection.find({"embedding": {"$exists": True}}))

    def cosine_sim(a, b):
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    scored = []
    for q in past_queries:
        emb = q.get("embedding")
        if emb is None:
            continue
        score = cosine_sim(query_emb, emb)
        scored.append((score, q))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]
    return [Document(page_content=f"{doc['question']}\n{doc['answer']}") for _, doc in top]


# 4. Prompt setup
prompt = ChatPromptTemplate.from_messages([
    ("system", "Use the following context to answer the question:\n\n{context}"),
    ("human", "{input}")
])


# 5. Final chain builder
def build_chain():
    vector_store, embeddings = build_faiss_store()

    def combined_retriever(query: str):
        faiss_docs = vector_store.similarity_search(query, k=3)
        past_query_docs = retrieve_past_queries(query, embeddings, top_k=3)
        if past_query_docs:
            return faiss_docs + past_query_docs
        else:
            return faiss_docs

    class CombinedRetriever(BaseRetriever):
        def get_relevant_documents(self, query: str):
            return combined_retriever(query)

        async def aget_relevant_documents(self, query: str):
            return combined_retriever(query)

    retriever = CombinedRetriever()
    combine_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=combine_chain)

    return chain, embeddings

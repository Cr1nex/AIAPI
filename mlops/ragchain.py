import os
import pickle
import shutil
import tempfile
import faiss
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


FAISS_FILE_PATH = os.path.join(VECTOR_DB_PATH, "faiss_meta.pkl")


def safe_pickle_load(filepath):
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        with open(filepath, "rb") as f:
            return pickle.load(f)
    return {}


def atomic_pickle_dump(obj, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fd, temp_path = tempfile.mkstemp()
    with os.fdopen(fd, "wb") as tmp:
        pickle.dump(obj, tmp)
    shutil.move(temp_path, filepath)


def build_faiss_store():
    try:
        index_file_path = os.path.join(VECTOR_DB_PATH, "index.faiss")
        if os.path.exists(index_file_path):
            vector_store = FAISS.load_local(
                VECTOR_DB_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
            #meta = safe_pickle_load(FAISS_FILE_PATH )
            return vector_store, embeddings
        else:
            raise FileNotFoundError("FAISS index file not found.")
    except Exception as e:
        print(f"[!] Failed to load FAISS vector store: {e}")
        print("[!] Rebuilding FAISS vector store from scratch...")

    
    splits = load_docs()
    vector_store = FAISS.from_documents(splits, embeddings)

    
    os.makedirs(VECTOR_DB_PATH, exist_ok=True)

    
    vector_store.save_local(VECTOR_DB_PATH)

    
    atomic_pickle_dump({}, FAISS_FILE_PATH )

  
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
    scored.sort(key=lambda x: x[0], reverse=True) #Improve later on too many past context might lead to failure
    top = scored[:top_k]
    return [Document(page_content=f"{doc['question']}\n{doc['answer']}") for _, doc in top]


prompt = ChatPromptTemplate.from_messages([
    ("system", "Use the following context to answer the question:\n\n{context}"),
    ("human", "{input}")
])


def build_chain():
    vector_store, embeddings = build_faiss_store()

    def combined_retriever(query: str):
        faiss_docs = vector_store.similarity_search(query, k=3)
        past_query_docs = retrieve_past_queries(query, embeddings, top_k=3)
        return faiss_docs + past_query_docs

    class CombinedRetriever(BaseRetriever):
        def get_relevant_documents(self, query: str):
            return combined_retriever(query)

        async def aget_relevant_documents(self, query: str):
            return combined_retriever(query)

    retriever = CombinedRetriever()
    combine_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=combine_chain)

    return chain, embeddings

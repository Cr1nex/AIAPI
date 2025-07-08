from pymongo import MongoClient
from dotenv import load_dotenv
import os
load_dotenv()
MONGODB_DATABASE_URL = os.getenv('MONGODB_DATABASE_URL')
uri = MONGODB_DATABASE_URL
client = MongoClient(uri)
rag_collection = client.mcp.embeddings
queries_collection = client.mcp.queries
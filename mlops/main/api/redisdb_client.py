import os
import redis
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_redis import RedisConfig, RedisVectorStore
from .llm import embeddings
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
print(f"Connecting to Redis at: {REDIS_URL}")
redis_client = redis.from_url(REDIS_URL)
print(redis_client.ping())


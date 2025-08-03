import os
import redis


REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
print(f"Connecting to Redis at: {REDIS_URL}")
redis_client = redis.from_url(REDIS_URL)
print(redis_client.ping())


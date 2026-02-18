import os
import sys
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")

_client = None
_db = None


async def connect_to_database():
    global _client, _db

    try:
        _client = AsyncIOMotorClient(MONGO_URI)
        _db = _client[DB_NAME]

        # Force a connection check
        await _client.admin.command("ping")

        print("Connected to MongoDB")
        print(f"Connected to database: {DB_NAME}")

    except Exception as error:
        print("MongoDB connection error:", str(error))
        sys.exit(1)


def get_db():
    if _db is None:
        raise RuntimeError("Database not initialized. Call connect_to_database first.")
    return _db

"""
Database connection module for MongoDB.
Uses motor (async MongoDB driver) for FastAPI compatibility.
"""
import os
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

# Load environment variables from .env file
load_dotenv()

# MongoDB connection settings from environment
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "deepfake_detector")

# Global client and database references
client = None
db = None


async def connect_to_mongo():
    """Connect to MongoDB on startup."""
    global client, db
    print(f"Connecting to MongoDB...")
    client = AsyncIOMotorClient(MONGODB_URI)
    db = client[DATABASE_NAME]
    
    # Create indexes for faster lookups
    await db.users.create_index("email", unique=True)
    await db.users.create_index("api_key", unique=True)
    
    print(f"MongoDB connected successfully to database: {DATABASE_NAME}")


async def close_mongo_connection():
    """Close MongoDB connection on shutdown."""
    global client
    if client:
        client.close()
        print("MongoDB connection closed.")


def get_database():
    """Get database instance."""
    return db

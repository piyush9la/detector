"""
Authentication module for API key management.
Handles password hashing, API key generation (hashed), rate limiting, and validation.
"""
import secrets
import hashlib
import bcrypt
from datetime import datetime, date
from fastapi import Header, HTTPException, Depends
from pydantic import BaseModel, EmailStr


# --- Configuration ---
DEFAULT_RATE_LIMIT = 100  # Requests per day


# --- Pydantic Models ---

class UserSignup(BaseModel):
    """Request body for signup."""
    email: EmailStr
    password: str


class UserLogin(BaseModel):
    """Request body for login."""
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    """Response after signup/login."""
    email: str
    api_key: str
    message: str


class UsageResponse(BaseModel):
    """Response for usage endpoint."""
    email: str
    requests_today: int
    rate_limit: int
    remaining: int
    total_requests: int


# --- Helper Functions ---

def hash_password(password: str) -> str:
    """Hash password using bcrypt."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')


def verify_password(password: str, hashed: str) -> bool:
    """Verify password against hash."""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))


def generate_api_key() -> tuple[str, str, str]:
    """
    Generate a unique API key in OpenAI style: sk-live-xxxx.
    Returns: (raw_key, key_hash, key_prefix)
    - raw_key: The full key to show user ONCE
    - key_hash: SHA-256 hash to store in DB
    - key_prefix: First 12 chars for display (sk-live-abc1...)
    """
    random_part = secrets.token_hex(24)  # 48 character hex string
    raw_key = f"sk-live-{random_part}"
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    key_prefix = raw_key[:16] + "..."  # e.g., "sk-live-abc123..."
    return raw_key, key_hash, key_prefix


def hash_api_key(api_key: str) -> str:
    """Hash an API key using SHA-256."""
    return hashlib.sha256(api_key.encode()).hexdigest()


def create_user_document(email: str, password: str) -> tuple[dict, str]:
    """
    Create a new user document for MongoDB.
    Returns: (user_doc, raw_api_key)
    """
    raw_key, key_hash, key_prefix = generate_api_key()
    
    user_doc = {
        "email": email,
        "password_hash": hash_password(password),
        "api_key_hash": key_hash,
        "api_key_prefix": key_prefix,
        "requests_today": 0,
        "last_request_date": str(date.today()),
        "total_requests": 0,
        "rate_limit": DEFAULT_RATE_LIMIT,
        "created_at": datetime.utcnow(),
        "last_login": None
    }
    
    return user_doc, raw_key


async def check_rate_limit(user: dict, db) -> bool:
    """
    Check and update rate limit for user.
    Returns True if within limit, raises HTTPException if exceeded.
    """
    today = str(date.today())
    
    # Reset counter if new day
    if user.get("last_request_date") != today:
        await db.users.update_one(
            {"_id": user["_id"]},
            {"$set": {"requests_today": 0, "last_request_date": today}}
        )
        user["requests_today"] = 0
    
    # Check if exceeded
    rate_limit = user.get("rate_limit", DEFAULT_RATE_LIMIT)
    if user.get("requests_today", 0) >= rate_limit:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Limit: {rate_limit} requests/day. Try again tomorrow."
        )
    
    # Increment counters
    await db.users.update_one(
        {"_id": user["_id"]},
        {
            "$inc": {"requests_today": 1, "total_requests": 1},
            "$set": {"last_request_date": today}
        }
    )
    
    return True


# --- API Key Validation Dependency ---

async def validate_api_key(x_api_key: str = Header(..., description="Your API key")):
    """
    FastAPI dependency to validate API key.
    Use this to protect endpoints.
    """
    from .database import get_database
    
    db = get_database()
    if db is None:
        raise HTTPException(status_code=500, detail="Database not connected")
    
    # Hash the incoming key and search
    incoming_hash = hash_api_key(x_api_key)
    user = await db.users.find_one({"api_key_hash": incoming_hash})
    
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key. Please login to get your API key."
        )
    
    # Check rate limit
    await check_rate_limit(user, db)
    
    return user

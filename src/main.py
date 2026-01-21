import os
import shutil
import tempfile
import uvicorn
import warnings
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from mtcnn.mtcnn import MTCNN
import tensorflow as tf

# --- Suppress TensorFlow & MTCNN Warnings ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore')
# Trigger reload for audio client

# --- Imports for config and prediction functions ---
try:
    from . import config
    from .predict import get_image_prediction
    from .predict_video_model import get_video_prediction
    from .video_model import build_video_model
    from .database import connect_to_mongo, close_mongo_connection, get_database
    from .auth import (
        UserSignup, UserLogin, UserResponse, UsageResponse,
        hash_password, verify_password, generate_api_key, create_user_document,
        validate_api_key, hash_api_key
    )
except ImportError:
    # This fallback lets us run the file directly if needed
    import config
    from predict import get_image_prediction
    from predict_video_model import get_video_prediction
    from database import connect_to_mongo, close_mongo_connection, get_database
    from auth import (
        UserSignup, UserLogin, UserResponse, UsageResponse,
        hash_password, verify_password, generate_api_key, create_user_document,
        validate_api_key, hash_api_key
    )

# --- 1. Create the FastAPI app ---
app = FastAPI(
    title="Deepfake Detector API",
    description="An API to detect deepfake images and videos using advanced ML models.",
    version="1.0.0"
)

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- 2. Load Models at Startup (Best Practice) ---
# This dictionary will hold our models, loaded ONCE.
# This is far more efficient than loading them for every request.
models = {}

@app.on_event("startup")
async def startup_event():
    """
    Startup: Connect to MongoDB and load ML models.
    """
    # Connect to MongoDB first
    await connect_to_mongo()
    
    # Then load ML models
    await load_models()


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown: Close MongoDB connection."""
    await close_mongo_connection()


async def load_models():
    """
    Load all ML models into memory when the API server starts.
    """
    print("--- Loading models into memory... ---")
    
    # Load Image Model (baseline_model.h5)
    image_model_path = os.path.join(config.MODEL_DIR, "baseline_model.h5")
    if os.path.exists(image_model_path):
        models["image_model"] = tf.keras.models.load_model(image_model_path, compile=False)
        print("Image model (baseline_model.h5) loaded successfully.")
    else:
        print(f"WARNING: Image model not found at {image_model_path}")


    video_model_path = os.path.join(config.MODEL_DIR, "video_model_v2.keras")
    if os.path.exists(video_model_path):
        try:
            # 1. Build the "empty" model architecture from your code
            print("Building video model architecture from video_model.py...")
            video_model = build_video_model()
            
            # 2. Load *only the weights* from your saved file
            print(f"Loading weights from {video_model_path}...")
            video_model.load_weights(video_model_path)
            
            # 3. Assign the working model
            models["video_model"] = video_model
            print("Video model (video_model_v2.keras) loaded successfully from weights.")
            
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to load video model from weights: {e}")
            print("This might be due to a mismatch between video_model.py and your saved file.")
    else:
        print(f"WARNING: Video model not found at {video_model_path}")
    # --- END: ULTIMATE FIX FOR VIDEO MODEL ---
    # Load MTCNN Detector
    models["mtcnn_detector"] = MTCNN()
    print("MTCNN detector initialized.")

    # --- Load Audio Model (Local HuggingFace) ---
    try:
        from transformers import pipeline
        print("Loading Audio Model (deepfake_audio.h5)...")
        models["audio_pipeline"] = pipeline("audio-classification", model="motheecreator/Deepfake-audio-detection")
        print("Audio Model loaded successfully.")
    except Exception as e:
        print(f"WARNING: Failed to load Audio Model: {e}")

    print("--- All models loaded. API is ready. ---")
    
# --- 3. Define API Endpoints ---

@app.get("/")
def read_root():
    """A simple 'health check' endpoint to see if the server is running."""
    return {"status": "Deepfake Detector API is online and running."}


# --- 4. Authentication Endpoints ---

@app.post("/signup", response_model=UserResponse)
async def signup(user: UserSignup):
    """
    Create a new user account and get your API key.
    ⚠️ IMPORTANT: Save your API key! It will only be shown ONCE.
    """
    db = get_database()
    
    # Check if email already exists
    existing = await db.users.find_one({"email": user.email})
    if existing:
        raise HTTPException(
            status_code=400,
            detail="Email already registered. Please login to get your API key."
        )
    
    # Create new user (returns user_doc and raw_api_key)
    user_doc, raw_api_key = create_user_document(user.email, user.password)
    await db.users.insert_one(user_doc)
    
    return UserResponse(
        email=user.email,
        api_key=raw_api_key,
        message="Account created! ⚠️ SAVE YOUR API KEY NOW - it will NOT be shown again!"
    )


@app.post("/login", response_model=UserResponse)
async def login(user: UserLogin):
    """
    Login to view your API key prefix.
    Note: For security, full key is only shown at signup.
    Use /regenerate-key to get a new key if lost.
    """
    db = get_database()
    
    # Find user
    existing = await db.users.find_one({"email": user.email})
    if not existing:
        raise HTTPException(status_code=404, detail="User not found. Please signup first.")
    
    # Verify password
    if not verify_password(user.password, existing["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid password.")
    
    # Update last login
    from datetime import datetime
    await db.users.update_one(
        {"email": user.email},
        {"$set": {"last_login": datetime.utcnow()}}
    )
    
    return UserResponse(
        email=user.email,
        api_key=existing.get("api_key_prefix", "Key hidden for security"),
        message="Login successful. Use /regenerate-key if you need a new API key."
    )


@app.post("/regenerate-key", response_model=UserResponse)
async def regenerate_key(user: UserLogin):
    """
    Generate a new API key. The old key will stop working.
    ⚠️ IMPORTANT: Save your new API key! It will only be shown ONCE.
    """
    db = get_database()
    
    # Find user
    existing = await db.users.find_one({"email": user.email})
    if not existing:
        raise HTTPException(status_code=404, detail="User not found.")
    
    # Verify password
    if not verify_password(user.password, existing["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid password.")
    
    # Generate new key (returns tuple)
    raw_key, key_hash, key_prefix = generate_api_key()
    await db.users.update_one(
        {"email": user.email},
        {"$set": {"api_key_hash": key_hash, "api_key_prefix": key_prefix}}
    )
    
    return UserResponse(
        email=user.email,
        api_key=raw_key,
        message="New API key generated! ⚠️ SAVE IT NOW - old key is now invalid!"
    )


@app.get("/usage", response_model=UsageResponse)
async def get_usage(user: dict = Depends(validate_api_key)):
    """
    Check your API usage and remaining quota.
    Requires x-api-key header.
    """
    rate_limit = user.get("rate_limit", 100)
    requests_today = user.get("requests_today", 0)
    
    return UsageResponse(
        email=user["email"],
        requests_today=requests_today,
        rate_limit=rate_limit,
        remaining=max(0, rate_limit - requests_today),
        total_requests=user.get("total_requests", 0)
    )


# --- 5. Prediction Endpoints (Protected) ---

@app.post("/predict_image")
async def predict_image_api(
    file: UploadFile = File(...),
    user: dict = Depends(validate_api_key)
):
    """
    Endpoint for predicting a single deepfake image.
    Requires API key in x-api-key header.
    """
    if "image_model" not in models:
        raise HTTPException(status_code=500, detail="Image model is not loaded.")
    
    # We must save the uploaded file to a temporary path
    # because our prediction function expects a file path.
    temp_file_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
        
        print(f"Processing image: {temp_file_path}")
        
        # Call our prediction function and pass it the pre-loaded model
        result = get_image_prediction(
            image_path=temp_file_path,
            model=models["image_model"]
        )
        return result
        
    except Exception as e:
        # If anything goes wrong, return an error
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    finally:
        # CRITICAL: Always clean up the temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/predict_video")
async def predict_video_api(
    file: UploadFile = File(...),
    user: dict = Depends(validate_api_key)
):
    """
    Endpoint for predicting a single deepfake video.
    Requires API key in x-api-key header.
    """
    if "video_model" not in models or "mtcnn_detector" not in models:
        raise HTTPException(status_code=500, detail="Video models are not loaded.")

    temp_file_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
            
        print(f"Processing video: {temp_file_path}")

        # Call the video prediction function
        result = get_video_prediction(
            video_path=temp_file_path,
            video_model=models["video_model"],
            detector=models["mtcnn_detector"]
        )
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    finally:
        # CRITICAL: Always clean up the temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/predict_audio")
async def predict_audio_api(
    file: UploadFile = File(...),
    user: dict = Depends(validate_api_key)
):
    """
    Endpoint for predicting a single deepfake audio.
    Requires API key in x-api-key header.
    """
    if "audio_pipeline" not in models:
        # Try to reload if missing
        try:
            from transformers import pipeline
            models["audio_pipeline"] = pipeline("audio-classification", model="motheecreator/Deepfake-audio-detection")
        except:
            raise HTTPException(status_code=500, detail="Audio model is not loaded.")

    temp_file_path = ""
    try:
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name

        print(f"Processing audio: {temp_file_path}")
        
        # Make prediction using local model
        audio_pipeline = models["audio_pipeline"]
        result = audio_pipeline(temp_file_path)[0]
        
        label = result['label'].upper()  # 'fake' or 'real'
        score = result['score'] * 100  # Convert to percentage
        
        print(f"Audio Model Result: {label} ({score:.2f}%)")
        
        # Map label to our format
        prediction = "FAKE" if "FAKE" in label else "REAL"

        return {
            "prediction": prediction,
            "confidence": round(score, 2),
            "raw": f"{label} ({score:.2f}%)"
        }

    except Exception as e:
        print(f"Audio prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    finally:
        # CRITICAL: Always clean up the temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# --- 4. How to run this file for development ---
if __name__ == "__main__":
    print("--- Starting FastAPI server directly (for development) ---")
    print("--- Go to http://127.0.0.1:8000 for the API ---")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True, app_dir="src")
import streamlit as st
import requests
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🕵️",
    layout="wide"
)

# --- Define API URL ---
# For local: http://127.0.0.1:8000
# For production: Update to your Railway/Render URL
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# --- Session State for API Key ---
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_email" not in st.session_state:
    st.session_state.user_email = ""


# --- Helper function to make authenticated requests ---
def make_api_request(endpoint, files=None, timeout=30):
    """Make authenticated API request with API key header."""
    headers = {"x-api-key": st.session_state.api_key}
    try:
        response = requests.post(
            f"{API_URL}/{endpoint}",
            files=files,
            headers=headers,
            timeout=timeout
        )
        return response
    except requests.exceptions.RequestException as e:
        return None


# --- Sidebar: Authentication ---
with st.sidebar:
    st.header("🔐 Authentication")
    
    if not st.session_state.logged_in:
        tab1, tab2 = st.tabs(["Login", "Signup"])
        
        with tab1:
            st.subheader("Login")
            login_email = st.text_input("Email", key="login_email")
            login_password = st.text_input("Password", type="password", key="login_pass")
            
            if st.button("Login", key="login_btn"):
                with st.spinner("Logging in..."):
                    try:
                        response = requests.post(
                            f"{API_URL}/login",
                            json={"email": login_email, "password": login_password},
                            timeout=10
                        )
                        if response.status_code == 200:
                            result = response.json()
                            st.success("Login successful!")
                            st.info("⚠️ Your API key prefix: " + result["api_key"])
                            st.warning("Enter your full API key below if you have it saved.")
                            st.session_state.user_email = login_email
                        else:
                            st.error(response.json().get("detail", "Login failed"))
                    except Exception as e:
                        st.error(f"Connection error: {e}")
        
        with tab2:
            st.subheader("Signup")
            signup_email = st.text_input("Email", key="signup_email")
            signup_password = st.text_input("Password", type="password", key="signup_pass")
            
            if st.button("Create Account", key="signup_btn"):
                with st.spinner("Creating account..."):
                    try:
                        response = requests.post(
                            f"{API_URL}/signup",
                            json={"email": signup_email, "password": signup_password},
                            timeout=10
                        )
                        if response.status_code == 200:
                            result = response.json()
                            st.success("Account created!")
                            st.warning("⚠️ SAVE YOUR API KEY NOW! It will NOT be shown again:")
                            st.code(result["api_key"], language=None)
                            st.session_state.api_key = result["api_key"]
                            st.session_state.logged_in = True
                            st.session_state.user_email = signup_email
                            st.rerun()
                        else:
                            st.error(response.json().get("detail", "Signup failed"))
                    except Exception as e:
                        st.error(f"Connection error: {e}")
        
        st.divider()
        st.subheader("Enter API Key")
        api_key_input = st.text_input("API Key (sk-live-...)", type="password", key="api_key_input")
        if st.button("Use API Key"):
            if api_key_input.startswith("sk-live-"):
                st.session_state.api_key = api_key_input
                st.session_state.logged_in = True
                st.success("API key saved!")
                st.rerun()
            else:
                st.error("Invalid API key format. Should start with 'sk-live-'")
    
    else:
        st.success(f"✅ Logged in as: {st.session_state.user_email or 'User'}")
        st.text(f"Key: {st.session_state.api_key[:20]}...")
        
        # Check usage
        if st.button("Check Usage"):
            headers = {"x-api-key": st.session_state.api_key}
            try:
                response = requests.get(f"{API_URL}/usage", headers=headers, timeout=10)
                if response.status_code == 200:
                    usage = response.json()
                    st.metric("Requests Today", usage["requests_today"])
                    st.metric("Remaining", usage["remaining"])
                    st.metric("Daily Limit", usage["rate_limit"])
                else:
                    st.error("Could not fetch usage")
            except:
                st.error("Connection error")
        
        if st.button("Logout"):
            st.session_state.api_key = ""
            st.session_state.logged_in = False
            st.session_state.user_email = ""
            st.rerun()


# --- Main App ---
st.title("Deepfake Detector 🕵️")
st.write("Welcome! This app uses advanced AI models to detect deepfakes in images, videos, and audio.")

# Check if logged in
if not st.session_state.logged_in:
    st.warning("⚠️ Please login or enter your API key in the sidebar to use the detector.")
    st.stop()

# --- Create three columns for the detectors ---
col1, col2, col3 = st.columns(3)

# --- Column 1: Image Detector ---
with col1:
    st.header("🖼️ Image Detector")
    image_file = st.file_uploader("Upload an Image", type=['jpg', 'png', 'jpeg'], key="image")

    if image_file:
        st.image(image_file, caption="Uploaded Image.", use_container_width=True)
        
        if st.button("Detect Image"):
            files = {"file": (image_file.name, image_file, image_file.type)}
            
            with st.spinner("Analyzing image... Please wait."):
                response = make_api_request("predict_image", files=files, timeout=30)
                
                if response is None:
                    st.error("Could not connect to the backend API")
                elif response.status_code == 200:
                    result = response.json()
                    confidence = result.get('confidence', 0.0)
                    
                    if result['prediction'] == 'FAKE':
                        st.error(f"**Prediction: FAKE** (Confidence: {confidence:.2f}%)")
                    else:
                        st.success(f"**Prediction: REAL** (Confidence: {confidence:.2f}%)")
                elif response.status_code == 429:
                    st.error("⚠️ Rate limit exceeded! Try again tomorrow.")
                elif response.status_code == 401:
                    st.error("❌ Invalid API key. Please check your key.")
                else:
                    st.error(f"Error from API: {response.text}")

# --- Column 2: Video Detector ---
with col2:
    st.header("🎬 Video Detector")
    video_file = st.file_uploader("Upload a Video", type=['mp4', 'mov', 'avi'], key="video")

    if video_file:
        st.video(video_file)
        
        if st.button("Detect Video"):
            files = {"file": (video_file.name, video_file, video_file.type)}
            
            with st.spinner("Analyzing video... This may take a minute or two."):
                response = make_api_request("predict_video", files=files, timeout=300)
                
                if response is None:
                    st.error("Could not connect to the backend API")
                elif response.status_code == 200:
                    result = response.json()
                    confidence = result.get('confidence', 0.0)
                    
                    if result['prediction'] == 'FAKE':
                        st.error(f"**Prediction: FAKE** (Confidence: {confidence:.2f}%)")
                    elif result['prediction'] == 'REAL':
                        st.success(f"**Prediction: REAL** (Confidence: {confidence:.2f}%)")
                    else:
                        st.warning(f"Prediction Error: {result.get('detail', 'Unknown error')}")
                elif response.status_code == 429:
                    st.error("⚠️ Rate limit exceeded! Try again tomorrow.")
                elif response.status_code == 401:
                    st.error("❌ Invalid API key. Please check your key.")
                else:
                    st.error(f"Error from API: {response.text}")

# --- Column 3: Audio Detector ---
with col3:
    st.header("🎵 Audio Detector")
    audio_file = st.file_uploader("Upload Audio", type=['wav', 'mp3', 'flac', 'ogg', 'm4a'], key="audio")

    if audio_file:
        st.audio(audio_file)
        
        if st.button("Detect Audio"):
            files = {"file": (audio_file.name, audio_file, audio_file.type)}
            
            with st.spinner("Analyzing audio..."):
                response = make_api_request("predict_audio", files=files, timeout=300)
                
                if response is None:
                    st.error("Could not connect to the backend API")
                elif response.status_code == 200:
                    result = response.json()
                    confidence = result.get('confidence', 0.0)
                    prediction = result.get('prediction', 'UNKNOWN')
                    
                    if prediction == 'FAKE':
                        st.error(f"**Prediction: FAKE** (Confidence: {confidence:.2f}%)")
                    elif prediction == 'REAL':
                        st.success(f"**Prediction: REAL** (Confidence: {confidence:.2f}%)")
                    else:
                        st.warning(f"Prediction: {prediction}")
                elif response.status_code == 429:
                    st.error("⚠️ Rate limit exceeded! Try again tomorrow.")
                elif response.status_code == 401:
                    st.error("❌ Invalid API key. Please check your key.")
                else:
                    st.error(f"Error from API: {response.text}")


# --- Footer ---
st.divider()
st.caption("Powered by FastAPI + TensorFlow + HuggingFace Transformers")
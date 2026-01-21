# Start from a standard, slim Python 3.10 image
FROM python:3.10-slim

# Install system dependencies for OpenCV and audio processing
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker's cache
COPY requirements.txt .

# Install all of your Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port (Render uses PORT env variable)
EXPOSE 8000

# Run FastAPI with uvicorn - use PORT from environment variable
CMD uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8000}
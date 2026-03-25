# Dockerfile for Render Deployment
# Based on python:3.11-slim for a lean image
FROM python:3.11-slim

# Set environment variables for non-interactive installs
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
# Tesseract OCR and its Spanish training data are required for ocr.py
# libgl1-mesa-glx and libglib2.0-0 are required by OpenCV (cv2)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-spa \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements files first to leverage Docker layer caching
COPY Local/requirements_base.txt .
COPY Local/requirements_linux.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_linux.txt

# Copy the core application files (Respecting .dockerignore)
# This will exclude tests, datasets, and local sqlite databases.
COPY Local/ .

# Production configuration
# Render will mount a persistent disk at /var/lib/naftaec
ENV PERSISTENT_STORAGE_PATH=/var/lib/naftaec
ENV PYTHONUNBUFFERED=1
ENV PORT=5001

# Expose the port used by the Flask app
EXPOSE 5001

# Run the app with Gunicorn. 
# We use 1 worker and 8 threads to ensure thread-safety for SQLite in WAL mode.
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "1", "--threads", "8", "--timeout", "120", "app:app"]

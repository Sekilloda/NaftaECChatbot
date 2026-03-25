# Refined Dockerfile for Low-RAM Render Deployment
FROM python:3.11-slim

# Set environment variables for non-interactive installs
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies: Tesseract OCR (Spanish) and OpenCV dependencies
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

# Install Python dependencies (Reduced set for Low-RAM)
RUN pip install --no-cache-dir -r requirements_linux.txt

# Copy the application files (Respecting .dockerignore)
COPY Local/ .

# Production configuration
ENV PERSISTENT_STORAGE_PATH=/var/lib/naftaec
ENV PYTHONUNBUFFERED=1
ENV PORT=5001

# Expose port
EXPOSE 5001

# Start using Gunicorn (Single worker for SQLite safety)
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "1", "--threads", "8", "--timeout", "120", "app:app"]

#!/bin/bash
# Robust Linux Deployment Script for NaftaEC Chatbot
set -e

echo "🐧 Linux Deployment Setup for NaftaEC Chatbot"

# 1. Environment Check
if [ ! -f ".env" ]; then
    echo "⚠️  WARNING: .env file not found!"
    echo "Please create a .env file with your API keys before running."
    exit 1
fi

# 2. Required Directories
echo "📁 Creating necessary directories..."
mkdir -p media reportes_descargados logs

# 3. Detect Distro and Install Dependencies
echo "📦 Checking system dependencies..."
if [ -f /etc/debian_version ]; then
    sudo apt-get update
    sudo apt-get install -y tesseract-ocr tesseract-ocr-spa python3-venv python3-pip
elif [ -f /etc/fedora-release ]; then
    sudo dnf install -y tesseract tesseract-langpack-spa python3 python3-pip
else
    echo "⚠️  Unknown Linux distribution. Please ensure Tesseract and Python 3.11+ are installed."
fi

# 4. Setup Virtual Environment
if [ ! -d "venv" ] && [ ! -d ".venv" ]; then
    echo "🐍 Creating virtual environment..."
    python3 -m venv .venv
fi

echo "🔄 Activating virtual environment and installing packages..."
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    source venv/bin/activate
fi

pip install --upgrade pip
pip install -r requirements_linux.txt

# 5. Start the Server
echo "🚀 Starting server (Logs will be output to console)..."
echo "💡 Tip: To run in the background, use: nohup ./run_linux.sh > logs/chatbot.log 2>&1 &"
python start.py

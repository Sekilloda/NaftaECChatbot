#!/bin/bash
# Robust MacOS Deployment Script for NaftaEC Chatbot
set -e

echo "🍎 MacOS Deployment Setup for NaftaEC Chatbot"

# 1. Environment Check
if [ ! -f ".env" ]; then
    echo "⚠️  WARNING: .env file not found!"
    echo "Please create a .env file with your API keys before running."
    exit 1
fi

# 2. Required Directories
echo "📁 Creating necessary directories..."
mkdir -p media reportes_descargados logs

# 3. Homebrew & Tesseract Check
echo "📦 Checking system dependencies..."
if ! command -v brew &> /dev/null; then
    echo "❌ Homebrew not found. Please install it from https://brew.sh/"
    exit 1
fi

if ! command -v tesseract &> /dev/null; then
    echo "Installing Tesseract..."
    brew install tesseract tesseract-lang
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
pip install -r requirements_macos.txt

# 5. Start Server
echo "🚀 Starting server (Logs will be output to console)..."
echo "💡 Tip: To run in the background, use: nohup ./run_macos.sh > logs/chatbot.log 2>&1 &"
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
python start.py

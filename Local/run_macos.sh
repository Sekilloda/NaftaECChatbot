#!/bin/bash
# MacOS Entry Point for NaftaEC Chatbot

echo "🍎 MacOS Setup for NaftaEC Chatbot"

# 1. Check for Homebrew
if ! command -v brew &> /dev/null; then
    echo "Homebrew not found. Please install it from https://brew.sh/"
    exit 1
fi

# 2. Install Tesseract if missing
if ! command -v tesseract &> /dev/null; then
    echo "Installing Tesseract..."
    brew install tesseract tesseract-lang
fi

# 3. Setup Virtual Environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate
pip install --upgrade pip
pip install -r requirements_macos.txt

# 4. Start Server
python start.py

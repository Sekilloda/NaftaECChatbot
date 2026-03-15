#!/bin/bash
# Linux Entry Point for NaftaEC Chatbot (Debian/Ubuntu/Fedora)

echo "🐧 Linux Setup for NaftaEC Chatbot"

# 1. Detect Distro and Install Tesseract
if [ -f /etc/debian_version ]; then
    echo "Detected Debian/Ubuntu..."
    sudo apt-get update && sudo apt-get install -y tesseract-ocr tesseract-ocr-spa python3-venv python3-pip
elif [ -f /etc/fedora-release ]; then
    echo "Detected Fedora..."
    sudo dnf install -y tesseract tesseract-langpack-spa python3.11 python3-pip
else
    echo "Unknown Linux distribution. Please ensure Tesseract and Python 3.11+ are installed."
fi

# 2. Setup Virtual Environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate
pip install --upgrade pip
pip install -r requirements_linux.txt

# 3. Start Server
python start.py

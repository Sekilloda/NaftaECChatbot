@echo off
:: Windows Entry Point for NaftaEC Chatbot

echo 🪟 Windows Setup for NaftaEC Chatbot

:: 1. Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python not found. Please install Python 3.11+ from python.org
    pause
    exit /b 1
)

:: 2. Setup Virtual Environment
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements_windows.txt

:: 3. Check for Tesseract (User manual check)
tesseract --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Tesseract not found in PATH. 
    echo Please install Tesseract-OCR and add it to your PATH or configure 'tesseract_cmd' in ocr_params.txt
)

:: 4. Start Server
python start.py
pause

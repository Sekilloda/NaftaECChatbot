@echo off
setlocal enabledelayedexpansion

echo 🪟 Windows Deployment Setup for NaftaEC Chatbot

:: 1. Environment Check
if not exist ".env" (
    echo ⚠️  WARNING: .env file not found!
    echo Please create a .env file with your API keys before running.
    pause
    exit /b 1
)

:: 2. Required Directories
echo 📁 Creating necessary directories...
if not exist "media" mkdir media
if not exist "reportes_descargados" mkdir reportes_descargados
if not exist "logs" mkdir logs

:: 3. Check for Python
echo 📦 Checking system dependencies...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found. Please install Python 3.11+ from python.org
    pause
    exit /b 1
)

:: 4. Check for Tesseract
tesseract --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  [WARNING] Tesseract not found in PATH. 
    echo Please install Tesseract-OCR and add it to your PATH or configure 'tesseract_cmd' in ocr_params.txt
)

:: 5. Setup Virtual Environment
set VENV_DIR=venv
if exist ".venv" set VENV_DIR=.venv

if not exist "%VENV_DIR%" (
    echo 🐍 Creating virtual environment...
    python -m venv .venv
    set VENV_DIR=.venv
)

echo 🔄 Activating virtual environment and installing packages...
call %VENV_DIR%\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements_windows.txt

:: 6. Start Server
echo 🚀 Starting server...
python start.py

pause

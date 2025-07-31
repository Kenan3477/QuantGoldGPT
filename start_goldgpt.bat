@echo off
echo 🏆 GoldGPT - Advanced Gold Trading AI Platform
echo ================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo 💡 Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

REM Navigate to the correct directory
cd /d "%~dp0"

echo 🚀 Starting GoldGPT with all ML capabilities...
echo.

REM Run the startup script
python start_goldgpt.py

echo.
echo 👋 GoldGPT session ended.
pause

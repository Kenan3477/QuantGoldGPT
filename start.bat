@echo off
echo 🏆 GoldGPT - Advanced AI Trading Web Application
echo ================================================
echo.
echo 🚀 Starting GoldGPT Web Application...
echo 📊 Trading 212 Inspired Dashboard
echo 🤖 Advanced AI Trading Features
echo.

cd /d "%~dp0"

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

REM Install dependencies if needed
echo 📦 Checking dependencies...
python -c "import flask" >nul 2>&1
if errorlevel 1 (
    echo Installing Flask...
    python -m pip install Flask Flask-SocketIO
)

REM Start the application
echo.
echo ✅ Starting GoldGPT Web Server...
echo 🌐 Open http://localhost:5000 in your browser
echo.
python run.py

echo.
echo 👋 GoldGPT Application stopped
pause

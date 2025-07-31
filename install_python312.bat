@echo off
echo 🏆 GoldGPT - Python 3.12 Compatible Installation
echo ===================================================
echo.

REM Check Python version
python --version | findstr "3.12" >nul
if %errorlevel%==0 (
    echo ✅ Python 3.12 detected - applying compatibility fixes
) else (
    echo ⚠️ Python 3.12 recommended for best compatibility
)

echo.
echo 📦 Installing Python 3.12 compatible dependencies...
echo.

REM Install core dependencies
pip install Flask==3.0.0
pip install Flask-SocketIO==5.3.6
pip install numpy==1.24.3
pip install pandas==2.0.3
pip install requests==2.31.0
pip install scikit-learn==1.3.0
pip install ta==0.10.2
pip install textblob==0.17.1
pip install python-dotenv==1.0.0

REM Install eventlet with Python 3.12 compatibility
pip install eventlet==0.36.1

echo.
echo ✅ Installation complete!
echo.
echo 🚀 You can now run: python app.py
echo.
pause

@echo off
echo 🚀 FORCING RAILWAY DEPLOYMENT...
cd /d "c:\Users\kenne\Downloads\ml-model-training\bot_modules\GoldGPT"

echo 📁 Current directory:
cd

echo 📋 Git status:
git status

echo 📝 Adding all files...
git add -A

echo 📝 Committing changes...
git commit -m "🚀 FORCE DEPLOY - Railway module import fixes

- Fixed Dockerfile to explicitly copy Python files
- Updated Procfile to use app.py
- Added missing dependencies: yfinance, ta
- Created deployment trigger files

This should resolve ModuleNotFoundError issues on Railway."

echo 🚀 Pushing to Railway...
git push origin main

echo ✅ Push completed!
pause

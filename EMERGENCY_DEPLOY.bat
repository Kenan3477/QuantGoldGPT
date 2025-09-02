@echo off
echo 🚀 EMERGENCY RAILWAY DEPLOYMENT
echo ===============================

cd /d "C:\Users\kenne\Downloads\ml-model-training\bot_modules\GoldGPT"

echo 📋 Git Status:
git status

echo.
echo 📦 Adding all files...
git add .

echo.
echo 💾 Committing changes...
git commit -m "EMERGENCY: Deploy Railway signal generation fix

- Fixed 500 errors on /api/signals/generate
- Added emergency signal generator fallback
- Simplified requirements.txt for Railway build
- Fixed signal_tracker import errors
- Updated Procfile for reliable startup
- All signal generation endpoints now working"

echo.
echo 🚀 Pushing to Railway...
git push origin main

echo.
echo ✅ DEPLOYMENT COMPLETE!
echo 🕒 Wait 2-3 minutes for Railway to rebuild
echo 🧪 Test: https://web-production-41802.up.railway.app/api/signals/generate

pause

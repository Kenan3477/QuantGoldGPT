@echo off
echo 🚀 Deploying Railway Signal Generation Fix
echo ========================================

echo 📁 Navigating to project directory...
cd /d "c:\Users\kenne\Downloads\ml-model-training\bot_modules\GoldGPT"

echo 📋 Checking git status...
git status

echo 📦 Adding files to git...
git add app.py
git add emergency_signal_generator.py
git add signal_tracker.py
git add enhanced_signal_tracker.py
git add auto_signal_tracker.py
git add railway_signal_test.py
git add test_local_signals.py
git add test_signal_endpoint.py
git add test_railway_deployment.py
git add DEPLOYMENT_INSTRUCTIONS.md

echo 💾 Committing changes...
git commit -m "Fix Railway 500 errors - Deploy signal generation system

- Fixed unprotected signal_tracker imports causing 500 errors
- Added proper error handling for missing modules
- Deploy emergency_signal_generator.py to Railway
- Deploy all signal tracking modules
- Fixed 'No module named signal_tracker' error
- Added comprehensive fallback system for production"

echo 🚀 Pushing to Railway...
git push origin main

echo ✅ Deployment complete! 
echo 🕒 Wait 2-3 minutes for Railway to rebuild...
echo 🧪 Then test: https://web-production-41802.up.railway.app/api/signals/generate

pause

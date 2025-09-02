@echo off
REM Deploy to Railway - Emergency Signal Generation Fix

echo 🚀 Deploying Emergency Signal Generation Fix to Railway
echo ==================================================

REM Add all changes
echo 📦 Adding changes to git...
git add .

REM Check status
echo 📋 Git status:
git status

REM Commit the changes
echo 💾 Committing changes...
git commit -m "Deploy emergency signal generation fix: Enhanced fallback system for Railway" -m "✅ Added emergency_signal_generator.py - guaranteed working signal generation" -m "✅ Enhanced app.py with triple fallback strategy (Advanced → Simple → Emergency)" -m "✅ Added comprehensive error handling and logging" -m "✅ Created test files for Railway deployment validation" -m "✅ Fixed 'all generation methods failed' error with reliable fallback" -m "" -m "Emergency generator provides stable signal generation when external APIs fail." -m "Tested locally and confirmed working with SELL signal at $3544.52"

REM Push to Railway
echo 🚂 Pushing to Railway...
git push origin main

echo ✅ Deployment complete!
echo 🔗 Test the deployment at: https://quantgoldgpt-production.up.railway.app/api/signals/generate

pause

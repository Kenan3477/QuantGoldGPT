@echo off
echo 🚀 FORCING RAILWAY DEPLOYMENT - Real-Time Pattern Detection...
cd /d "c:\Users\kenne\Downloads\ml-model-training\bot_modules\GoldGPT"

echo 📁 Current directory:
cd

echo 📋 Git status:
git status

echo 📝 Adding all files...
git add -A

echo 📝 Committing changes...
git commit -m "🚀 RAILWAY DEPLOY - Real-Time Pattern Detection v2.0

✅ LIVE MARKET SCANNING:
- Multi-source data (Yahoo Finance, Gold API, Alpha Vantage)
- Exact timestamp tracking for every pattern
- Enhanced detection algorithms
- Live vs historical classification
- Pattern significance scoring
- Continuous monitoring service

🎯 ELIMINATES FAKE PATTERNS:
- No more simulated 'doji from 1 hour ago'
- Authentic real-time market data
- Live pattern formation timestamps
- Market effect predictions

🔴 READY FOR TESTING!"

echo 🚀 Pushing to Railway...
git push origin main --force

echo ✅ Deployment triggered!
echo 🎯 Check Railway dashboard for deployment progress
pause

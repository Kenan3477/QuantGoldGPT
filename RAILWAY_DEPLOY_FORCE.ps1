Write-Host "🚀 FORCING RAILWAY DEPLOYMENT..." -ForegroundColor Green

$projectPath = "c:\Users\kenne\Downloads\ml-model-training\bot_modules\GoldGPT"
Set-Location $projectPath

Write-Host "📁 Current directory: $(Get-Location)" -ForegroundColor Blue

Write-Host "📋 Checking git status..." -ForegroundColor Yellow
git status

Write-Host "📝 Adding all files..." -ForegroundColor Yellow
git add -A

Write-Host "📝 Committing changes..." -ForegroundColor Yellow
git commit -m "🚀 RAILWAY DEPLOY - Real-Time Pattern Detection v2.0

✅ LIVE MARKET SCANNING FEATURES:
- Multi-source real-time data (Yahoo Finance, Gold API, Alpha Vantage)
- Exact timestamp tracking for every pattern formation
- Enhanced candlestick detection algorithms
- Live vs historical pattern classification
- Pattern significance scoring and market effect analysis
- Continuous background monitoring service (60s intervals)

🎯 ELIMINATES FAKE PATTERN ISSUE:
- No more simulated 'doji from 1 hour ago' patterns
- Authentic real-time market data scanning
- Live pattern formation timestamps
- Market effect predictions and urgency classification

🔴 DEPLOYMENT READY FOR IMMEDIATE TESTING!
📊 Enhanced /api/live/patterns endpoint with comprehensive live data
🚨 Pattern alerts for high-impact formations
📈 Real-time monitoring dashboard with live indicators

$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

Write-Host "🚀 Force pushing to Railway..." -ForegroundColor Green
git push origin main --force

Write-Host "✅ Deployment push completed!" -ForegroundColor Green
Write-Host "🎯 Check your Railway dashboard for deployment status" -ForegroundColor Magenta
Write-Host "🔗 Test the live pattern detection once deployment completes" -ForegroundColor Cyan

Read-Host "Press Enter to continue..."

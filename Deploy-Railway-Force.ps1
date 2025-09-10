Write-Host "🚀 FORCING RAILWAY DEPLOYMENT..." -ForegroundColor Green

$projectPath = "c:\Users\kenne\Downloads\ml-model-training\bot_modules\GoldGPT"
Set-Location $projectPath

Write-Host "📁 Current directory: $(Get-Location)" -ForegroundColor Blue

Write-Host "📋 Checking git status..." -ForegroundColor Yellow
& git status

Write-Host "📝 Adding all files..." -ForegroundColor Yellow
& git add -A

Write-Host "📝 Committing changes..." -ForegroundColor Yellow
& git commit -m "🚀 FORCE DEPLOY - Railway fixes - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

Write-Host "🚀 Pushing to Railway..." -ForegroundColor Green
& git push origin main --force

Write-Host "✅ Deployment push completed!" -ForegroundColor Green

# Also try railway CLI if available
Write-Host "🚂 Attempting Railway CLI deployment..." -ForegroundColor Cyan
try {
    & railway up
    Write-Host "✅ Railway CLI deployment triggered!" -ForegroundColor Green
} catch {
    Write-Host "ℹ️ Railway CLI not available, using git push only" -ForegroundColor Yellow
}

Write-Host "🎯 Check your Railway dashboard for deployment status" -ForegroundColor Magenta

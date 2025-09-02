# PowerShell deployment script for Railway
Write-Host "🚀 Deploying Signal Generation Fix to Railway" -ForegroundColor Green
Write-Host "=" * 50

# Set location
Set-Location "C:\Users\kenne\Downloads\ml-model-training\bot_modules\GoldGPT"
Write-Host "📁 Current directory: $(Get-Location)" -ForegroundColor Yellow

# Check git status
Write-Host "📋 Checking git status..." -ForegroundColor Cyan
& git status

# Add files
Write-Host "📦 Adding files to git..." -ForegroundColor Cyan
& git add app.py
& git add emergency_signal_generator.py
& git add signal_tracker.py
& git add enhanced_signal_tracker.py
& git add auto_signal_tracker.py
& git add validate_signal_fix.py
& git add deploy_signal_fix.bat

Write-Host "✅ Files staged" -ForegroundColor Green

# Commit
Write-Host "💾 Committing changes..." -ForegroundColor Cyan
$commitMessage = @"
Fix Railway 500 errors - Deploy signal generation system

- Fixed unprotected signal_tracker imports causing 500 errors
- Added proper error handling for missing modules
- Deploy emergency_signal_generator.py to Railway
- Deploy all signal tracking modules
- Fixed 'No module named signal_tracker' error
- Added comprehensive fallback system for production
- Emergency generator guarantees signal generation always works
"@

& git commit -m $commitMessage

Write-Host "✅ Changes committed" -ForegroundColor Green

# Push to Railway
Write-Host "🚀 Pushing to Railway..." -ForegroundColor Cyan
& git push origin main

Write-Host "✅ DEPLOYMENT COMPLETE!" -ForegroundColor Green
Write-Host "🕒 Wait 2-3 minutes for Railway to rebuild..." -ForegroundColor Yellow
Write-Host "🧪 Then test: https://web-production-41802.up.railway.app/api/signals/generate" -ForegroundColor Cyan

Write-Host ""
Write-Host "🎯 This should fix all the 500 errors you're seeing!" -ForegroundColor Green

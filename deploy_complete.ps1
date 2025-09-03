# QuantGold Complete Deployment Script for Railway (PowerShell)
# Deploys the complete QuantGold trading platform with all features

Write-Host "🚀 QuantGold Complete Deployment to Railway" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green

# Check if Railway CLI is installed
if (!(Get-Command railway -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Railway CLI not found. Please install it first:" -ForegroundColor Red
    Write-Host "npm install -g @railway/cli" -ForegroundColor Yellow
    exit 1
}

# Login to Railway (if not already logged in)
Write-Host "🔐 Checking Railway authentication..." -ForegroundColor Cyan
railway login

# Set up deployment files
Write-Host "📦 Preparing deployment files..." -ForegroundColor Cyan

# Copy complete app as main app.py
Copy-Item "quantgold_complete_app.py" "app.py" -Force

# Copy complete requirements
Copy-Item "requirements_complete.txt" "requirements.txt" -Force

# Copy complete Procfile
Copy-Item "Procfile_complete" "Procfile" -Force

# Copy complete Dockerfile
Copy-Item "Dockerfile_complete" "Dockerfile" -Force

Write-Host "✅ Deployment files prepared" -ForegroundColor Green

# Show what we're deploying
Write-Host ""
Write-Host "📋 Deployment Summary:" -ForegroundColor Yellow
Write-Host "----------------------" -ForegroundColor Yellow
Write-Host "App File: quantgold_complete_app.py → app.py"
Write-Host "Requirements: requirements_complete.txt → requirements.txt"
Write-Host "Procfile: Procfile_complete → Procfile"
Write-Host "Dockerfile: Dockerfile_complete → Dockerfile"
Write-Host ""
Write-Host "🎯 Features Included:" -ForegroundColor Magenta
Write-Host "- Advanced AI Signal Generation"
Write-Host "- Real-time Signal Tracking with P&L"
Write-Host "- Enhanced ML Prediction Engine"
Write-Host "- Multi-timeframe Predictions (5M to 1W)"
Write-Host "- Professional Trading Dashboard"
Write-Host "- Live WebSocket Updates"
Write-Host "- Market News Integration"
Write-Host "- Technical Analysis Engine"
Write-Host "- Performance Analytics"
Write-Host "- Emergency Fallback Systems"
Write-Host ""

# Ask for confirmation
$confirmation = Read-Host "🤔 Deploy QuantGold Complete to Railway? (y/N)"
if ($confirmation -ne 'y' -and $confirmation -ne 'Y') {
    Write-Host "❌ Deployment cancelled" -ForegroundColor Red
    exit 1
}

# Initialize Railway project (if not already initialized)
Write-Host "🔧 Initializing Railway project..." -ForegroundColor Cyan
railway init

# Deploy to Railway
Write-Host "🚀 Deploying to Railway..." -ForegroundColor Green
railway up

# Show deployment status
Write-Host ""
Write-Host "✅ Deployment initiated!" -ForegroundColor Green
Write-Host ""
Write-Host "🔍 Check deployment status:" -ForegroundColor Cyan
Write-Host "railway logs" -ForegroundColor Yellow
Write-Host ""
Write-Host "🌐 Get your app URL:" -ForegroundColor Cyan
Write-Host "railway domain" -ForegroundColor Yellow
Write-Host ""
Write-Host "📊 Monitor your app:" -ForegroundColor Cyan
Write-Host "railway open" -ForegroundColor Yellow
Write-Host ""
Write-Host "🛠️ If there are issues, check logs:" -ForegroundColor Cyan
Write-Host "railway logs --follow" -ForegroundColor Yellow
Write-Host ""
Write-Host "🎯 QuantGold Complete Features:" -ForegroundColor Magenta
Write-Host "- Visit /quantgold for the professional dashboard"
Write-Host "- Visit /health for system status"
Write-Host "- Visit /debug for detailed system information"
Write-Host ""
Write-Host "🎉 Happy Trading! 📈" -ForegroundColor Green

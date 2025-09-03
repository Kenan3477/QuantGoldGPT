#!/bin/bash

# QuantGold Complete Deployment Script for Railway
# Deploys the complete QuantGold trading platform with all features

echo "🚀 QuantGold Complete Deployment to Railway"
echo "============================================="

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found. Please install it first:"
    echo "npm install -g @railway/cli"
    exit 1
fi

# Login to Railway (if not already logged in)
echo "🔐 Checking Railway authentication..."
railway login

# Set up deployment files
echo "📦 Preparing deployment files..."

# Copy complete app as main app.py
cp quantgold_complete_app.py app.py

# Copy complete requirements
cp requirements_complete.txt requirements.txt

# Copy complete Procfile
cp Procfile_complete Procfile

# Copy complete Dockerfile
cp Dockerfile_complete Dockerfile

echo "✅ Deployment files prepared"

# Show what we're deploying
echo ""
echo "📋 Deployment Summary:"
echo "----------------------"
echo "App File: quantgold_complete_app.py → app.py"
echo "Requirements: requirements_complete.txt → requirements.txt"
echo "Procfile: Procfile_complete → Procfile"
echo "Dockerfile: Dockerfile_complete → Dockerfile"
echo ""
echo "🎯 Features Included:"
echo "- Advanced AI Signal Generation"
echo "- Real-time Signal Tracking with P&L"
echo "- Enhanced ML Prediction Engine"
echo "- Multi-timeframe Predictions (5M to 1W)"
echo "- Professional Trading Dashboard"
echo "- Live WebSocket Updates"
echo "- Market News Integration"
echo "- Technical Analysis Engine"
echo "- Performance Analytics"
echo "- Emergency Fallback Systems"
echo ""

# Ask for confirmation
read -p "🤔 Deploy QuantGold Complete to Railway? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Deployment cancelled"
    exit 1
fi

# Initialize Railway project (if not already initialized)
echo "🔧 Initializing Railway project..."
railway init

# Deploy to Railway
echo "🚀 Deploying to Railway..."
railway up

# Show deployment status
echo ""
echo "✅ Deployment initiated!"
echo ""
echo "🔍 Check deployment status:"
echo "railway logs"
echo ""
echo "🌐 Get your app URL:"
echo "railway domain"
echo ""
echo "📊 Monitor your app:"
echo "railway open"
echo ""
echo "🛠️ If there are issues, check logs:"
echo "railway logs --follow"
echo ""
echo "🎯 QuantGold Complete Features:"
echo "- Visit /quantgold for the professional dashboard"
echo "- Visit /health for system status"
echo "- Visit /debug for detailed system information"
echo ""
echo "🎉 Happy Trading! 📈"

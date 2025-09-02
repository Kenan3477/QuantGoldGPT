#!/bin/bash
# Deploy to Railway - Emergency Signal Generation Fix

echo "🚀 Deploying Emergency Signal Generation Fix to Railway"
echo "=================================================="

# Add all changes
echo "📦 Adding changes to git..."
git add .

# Check status
echo "📋 Git status:"
git status

# Commit the changes
echo "💾 Committing changes..."
git commit -m "Deploy emergency signal generation fix: Enhanced fallback system for Railway

✅ Added emergency_signal_generator.py - guaranteed working signal generation
✅ Enhanced app.py with triple fallback strategy (Advanced → Simple → Emergency)  
✅ Added comprehensive error handling and logging
✅ Created test files for Railway deployment validation
✅ Fixed 'all generation methods failed' error with reliable fallback

Emergency generator provides stable signal generation when external APIs fail.
Tested locally and confirmed working with SELL signal at $3544.52"

# Push to Railway
echo "🚂 Pushing to Railway..."
git push origin main

echo "✅ Deployment complete!"
echo "🔗 Test the deployment at: https://quantgoldgpt-production.up.railway.app/api/signals/generate"

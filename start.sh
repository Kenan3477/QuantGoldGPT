#!/bin/bash
# Railway start script - ensures correct app.py execution
echo "🚀 Starting GoldGPT Advanced Dashboard..."
echo "📁 Current directory: $(pwd)"
echo "📋 Files in directory:"
ls -la *.py | head -5
echo "🎯 Executing app.py..."
exec python app.py

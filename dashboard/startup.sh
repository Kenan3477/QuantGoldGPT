#!/bin/bash
set -e

echo "===================="
echo "QuantGold API Startup"
echo "===================="
echo ""

# Show environment
echo "📍 Working directory: $(pwd)"
echo "📂 Files in /app:"
ls -la /app/ || echo "  /app/ not found"
echo ""
echo "📂 Files in /app/dashboard:"
ls -la /app/dashboard/ || echo "  /app/dashboard/ not found"
echo ""
echo "📂 Files in /app/paper_trading:"
ls -la /app/paper_trading/ || echo "  /app/paper_trading/ not found"
echo ""

# Check Python and modules
echo "🐍 Python version:"
python --version
echo ""
echo "📦 Installed packages:"
pip list | grep -E "(fastapi|uvicorn|pandas)" || echo "  Required packages not found!"
echo ""

# Show port
PORT=${PORT:-8080}
echo "🌐 Starting server on 0.0.0.0:$PORT"
echo ""

# Start uvicorn with verbose logging
exec uvicorn dashboard.api:app \
    --host 0.0.0.0 \
    --port $PORT \
    --log-level info \
    --access-log

#!/bin/sh
set -e

# Railway sets PORT env var
PORT=${PORT:-8080}

echo "Starting QuantGold API on port $PORT..."
echo "Working directory: $(pwd)"
echo "Paper trading files:"
ls -la /app/paper_trading/ 2>/dev/null || echo "  No paper_trading directory found"
echo ""

exec uvicorn dashboard.api:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --log-level info \
    --access-log

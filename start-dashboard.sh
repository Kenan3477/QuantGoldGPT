#!/bin/bash
# Start QuantGold system with dashboard

echo "================================"
echo "QuantGold System Startup"
echo "================================"
echo ""

# Check if paper trading data exists
if [ ! -d "/workspace/paper_trading" ] || [ -z "$(ls -A /workspace/paper_trading/*.parquet 2>/dev/null)" ]; then
    echo "📊 No paper trading data found. Running initial deployment..."
    python3 /workspace/quantgold/execution/deploy_paper_trading.py \
        --symbol XAUUSD \
        --timeframe H4 \
        --model xgboost
    echo ""
fi

echo "🚀 Starting FastAPI backend on port 8000..."
cd /workspace
uvicorn dashboard.api:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!

# Wait for API to start
sleep 3

echo ""
echo "================================"
echo "✅ System Started!"
echo "================================"
echo ""
echo "📊 Dashboard API: http://localhost:8000"
echo "📖 API Docs: http://localhost:8000/docs"
echo ""
echo "🌐 Next steps:"
echo "   1. In another terminal, run dashboard:"
echo "      cd /workspace/dashboard && npm install && npm run dev"
echo "   2. Visit: http://localhost:3000"
echo ""
echo "   OR deploy to Vercel:"
echo "      cd /workspace && vercel"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Wait for API process
wait $API_PID

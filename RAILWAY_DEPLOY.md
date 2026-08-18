# Railway Deployment Instructions

## What to Deploy
This project contains a FastAPI dashboard backend in `dashboard/api.py`.

## Start Command
Use the Procfile:
```
web: uvicorn dashboard.api:app --host 0.0.0.0 --port $PORT
```

## Test Endpoints
- Health: GET /
- Status: GET /api/status
- Metrics: GET /api/metrics
- Trades: GET /api/trades
- Live Feed: GET /api/live-feed

## Requirements
Install from: requirements-dashboard.txt

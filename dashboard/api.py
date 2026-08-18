"""
FastAPI backend for QuantGold dashboard.

Serves real-time predictions, performance metrics, and system status.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import pandas as pd
import json
from datetime import datetime
from typing import Optional, List, Dict
import glob

app = FastAPI(title="QuantGold Dashboard API")

# Enable CORS for Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your Vercel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_latest_predictions_file() -> Optional[str]:
    """Get the most recent predictions parquet file."""
    files = glob.glob("/workspace/paper_trading/predictions_*.parquet")
    if not files:
        return None
    return max(files, key=lambda x: Path(x).stat().st_mtime)


def get_latest_summary_file() -> Optional[str]:
    """Get the most recent summary JSON file."""
    files = glob.glob("/workspace/paper_trading/summary_*.json")
    if not files:
        return None
    return max(files, key=lambda x: Path(x).stat().st_mtime)


@app.get("/")
def read_root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "QuantGold Dashboard API",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/live-gold-price")
def legacy_healthcheck():
    """Legacy healthcheck endpoint for Railway compatibility."""
    return {
        "status": "ok",
        "price": 0,
        "message": "Legacy endpoint - use /api/status instead"
    }


@app.get("/api/status")
def get_status():
    """Get overall system status."""
    predictions_file = get_latest_predictions_file()
    summary_file = get_latest_summary_file()
    
    if not predictions_file or not summary_file:
        return {
            "status": "no_data",
            "message": "Paper trading not yet started",
        }
    
    # Load summary
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    # Load predictions
    df = pd.read_parquet(predictions_file)
    
    # Calculate metrics
    executed = df[df['side'].isin(['BUY', 'SELL'])]
    
    if len(executed) == 0:
        return {
            "status": "no_trades",
            "message": "No trades executed yet",
        }
    
    successful = (executed['success'] == True).sum()
    win_rate = successful / len(executed)
    
    # Recent performance
    recent = executed.tail(50)
    recent_win_rate = (recent['success'] == True).sum() / len(recent) if len(recent) > 0 else 0
    
    # Drift detection
    expected_win_rate = 0.943
    drift_threshold = 0.15
    win_rate_drop = expected_win_rate - win_rate
    
    if win_rate_drop > drift_threshold:
        drift_status = "DRIFT_DETECTED"
        drift_severity = "high"
    elif win_rate_drop > 0.05:
        drift_status = "MINOR_DEGRADATION"
        drift_severity = "medium"
    else:
        drift_status = "HEALTHY"
        drift_severity = "low"
    
    return {
        "status": "active",
        "timestamp": datetime.now().isoformat(),
        "win_rate": round(win_rate, 4),
        "recent_win_rate": round(recent_win_rate, 4),
        "total_trades": len(executed),
        "total_predictions": len(df),
        "coverage": round(len(executed) / len(df), 4),
        "drift_status": drift_status,
        "drift_severity": drift_severity,
        "expected_win_rate": expected_win_rate,
        "drift_amount": round(win_rate_drop, 4),
    }


@app.get("/api/predictions")
def get_predictions(limit: int = 100, offset: int = 0):
    """Get recent predictions."""
    predictions_file = get_latest_predictions_file()
    
    if not predictions_file:
        raise HTTPException(status_code=404, detail="No predictions found")
    
    df = pd.read_parquet(predictions_file)
    
    # Sort by timestamp descending (most recent first)
    df = df.sort_values('timestamp', ascending=False)
    
    # Apply pagination
    paginated = df.iloc[offset:offset+limit]
    
    # Convert to JSON-friendly format
    records = []
    for _, row in paginated.iterrows():
        records.append({
            'timestamp': row['timestamp'].isoformat(),
            'symbol': row['symbol'],
            'side': row['side'],
            'calibrated_probability': float(row['calibrated_probability']),
            'reason': row['reason'],
            'success': bool(row['success']) if pd.notna(row['success']) else None,
            'close': float(row['close']),
        })
    
    return {
        'predictions': records,
        'total': len(df),
        'offset': offset,
        'limit': limit,
    }


@app.get("/api/trades")
def get_trades(limit: int = 50):
    """Get executed trades only."""
    predictions_file = get_latest_predictions_file()
    
    if not predictions_file:
        raise HTTPException(status_code=404, detail="No predictions found")
    
    df = pd.read_parquet(predictions_file)
    
    # Filter to executed trades
    executed = df[df['side'].isin(['BUY', 'SELL'])].copy()
    executed = executed.sort_values('timestamp', ascending=False)
    
    # Apply limit
    trades = executed.head(limit)
    
    # Convert to JSON
    records = []
    for _, row in trades.iterrows():
        records.append({
            'timestamp': row['timestamp'].isoformat(),
            'symbol': row['symbol'],
            'side': row['side'],
            'calibrated_probability': float(row['calibrated_probability']),
            'success': bool(row['success']) if pd.notna(row['success']) else None,
            'close': float(row['close']),
        })
    
    return {
        'trades': records,
        'total': len(executed),
    }


@app.get("/api/metrics")
def get_metrics():
    """Get detailed performance metrics."""
    predictions_file = get_latest_predictions_file()
    
    if not predictions_file:
        raise HTTPException(status_code=404, detail="No predictions found")
    
    df = pd.read_parquet(predictions_file)
    executed = df[df['side'].isin(['BUY', 'SELL'])].copy()
    
    if len(executed) == 0:
        raise HTTPException(status_code=404, detail="No trades executed yet")
    
    # Overall metrics
    successful = (executed['success'] == True).sum()
    failed = (executed['success'] == False).sum()
    win_rate = successful / len(executed)
    
    # By side
    buy_trades = executed[executed['side'] == 'BUY']
    sell_trades = executed[executed['side'] == 'SELL']
    
    buy_win_rate = (buy_trades['success'] == True).sum() / len(buy_trades) if len(buy_trades) > 0 else 0
    sell_win_rate = (sell_trades['success'] == True).sum() / len(sell_trades) if len(sell_trades) > 0 else 0
    
    # Recent performance (last 50 trades)
    recent = executed.tail(50)
    recent_win_rate = (recent['success'] == True).sum() / len(recent) if len(recent) > 0 else 0
    
    # Hourly breakdown
    executed['hour'] = pd.to_datetime(executed['timestamp']).dt.hour
    hourly_stats = []
    for hour in range(24):
        hour_trades = executed[executed['hour'] == hour]
        if len(hour_trades) > 0:
            hourly_stats.append({
                'hour': hour,
                'trades': len(hour_trades),
                'win_rate': float((hour_trades['success'] == True).sum() / len(hour_trades)),
            })
    
    return {
        'overall': {
            'total_predictions': len(df),
            'total_trades': len(executed),
            'successful': int(successful),
            'failed': int(failed),
            'win_rate': round(win_rate, 4),
            'coverage': round(len(executed) / len(df), 4),
            'avg_probability': round(executed['calibrated_probability'].mean(), 4),
        },
        'by_side': {
            'buy': {
                'count': len(buy_trades),
                'win_rate': round(buy_win_rate, 4),
            },
            'sell': {
                'count': len(sell_trades),
                'win_rate': round(sell_win_rate, 4),
            },
        },
        'recent': {
            'trades': len(recent),
            'win_rate': round(recent_win_rate, 4),
        },
        'hourly': hourly_stats,
    }


@app.get("/api/live-feed")
def get_live_feed(limit: int = 20):
    """Get live feed of most recent activity."""
    predictions_file = get_latest_predictions_file()
    
    if not predictions_file:
        return {'feed': []}
    
    df = pd.read_parquet(predictions_file)
    df = df.sort_values('timestamp', ascending=False).head(limit)
    
    feed = []
    for _, row in df.iterrows():
        feed.append({
            'timestamp': row['timestamp'].isoformat(),
            'type': 'trade' if row['side'] in ['BUY', 'SELL'] else 'prediction',
            'side': row['side'],
            'probability': float(row['calibrated_probability']),
            'success': bool(row['success']) if pd.notna(row['success']) else None,
            'reason': row['reason'],
        })
    
    return {'feed': feed}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

'use client'

import { useState, useEffect } from 'react'
import useSWR from 'swr'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

const fetcher = (url: string) => fetch(url).then(res => res.json())

interface SystemStatus {
  status: string
  win_rate: number
  recent_win_rate: number
  total_trades: number
  total_predictions: number
  coverage: number
  drift_status: string
  drift_severity: string
  expected_win_rate: number
  drift_amount: number
}

interface Trade {
  timestamp: string
  side: string
  calibrated_probability: number
  success: boolean | null
  close: number
}

interface Metrics {
  overall: {
    total_predictions: number
    total_trades: number
    successful: number
    failed: number
    win_rate: number
    coverage: number
    avg_probability: number
  }
  by_side: {
    buy: { count: number; win_rate: number }
    sell: { count: number; win_rate: number }
  }
  recent: {
    trades: number
    win_rate: number
  }
}

interface FeedItem {
  timestamp: string
  type: string
  side: string
  probability: number
  success: boolean | null
  reason: string
}

export default function Dashboard() {
  const [autoRefresh, setAutoRefresh] = useState(true)
  
  // Poll every 10 seconds if auto-refresh enabled
  const refreshInterval = autoRefresh ? 10000 : 0
  
  const { data: status } = useSWR<SystemStatus>(
    `${API_URL}/api/status`,
    fetcher,
    { refreshInterval }
  )
  
  const { data: metrics } = useSWR<Metrics>(
    `${API_URL}/api/metrics`,
    fetcher,
    { refreshInterval }
  )
  
  const { data: tradesData } = useSWR<{ trades: Trade[] }>(
    `${API_URL}/api/trades?limit=10`,
    fetcher,
    { refreshInterval }
  )
  
  const { data: feedData } = useSWR<{ feed: FeedItem[] }>(
    `${API_URL}/api/live-feed?limit=20`,
    fetcher,
    { refreshInterval }
  )

  const getDriftColor = (severity: string) => {
    switch (severity) {
      case 'low': return 'text-green-500'
      case 'medium': return 'text-yellow-500'
      case 'high': return 'text-red-500'
      default: return 'text-gray-500'
    }
  }

  const getSideColor = (side: string) => {
    switch (side) {
      case 'BUY': return 'text-green-600'
      case 'SELL': return 'text-red-600'
      default: return 'text-gray-600'
    }
  }

  const getSuccessIcon = (success: boolean | null) => {
    if (success === null) return '⏳'
    return success ? '✅' : '❌'
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-black text-white">
      {/* Header */}
      <header className="bg-gray-800/50 backdrop-blur-sm border-b border-gold-500/20">
        <div className="max-w-7xl mx-auto px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="text-3xl">📊</div>
              <div>
                <h1 className="text-2xl font-bold text-gold-400">QuantGold</h1>
                <p className="text-sm text-gray-400">Real-Time Trading Dashboard</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <button
                onClick={() => setAutoRefresh(!autoRefresh)}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  autoRefresh
                    ? 'bg-green-600 hover:bg-green-700'
                    : 'bg-gray-600 hover:bg-gray-700'
                }`}
              >
                {autoRefresh ? '🟢 Live' : '⏸️ Paused'}
              </button>
              <div className="text-sm text-gray-400">
                {new Date().toLocaleTimeString()}
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8 sm:px-6 lg:px-8">
        {/* Status Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          {/* Win Rate */}
          <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gold-500/20">
            <div className="text-sm text-gray-400 mb-2">Win Rate</div>
            <div className="text-3xl font-bold text-gold-400">
              {status ? `${(status.win_rate * 100).toFixed(1)}%` : '---'}
            </div>
            <div className="text-xs text-gray-500 mt-2">
              Recent: {status ? `${(status.recent_win_rate * 100).toFixed(1)}%` : '---'}
            </div>
          </div>

          {/* Total Trades */}
          <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gold-500/20">
            <div className="text-sm text-gray-400 mb-2">Total Trades</div>
            <div className="text-3xl font-bold text-white">
              {status?.total_trades?.toLocaleString() || '---'}
            </div>
            <div className="text-xs text-gray-500 mt-2">
              Coverage: {status ? `${(status.coverage * 100).toFixed(1)}%` : '---'}
            </div>
          </div>

          {/* Drift Status */}
          <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gold-500/20">
            <div className="text-sm text-gray-400 mb-2">Drift Status</div>
            <div className={`text-2xl font-bold ${status ? getDriftColor(status.drift_severity) : 'text-gray-500'}`}>
              {status?.drift_status || '---'}
            </div>
            <div className="text-xs text-gray-500 mt-2">
              Drop: {status ? `${(status.drift_amount * 100).toFixed(1)}%` : '---'}
            </div>
          </div>

          {/* System Status */}
          <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gold-500/20">
            <div className="text-sm text-gray-400 mb-2">System Status</div>
            <div className="text-2xl font-bold text-green-400">
              {status?.status === 'active' ? '🟢 ACTIVE' : '🔴 INACTIVE'}
            </div>
            <div className="text-xs text-gray-500 mt-2">
              Predictions: {status?.total_predictions?.toLocaleString() || '---'}
            </div>
          </div>
        </div>

        {/* Performance Metrics */}
        {metrics && (
          <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gold-500/20 mb-8">
            <h2 className="text-xl font-bold mb-4 text-gold-400">Performance Metrics</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* Overall */}
              <div>
                <h3 className="text-sm font-semibold text-gray-400 mb-3">Overall</h3>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-gray-300">Successful:</span>
                    <span className="text-green-400 font-medium">{metrics.overall.successful}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-300">Failed:</span>
                    <span className="text-red-400 font-medium">{metrics.overall.failed}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-300">Avg Probability:</span>
                    <span className="text-white font-medium">{(metrics.overall.avg_probability * 100).toFixed(1)}%</span>
                  </div>
                </div>
              </div>

              {/* By Side */}
              <div>
                <h3 className="text-sm font-semibold text-gray-400 mb-3">By Signal Type</h3>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-green-600 font-medium">BUY:</span>
                    <span className="text-white">{metrics.by_side.buy.count} ({(metrics.by_side.buy.win_rate * 100).toFixed(1)}%)</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-red-600 font-medium">SELL:</span>
                    <span className="text-white">{metrics.by_side.sell.count} ({(metrics.by_side.sell.win_rate * 100).toFixed(1)}%)</span>
                  </div>
                </div>
              </div>

              {/* Recent */}
              <div>
                <h3 className="text-sm font-semibold text-gray-400 mb-3">Recent Performance</h3>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-gray-300">Last {metrics.recent.trades} trades:</span>
                    <span className="text-gold-400 font-medium">{(metrics.recent.win_rate * 100).toFixed(1)}%</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Recent Trades */}
          <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gold-500/20">
            <h2 className="text-xl font-bold mb-4 text-gold-400">Recent Trades</h2>
            <div className="space-y-3">
              {tradesData?.trades?.slice(0, 10).map((trade, idx) => (
                <div key={idx} className="bg-gray-700/30 rounded-lg p-4 border border-gray-600/30">
                  <div className="flex items-center justify-between mb-2">
                    <span className={`font-bold text-lg ${getSideColor(trade.side)}`}>
                      {trade.side}
                    </span>
                    <span className="text-2xl">{getSuccessIcon(trade.success)}</span>
                  </div>
                  <div className="text-sm space-y-1">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Probability:</span>
                      <span className="text-white font-medium">{(trade.calibrated_probability * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Price:</span>
                      <span className="text-white font-medium">${trade.close.toFixed(2)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Time:</span>
                      <span className="text-gray-300 text-xs">{new Date(trade.timestamp).toLocaleString()}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Live Feed */}
          <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gold-500/20">
            <h2 className="text-xl font-bold mb-4 text-gold-400">Live Feed</h2>
            <div className="space-y-2 max-h-[600px] overflow-y-auto">
              {feedData?.feed?.map((item, idx) => (
                <div key={idx} className="bg-gray-700/30 rounded-lg p-3 border border-gray-600/20">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <span className={`font-semibold ${getSideColor(item.side)}`}>
                        {item.side}
                      </span>
                      {item.success !== null && (
                        <span className="text-lg">{getSuccessIcon(item.success)}</span>
                      )}
                    </div>
                    <span className="text-xs text-gray-400">
                      {new Date(item.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                  <div className="mt-1 text-xs text-gray-400">
                    {item.type === 'trade' ? (
                      <span>Prob: {(item.probability * 100).toFixed(1)}%</span>
                    ) : (
                      <span className="text-gray-500">Reason: {item.reason}</span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-gray-800/50 backdrop-blur-sm border-t border-gold-500/20 mt-12">
        <div className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
          <div className="text-center text-sm text-gray-400">
            QuantGold Trading System | Auto-refresh: {autoRefresh ? 'ON (10s)' : 'OFF'}
          </div>
        </div>
      </footer>
    </div>
  )
}

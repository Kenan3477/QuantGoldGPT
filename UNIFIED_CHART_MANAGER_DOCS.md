# 📊 UnifiedChartManager - Complete Documentation

## 🎯 Overview

The UnifiedChartManager is a sophisticated charting system that provides a consistent API across multiple charting libraries with full WebSocket integration for real-time trading data.

## ✨ Key Features

### 📚 Multi-Library Support (Priority Order)
1. **LightweightCharts** (Highest Priority) - Professional trading charts
2. **Chart.js** (Fallback) - Versatile web charts with OHLC support
3. **TradingView** (Enterprise Fallback) - Advanced trading platform widgets

### 📈 Chart Types
- **Candlestick Charts** - Traditional OHLC candlestick visualization
- **OHLC Charts** - Open-High-Low-Close bar charts
- **Line Charts** - Simple price line visualization

### ⏰ Timeframe Support
- **1m** - 1-minute intervals
- **5m** - 5-minute intervals  
- **15m** - 15-minute intervals
- **1h** - 1-hour intervals
- **4h** - 4-hour intervals
- **1d** - Daily intervals

### 🔗 WebSocket Integration
- Real-time price updates every 2 seconds
- Automatic data buffering and chart updates
- Connection state management
- Fallback support for offline mode

## 🚀 Quick Start

### Basic Chart Creation
```javascript
// Create a basic candlestick chart
const chartManager = window.UnifiedChartManagerFactory.createChart('my-chart', {
    chartType: 'candlestick',
    timeframe: '1h',
    theme: 'dark',
    height: 400
});
```

### WebSocket Integration
```javascript
// Get WebSocket manager
const wsManager = window.WebSocketManager;

// Create chart with real-time updates
const realtimeChart = window.UnifiedChartManagerFactory.createChart('realtime-chart', {
    chartType: 'candlestick',
    timeframe: '5m',
    wsManager: wsManager, // Enable WebSocket integration
    realtime: true
});
```

### Chart Type Switching
```javascript
// Change chart type dynamically
await chartManager.setChartType('line');
await chartManager.setTimeframe('15m');
```

## 📖 API Reference

### UnifiedChartManager Class

#### Constructor Options
```javascript
const options = {
    chartType: 'candlestick',    // 'candlestick', 'ohlc', 'line'
    timeframe: '1h',             // '1m', '5m', '15m', '1h', '4h', '1d'
    theme: 'dark',               // 'light', 'dark'
    height: 400,                 // Chart height in pixels
    width: null,                 // Chart width (null = responsive)
    realtime: true,              // Enable real-time updates
    maxDataPoints: 1000,         // Maximum data points for performance
    enableVolume: true,          // Show volume bars
    enableIndicators: true,      // Enable technical indicators
    wsManager: null,             // WebSocket manager instance
    debug: false                 // Enable debug logging
};
```

#### Methods

##### Connection and Setup
```javascript
// Initialize chart with auto-library detection
await chartManager.initializeChart();

// Set chart data
await chartManager.setData(dataArray);

// Add real-time data point
await chartManager.addDataPoint(dataPoint);
```

##### Chart Configuration
```javascript
// Change chart type
await chartManager.setChartType('candlestick');

// Change timeframe
await chartManager.setTimeframe('1h');

// Get current status
const status = chartManager.getStatus();
```

##### WebSocket Integration
```javascript
// Setup WebSocket integration
chartManager.setupWebSocketIntegration();

// Handle price updates manually
chartManager.handleWebSocketPriceUpdate(priceData);
```

##### Cleanup
```javascript
// Destroy chart and cleanup resources
await chartManager.destroy();
```

### UnifiedChartManagerFactory Class

#### Factory Methods
```javascript
// Create new chart instance
const chart = window.UnifiedChartManagerFactory.createChart(containerId, options);

// Get existing chart
const chart = window.UnifiedChartManagerFactory.getChart(containerId);

// Remove chart
await window.UnifiedChartManagerFactory.removeChart(containerId);

// Get all charts
const allCharts = window.UnifiedChartManagerFactory.getAllCharts();

// Destroy all charts
await window.UnifiedChartManagerFactory.destroyAll();
```

## 🎛️ Integration Examples

### HTML Setup
```html
<!DOCTYPE html>
<html>
<head>
    <!-- Chart Libraries (load in priority order) -->
    <script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-chart-financial/dist/chartjs-chart-financial.min.js"></script>
    
    <!-- Socket.IO for WebSocket -->
    <script src="https://cdn.socket.io/4.0.0/socket.io.min.js"></script>
</head>
<body>
    <!-- Chart Container -->
    <div id="trading-chart" style="height: 600px;"></div>
    
    <!-- Load GoldGPT Scripts -->
    <script src="{{ url_for('static', filename='js/websocket_manager.js') }}"></script>
    <script src="{{ url_for('static', filename='js/unified_chart_manager.js') }}"></script>
</body>
</html>
```

### JavaScript Integration
```javascript
document.addEventListener('DOMContentLoaded', async () => {
    // Get WebSocket manager
    const wsManager = window.WebSocketManagerFactory.getInstance('default');
    
    // Connect WebSocket
    await wsManager.connect();
    
    // Create chart with WebSocket integration
    const chart = window.UnifiedChartManagerFactory.createChart('trading-chart', {
        chartType: 'candlestick',
        timeframe: '1h',
        wsManager: wsManager,
        realtime: true,
        enableVolume: true,
        theme: 'dark'
    });
    
    console.log('Chart created:', chart.getStatus());
});
```

### Advanced Dashboard Integration
```javascript
class TradingDashboard {
    constructor() {
        this.wsManager = null;
        this.charts = {};
        this.initialize();
    }
    
    async initialize() {
        // Setup WebSocket
        this.wsManager = window.WebSocketManagerFactory.getInstance('dashboard');
        await this.wsManager.connect();
        
        // Create main chart
        this.charts.main = window.UnifiedChartManagerFactory.createChart('main-chart', {
            chartType: 'candlestick',
            timeframe: '1h',
            wsManager: this.wsManager,
            height: 600,
            enableVolume: true
        });
        
        // Create overview chart
        this.charts.overview = window.UnifiedChartManagerFactory.createChart('overview-chart', {
            chartType: 'line',
            timeframe: '1d',
            wsManager: this.wsManager,
            height: 200,
            enableVolume: false
        });
    }
    
    async changeChartType(chartType) {
        await this.charts.main.setChartType(chartType);
    }
    
    async changeTimeframe(timeframe) {
        await this.charts.main.setTimeframe(timeframe);
    }
}

// Initialize dashboard
const dashboard = new TradingDashboard();
```

## 📊 Data Format

### Price Data Structure
```javascript
// For Candlestick/OHLC charts
const dataPoint = {
    time: 1627776000,          // Unix timestamp (seconds)
    open: 1800.50,             // Opening price
    high: 1820.75,             // Highest price
    low: 1795.25,              // Lowest price
    close: 1815.00,            // Closing price
    volume: 1500000,           // Volume (optional)
    timestamp: 1627776000000   // Milliseconds timestamp (optional)
};

// For Line charts
const linePoint = {
    time: 1627776000,          // Unix timestamp (seconds)
    value: 1815.00,            // Price value
    timestamp: 1627776000000   // Milliseconds timestamp (optional)
};
```

### WebSocket Price Update Format
```javascript
const priceUpdate = {
    price: 1815.00,            // Current price
    change: 5.50,              // Price change
    change_percent: 0.31,      // Percentage change
    timestamp: 1627776000000,  // Update timestamp
    high: 1820.75,             // Daily high (optional)
    low: 1795.25,              // Daily low (optional)
    volume: 1500000            // Volume (optional)
};
```

## 🔧 Library-Specific Features

### LightweightCharts (Priority 1)
- ✅ Professional trading chart appearance
- ✅ High performance with large datasets
- ✅ Native candlestick/OHLC support
- ✅ Built-in volume indicators
- ✅ Responsive design

### Chart.js (Priority 2)  
- ✅ OHLC plugin support
- ✅ Extensive customization options
- ✅ Animation support
- ✅ Plugin ecosystem
- ✅ Wide browser compatibility

### TradingView (Priority 3)
- ✅ Professional trading platform
- ✅ Advanced technical indicators
- ✅ Multiple timeframe support
- ✅ Real-time data feeds
- ⚠️ Requires TradingView account for advanced features

## ⚡ Performance Optimization

### Data Management
- **Buffer Management**: Automatic price update buffering (100ms intervals)
- **Data Limits**: Configurable max data points (default: 1000)
- **Memory Cleanup**: Automatic cleanup on chart destruction
- **Update Batching**: Batched DOM updates for smooth performance

### Real-time Updates
```javascript
// Optimized for real-time performance
const chart = window.UnifiedChartManagerFactory.createChart('fast-chart', {
    chartType: 'line',         // Line charts are fastest
    maxDataPoints: 500,        // Limit data points
    realtime: true,
    wsManager: wsManager
});
```

## 🔍 Debugging and Monitoring

### Enable Debug Mode
```javascript
const chart = window.UnifiedChartManagerFactory.createChart('debug-chart', {
    debug: true               // Enable detailed logging
});
```

### Check Chart Status
```javascript
const status = chart.getStatus();
console.log('Chart Status:', {
    activeLibrary: status.activeLibrary,     // Which library is being used
    chartType: status.chartType,             // Current chart type
    timeframe: status.timeframe,             // Current timeframe
    dataPoints: status.dataPoints,           // Number of data points
    isRealtime: status.isRealtime,          // Real-time status
    hasWebSocket: status.hasWebSocket       // WebSocket integration status
});
```

### Monitor Performance
```javascript
// Check available libraries
const availability = chart.availableLibraries;
console.log('Available Libraries:', availability);

// Monitor update frequency
chart.subscribe('priceUpdate', (data) => {
    console.log('Price update received:', data.timestamp);
});
```

## 🛠️ Testing and Development

### Demo Page
Visit the demo page to test all features:
```
http://localhost:5000/chart-demo
```

### Test Functions
```javascript
// Test basic chart creation
window.createBasicChart();

// Test real-time chart
window.createRealtimeChart();

// Test multiple charts
window.createMultipleCharts();

// Check system status
window.checkChartStatus();
```

### Manual Data Testing
```javascript
// Add test data point
chart.addDataPoint({
    time: Math.floor(Date.now() / 1000),
    open: 1800,
    high: 1820,
    low: 1795,
    close: 1815,
    volume: 1000000
});
```

## 📋 Best Practices

### 1. Library Priority
Always include libraries in priority order for best user experience:
1. LightweightCharts first
2. Chart.js as fallback
3. TradingView for enterprise features

### 2. WebSocket Integration
```javascript
// Always check WebSocket status before creating charts
const wsManager = window.WebSocketManager;
if (wsManager && wsManager.getStatus().connected) {
    // Create chart with WebSocket integration
    const chart = createChartWithWebSocket();
} else {
    // Create chart without real-time features
    const chart = createBasicChart();
}
```

### 3. Error Handling
```javascript
try {
    const chart = window.UnifiedChartManagerFactory.createChart('my-chart', options);
    console.log('Chart created successfully');
} catch (error) {
    console.error('Chart creation failed:', error);
    // Provide fallback or user notification
}
```

### 4. Resource Management
```javascript
// Always cleanup charts when done
window.addEventListener('beforeunload', async () => {
    await window.UnifiedChartManagerFactory.destroyAll();
});
```

### 5. Responsive Design
```javascript
// Make charts responsive
const chart = window.UnifiedChartManagerFactory.createChart('responsive-chart', {
    width: null,              // Responsive width
    height: 400               // Fixed height
});

// Handle window resize
window.addEventListener('resize', () => {
    // Charts handle resize automatically
});
```

## 🔗 Integration with GoldGPT Features

### WebSocket Manager Integration
The UnifiedChartManager seamlessly integrates with the existing WebSocket system:

```javascript
// Use existing WebSocket manager
const wsManager = window.WebSocketManager;

// Create chart with automatic real-time updates
const chart = window.UnifiedChartManagerFactory.createChart('integrated-chart', {
    wsManager: wsManager,     // Automatic integration
    realtime: true           // Enable real-time features
});
```

### Dashboard Integration
Perfect integration with the GoldGPT dashboard:

```javascript
// Auto-detect dashboard environment
if (document.querySelector('.dashboard-container')) {
    // Initialize chart dashboard automatically
    window.goldGPTChartIntegration = new GoldGPTChartIntegration();
}
```

## 🎉 System Status

### ✅ Completed Features
- ✅ Multi-library support with automatic fallback
- ✅ Consistent API across all chart libraries
- ✅ Complete OHLC, candlestick, and line chart support
- ✅ Full timeframe switching (1m to 1d)
- ✅ WebSocket integration with real-time updates
- ✅ Connection management and error handling
- ✅ Performance optimization with data buffering
- ✅ Responsive design and theme support
- ✅ Factory pattern for multiple chart instances
- ✅ Comprehensive debugging and monitoring
- ✅ Demo page with full testing capabilities

### 🚀 Ready for Production
The UnifiedChartManager is **production-ready** and provides enterprise-level charting capabilities for your GoldGPT trading platform! 📊⚡

/**
 * UnifiedChartManager Integration Examples
 * Demonstrates how to use the UnifiedChartManager with WebSocket integration
 */

// =============================================================================
// BASIC USAGE EXAMPLES
// =============================================================================

/**
 * Example 1: Simple Chart Creation
 */
function createBasicChart() {
    // Create a basic candlestick chart
    const chartManager = window.UnifiedChartManagerFactory.createChart('price-chart', {
        chartType: 'candlestick',
        timeframe: '1h',
        theme: 'dark',
        height: 400,
        realtime: true
    });

    console.log('Basic chart created:', chartManager.getStatus());
    return chartManager;
}

/**
 * Example 2: Chart with WebSocket Integration
 */
function createRealtimeChart() {
    // Get WebSocket manager (assumes it's already initialized)
    const wsManager = window.WebSocketManager || window.WebSocketManagerFactory.getInstance('default');

    // Create chart with WebSocket integration
    const chartManager = window.UnifiedChartManagerFactory.createChart('realtime-chart', {
        chartType: 'candlestick',
        timeframe: '5m',
        theme: 'dark',
        height: 500,
        realtime: true,
        wsManager: wsManager, // Enable WebSocket integration
        maxDataPoints: 500,
        enableVolume: true,
        enableIndicators: true,
        debug: true
    });

    console.log('Real-time chart created:', chartManager.getStatus());
    return chartManager;
}

/**
 * Example 3: Multiple Chart Types
 */
function createMultipleCharts() {
    const wsManager = window.WebSocketManager;

    // Main candlestick chart
    const mainChart = window.UnifiedChartManagerFactory.createChart('main-chart', {
        chartType: 'candlestick',
        timeframe: '1h',
        wsManager: wsManager,
        height: 600,
        enableVolume: true
    });

    // Line chart for quick overview
    const overviewChart = window.UnifiedChartManagerFactory.createChart('overview-chart', {
        chartType: 'line',
        timeframe: '1d',
        wsManager: wsManager,
        height: 200,
        enableVolume: false
    });

    // OHLC chart for detailed analysis
    const detailChart = window.UnifiedChartManagerFactory.createChart('detail-chart', {
        chartType: 'ohlc',
        timeframe: '15m',
        wsManager: wsManager,
        height: 400,
        enableIndicators: true
    });

    return { mainChart, overviewChart, detailChart };
}

// =============================================================================
// ADVANCED USAGE EXAMPLES
// =============================================================================

/**
 * Example 4: Dynamic Chart Configuration
 */
class DynamicChartController {
    constructor(containerId, wsManager) {
        this.containerId = containerId;
        this.wsManager = wsManager;
        this.chartManager = null;
        this.currentSettings = {
            chartType: 'candlestick',
            timeframe: '1h',
            theme: 'dark'
        };

        this.initializeChart();
        this.setupEventListeners();
    }

    initializeChart() {
        this.chartManager = window.UnifiedChartManagerFactory.createChart(this.containerId, {
            ...this.currentSettings,
            wsManager: this.wsManager,
            height: 500,
            realtime: true,
            debug: true
        });
    }

    async changeChartType(newType) {
        if (this.currentSettings.chartType === newType) return;

        console.log(`Changing chart type to: ${newType}`);
        this.currentSettings.chartType = newType;
        await this.chartManager.setChartType(newType);
    }

    async changeTimeframe(newTimeframe) {
        if (this.currentSettings.timeframe === newTimeframe) return;

        console.log(`Changing timeframe to: ${newTimeframe}`);
        this.currentSettings.timeframe = newTimeframe;
        await this.chartManager.setTimeframe(newTimeframe);
    }

    async changeTheme(newTheme) {
        if (this.currentSettings.theme === newTheme) return;

        console.log(`Changing theme to: ${newTheme}`);
        this.currentSettings.theme = newTheme;
        
        // Recreate chart with new theme
        await this.chartManager.destroy();
        this.currentSettings.theme = newTheme;
        this.initializeChart();
    }

    setupEventListeners() {
        // Listen for chart type changes
        document.addEventListener('chartTypeChange', (event) => {
            this.changeChartType(event.detail.chartType);
        });

        // Listen for timeframe changes
        document.addEventListener('timeframeChange', (event) => {
            this.changeTimeframe(event.detail.timeframe);
        });

        // Listen for theme changes
        document.addEventListener('themeChange', (event) => {
            this.changeTheme(event.detail.theme);
        });
    }

    getStatus() {
        return {
            settings: this.currentSettings,
            chartStatus: this.chartManager ? this.chartManager.getStatus() : null
        };
    }
}

/**
 * Example 5: Chart Dashboard Integration
 */
class ChartDashboard {
    constructor(wsManager) {
        this.wsManager = wsManager;
        this.charts = {};
        this.activeChart = null;
        this.isInitialized = false;

        this.initialize();
    }

    async initialize() {
        console.log('🚀 Initializing Chart Dashboard...');

        // Create main trading chart
        this.charts.main = window.UnifiedChartManagerFactory.createChart('trading-chart', {
            chartType: 'candlestick',
            timeframe: '1h',
            wsManager: this.wsManager,
            height: 600,
            enableVolume: true,
            enableIndicators: true,
            theme: 'dark'
        });

        // Create mini overview chart
        this.charts.overview = window.UnifiedChartManagerFactory.createChart('overview-chart', {
            chartType: 'line',
            timeframe: '1d',
            wsManager: this.wsManager,
            height: 150,
            enableVolume: false,
            enableIndicators: false,
            theme: 'dark'
        });

        this.activeChart = this.charts.main;
        this.setupDashboardControls();
        this.isInitialized = true;

        console.log('✅ Chart Dashboard initialized');
    }

    setupDashboardControls() {
        // Chart type selector
        this.createChartTypeSelector();
        
        // Timeframe selector
        this.createTimeframeSelector();
        
        // Theme toggle
        this.createThemeToggle();
        
        // Chart library info
        this.createLibraryInfo();
    }

    createChartTypeSelector() {
        const selector = document.getElementById('chart-type-selector');
        if (!selector) return;

        const chartTypes = ['candlestick', 'ohlc', 'line'];
        
        selector.innerHTML = chartTypes.map(type => 
            `<button class="chart-type-btn" data-type="${type}">${type.toUpperCase()}</button>`
        ).join('');

        selector.addEventListener('click', (e) => {
            if (e.target.classList.contains('chart-type-btn')) {
                const chartType = e.target.dataset.type;
                this.changeChartType(chartType);
                
                // Update active button
                selector.querySelectorAll('.chart-type-btn').forEach(btn => btn.classList.remove('active'));
                e.target.classList.add('active');
            }
        });
    }

    createTimeframeSelector() {
        const selector = document.getElementById('timeframe-selector');
        if (!selector) return;

        const timeframes = ['1m', '5m', '15m', '1h', '4h', '1d'];
        
        selector.innerHTML = timeframes.map(tf => 
            `<button class="timeframe-btn" data-timeframe="${tf}">${tf}</button>`
        ).join('');

        selector.addEventListener('click', (e) => {
            if (e.target.classList.contains('timeframe-btn')) {
                const timeframe = e.target.dataset.timeframe;
                this.changeTimeframe(timeframe);
                
                // Update active button
                selector.querySelectorAll('.timeframe-btn').forEach(btn => btn.classList.remove('active'));
                e.target.classList.add('active');
            }
        });
    }

    createThemeToggle() {
        const toggle = document.getElementById('theme-toggle');
        if (!toggle) return;

        toggle.addEventListener('click', () => {
            const currentTheme = this.activeChart.options.theme;
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            this.changeTheme(newTheme);
        });
    }

    createLibraryInfo() {
        const info = document.getElementById('chart-library-info');
        if (!info) return;

        const status = this.activeChart.getStatus();
        info.innerHTML = `
            <div class="library-info">
                <strong>Active Library:</strong> ${status.activeLibrary}
                <br>
                <strong>Chart Type:</strong> ${status.chartType}
                <br>
                <strong>Timeframe:</strong> ${status.timeframe}
                <br>
                <strong>Data Points:</strong> ${status.dataPoints}
                <br>
                <strong>Real-time:</strong> ${status.isRealtime ? 'Enabled' : 'Disabled'}
            </div>
        `;
    }

    async changeChartType(chartType) {
        if (this.activeChart) {
            await this.activeChart.setChartType(chartType);
            this.updateLibraryInfo();
        }
    }

    async changeTimeframe(timeframe) {
        if (this.activeChart) {
            await this.activeChart.setTimeframe(timeframe);
            this.updateLibraryInfo();
        }
    }

    async changeTheme(theme) {
        // Change theme for all charts
        for (const [name, chart] of Object.entries(this.charts)) {
            if (chart) {
                chart.options.theme = theme;
                await chart.destroy();
                // Recreate chart with new theme would happen automatically
            }
        }
        this.updateLibraryInfo();
    }

    updateLibraryInfo() {
        setTimeout(() => this.createLibraryInfo(), 100);
    }

    getDashboardStatus() {
        return {
            isInitialized: this.isInitialized,
            charts: Object.keys(this.charts),
            activeChart: this.activeChart ? this.activeChart.getStatus() : null,
            wsConnected: this.wsManager ? this.wsManager.getStatus().connected : false
        };
    }
}

// =============================================================================
// INTEGRATION WITH GOLDGPT DASHBOARD
// =============================================================================

/**
 * GoldGPT Dashboard Chart Integration
 */
class GoldGPTChartIntegration {
    constructor() {
        this.wsManager = null;
        this.chartDashboard = null;
        this.isReady = false;

        this.initialize();
    }

    async initialize() {
        console.log('🚀 Initializing GoldGPT Chart Integration...');

        try {
            // Wait for WebSocket manager
            await this.waitForWebSocketManager();
            
            // Initialize chart dashboard
            this.chartDashboard = new ChartDashboard(this.wsManager);
            
            // Setup integration events
            this.setupIntegrationEvents();
            
            this.isReady = true;
            console.log('✅ GoldGPT Chart Integration ready');

        } catch (error) {
            console.error('❌ Error initializing GoldGPT Chart Integration:', error);
        }
    }

    async waitForWebSocketManager() {
        return new Promise((resolve, reject) => {
            const checkWebSocket = () => {
                if (window.WebSocketManager || window.wsManager) {
                    this.wsManager = window.WebSocketManager || window.wsManager;
                    console.log('📡 WebSocket manager found');
                    resolve();
                } else if (window.WebSocketManagerFactory) {
                    this.wsManager = window.WebSocketManagerFactory.getInstance('default');
                    console.log('📡 WebSocket manager created');
                    resolve();
                } else {
                    console.log('⏳ Waiting for WebSocket manager...');
                    setTimeout(checkWebSocket, 1000);
                }
            };

            checkWebSocket();
            
            // Timeout after 10 seconds
            setTimeout(() => {
                if (!this.wsManager) {
                    reject(new Error('WebSocket manager not available after 10 seconds'));
                }
            }, 10000);
        });
    }

    setupIntegrationEvents() {
        // Listen for WebSocket connection changes
        if (this.wsManager) {
            this.wsManager.subscribe('connectionStateChanged', (data) => {
                console.log(`📊 Chart integration: WebSocket ${data.state}`);
                this.updateConnectionStatus(data.state);
            });
        }

        // Listen for dashboard events
        document.addEventListener('dashboardReady', () => {
            console.log('📊 Dashboard ready, updating chart integration');
            this.updateChartDisplay();
        });
    }

    updateConnectionStatus(state) {
        const statusElement = document.getElementById('chart-connection-status');
        if (statusElement) {
            statusElement.textContent = `Charts: ${state}`;
            statusElement.className = `connection-status ${state}`;
        }
    }

    updateChartDisplay() {
        // Update chart display based on dashboard state
        if (this.chartDashboard && this.chartDashboard.isInitialized) {
            const status = this.chartDashboard.getDashboardStatus();
            console.log('📊 Chart dashboard status:', status);
        }
    }

    getIntegrationStatus() {
        return {
            isReady: this.isReady,
            wsManager: !!this.wsManager,
            chartDashboard: this.chartDashboard ? this.chartDashboard.getDashboardStatus() : null
        };
    }
}

// =============================================================================
// AUTO-INITIALIZATION AND EXAMPLES
// =============================================================================

// Auto-initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    console.log('📊 Chart integration examples loaded');

    // Global functions for easy testing
    window.createBasicChart = createBasicChart;
    window.createRealtimeChart = createRealtimeChart;
    window.createMultipleCharts = createMultipleCharts;
    window.DynamicChartController = DynamicChartController;
    window.ChartDashboard = ChartDashboard;
    window.GoldGPTChartIntegration = GoldGPTChartIntegration;

    // Initialize GoldGPT integration if in dashboard
    if (document.getElementById('trading-chart') || document.querySelector('.dashboard-container')) {
        console.log('📊 Dashboard detected, initializing chart integration...');
        window.goldGPTChartIntegration = new GoldGPTChartIntegration();
    }
});

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        createBasicChart,
        createRealtimeChart,
        createMultipleCharts,
        DynamicChartController,
        ChartDashboard,
        GoldGPTChartIntegration
    };
}

console.log('📊 UnifiedChartManager integration examples ready');
console.log('💡 Available functions:');
console.log('  - createBasicChart()');
console.log('  - createRealtimeChart()');
console.log('  - createMultipleCharts()');
console.log('  - new DynamicChartController(containerId, wsManager)');
console.log('  - new ChartDashboard(wsManager)');
console.log('  - new GoldGPTChartIntegration()');

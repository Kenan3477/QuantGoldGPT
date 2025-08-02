/**
 * UnifiedChartManager - Unified charting system for GoldGPT
 * 
 * Features:
 * - Multi-library support: LightweightCharts, Chart.js, TradingView (in priority order)
 * - Consistent API across all chart libraries
 * - OHLC, candlestick, and line chart types
 * - Timeframe switching (1m, 5m, 15m, 1h, 4h, 1d)
 * - WebSocket integration for real-time updates
 * - Automatic fallback to available libraries
 * - Performance optimization for real-time data
 * - Robust error handling and recovery
 */

/**
 * Custom error class for chart-related errors
 */
class ChartError extends Error {
    constructor(code, message, context = {}) {
        super(message);
        this.name = 'ChartError';
        this.code = code;
        this.context = context;
        this.timestamp = new Date();
        this.recoverable = this.isRecoverable(code);
    }

    isRecoverable(code) {
        const recoverableErrors = [
            'NETWORK_ERROR',
            'DATA_LOAD_FAILED',
            'WEBSOCKET_DISCONNECTED',
            'LIBRARY_LOAD_FAILED',
            'CHART_RENDER_FAILED',
            'REALTIME_UPDATE_FAILED'
        ];
        return recoverableErrors.includes(code);
    }

    toJSON() {
        return {
            name: this.name,
            code: this.code,
            message: this.message,
            context: this.context,
            timestamp: this.timestamp,
            recoverable: this.recoverable,
            stack: this.stack
        };
    }
}

/**
 * Error display utility
 */
class ChartErrorDisplay {
    static createErrorElement(error, containerId) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'chart-error-display';
        errorDiv.innerHTML = `
            <div class="error-content">
                <div class="error-icon">⚠️</div>
                <div class="error-message">
                    <h4>Chart Error</h4>
                    <p>${error.message}</p>
                    <small>Error Code: ${error.code}</small>
                </div>
                <div class="error-actions">
                    ${error.recoverable ? '<button class="retry-btn" onclick="window.UnifiedChartManagerFactory.retryChart(\'' + containerId + '\')">Retry</button>' : ''}
                    <button class="details-btn" onclick="this.parentElement.parentElement.querySelector(\'.error-details\').style.display = this.parentElement.parentElement.querySelector(\'.error-details\').style.display === \'none\' ? \'block\' : \'none\'">Details</button>
                </div>
                <div class="error-details" style="display: none;">
                    <pre>${JSON.stringify(error.context, null, 2)}</pre>
                </div>
            </div>
        `;
        return errorDiv;
    }

    static getErrorStyles() {
        return `
            .chart-error-display {
                display: flex;
                align-items: center;
                justify-content: center;
                min-height: 200px;
                background: rgba(255, 0, 0, 0.1);
                border: 2px solid rgba(255, 0, 0, 0.3);
                border-radius: 8px;
                padding: 20px;
                margin: 10px;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }
            .chart-error-display .error-content {
                text-align: center;
                max-width: 400px;
            }
            .chart-error-display .error-icon {
                font-size: 48px;
                margin-bottom: 15px;
            }
            .chart-error-display .error-message h4 {
                color: #ff4444;
                margin: 0 0 10px 0;
                font-size: 18px;
            }
            .chart-error-display .error-message p {
                color: #666;
                margin: 0 0 10px 0;
                font-size: 14px;
            }
            .chart-error-display .error-message small {
                color: #999;
                font-size: 12px;
            }
            .chart-error-display .error-actions {
                margin: 20px 0;
                display: flex;
                gap: 10px;
                justify-content: center;
            }
            .chart-error-display button {
                padding: 8px 16px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 12px;
                transition: all 0.2s ease;
            }
            .chart-error-display .retry-btn {
                background: #4CAF50;
                color: white;
            }
            .chart-error-display .retry-btn:hover {
                background: #45a049;
            }
            .chart-error-display .details-btn {
                background: #2196F3;
                color: white;
            }
            .chart-error-display .details-btn:hover {
                background: #1976D2;
            }
            .chart-error-display .error-details {
                background: rgba(0, 0, 0, 0.1);
                border-radius: 4px;
                padding: 10px;
                margin-top: 15px;
                text-align: left;
            }
            .chart-error-display .error-details pre {
                margin: 0;
                font-size: 11px;
                color: #666;
                white-space: pre-wrap;
                word-wrap: break-word;
            }
        `;
    }
}

class UnifiedChartManager {
    constructor(containerId, options = {}) {
        this.containerId = containerId;
        this.container = document.getElementById(containerId);
        
        // Error handling state
        this.errors = [];
        this.lastError = null;
        this.errorCallback = options.onError || null;
        this.maxRetries = options.maxRetries || 3;
        this.retryAttempts = 0;
        this.isInitializing = false;
        this.hasRecoveryMode = options.enableRecovery !== false;
        
        if (!this.container) {
            const error = new ChartError('CONTAINER_NOT_FOUND', `Chart container '${containerId}' not found`, {
                containerId,
                availableContainers: Array.from(document.querySelectorAll('[id]')).map(el => el.id).filter(Boolean)
            });
            this.handleError(error);
            throw error;
        }

        this.options = {
            chartType: options.chartType || 'candlestick', // 'candlestick', 'ohlc', 'line'
            timeframe: options.timeframe || '1h', // '1m', '5m', '15m', '1h', '4h', '1d'
            theme: options.theme || 'dark', // 'light', 'dark'
            height: options.height || 400,
            width: options.width || null, // null = responsive
            realtime: options.realtime !== false, // Enable real-time updates
            maxDataPoints: options.maxDataPoints || 1000,
            enableVolume: options.enableVolume !== false,
            enableIndicators: options.enableIndicators !== false,
            wsManager: options.wsManager || null, // WebSocket manager instance
            debug: options.debug || false,
            fallbackToBasicChart: options.fallbackToBasicChart !== false,
            showErrorMessages: options.showErrorMessages !== false,
            enableAutoRecovery: options.enableAutoRecovery !== false,
            recoveryInterval: options.recoveryInterval || 30000, // 30 seconds
            ...options
        };

        // Chart state
        this.activeLibrary = null;
        this.chart = null;
        this.series = null;
        this.volumeSeries = null;
        this.data = [];
        this.indicators = {};
        this.isRealtime = false;
        this.lastUpdate = null;
        this.initializationPromise = null;

        // WebSocket integration
        this.wsSubscriptions = [];
        this.priceBuffer = [];
        this.bufferTimeout = null;

        // Recovery system
        this.recoveryTimeout = null;
        this.healthCheckInterval = null;
        this.connectionStatus = 'initializing';

        // Library availability check with error handling
        try {
            this.availableLibraries = this.checkAvailableLibraries();
        } catch (error) {
            this.handleError(new ChartError('LIBRARY_CHECK_FAILED', 'Failed to check available chart libraries', { error }));
            this.availableLibraries = { lightweightCharts: false, chartjs: false, tradingview: false };
        }
        
        this.log('UnifiedChartManager initialized', {
            container: containerId,
            options: this.options,
            availableLibraries: this.availableLibraries
        });

        // Show loading indicator
        this.showLoadingIndicator();

        // Initialize chart with error handling
        this.initializeChartWithRecovery();
        
        // Setup WebSocket if provided
        if (this.options.wsManager) {
            this.setupWebSocketIntegrationWithRecovery();
        }

        // Setup health monitoring
        if (this.options.enableAutoRecovery) {
            this.setupHealthMonitoring();
        }
    }

    /**
     * Handle errors with logging, user notification, and recovery attempts
     */
    handleError(error, context = {}) {
        // Convert to ChartError if it's not already
        if (!(error instanceof ChartError)) {
            error = new ChartError('UNKNOWN_ERROR', error.message || 'Unknown error occurred', {
                originalError: error,
                ...context
            });
        }

        // Add to error history
        this.errors.push(error);
        this.lastError = error;

        // Update connection status
        this.connectionStatus = 'error';

        // Log error
        this.logError(error);

        // Call error callback if provided
        if (this.errorCallback) {
            try {
                this.errorCallback(error);
            } catch (callbackError) {
                console.error('Error callback failed:', callbackError);
            }
        }

        // Show user-friendly error message
        if (this.options.showErrorMessages) {
            this.showErrorMessage(error);
        }

        // Attempt recovery if error is recoverable
        if (error.recoverable && this.hasRecoveryMode && this.retryAttempts < this.maxRetries) {
            this.scheduleRecovery(error);
        } else if (!error.recoverable) {
            this.showFatalError(error);
        }

        return error;
    }

    /**
     * Log error with detailed information
     */
    logError(error) {
        const errorInfo = {
            code: error.code,
            message: error.message,
            context: error.context,
            timestamp: error.timestamp,
            containerId: this.containerId,
            activeLibrary: this.activeLibrary,
            availableLibraries: this.availableLibraries,
            chartState: {
                hasChart: !!this.chart,
                hasSeries: !!this.series,
                dataPoints: this.data.length,
                isRealtime: this.isRealtime
            },
            retryAttempts: this.retryAttempts
        };

        if (this.options.debug) {
            console.group('🚨 UnifiedChartManager Error');
            console.error('Error:', error);
            console.table(errorInfo);
            console.trace('Stack trace');
            console.groupEnd();
        } else {
            console.error(`[UnifiedChartManager] ${error.code}: ${error.message}`, errorInfo);
        }

        // Send to external error tracking if available
        if (window.errorTracker && typeof window.errorTracker.logError === 'function') {
            window.errorTracker.logError('UnifiedChartManager', errorInfo);
        }
    }

    /**
     * Show user-friendly error message
     */
    showErrorMessage(error) {
        // Remove existing error displays
        this.hideLoadingIndicator();
        this.clearErrorDisplays();

        // Show connection indicator
        this.updateConnectionIndicator('error', error.message);

        // Create error display if container is empty or only has basic content
        if (!this.chart || error.code === 'FATAL_ERROR') {
            const errorElement = ChartErrorDisplay.createErrorElement(error, this.containerId);
            this.container.appendChild(errorElement);
        }
    }

    /**
     * Show fatal error that cannot be recovered
     */
    showFatalError(error) {
        this.connectionStatus = 'fatal';
        this.clearContainer();
        
        const fatalError = new ChartError('FATAL_ERROR', 'Chart initialization failed completely. Please refresh the page.', {
            originalError: error,
            availableLibraries: this.availableLibraries,
            troubleshooting: [
                'Check if chart libraries are properly loaded',
                'Verify container element exists',
                'Check browser console for additional errors',
                'Try refreshing the page'
            ]
        });

        this.showErrorMessage(fatalError);
        this.log('Fatal error occurred, chart cannot be initialized');
    }

    /**
     * Schedule recovery attempt
     */
    scheduleRecovery(error) {
        if (this.recoveryTimeout) {
            clearTimeout(this.recoveryTimeout);
        }

        const delay = Math.min(1000 * Math.pow(2, this.retryAttempts), this.options.recoveryInterval);
        
        this.log(`Scheduling recovery attempt ${this.retryAttempts + 1}/${this.maxRetries} in ${delay}ms`);
        
        this.updateConnectionIndicator('recovering', `Retrying in ${Math.ceil(delay / 1000)}s...`);

        this.recoveryTimeout = setTimeout(async () => {
            this.retryAttempts++;
            await this.attemptRecovery(error);
        }, delay);
    }

    /**
     * Attempt to recover from error
     */
    async attemptRecovery(originalError) {
        this.log(`Attempting recovery (${this.retryAttempts}/${this.maxRetries})...`);
        
        try {
            this.updateConnectionIndicator('recovering', 'Attempting recovery...');

            // Clear current state
            await this.cleanup(false);

            // Re-check available libraries
            this.availableLibraries = this.checkAvailableLibraries();

            // Try to reinitialize
            await this.initializeChartWithRecovery();

            // If successful, reset retry count
            this.retryAttempts = 0;
            this.connectionStatus = 'connected';
            this.updateConnectionIndicator('connected', 'Chart recovered');
            this.clearErrorDisplays();
            
            this.log('✅ Recovery successful');

        } catch (recoveryError) {
            this.log('❌ Recovery failed:', recoveryError);
            
            const wrappedError = new ChartError('RECOVERY_FAILED', 'Recovery attempt failed', {
                originalError,
                recoveryError,
                attempt: this.retryAttempts
            });

            this.handleError(wrappedError);
        }
    }

    /**
     * Initialize chart with comprehensive error handling
     */
    async initializeChartWithRecovery() {
        if (this.isInitializing) {
            return this.initializationPromise;
        }

        this.isInitializing = true;
        this.updateConnectionIndicator('connecting', 'Initializing chart...');

        this.initializationPromise = this.performChartInitialization();
        
        try {
            await this.initializationPromise;
        } finally {
            this.isInitializing = false;
        }

        return this.initializationPromise;
    }

    /**
     * Perform the actual chart initialization with error handling
     */
    async performChartInitialization() {
        const libraries = ['lightweightCharts', 'chartjs', 'tradingview'];
        let initializationErrors = [];
        
        for (const lib of libraries) {
            if (!this.availableLibraries[lib]) {
                this.log(`Skipping ${lib} - not available`);
                continue;
            }

            try {
                this.log(`Attempting to initialize with ${lib}...`);
                await this.initializeLibraryWithErrorHandling(lib);
                this.activeLibrary = lib;
                this.connectionStatus = 'connected';
                this.log(`✅ Successfully initialized with ${lib}`);
                
                // Load initial data
                await this.loadInitialDataWithErrorHandling();
                
                // Hide loading indicator and show success
                this.hideLoadingIndicator();
                this.updateConnectionIndicator('connected', `Using ${lib}`);
                
                return;
                
            } catch (error) {
                this.log(`❌ Failed to initialize ${lib}:`, error);
                initializationErrors.push({ library: lib, error });
                
                // Clean up any partial initialization
                await this.cleanup(false);
            }
        }

        // If we get here, all libraries failed
        const error = new ChartError('NO_LIBRARY_AVAILABLE', 
            'No charting library could be initialized', {
            availableLibraries: this.availableLibraries,
            initializationErrors,
            suggestions: [
                'Check if chart libraries are properly loaded',
                'Verify network connectivity',
                'Try refreshing the page'
            ]
        });

        throw error;
    }
    /**
     * Check which charting libraries are available with error handling
     */
    checkAvailableLibraries() {
        const libraries = {};

        try {
            libraries.lightweightCharts = typeof LightweightCharts !== 'undefined';
        } catch (error) {
            libraries.lightweightCharts = false;
            this.log('Error checking LightweightCharts availability:', error);
        }

        try {
            libraries.chartjs = typeof Chart !== 'undefined';
        } catch (error) {
            libraries.chartjs = false;
            this.log('Error checking Chart.js availability:', error);
        }

        try {
            libraries.tradingview = typeof TradingView !== 'undefined' || 
                        document.querySelector('script[src*="tradingview"]') !== null;
        } catch (error) {
            libraries.tradingview = false;
            this.log('Error checking TradingView availability:', error);
        }

        this.log('Library availability check:', libraries);
        return libraries;
    }

    /**
     * Initialize specific charting library with error handling
     */
    async initializeLibraryWithErrorHandling(library) {
        try {
            switch (library) {
                case 'lightweightCharts':
                    await this.initializeLightweightChartsWithErrorHandling();
                    break;
                case 'chartjs':
                    await this.initializeChartJSWithErrorHandling();
                    break;
                case 'tradingview':
                    await this.initializeTradingViewWithErrorHandling();
                    break;
                default:
                    throw new ChartError('UNKNOWN_LIBRARY', `Unknown library: ${library}`, { library });
            }
        } catch (error) {
            throw new ChartError('LIBRARY_INITIALIZATION_FAILED', 
                `Failed to initialize ${library}`, { library, originalError: error });
        }
    }

    /**
     * Initialize LightweightCharts with comprehensive error handling
     */
    async initializeLightweightChartsWithErrorHandling() {
        this.log('Initializing LightweightCharts...');

        if (typeof LightweightCharts === 'undefined') {
            throw new ChartError('LIBRARY_NOT_AVAILABLE', 'LightweightCharts library not loaded');
        }

        try {
            // Validate container
            if (!this.container || !this.container.isConnected) {
                throw new ChartError('CONTAINER_INVALID', 'Chart container is not valid or not in DOM');
            }

            const chartOptions = {
                width: this.options.width || this.container.clientWidth,
                height: this.options.height,
                layout: {
                    backgroundColor: this.options.theme === 'dark' ? '#131722' : '#FFFFFF',
                    textColor: this.options.theme === 'dark' ? '#d1d4dc' : '#191919',
                },
                grid: {
                    vertLines: { color: this.options.theme === 'dark' ? '#334158' : '#e1e3e6' },
                    horzLines: { color: this.options.theme === 'dark' ? '#334158' : '#e1e3e6' },
                },
                crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
                rightPriceScale: { borderColor: this.options.theme === 'dark' ? '#485c7b' : '#cccccc' },
                timeScale: { borderColor: this.options.theme === 'dark' ? '#485c7b' : '#cccccc' },
            };

            this.chart = LightweightCharts.createChart(this.container, chartOptions);

            if (!this.chart) {
                throw new ChartError('CHART_CREATION_FAILED', 'Failed to create LightweightCharts instance');
            }

            // Create appropriate series based on chart type
            try {
                if (this.options.chartType === 'candlestick') {
                    this.series = this.chart.addCandlestickSeries({
                        upColor: '#26a69a',
                        downColor: '#ef5350',
                        borderVisible: false,
                        wickUpColor: '#26a69a',
                        wickDownColor: '#ef5350',
                    });
                } else if (this.options.chartType === 'ohlc') {
                    this.series = this.chart.addCandlestickSeries({
                        upColor: '#26a69a',
                        downColor: '#ef5350',
                        borderVisible: true,
                        wickUpColor: '#26a69a',
                        wickDownColor: '#ef5350',
                    });
                } else {
                    this.series = this.chart.addLineSeries({
                        color: '#2196F3',
                        lineWidth: 2,
                    });
                }

                if (!this.series) {
                    throw new ChartError('SERIES_CREATION_FAILED', 'Failed to create chart series');
                }
            } catch (error) {
                throw new ChartError('SERIES_SETUP_FAILED', 'Failed to setup chart series', { error });
            }

            // Add volume series if enabled
            if (this.options.enableVolume && this.options.chartType !== 'line') {
                try {
                    this.volumeSeries = this.chart.addHistogramSeries({
                        color: '#26a69a',
                        priceFormat: { type: 'volume' },
                        priceScaleId: 'volume',
                    });
                    
                    this.chart.priceScale('volume').applyOptions({
                        scaleMargins: { top: 0.8, bottom: 0 },
                    });
                } catch (error) {
                    this.log('Warning: Volume series setup failed:', error);
                    // Volume is optional, don't fail the entire initialization
                }
            }

            // Handle resize with error handling
            this.setupResizeHandlerWithErrorHandling();

            this.log('✅ LightweightCharts initialized successfully');

        } catch (error) {
            // Clean up any partial initialization
            if (this.chart) {
                try {
                    this.chart.remove();
                } catch (cleanupError) {
                    this.log('Error during cleanup:', cleanupError);
                }
                this.chart = null;
                this.series = null;
                this.volumeSeries = null;
            }
            throw error;
        }
    }

    /**
     * Initialize Chart.js with error handling
     */
    async initializeChartJSWithErrorHandling() {
        this.log('Initializing Chart.js...');

        if (typeof Chart === 'undefined') {
            throw new ChartError('LIBRARY_NOT_AVAILABLE', 'Chart.js library not loaded');
        }

        try {
            // Clear container and create canvas
            this.container.innerHTML = '';
            const canvas = document.createElement('canvas');
            canvas.width = this.options.width || this.container.clientWidth;
            canvas.height = this.options.height;
            this.container.appendChild(canvas);

            const ctx = canvas.getContext('2d');
            if (!ctx) {
                throw new ChartError('CANVAS_CONTEXT_FAILED', 'Failed to get canvas 2D context');
            }

            const chartConfig = {
                type: this.getChartJSType(),
                data: {
                    datasets: [{
                        label: 'Gold Price',
                        data: [],
                        borderColor: '#2196F3',
                        backgroundColor: this.options.chartType === 'line' ? 'rgba(33, 150, 243, 0.1)' : undefined,
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            type: 'time',
                            time: { unit: this.getTimeUnit() },
                            grid: { color: this.options.theme === 'dark' ? '#334158' : '#e1e3e6' },
                            ticks: { color: this.options.theme === 'dark' ? '#d1d4dc' : '#191919' }
                        },
                        y: {
                            grid: { color: this.options.theme === 'dark' ? '#334158' : '#e1e3e6' },
                            ticks: { color: this.options.theme === 'dark' ? '#d1d4dc' : '#191919' }
                        }
                    },
                    plugins: {
                        legend: { display: false },
                        tooltip: { mode: 'index', intersect: false }
                    },
                    backgroundColor: this.options.theme === 'dark' ? '#131722' : '#FFFFFF',
                    onResize: (chart, size) => {
                        this.handleChartResize(size);
                    }
                }
            };

            this.chart = new Chart(ctx, chartConfig);
            this.series = this.chart.data.datasets[0];

            if (!this.chart) {
                throw new ChartError('CHART_CREATION_FAILED', 'Failed to create Chart.js instance');
            }

            this.log('✅ Chart.js initialized successfully');

        } catch (error) {
            throw new ChartError('CHARTJS_INITIALIZATION_FAILED', 'Chart.js initialization failed', { error });
        }
    }

    /**
     * Initialize TradingView with error handling
     */
    async initializeTradingViewWithErrorHandling() {
        this.log('Initializing TradingView...');

        if (typeof TradingView === 'undefined') {
            throw new ChartError('LIBRARY_NOT_AVAILABLE', 'TradingView library not loaded');
        }

        try {
            // Clear container
            this.container.innerHTML = '';

            const widgetConfig = {
                container_id: this.containerId,
                width: this.options.width || '100%',
                height: this.options.height,
                symbol: 'GOLD',
                interval: this.getTradingViewInterval(),
                timezone: 'Etc/UTC',
                theme: this.options.theme,
                style: this.getTradingViewStyle(),
                locale: 'en',
                toolbar_bg: this.options.theme === 'dark' ? '#131722' : '#FFFFFF',
                enable_publishing: false,
                hide_top_toolbar: true,
                hide_legend: true,
                save_image: false,
                studies: this.options.enableIndicators ? ['RSI', 'MACD'] : [],
                overrides: {
                    'paneProperties.background': this.options.theme === 'dark' ? '#131722' : '#FFFFFF',
                    'paneProperties.vertGridProperties.color': this.options.theme === 'dark' ? '#334158' : '#e1e3e6',
                    'paneProperties.horzGridProperties.color': this.options.theme === 'dark' ? '#334158' : '#e1e3e6',
                }
            };

            this.chart = new TradingView.widget(widgetConfig);

            // TradingView widget initialization is async, wait for it to load
            await new Promise((resolve, reject) => {
                const timeout = setTimeout(() => {
                    reject(new ChartError('TRADINGVIEW_TIMEOUT', 'TradingView widget failed to load within timeout'));
                }, 10000);

                this.chart.onChartReady(() => {
                    clearTimeout(timeout);
                    resolve();
                });
            });

            this.log('✅ TradingView initialized successfully');

        } catch (error) {
            throw new ChartError('TRADINGVIEW_INITIALIZATION_FAILED', 'TradingView initialization failed', { error });
        }
    }

    /**
     * Load initial chart data with error handling
     */
    async loadInitialDataWithErrorHandling() {
        try {
            this.log('Loading initial chart data...');
            
            // Try to load from API first, fallback to sample data
            let data;
            try {
                data = await this.loadDataFromAPI();
            } catch (apiError) {
                this.log('API data load failed, using sample data:', apiError);
                data = this.generateSampleData();
            }
            
            await this.setDataWithErrorHandling(data);
            
            this.log('Initial data loaded successfully');
        } catch (error) {
            throw new ChartError('DATA_LOAD_FAILED', 'Failed to load initial chart data', { error });
        }
    }

    /**
     * Set chart data with error handling
     */
    async setDataWithErrorHandling(data) {
        if (!Array.isArray(data) || data.length === 0) {
            throw new ChartError('INVALID_DATA', 'Chart data is invalid or empty', { data });
        }

        try {
            this.data = data;
            
            switch (this.activeLibrary) {
                case 'lightweightCharts':
                    await this.setLightweightChartsDataWithErrorHandling(this.data);
                    break;
                case 'chartjs':
                    await this.setChartJSDataWithErrorHandling(this.data);
                    break;
                case 'tradingview':
                    // TradingView handles data differently
                    break;
            }

            this.lastUpdate = new Date();
        } catch (error) {
            throw new ChartError('DATA_SET_FAILED', 'Failed to set chart data', { error, dataLength: data.length });
        }
    }

    /**
     * Set data for LightweightCharts with error handling
     */
    async setLightweightChartsDataWithErrorHandling(data) {
        if (!this.series) {
            throw new ChartError('SERIES_NOT_AVAILABLE', 'Chart series not available');
        }

        try {
            const formattedData = data.map((item, index) => {
                try {
                    if (this.options.chartType === 'line') {
                        return {
                            time: item.time || item.timestamp,
                            value: item.close || item.price || item.value
                        };
                    } else {
                        return {
                            time: item.time || item.timestamp,
                            open: item.open,
                            high: item.high,
                            low: item.low,
                            close: item.close
                        };
                    }
                } catch (itemError) {
                    this.log(`Error formatting data item ${index}:`, itemError);
                    return null;
                }
            }).filter(item => item !== null);

            if (formattedData.length === 0) {
                throw new ChartError('DATA_FORMAT_FAILED', 'No valid data points after formatting');
            }

            this.series.setData(formattedData);

            if (this.volumeSeries && data[0]?.volume !== undefined) {
                try {
                    const volumeData = data.map(item => ({
                        time: item.time || item.timestamp,
                        value: item.volume,
                        color: item.close >= item.open ? '#26a69a' : '#ef5350'
                    })).filter(item => item.value !== undefined);
                    
                    this.volumeSeries.setData(volumeData);
                } catch (volumeError) {
                    this.log('Warning: Volume data setup failed:', volumeError);
                    // Volume is optional, don't fail the entire operation
                }
            }
        } catch (error) {
            throw new ChartError('LIGHTWEIGHT_DATA_SET_FAILED', 'Failed to set LightweightCharts data', { error });
        }
    }

    /**
     * Set data for Chart.js with error handling
     */
    async setChartJSDataWithErrorHandling(data) {
        if (!this.chart) {
            throw new ChartError('CHART_NOT_AVAILABLE', 'Chart.js instance not available');
        }

        try {
            const formattedData = data.map((item, index) => {
                try {
                    return {
                        x: item.time || item.timestamp,
                        y: this.options.chartType === 'line' ? 
                            (item.close || item.price || item.value) : item,
                        o: item.open,
                        h: item.high,
                        l: item.low,
                        c: item.close
                    };
                } catch (itemError) {
                    this.log(`Error formatting Chart.js data item ${index}:`, itemError);
                    return null;
                }
            }).filter(item => item !== null);

            if (formattedData.length === 0) {
                throw new ChartError('DATA_FORMAT_FAILED', 'No valid data points after formatting');
            }

            this.chart.data.datasets[0].data = formattedData;
            this.chart.update('none');
        } catch (error) {
            throw new ChartError('CHARTJS_DATA_SET_FAILED', 'Failed to set Chart.js data', { error });
        }
    }

    /**
     * Setup WebSocket integration with error handling
     */
    setupWebSocketIntegrationWithRecovery() {
        if (!this.options.wsManager) return;

        this.log('Setting up WebSocket integration...');

        try {
            // Subscribe to price updates with error handling
            const priceUnsubscribe = this.options.wsManager.subscribe('priceUpdate', (data) => {
                try {
                    this.handleWebSocketPriceUpdate(data);
                } catch (error) {
                    this.handleError(new ChartError('WEBSOCKET_UPDATE_FAILED', 'WebSocket price update failed', { error, data }));
                }
            });

            this.wsSubscriptions.push(priceUnsubscribe);

            // Monitor WebSocket connection status
            if (this.options.wsManager.onConnectionChange) {
                const connectionUnsubscribe = this.options.wsManager.onConnectionChange((status) => {
                    this.handleWebSocketConnectionChange(status);
                });
                this.wsSubscriptions.push(connectionUnsubscribe);
            }

            // Enable real-time mode
            this.isRealtime = true;
            this.log('WebSocket integration setup complete');

        } catch (error) {
            this.handleError(new ChartError('WEBSOCKET_SETUP_FAILED', 'WebSocket integration setup failed', { error }));
        }
    }

    /**
     * Handle WebSocket connection status changes
     */
    handleWebSocketConnectionChange(status) {
        this.log('WebSocket connection status changed:', status);
        
        switch (status) {
            case 'connected':
                this.updateConnectionIndicator('connected', 'Real-time updates active');
                break;
            case 'connecting':
                this.updateConnectionIndicator('connecting', 'Connecting to real-time data...');
                break;
            case 'disconnected':
                this.updateConnectionIndicator('warning', 'Real-time updates paused');
                this.handleError(new ChartError('WEBSOCKET_DISCONNECTED', 'WebSocket connection lost', { status }));
                break;
            case 'error':
                this.updateConnectionIndicator('error', 'Real-time connection error');
                this.handleError(new ChartError('WEBSOCKET_ERROR', 'WebSocket connection error', { status }));
                break;
        }
    }

    /**
     * Setup health monitoring system
     */
    setupHealthMonitoring() {
        if (this.healthCheckInterval) {
            clearInterval(this.healthCheckInterval);
        }

        this.healthCheckInterval = setInterval(() => {
            this.performHealthCheck();
        }, this.options.recoveryInterval);

        this.log('Health monitoring enabled');
    }

    /**
     * Perform health check
     */
    performHealthCheck() {
        try {
            const health = {
                hasChart: !!this.chart,
                hasSeries: !!this.series,
                hasData: this.data.length > 0,
                isConnected: this.connectionStatus === 'connected',
                lastUpdate: this.lastUpdate,
                timeSinceLastUpdate: this.lastUpdate ? Date.now() - this.lastUpdate.getTime() : Infinity
            };

            // Check if chart is still responsive
            if (health.hasChart && this.activeLibrary === 'lightweightCharts') {
                try {
                    // Try to get chart time scale - this will fail if chart is broken
                    this.chart.timeScale();
                } catch (error) {
                    health.isConnected = false;
                    this.handleError(new ChartError('CHART_UNRESPONSIVE', 'Chart is no longer responsive', { health, error }));
                    return;
                }
            }

            // Check if data is stale (no updates for too long)
            if (this.isRealtime && health.timeSinceLastUpdate > 300000) { // 5 minutes
                this.handleError(new ChartError('DATA_STALE', 'Chart data appears to be stale', { health }));
            }

            this.log('Health check passed', health);

        } catch (error) {
            this.handleError(new ChartError('HEALTH_CHECK_FAILED', 'Health check failed', { error }));
        }
    }

    /**
     * Show loading indicator
     */
    showLoadingIndicator() {
        this.clearContainer();
        
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'chart-loading-indicator';
        loadingDiv.innerHTML = `
            <div class="loading-content">
                <div class="loading-spinner"></div>
                <div class="loading-text">Loading Chart...</div>
                <div class="loading-subtext">Initializing charting libraries</div>
            </div>
        `;
        
        this.container.appendChild(loadingDiv);
        
        // Add styles if not already present
        if (!document.querySelector('#chart-loading-styles')) {
            const style = document.createElement('style');
            style.id = 'chart-loading-styles';
            style.textContent = this.getLoadingStyles();
            document.head.appendChild(style);
        }
    }

    /**
     * Hide loading indicator
     */
    hideLoadingIndicator() {
        const loadingIndicator = this.container.querySelector('.chart-loading-indicator');
        if (loadingIndicator) {
            loadingIndicator.remove();
        }
    }

    /**
     * Update connection status indicator
     */
    updateConnectionIndicator(status, message = '') {
        let indicator = this.container.querySelector('.chart-connection-indicator');
        
        if (!indicator) {
            indicator = document.createElement('div');
            indicator.className = 'chart-connection-indicator';
            this.container.appendChild(indicator);
        }

        const statusIcons = {
            connecting: '🔄',
            connected: '🟢',
            warning: '🟡',
            error: '🔴',
            recovering: '⚠️',
            fatal: '💥'
        };

        indicator.innerHTML = `${statusIcons[status] || '❓'} ${message}`;
        indicator.className = `chart-connection-indicator status-${status}`;
    }

    /**
     * Clear error displays
     */
    clearErrorDisplays() {
        const errorDisplays = this.container.querySelectorAll('.chart-error-display');
        errorDisplays.forEach(display => display.remove());
    }

    /**
     * Clear container
     */
    clearContainer() {
        this.container.innerHTML = '';
    }

    /**
     * Cleanup with error handling
     */
    async cleanup(destroyChart = true) {
        try {
            this.log('Performing cleanup...');

            // Clear timers
            if (this.bufferTimeout) {
                clearTimeout(this.bufferTimeout);
                this.bufferTimeout = null;
            }

            if (this.recoveryTimeout) {
                clearTimeout(this.recoveryTimeout);
                this.recoveryTimeout = null;
            }

            if (this.healthCheckInterval) {
                clearInterval(this.healthCheckInterval);
                this.healthCheckInterval = null;
            }

            // Cleanup WebSocket subscriptions
            this.wsSubscriptions.forEach(unsubscribe => {
                try {
                    unsubscribe();
                } catch (error) {
                    this.log('Error unsubscribing from WebSocket:', error);
                }
            });
            this.wsSubscriptions = [];

            // Destroy chart if requested
            if (destroyChart) {
                await this.destroyChart();
            }

            this.log('Cleanup completed');

        } catch (error) {
            this.log('Error during cleanup:', error);
        }
    }

    /**
     * Destroy chart based on library
     */
    async destroyChart() {
        try {
            switch (this.activeLibrary) {
                case 'lightweightCharts':
                    if (this.chart) {
                        this.chart.remove();
                    }
                    break;
                case 'chartjs':
                    if (this.chart) {
                        this.chart.destroy();
                    }
                    break;
                case 'tradingview':
                    // TradingView widget cleanup
                    this.clearContainer();
                    break;
            }
        } catch (error) {
            this.log('Error destroying chart:', error);
        } finally {
            // Reset state
            this.chart = null;
            this.series = null;
            this.volumeSeries = null;
            this.activeLibrary = null;
        }
    }

    /**
     * Get loading styles
     */
    getLoadingStyles() {
        return `
            .chart-loading-indicator {
                display: flex;
                align-items: center;
                justify-content: center;
                min-height: 200px;
                background: rgba(0,0,0,0.02);
                border-radius: 8px;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }
            .loading-content {
                text-align: center;
                color: #666;
            }
            .loading-spinner {
                width: 40px;
                height: 40px;
                border: 3px solid #f3f3f3;
                border-top: 3px solid #2196F3;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin: 0 auto 15px;
            }
            .loading-text {
                font-size: 16px;
                font-weight: 500;
                margin-bottom: 5px;
            }
            .loading-subtext {
                font-size: 12px;
                color: #999;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        `;
    }

    /**
     * Setup resize handler with error handling
     */
    setupResizeHandlerWithErrorHandling() {
        if (this.activeLibrary === 'lightweightCharts' && this.chart) {
            try {
                const resizeObserver = new ResizeObserver((entries) => {
                    try {
                        for (const entry of entries) {
                            if (this.chart && this.container) {
                                this.chart.applyOptions({
                                    width: this.container.clientWidth,
                                    height: this.options.height
                                });
                            }
                        }
                    } catch (error) {
                        this.handleError(new ChartError('RESIZE_FAILED', 'Chart resize failed', { error }));
                    }
                });

                resizeObserver.observe(this.container);
                this.resizeObserver = resizeObserver;
            } catch (error) {
                this.log('Warning: Resize observer setup failed:', error);
                // Fallback to window resize event
                this.setupFallbackResizeHandler();
            }
        }
    }

    /**
     * Fallback resize handler
     */
    setupFallbackResizeHandler() {
        const handleResize = () => {
            try {
                if (this.chart && this.container && this.activeLibrary === 'lightweightCharts') {
                    this.chart.applyOptions({
                        width: this.container.clientWidth,
                        height: this.options.height
                    });
                }
            } catch (error) {
                this.log('Error in fallback resize handler:', error);
            }
        };

        window.addEventListener('resize', handleResize);
        this.windowResizeHandler = handleResize;
    }

    /**
     * Handle chart resize
     */
    handleChartResize(size) {
        try {
            this.log('Chart resized:', size);
            // Additional resize handling logic if needed
        } catch (error) {
            this.handleError(new ChartError('RESIZE_HANDLER_FAILED', 'Chart resize handler failed', { error, size }));
        }
    }

    /**
     * Load data from API with timeout and retry
     */
    async loadDataFromAPI() {
        // This would connect to your actual API
        // For now, we'll simulate with sample data
        return new Promise((resolve, reject) => {
            setTimeout(() => {
                // Simulate API call
                resolve(this.generateSampleData());
            }, 100);
        });
    }

    /**
     * Set chart data
     */
    async setData(data) {
        this.data = Array.isArray(data) ? data : [];
        
        switch (this.activeLibrary) {
            case 'lightweightCharts':
                await this.setLightweightChartsData(this.data);
                break;
            case 'chartjs':
                await this.setChartJSData(this.data);
                break;
            case 'tradingview':
                // TradingView handles data differently
                break;
        }

        this.lastUpdate = new Date();
    }

    /**
     * Set data for LightweightCharts
     */
    async setLightweightChartsData(data) {
        if (!this.series) return;

        const formattedData = data.map(item => {
            if (this.options.chartType === 'line') {
                return {
                    time: item.time || item.timestamp,
                    value: item.close || item.price || item.value
                };
            } else {
                return {
                    time: item.time || item.timestamp,
                    open: item.open,
                    high: item.high,
                    low: item.low,
                    close: item.close
                };
            }
        });

        this.series.setData(formattedData);

        if (this.volumeSeries && data[0]?.volume !== undefined) {
            const volumeData = data.map(item => ({
                time: item.time || item.timestamp,
                value: item.volume,
                color: item.close >= item.open ? '#26a69a' : '#ef5350'
            }));
            this.volumeSeries.setData(volumeData);
        }
    }

    /**
     * Set data for Chart.js
     */
    async setChartJSData(data) {
        if (!this.chart) return;

        const formattedData = data.map(item => ({
            x: item.time || item.timestamp,
            y: this.options.chartType === 'line' ? 
                (item.close || item.price || item.value) : item,
            o: item.open,
            h: item.high,
            l: item.low,
            c: item.close
        }));

        this.chart.data.datasets[0].data = formattedData;
        this.chart.update('none');
    }

    /**
     * Add real-time data point
     */
    async addDataPoint(dataPoint) {
        if (!dataPoint) return;

        this.log('Adding real-time data point:', dataPoint);

        // Add to internal data array
        this.data.push(dataPoint);

        // Limit data points for performance
        if (this.data.length > this.options.maxDataPoints) {
            this.data = this.data.slice(-this.options.maxDataPoints);
        }

        switch (this.activeLibrary) {
            case 'lightweightCharts':
                await this.addLightweightChartsPoint(dataPoint);
                break;
            case 'chartjs':
                await this.addChartJSPoint(dataPoint);
                break;
            case 'tradingview':
                // TradingView handles real-time differently
                break;
        }

        this.lastUpdate = new Date();
    }

    /**
     * Add point to LightweightCharts
     */
    async addLightweightChartsPoint(dataPoint) {
        if (!this.series) return;

        const formattedPoint = this.options.chartType === 'line' ? {
            time: dataPoint.time || dataPoint.timestamp,
            value: dataPoint.close || dataPoint.price || dataPoint.value
        } : {
            time: dataPoint.time || dataPoint.timestamp,
            open: dataPoint.open,
            high: dataPoint.high,
            low: dataPoint.low,
            close: dataPoint.close
        };

        this.series.update(formattedPoint);

        if (this.volumeSeries && dataPoint.volume !== undefined) {
            this.volumeSeries.update({
                time: dataPoint.time || dataPoint.timestamp,
                value: dataPoint.volume,
                color: dataPoint.close >= dataPoint.open ? '#26a69a' : '#ef5350'
            });
        }
    }

    /**
     * Add point to Chart.js
     */
    async addChartJSPoint(dataPoint) {
        if (!this.chart) return;

        const formattedPoint = {
            x: dataPoint.time || dataPoint.timestamp,
            y: this.options.chartType === 'line' ? 
                (dataPoint.close || dataPoint.price || dataPoint.value) : dataPoint,
            o: dataPoint.open,
            h: dataPoint.high,
            l: dataPoint.low,
            c: dataPoint.close
        };

        this.chart.data.datasets[0].data.push(formattedPoint);

        // Limit data points
        if (this.chart.data.datasets[0].data.length > this.options.maxDataPoints) {
            this.chart.data.datasets[0].data.shift();
        }

        this.chart.update('none');
    }

    /**
     * Change chart type
     */
    async setChartType(chartType) {
        if (this.options.chartType === chartType) return;

        this.log(`Changing chart type from ${this.options.chartType} to ${chartType}`);
        this.options.chartType = chartType;

        // Reinitialize chart with new type
        await this.destroy();
        await this.initializeChart();
    }

    /**
     * Change timeframe
     */
    async setTimeframe(timeframe) {
        if (this.options.timeframe === timeframe) return;

        this.log(`Changing timeframe from ${this.options.timeframe} to ${timeframe}`);
        this.options.timeframe = timeframe;

        // Update chart configuration and reload data
        switch (this.activeLibrary) {
            case 'lightweightCharts':
                // LightweightCharts doesn't need special timeframe handling
                break;
            case 'chartjs':
                this.chart.options.scales.x.time.unit = this.getTimeUnit();
                this.chart.update();
                break;
            case 'tradingview':
                // TradingView widget needs to be recreated
                await this.destroy();
                await this.initializeChart();
                break;
        }

        // Reload data for new timeframe
        await this.loadInitialData();
    }

    /**
     * Set up WebSocket integration
     */
    setupWebSocketIntegration() {
        if (!this.options.wsManager) return;

        this.log('Setting up WebSocket integration...');

        // Subscribe to price updates
        const priceUnsubscribe = this.options.wsManager.subscribe('priceUpdate', (data) => {
            this.handleWebSocketPriceUpdate(data);
        });

        this.wsSubscriptions.push(priceUnsubscribe);

        // Enable real-time mode
        this.isRealtime = true;
        this.log('WebSocket integration setup complete');
    }

    /**
     * Handle WebSocket price updates
     */
    handleWebSocketPriceUpdate(data) {
        if (!this.isRealtime) return;

        // Buffer price updates to avoid excessive chart updates
        this.priceBuffer.push(data);

        if (this.bufferTimeout) {
            clearTimeout(this.bufferTimeout);
        }

        this.bufferTimeout = setTimeout(() => {
            this.processBufferedUpdates();
        }, 100); // Process updates every 100ms
    }

    /**
     * Process buffered price updates
     */
    async processBufferedUpdates() {
        if (this.priceBuffer.length === 0) return;

        // Use the latest price update
        const latestUpdate = this.priceBuffer[this.priceBuffer.length - 1];
        this.priceBuffer = [];

        // Convert price update to chart data point
        const dataPoint = this.convertPriceUpdateToDataPoint(latestUpdate);
        
        if (dataPoint) {
            await this.addDataPoint(dataPoint);
        }
    }

    /**
     * Convert price update to chart data point
     */
    convertPriceUpdateToDataPoint(priceUpdate) {
        const timestamp = priceUpdate.timestamp || Date.now();
        const price = priceUpdate.price || priceUpdate.close;

        if (!price) return null;

        if (this.options.chartType === 'line') {
            return {
                time: Math.floor(timestamp / 1000),
                value: price,
                timestamp: timestamp
            };
        } else {
            // For OHLC/candlestick, we need to aggregate data
            // This is a simplified version - in production, you'd aggregate by timeframe
            return {
                time: Math.floor(timestamp / 1000),
                open: price,
                high: price,
                low: price,
                close: price,
                volume: priceUpdate.volume || 0,
                timestamp: timestamp
            };
        }
    }

    /**
     * Helper methods for different libraries
     */
    getChartJSType() {
        switch (this.options.chartType) {
            case 'candlestick':
            case 'ohlc':
                return 'candlestick';
            default:
                return 'line';
        }
    }

    getTimeUnit() {
        const timeframe = this.options.timeframe;
        if (timeframe.includes('m')) return 'minute';
        if (timeframe.includes('h')) return 'hour';
        if (timeframe.includes('d')) return 'day';
        return 'hour';
    }

    getTradingViewInterval() {
        const mapping = {
            '1m': '1',
            '5m': '5',
            '15m': '15',
            '1h': '60',
            '4h': '240',
            '1d': '1D'
        };
        return mapping[this.options.timeframe] || '60';
    }

    getTradingViewStyle() {
        const mapping = {
            'candlestick': '1',
            'ohlc': '8',
            'line': '2'
        };
        return mapping[this.options.chartType] || '1';
    }

    /**
     * Generate sample data for testing
     */
    generateSampleData() {
        const data = [];
        const now = Date.now();
        const intervals = this.getIntervalMilliseconds();
        let basePrice = 2000;

        for (let i = 100; i >= 0; i--) {
            const time = Math.floor((now - (i * intervals)) / 1000);
            const change = (Math.random() - 0.5) * 20;
            const open = basePrice;
            const high = open + Math.random() * 10;
            const low = open - Math.random() * 10;
            const close = open + change;
            
            data.push({
                time: time,
                open: Math.round(open * 100) / 100,
                high: Math.round(high * 100) / 100,
                low: Math.round(low * 100) / 100,
                close: Math.round(close * 100) / 100,
                volume: Math.floor(Math.random() * 1000000),
                timestamp: time * 1000
            });

            basePrice = close;
        }

        return data;
    }

    /**
     * Get interval in milliseconds
     */
    getIntervalMilliseconds() {
        const mapping = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000
        };
        return mapping[this.options.timeframe] || 60 * 60 * 1000;
    }

    /**
     * Setup resize handler
     */
    setupResizeHandler() {
        if (this.activeLibrary === 'lightweightCharts' && this.chart) {
            const resizeObserver = new ResizeObserver(() => {
                if (this.chart && this.container) {
                    this.chart.applyOptions({
                        width: this.container.clientWidth,
                        height: this.options.height
                    });
                }
            });

            resizeObserver.observe(this.container);
        }
    }

    /**
     * Get current chart status
     */
    getStatus() {
        return {
            activeLibrary: this.activeLibrary,
            chartType: this.options.chartType,
            timeframe: this.options.timeframe,
            dataPoints: this.data.length,
            isRealtime: this.isRealtime,
            lastUpdate: this.lastUpdate,
            hasWebSocket: !!this.options.wsManager,
            availableLibraries: this.availableLibraries
        };
    }

    /**
     * Destroy chart and cleanup
     */
    async destroy() {
        this.log('Destroying chart...');

        // Cleanup WebSocket subscriptions
        this.wsSubscriptions.forEach(unsubscribe => unsubscribe());
        this.wsSubscriptions = [];

        // Clear timers
        if (this.bufferTimeout) {
            clearTimeout(this.bufferTimeout);
            this.bufferTimeout = null;
        }

        // Destroy chart based on library
        switch (this.activeLibrary) {
            case 'lightweightCharts':
                if (this.chart) {
                    this.chart.remove();
                }
                break;
            case 'chartjs':
                if (this.chart) {
                    this.chart.destroy();
                }
                break;
            case 'tradingview':
                // TradingView widget cleanup
                if (this.container) {
                    this.container.innerHTML = '';
                }
                break;
        }

        // Reset state
        this.chart = null;
        this.series = null;
        this.volumeSeries = null;
        this.data = [];
        this.isRealtime = false;
        this.activeLibrary = null;

        this.log('Chart destroyed');
    }

    /**
     * Internal logging method
     */
    log(message, data = null) {
        if (this.options.debug) {
            const timestamp = new Date().toLocaleTimeString();
            if (data) {
                console.log(`[${timestamp}] UnifiedChartManager: ${message}`, data);
            } else {
                console.log(`[${timestamp}] UnifiedChartManager: ${message}`);
            }
        }
    }
}

/**
 * UnifiedChartManagerFactory - Factory for managing multiple chart instances with error handling
 */
class UnifiedChartManagerFactory {
    constructor() {
        this.instances = new Map();
        this.errors = new Map();
        this.defaultOptions = {
            chartType: 'candlestick',
            timeframe: '1h',
            theme: 'dark',
            height: 400,
            realtime: true,
            enableVolume: true,
            enableIndicators: true,
            debug: true,
            maxRetries: 3,
            enableRecovery: true,
            enableAutoRecovery: true,
            showErrorMessages: true,
            onError: (error) => this.handleFactoryError(error)
        };

        // Inject error display styles
        this.injectErrorStyles();
    }

    /**
     * Create or get chart manager instance with error handling
     */
    async createChart(containerId, options = {}) {
        try {
            if (this.instances.has(containerId)) {
                console.warn(`Chart manager for '${containerId}' already exists`);
                return this.instances.get(containerId);
            }

            const mergedOptions = { ...this.defaultOptions, ...options };
            
            // Add factory-level error handler
            const originalErrorHandler = mergedOptions.onError;
            mergedOptions.onError = (error) => {
                this.handleInstanceError(containerId, error);
                if (originalErrorHandler) {
                    originalErrorHandler(error);
                }
            };

            const manager = new UnifiedChartManager(containerId, mergedOptions);
            this.instances.set(containerId, manager);
            
            console.log(`✅ UnifiedChartManager instance '${containerId}' created`);
            return manager;

        } catch (error) {
            const chartError = new ChartError('FACTORY_CREATE_FAILED', 
                `Failed to create chart manager for '${containerId}'`, { containerId, error });
            
            this.handleInstanceError(containerId, chartError);
            throw chartError;
        }
    }

    /**
     * Retry chart creation for a specific container
     */
    async retryChart(containerId) {
        try {
            console.log(`🔄 Retrying chart creation for '${containerId}'...`);
            
            // Remove existing instance if any
            await this.removeChart(containerId);
            
            // Clear previous errors
            this.errors.delete(containerId);
            
            // Get original options (you might want to store these)
            const options = this.getStoredOptions(containerId) || {};
            
            // Recreate chart
            const manager = await this.createChart(containerId, options);
            
            console.log(`✅ Chart '${containerId}' successfully recreated`);
            return manager;

        } catch (error) {
            console.error(`❌ Retry failed for '${containerId}':`, error);
            throw error;
        }
    }

    /**
     * Handle errors from individual chart instances
     */
    handleInstanceError(containerId, error) {
        // Store error for this instance
        if (!this.errors.has(containerId)) {
            this.errors.set(containerId, []);
        }
        this.errors.get(containerId).push(error);

        // Log at factory level
        console.error(`[Factory] Chart '${containerId}' error:`, error);

        // Emit factory-level error event
        this.emitFactoryEvent('chartError', { containerId, error });
    }

    /**
     * Handle factory-level errors
     */
    handleFactoryError(error) {
        console.error('[Factory] Factory-level error:', error);
        this.emitFactoryEvent('factoryError', { error });
    }

    /**
     * Get existing chart manager
     */
    getChart(containerId) {
        return this.instances.get(containerId);
    }

    /**
     * Remove chart manager instance
     */
    async removeChart(containerId) {
        const manager = this.instances.get(containerId);
        if (manager) {
            try {
                await manager.destroy();
                this.instances.delete(containerId);
                this.errors.delete(containerId);
                console.log(`🗑️ UnifiedChartManager instance '${containerId}' removed`);
            } catch (error) {
                console.error(`Error removing chart '${containerId}':`, error);
            }
        }
    }

    /**
     * Get all active chart instances with their status
     */
    getAllCharts() {
        return Array.from(this.instances.entries()).map(([containerId, manager]) => {
            try {
                return {
                    containerId,
                    status: manager.getStatus(),
                    errors: this.errors.get(containerId) || [],
                    hasErrors: (this.errors.get(containerId) || []).length > 0
                };
            } catch (error) {
                return {
                    containerId,
                    status: { error: 'Status check failed' },
                    errors: [error],
                    hasErrors: true
                };
            }
        });
    }

    /**
     * Get factory health status
     */
    getFactoryStatus() {
        const charts = this.getAllCharts();
        const totalCharts = charts.length;
        const healthyCharts = charts.filter(chart => !chart.hasErrors && 
                                          chart.status.connectionStatus === 'connected').length;
        const errorCharts = charts.filter(chart => chart.hasErrors).length;

        return {
            totalCharts,
            healthyCharts,
            errorCharts,
            healthPercentage: totalCharts > 0 ? (healthyCharts / totalCharts) * 100 : 0,
            lastCheck: new Date(),
            charts: charts.map(chart => ({
                containerId: chart.containerId,
                healthy: !chart.hasErrors,
                status: chart.status.connectionStatus || 'unknown'
            }))
        };
    }

    /**
     * Perform factory-wide health check
     */
    async performFactoryHealthCheck() {
        console.log('🏥 Performing factory health check...');
        
        const status = this.getFactoryStatus();
        console.log('Factory Health Status:', status);

        // Try to recover failed charts
        const failedCharts = status.charts.filter(chart => !chart.healthy);
        
        for (const chart of failedCharts) {
            try {
                console.log(`🔧 Attempting to recover chart '${chart.containerId}'...`);
                await this.retryChart(chart.containerId);
            } catch (error) {
                console.error(`❌ Recovery failed for '${chart.containerId}':`, error);
            }
        }

        return this.getFactoryStatus();
    }

    /**
     * Destroy all chart instances
     */
    async destroyAll() {
        const destroyPromises = Array.from(this.instances.entries()).map(async ([containerId, manager]) => {
            try {
                await manager.destroy();
            } catch (error) {
                console.error(`Error destroying chart '${containerId}':`, error);
            }
        });
        
        await Promise.all(destroyPromises);
        this.instances.clear();
        this.errors.clear();
        console.log('🧹 All UnifiedChartManager instances destroyed');
    }

    /**
     * Store options for potential retry
     */
    storeOptions(containerId, options) {
        if (!this.storedOptions) {
            this.storedOptions = new Map();
        }
        this.storedOptions.set(containerId, options);
    }

    /**
     * Get stored options for retry
     */
    getStoredOptions(containerId) {
        return this.storedOptions ? this.storedOptions.get(containerId) : null;
    }

    /**
     * Emit factory events
     */
    emitFactoryEvent(eventType, data) {
        const event = new CustomEvent(`unifiedChartFactory:${eventType}`, { detail: data });
        document.dispatchEvent(event);
    }

    /**
     * Inject error display styles
     */
    injectErrorStyles() {
        if (!document.querySelector('#unified-chart-error-styles')) {
            const style = document.createElement('style');
            style.id = 'unified-chart-error-styles';
            style.textContent = ChartErrorDisplay.getErrorStyles();
            document.head.appendChild(style);
        }
    }

    /**
     * Debug information
     */
    getDebugInfo() {
        return {
            factoryStatus: this.getFactoryStatus(),
            instances: Array.from(this.instances.keys()),
            errors: Object.fromEntries(this.errors.entries()),
            availableLibraries: {
                lightweightCharts: typeof LightweightCharts !== 'undefined',
                chartjs: typeof Chart !== 'undefined',
                tradingview: typeof TradingView !== 'undefined'
            }
        };
    }
}

// Global factory instance
window.UnifiedChartManagerFactory = window.UnifiedChartManagerFactory || new UnifiedChartManagerFactory();

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { UnifiedChartManager, UnifiedChartManagerFactory };
}

// Auto-setup when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 UnifiedChartManager system ready');
    console.log('📊 Available libraries:', {
        lightweightCharts: typeof LightweightCharts !== 'undefined',
        chartjs: typeof Chart !== 'undefined',
        tradingview: typeof TradingView !== 'undefined'
    });
    console.log('💡 Use UnifiedChartManagerFactory.createChart(containerId, options) to create charts');
});

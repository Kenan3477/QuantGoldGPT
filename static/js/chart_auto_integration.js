/**
 * UnifiedChartManager Auto-Integration for Existing GoldGPT Dashboard
 * This script automatically detects chart containers and initializes charts
 */

// Auto-integration configuration
const CHART_AUTO_CONFIG = {
    // Chart container selectors and their configurations
    containers: {
        '#price-chart': {
            chartType: 'candlestick',
            timeframe: '1h',
            height: 400,
            enableVolume: true
        },
        '#trading-chart': {
            chartType: 'candlestick',
            timeframe: '1h',
            height: 600,
            enableVolume: true,
            enableIndicators: true
        },
        '#overview-chart': {
            chartType: 'line',
            timeframe: '1d',
            height: 200,
            enableVolume: false
        },
        '.chart-container': {
            chartType: 'candlestick',
            timeframe: '1h',
            height: 400,
            enableVolume: true
        }
    },
    
    // Default options for all charts
    defaults: {
        theme: 'dark',
        realtime: true,
        debug: true,
        maxDataPoints: 1000
    }
};

class GoldGPTChartAutoIntegration {
    constructor() {
        this.initialized = false;
        this.charts = new Map();
        this.wsManager = null;
        this.initializationErrors = [];
        this.retryAttempts = 0;
        this.maxRetries = 3;
        
        console.log('🚀 GoldGPT Chart Auto-Integration starting...');
        this.initializeWithErrorHandling();
    }

    async initializeWithErrorHandling() {
        try {
            // Wait for required dependencies
            await this.waitForDependencies();
            
            // Get WebSocket manager
            await this.setupWebSocketManager();
            
            // Auto-detect and initialize charts
            await this.autoDetectChartsWithErrorHandling();
            
            // Setup event listeners
            this.setupEventListeners();
            
            this.initialized = true;
            console.log('✅ GoldGPT Chart Auto-Integration ready');

        } catch (error) {
            console.error('❌ Chart auto-integration failed:', error);
            this.handleInitializationError(error);
        }
    }

    async handleInitializationError(error) {
        this.initializationErrors.push({
            error,
            timestamp: new Date(),
            attempt: this.retryAttempts
        });

        console.error('🚨 Auto-integration error:', error);

        // Show error message to user
        this.showGlobalErrorMessage(error);

        // Attempt recovery if we haven't exceeded max retries
        if (this.retryAttempts < this.maxRetries) {
            const delay = Math.min(1000 * Math.pow(2, this.retryAttempts), 10000);
            console.log(`🔄 Retrying auto-integration in ${delay}ms... (${this.retryAttempts + 1}/${this.maxRetries})`);
            
            setTimeout(() => {
                this.retryAttempts++;
                this.initializeWithErrorHandling();
            }, delay);
        } else {
            console.error('❌ Auto-integration failed after maximum retries');
            this.showFatalError();
        }
    }

    async waitForDependencies() {
        return new Promise((resolve, reject) => {
            let attempts = 0;
            const maxAttempts = 30; // 15 seconds timeout

            const checkDependencies = () => {
                attempts++;
                
                try {
                    // Check if UnifiedChartManager is available
                    if (typeof window.UnifiedChartManagerFactory !== 'undefined') {
                        console.log('📊 UnifiedChartManager detected');
                        resolve();
                        return;
                    }

                    if (attempts >= maxAttempts) {
                        reject(new Error('UnifiedChartManager not available after timeout'));
                        return;
                    }

                    console.log(`⏳ Waiting for dependencies... (${attempts}/${maxAttempts})`);
                    setTimeout(checkDependencies, 500);
                } catch (error) {
                    reject(new Error(`Dependency check failed: ${error.message}`));
                }
            };

            checkDependencies();
        });
    }

    async setupWebSocketManager() {
        return new Promise((resolve) => {
            try {
                const checkWebSocket = () => {
                    if (window.WebSocketManager || window.wsManager) {
                        this.wsManager = window.WebSocketManager || window.wsManager;
                        console.log('📡 WebSocket manager found');
                        resolve();
                    } else if (window.WebSocketManagerFactory) {
                        this.wsManager = window.WebSocketManagerFactory.getInstance('auto-charts');
                        console.log('📡 WebSocket manager created');
                        resolve();
                    } else {
                        console.log('⚠️ WebSocket manager not available, continuing without real-time');
                        resolve();
                    }
                };

                checkWebSocket();
            } catch (error) {
                console.error('❌ WebSocket setup error:', error);
                resolve(); // Continue without WebSocket
            }
        });
    }

    async autoDetectChartsWithErrorHandling() {
        console.log('🔍 Auto-detecting chart containers...');
        const detectionErrors = [];

        for (const [selector, config] of Object.entries(CHART_AUTO_CONFIG.containers)) {
            try {
                const containers = document.querySelectorAll(selector);
                
                if (containers.length === 0) {
                    console.log(`⚠️ No containers found for selector: ${selector}`);
                    continue;
                }
                
                for (const [index, container] of containers.entries()) {
                    try {
                        if (!container.id) {
                            container.id = `auto-chart-${selector.replace(/[#.]/g, '')}-${index}`;
                        }

                        await this.createAutoChartWithErrorHandling(container.id, config, selector);
                    } catch (error) {
                        console.error(`❌ Error creating chart for container ${container.id}:`, error);
                        detectionErrors.push({ containerId: container.id, selector, error });
                        this.showContainerError(container, error);
                    }
                }
            } catch (error) {
                console.error(`❌ Error processing selector ${selector}:`, error);
                detectionErrors.push({ selector, error });
            }
        }

        console.log(`📊 Auto-detected ${this.charts.size} chart containers`);
        
        if (detectionErrors.length > 0) {
            console.warn(`⚠️ ${detectionErrors.length} containers failed to initialize:`, detectionErrors);
        }

        // If no charts were created at all, this is a problem
        if (this.charts.size === 0 && detectionErrors.length > 0) {
            throw new Error(`Failed to create any charts. Errors: ${detectionErrors.map(e => e.error.message).join(', ')}`);
        }
    }

    async createAutoChartWithErrorHandling(containerId, config, selector) {
        try {
            const mergedConfig = {
                ...CHART_AUTO_CONFIG.defaults,
                ...config,
                wsManager: this.wsManager,
                onError: (error) => this.handleChartError(containerId, error),
                enableAutoRecovery: true,
                showErrorMessages: true
            };

            // Store configuration for potential retries
            window.UnifiedChartManagerFactory.storeOptions(containerId, mergedConfig);

            const chart = await window.UnifiedChartManagerFactory.createChart(containerId, mergedConfig);
            this.charts.set(containerId, {
                chart,
                config: mergedConfig,
                selector,
                status: 'active'
            });

            // Add auto-controls if specified
            if (config.enableAutoControls !== false) {
                this.addAutoControlsWithErrorHandling(containerId, chart);
            }

            console.log(`✅ Auto-chart created: ${containerId}`);
            return chart;

        } catch (error) {
            console.error(`❌ Failed to create auto-chart ${containerId}:`, error);
            this.charts.set(containerId, {
                chart: null,
                config,
                selector,
                status: 'error',
                error
            });
            throw error;
        }
    }

    handleChartError(containerId, error) {
        console.error(`🚨 Chart error for ${containerId}:`, error);
        
        const chartInfo = this.charts.get(containerId);
        if (chartInfo) {
            chartInfo.status = 'error';
            chartInfo.error = error;
            chartInfo.lastErrorTime = new Date();
        }

        // Emit error event for external handling
        this.emitChartErrorEvent(containerId, error);
    }

    addAutoControlsWithErrorHandling(containerId, chart) {
        try {
            const container = document.getElementById(containerId);
            if (!container) {
                throw new Error(`Container ${containerId} not found for auto-controls`);
            }

            // Check if controls already exist
            if (container.querySelector('.chart-auto-controls')) {
                return;
            }

            const controlsDiv = document.createElement('div');
            controlsDiv.className = 'chart-auto-controls';
            controlsDiv.innerHTML = this.generateAutoControlsHTML(containerId);

            // Insert controls at the beginning of the container
            container.insertBefore(controlsDiv, container.firstChild);

            // Setup event listeners for controls
            this.setupAutoControlEventListeners(containerId, controlsDiv, chart);

            console.log(`🎛️ Auto-controls added for ${containerId}`);

        } catch (error) {
            console.error(`❌ Failed to add auto-controls for ${containerId}:`, error);
            // Don't fail the entire chart creation for control errors
        }
    }

    setupAutoControlEventListeners(containerId, controlsDiv, chart) {
        try {
            // Timeframe selector
            const timeframeSelect = controlsDiv.querySelector('.timeframe-select');
            if (timeframeSelect) {
                timeframeSelect.addEventListener('change', async (e) => {
                    try {
                        await chart.setTimeframe(e.target.value);
                    } catch (error) {
                        console.error(`Error changing timeframe for ${containerId}:`, error);
                        this.showControlError(controlsDiv, 'Timeframe change failed');
                    }
                });
            }

            // Chart type selector
            const chartTypeSelect = controlsDiv.querySelector('.chart-type-select');
            if (chartTypeSelect) {
                chartTypeSelect.addEventListener('change', async (e) => {
                    try {
                        await chart.setChartType(e.target.value);
                    } catch (error) {
                        console.error(`Error changing chart type for ${containerId}:`, error);
                        this.showControlError(controlsDiv, 'Chart type change failed');
                    }
                });
            }

            // Refresh button
            const refreshBtn = controlsDiv.querySelector('.refresh-btn');
            if (refreshBtn) {
                refreshBtn.addEventListener('click', async () => {
                    try {
                        refreshBtn.disabled = true;
                        refreshBtn.textContent = 'Refreshing...';
                        
                        await chart.loadInitialDataWithErrorHandling();
                        
                        refreshBtn.textContent = 'Refresh';
                    } catch (error) {
                        console.error(`Error refreshing chart ${containerId}:`, error);
                        this.showControlError(controlsDiv, 'Refresh failed');
                        refreshBtn.textContent = 'Retry';
                    } finally {
                        refreshBtn.disabled = false;
                    }
                });
            }

        } catch (error) {
            console.error(`Error setting up control listeners for ${containerId}:`, error);
        }
    }

    showControlError(controlsDiv, message) {
        const errorSpan = controlsDiv.querySelector('.control-error') || document.createElement('span');
        errorSpan.className = 'control-error';
        errorSpan.textContent = message;
        errorSpan.style.color = '#ff4444';
        errorSpan.style.fontSize = '11px';
        
        if (!controlsDiv.contains(errorSpan)) {
            controlsDiv.appendChild(errorSpan);
        }

        // Clear error after 3 seconds
        setTimeout(() => {
            if (controlsDiv.contains(errorSpan)) {
                controlsDiv.removeChild(errorSpan);
            }
        }, 3000);
    }

    showContainerError(container, error) {
        try {
            const errorDiv = document.createElement('div');
            errorDiv.className = 'auto-integration-error';
            errorDiv.innerHTML = `
                <div class="error-content">
                    <span class="error-icon">⚠️</span>
                    <span class="error-text">Chart initialization failed</span>
                    <button class="retry-container-btn" onclick="autoIntegration.retryContainer('${container.id}')">Retry</button>
                </div>
                <div class="error-details" style="display: none;">
                    ${error.message}
                </div>
            `;
            
            container.appendChild(errorDiv);
            
            // Add click handler to show details
            errorDiv.addEventListener('click', () => {
                const details = errorDiv.querySelector('.error-details');
                details.style.display = details.style.display === 'none' ? 'block' : 'none';
            });

        } catch (displayError) {
            console.error('Error showing container error:', displayError);
        }
    }

    async retryContainer(containerId) {
        try {
            console.log(`🔄 Retrying container ${containerId}...`);
            
            const chartInfo = this.charts.get(containerId);
            if (!chartInfo) {
                throw new Error(`Chart info not found for ${containerId}`);
            }

            // Remove error display
            const container = document.getElementById(containerId);
            if (container) {
                const errorDiv = container.querySelector('.auto-integration-error');
                if (errorDiv) {
                    errorDiv.remove();
                }
            }

            // Retry creation
            await this.createAutoChartWithErrorHandling(containerId, chartInfo.config, chartInfo.selector);
            
            console.log(`✅ Container ${containerId} retry successful`);

        } catch (error) {
            console.error(`❌ Container ${containerId} retry failed:`, error);
            const container = document.getElementById(containerId);
            if (container) {
                this.showContainerError(container, error);
            }
        }
    }

    showGlobalErrorMessage(error) {
        // Show a global error message for integration failures
        const errorDiv = document.createElement('div');
        errorDiv.className = 'global-auto-integration-error';
        errorDiv.innerHTML = `
            <div class="global-error-content">
                <h4>🚨 Chart System Error</h4>
                <p>Chart auto-integration encountered an error. Some charts may not be available.</p>
                <details>
                    <summary>Error Details</summary>
                    <pre>${error.message}</pre>
                </details>
                <button onclick="this.parentElement.parentElement.remove()">Dismiss</button>
            </div>
        `;
        
        document.body.appendChild(errorDiv);
    }

    showFatalError() {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'fatal-auto-integration-error';
        errorDiv.innerHTML = `
            <div class="fatal-error-content">
                <h4>💥 Chart System Fatal Error</h4>
                <p>Chart auto-integration has failed completely. Please refresh the page to retry.</p>
                <button onclick="window.location.reload()">Refresh Page</button>
                <button onclick="this.parentElement.parentElement.remove()">Dismiss</button>
            </div>
        `;
        
        document.body.appendChild(errorDiv);
    }

    emitChartErrorEvent(containerId, error) {
        const event = new CustomEvent('chartAutoIntegrationError', {
            detail: { containerId, error, timestamp: new Date() }
        });
        document.dispatchEvent(event);
    }

    getIntegrationStatus() {
        const totalCharts = this.charts.size;
        const activeCharts = Array.from(this.charts.values()).filter(info => info.status === 'active').length;
        const errorCharts = Array.from(this.charts.values()).filter(info => info.status === 'error').length;

        return {
            initialized: this.initialized,
            totalCharts,
            activeCharts,
            errorCharts,
            successRate: totalCharts > 0 ? (activeCharts / totalCharts) * 100 : 0,
            hasWebSocket: !!this.wsManager,
            initializationErrors: this.initializationErrors,
            charts: Object.fromEntries(
                Array.from(this.charts.entries()).map(([id, info]) => [
                    id, 
                    { 
                        status: info.status, 
                        selector: info.selector,
                        error: info.error?.message 
                    }
                ])
            )
        };
    }
            };

            console.log(`📊 Creating auto-chart: ${containerId}`, mergedConfig);

            const chart = window.UnifiedChartManagerFactory.createChart(containerId, mergedConfig);
            
            if (chart) {
                this.charts.set(containerId, chart);
                console.log(`✅ Auto-chart created: ${containerId}`);

                // Add chart controls if container has data attribute
                const container = document.getElementById(containerId);
                if (container && container.dataset.autoControls === 'true') {
                    this.addChartControls(containerId);
                }
            }

        } catch (error) {
            console.error(`❌ Failed to create auto-chart ${containerId}:`, error);
        }
    }

    addChartControls(containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;

        const controlsId = `${containerId}-controls`;
        
        // Skip if controls already exist
        if (document.getElementById(controlsId)) return;

        const controlsHtml = `
            <div id="${controlsId}" class="chart-auto-controls" style="
                margin-bottom: 10px;
                padding: 10px;
                background: rgba(255,255,255,0.1);
                border-radius: 4px;
                display: flex;
                gap: 10px;
                flex-wrap: wrap;
                align-items: center;
                font-size: 12px;
            ">
                <div>
                    <label style="margin-right: 5px; color: #ccc;">Type:</label>
                    <select class="chart-type-select" style="
                        background: #374151;
                        color: white;
                        border: 1px solid #4b5563;
                        border-radius: 3px;
                        padding: 4px;
                    ">
                        <option value="candlestick">Candlestick</option>
                        <option value="ohlc">OHLC</option>
                        <option value="line">Line</option>
                    </select>
                </div>
                <div>
                    <label style="margin-right: 5px; color: #ccc;">Timeframe:</label>
                    <select class="timeframe-select" style="
                        background: #374151;
                        color: white;
                        border: 1px solid #4b5563;
                        border-radius: 3px;
                        padding: 4px;
                    ">
                        <option value="1m">1m</option>
                        <option value="5m">5m</option>
                        <option value="15m">15m</option>
                        <option value="1h" selected>1h</option>
                        <option value="4h">4h</option>
                        <option value="1d">1d</option>
                    </select>
                </div>
                <button class="add-test-data-btn" style="
                    background: #2196F3;
                    color: white;
                    border: none;
                    border-radius: 3px;
                    padding: 4px 8px;
                    cursor: pointer;
                    font-size: 11px;
                ">Add Test Data</button>
            </div>
        `;

        container.insertAdjacentHTML('beforebegin', controlsHtml);

        // Setup control event listeners
        this.setupControlEvents(containerId, controlsId);
    }

    setupControlEvents(containerId, controlsId) {
        const controls = document.getElementById(controlsId);
        if (!controls) return;

        const chart = this.charts.get(containerId);
        if (!chart) return;

        // Chart type change
        const typeSelect = controls.querySelector('.chart-type-select');
        typeSelect.addEventListener('change', async (e) => {
            await chart.setChartType(e.target.value);
            console.log(`📊 Chart type changed to: ${e.target.value}`);
        });

        // Timeframe change
        const timeframeSelect = controls.querySelector('.timeframe-select');
        timeframeSelect.addEventListener('change', async (e) => {
            await chart.setTimeframe(e.target.value);
            console.log(`⏰ Timeframe changed to: ${e.target.value}`);
        });

        // Add test data
        const testDataBtn = controls.querySelector('.add-test-data-btn');
        testDataBtn.addEventListener('click', () => {
            this.addTestDataToChart(containerId);
        });
    }

    addTestDataToChart(containerId) {
        const chart = this.charts.get(containerId);
        if (!chart) return;

        const testData = {
            time: Math.floor(Date.now() / 1000),
            open: 1800 + Math.random() * 40,
            high: 1820 + Math.random() * 20,
            low: 1780 + Math.random() * 20,
            close: 1800 + Math.random() * 40,
            volume: Math.floor(Math.random() * 1000000),
            timestamp: Date.now()
        };

        chart.addDataPoint(testData);
        console.log(`📊 Test data added to ${containerId}:`, testData);
    }

    setupEventListeners() {
        // Listen for dynamic chart container additions
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                mutation.addedNodes.forEach((node) => {
                    if (node.nodeType === Node.ELEMENT_NODE) {
                        this.checkForNewChartContainers(node);
                    }
                });
            });
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true
        });

        // Listen for WebSocket status changes
        if (this.wsManager) {
            this.wsManager.subscribe('connectionStateChanged', (data) => {
                console.log(`📊 Chart integration: WebSocket ${data.state}`);
                this.updateChartsConnectionStatus(data.state);
            });
        }
    }

    checkForNewChartContainers(element) {
        for (const selector of Object.keys(CHART_AUTO_CONFIG.containers)) {
            if (element.matches && element.matches(selector)) {
                if (!element.id) {
                    element.id = `dynamic-chart-${Date.now()}`;
                }
                
                const config = CHART_AUTO_CONFIG.containers[selector];
                this.createAutoChart(element.id, config);
            }

            // Check child elements
            const childContainers = element.querySelectorAll ? element.querySelectorAll(selector) : [];
            childContainers.forEach((container, index) => {
                if (!container.id) {
                    container.id = `dynamic-chart-${selector.replace(/[#.]/g, '')}-${Date.now()}-${index}`;
                }
                
                const config = CHART_AUTO_CONFIG.containers[selector];
                this.createAutoChart(container.id, config);
            });
        }
    }

    updateChartsConnectionStatus(status) {
        // Update visual indicators for all charts
        this.charts.forEach((chart, containerId) => {
            const container = document.getElementById(containerId);
            if (container) {
                // Add connection status indicator
                let statusIndicator = container.querySelector('.chart-connection-indicator');
                if (!statusIndicator) {
                    statusIndicator = document.createElement('div');
                    statusIndicator.className = 'chart-connection-indicator';
                    statusIndicator.style.cssText = `
                        position: absolute;
                        top: 10px;
                        right: 10px;
                        padding: 4px 8px;
                        border-radius: 3px;
                        font-size: 10px;
                        font-weight: bold;
                        z-index: 1000;
                        pointer-events: none;
                    `;
                    container.style.position = 'relative';
                    container.appendChild(statusIndicator);
                }

                statusIndicator.textContent = `Real-time: ${status}`;
                statusIndicator.style.backgroundColor = 
                    status === 'connected' || status === 'authenticated' ? '#10b981' :
                    status === 'connecting' || status === 'reconnecting' ? '#f59e0b' : '#ef4444';
                statusIndicator.style.color = 'white';
            }
        });
    }

    // Public API methods
    getChart(containerId) {
        return this.charts.get(containerId);
    }

    getAllCharts() {
        return Array.from(this.charts.entries()).map(([id, chart]) => ({
            id,
            status: chart.getStatus()
        }));
    }

    async destroyChart(containerId) {
        const chart = this.charts.get(containerId);
        if (chart) {
            await chart.destroy();
            this.charts.delete(containerId);
            
            // Remove controls
            const controls = document.getElementById(`${containerId}-controls`);
            if (controls) {
                controls.remove();
            }
            
            console.log(`🗑️ Auto-chart destroyed: ${containerId}`);
        }
    }

    async destroyAllCharts() {
        const destroyPromises = Array.from(this.charts.keys()).map(id => this.destroyChart(id));
        await Promise.all(destroyPromises);
        console.log('🗑️ All auto-charts destroyed');
    }

    getIntegrationStatus() {
        return {
            initialized: this.initialized,
            chartsCount: this.charts.size,
            wsManagerAvailable: !!this.wsManager,
            wsConnected: this.wsManager ? this.wsManager.getStatus().connected : false,
            charts: this.getAllCharts()
        };
    }
}

// Auto-initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Wait a bit for other scripts to load
    setTimeout(() => {
        if (typeof window.UnifiedChartManagerFactory !== 'undefined') {
            window.goldGPTChartAutoIntegration = new GoldGPTChartAutoIntegration();
            
            // Make it globally accessible
            window.getAutoChart = (containerId) => window.goldGPTChartAutoIntegration.getChart(containerId);
            window.getAllAutoCharts = () => window.goldGPTChartAutoIntegration.getAllCharts();
            window.getChartIntegrationStatus = () => window.goldGPTChartAutoIntegration.getIntegrationStatus();
            
            console.log('🚀 GoldGPT Chart Auto-Integration loaded');
            console.log('💡 Available functions: getAutoChart(id), getAllAutoCharts(), getChartIntegrationStatus()');
        } else {
            console.log('⚠️ UnifiedChartManager not available for auto-integration');
        }
    }, 2000);
});

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { GoldGPTChartAutoIntegration, CHART_AUTO_CONFIG };
}

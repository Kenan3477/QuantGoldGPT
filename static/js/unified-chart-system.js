/**
 * GoldGPT Unified Chart System
 * Trading 212-inspired multi-timeframe chart system with AI annotations
 */

class UnifiedChartSystem {
    constructor() {
        this.charts = new Map();
        this.timeframes = ['1H', '4H', '1D'];
        this.activeTimeframe = '1H';
        this.syncEnabled = true;
        this.indicators = new Map();
        this.aiAnnotations = [];
        this.patternHighlights = [];
        
        this.init();
    }

    init() {
        this.createChartContainer();
        this.setupEventListeners();
        this.loadChartData();
        this.initializeIndicators();
        
        console.log('✅ Unified Chart System initialized');
    }

    createChartContainer() {
        const container = document.getElementById('unified-charts-container');
        if (!container) return;

        container.innerHTML = `
            <div class="chart-system-header">
                <div class="chart-controls">
                    <div class="timeframe-selector">
                        ${this.timeframes.map(tf => `
                            <button class="timeframe-btn ${tf === this.activeTimeframe ? 'active' : ''}" 
                                    data-timeframe="${tf}">${tf}</button>
                        `).join('')}
                    </div>
                    <div class="chart-tools">
                        <button class="tool-btn sync-btn ${this.syncEnabled ? 'active' : ''}" 
                                data-tooltip="Sync Charts">
                            <i class="fas fa-link"></i>
                        </button>
                        <button class="tool-btn indicators-btn" data-tooltip="Indicators">
                            <i class="fas fa-chart-line"></i>
                        </button>
                        <button class="tool-btn patterns-btn" data-tooltip="AI Patterns">
                            <i class="fas fa-brain"></i>
                        </button>
                        <button class="tool-btn fullscreen-btn" data-tooltip="Fullscreen">
                            <i class="fas fa-expand"></i>
                        </button>
                    </div>
                </div>
                <div class="chart-status">
                    <span class="status-indicator online"></span>
                    <span class="status-text">Live Data</span>
                    <span class="sync-status">${this.syncEnabled ? 'Synced' : 'Independent'}</span>
                </div>
            </div>
            
            <div class="charts-grid">
                ${this.timeframes.map(tf => `
                    <div class="chart-panel" data-timeframe="${tf}">
                        <div class="chart-header">
                            <h4>XAU/USD ${tf}</h4>
                            <div class="chart-info">
                                <span class="price-display">$0.00</span>
                                <span class="change-display">+0.00%</span>
                            </div>
                        </div>
                        <div class="chart-container" id="chart-${tf}"></div>
                        <div class="chart-footer">
                            <div class="timeframe-indicators">
                                <span class="indicator-pill trend-up">Trend: ↗</span>
                                <span class="indicator-pill momentum-strong">Momentum: Strong</span>
                            </div>
                        </div>
                    </div>
                `).join('')}
            </div>
            
            <div class="indicators-panel" id="indicators-panel">
                <div class="panel-header">
                    <h4>Technical Indicators</h4>
                    <button class="close-panel">×</button>
                </div>
                <div class="indicators-list">
                    <div class="indicator-group">
                        <h5>Trend Indicators</h5>
                        <label><input type="checkbox" data-indicator="sma20" checked> SMA 20</label>
                        <label><input type="checkbox" data-indicator="sma50" checked> SMA 50</label>
                        <label><input type="checkbox" data-indicator="ema12"> EMA 12</label>
                        <label><input type="checkbox" data-indicator="bollinger"> Bollinger Bands</label>
                    </div>
                    <div class="indicator-group">
                        <h5>Momentum Indicators</h5>
                        <label><input type="checkbox" data-indicator="rsi"> RSI</label>
                        <label><input type="checkbox" data-indicator="macd"> MACD</label>
                        <label><input type="checkbox" data-indicator="stoch"> Stochastic</label>
                    </div>
                    <div class="indicator-group">
                        <h5>Volume Indicators</h5>
                        <label><input type="checkbox" data-indicator="volume"> Volume</label>
                        <label><input type="checkbox" data-indicator="vwap"> VWAP</label>
                    </div>
                </div>
            </div>
        `;
    }

    setupEventListeners() {
        // Timeframe selection
        document.querySelectorAll('.timeframe-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const timeframe = e.target.dataset.timeframe;
                this.switchTimeframe(timeframe);
            });
        });

        // Chart tools
        document.querySelector('.sync-btn')?.addEventListener('click', () => {
            this.toggleSync();
        });

        document.querySelector('.indicators-btn')?.addEventListener('click', () => {
            this.toggleIndicatorsPanel();
        });

        document.querySelector('.patterns-btn')?.addEventListener('click', () => {
            this.togglePatternHighlights();
        });

        document.querySelector('.fullscreen-btn')?.addEventListener('click', () => {
            this.toggleFullscreen();
        });

        // Indicator toggles
        document.querySelectorAll('[data-indicator]').forEach(checkbox => {
            checkbox.addEventListener('change', (e) => {
                const indicator = e.target.dataset.indicator;
                this.toggleIndicator(indicator, e.target.checked);
            });
        });

        // Chart synchronization events
        if (this.syncEnabled) {
            this.setupChartSync();
        }

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            this.handleKeyboardShortcuts(e);
        });
    }

    async loadChartData() {
        for (const timeframe of this.timeframes) {
            try {
                const data = await this.fetchChartData(timeframe);
                await this.createChart(timeframe, data);
                this.updateChartInfo(timeframe, data);
            } catch (error) {
                console.error(`Error loading ${timeframe} chart:`, error);
                this.showChartError(timeframe);
            }
        }
    }

    async fetchChartData(timeframe) {
        const response = await fetch(`/api/chart-data/${timeframe}`);
        if (!response.ok) {
            throw new Error(`Failed to fetch ${timeframe} data`);
        }
        return await response.json();
    }

    async createChart(timeframe, data) {
        const container = document.getElementById(`chart-${timeframe}`);
        if (!container) return;

        // Use TradingView Lightweight Charts for performance
        const chart = LightweightCharts.createChart(container, {
            width: container.clientWidth,
            height: 300,
            layout: {
                backgroundColor: '#1a1a1a',
                textColor: '#d1d5db',
                fontSize: 12,
                fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, sans-serif'
            },
            grid: {
                vertLines: { color: '#2a2a2a' },
                horzLines: { color: '#2a2a2a' }
            },
            crosshair: {
                mode: LightweightCharts.CrosshairMode.Normal,
                vertLine: { color: '#fbbf24', width: 1, style: 3 },
                horzLine: { color: '#fbbf24', width: 1, style: 3 }
            },
            rightPriceScale: {
                borderColor: '#485563',
                textColor: '#d1d5db'
            },
            timeScale: {
                borderColor: '#485563',
                textColor: '#d1d5db',
                timeVisible: true,
                secondsVisible: false
            }
        });

        // Main candlestick series
        const candlestickSeries = chart.addCandlestickSeries({
            upColor: '#10b981',
            downColor: '#ef4444',
            borderDownColor: '#ef4444',
            borderUpColor: '#10b981',
            wickDownColor: '#ef4444',
            wickUpColor: '#10b981'
        });

        candlestickSeries.setData(data.candles);

        // Add default indicators
        this.addIndicatorToChart(chart, 'sma20', data.indicators?.sma20);
        this.addIndicatorToChart(chart, 'sma50', data.indicators?.sma50);

        // Store chart reference
        this.charts.set(timeframe, {
            chart,
            candlestickSeries,
            container,
            data
        });

        // Add AI annotations if available
        if (data.aiAnnotations) {
            this.addAIAnnotations(chart, data.aiAnnotations);
        }

        // Add pattern highlights
        if (data.patterns) {
            this.addPatternHighlights(chart, data.patterns);
        }

        // Setup chart event listeners
        this.setupChartEvents(timeframe, chart);
    }

    addIndicatorToChart(chart, indicatorType, data) {
        if (!data) return;

        switch (indicatorType) {
            case 'sma20':
                const sma20Series = chart.addLineSeries({
                    color: '#fbbf24',
                    lineWidth: 1,
                    title: 'SMA 20'
                });
                sma20Series.setData(data);
                break;

            case 'sma50':
                const sma50Series = chart.addLineSeries({
                    color: '#8b5cf6',
                    lineWidth: 1,
                    title: 'SMA 50'
                });
                sma50Series.setData(data);
                break;

            case 'bollinger':
                const upperBand = chart.addLineSeries({
                    color: '#6b7280',
                    lineWidth: 1,
                    title: 'BB Upper'
                });
                const lowerBand = chart.addLineSeries({
                    color: '#6b7280',
                    lineWidth: 1,
                    title: 'BB Lower'
                });
                upperBand.setData(data.upper);
                lowerBand.setData(data.lower);
                break;

            case 'rsi':
                // RSI would be in a separate pane
                this.addRSIPane(chart, data);
                break;
        }

        this.indicators.set(indicatorType, { chart, data, visible: true });
    }

    addAIAnnotations(chart, annotations) {
        annotations.forEach(annotation => {
            const marker = {
                time: annotation.time,
                position: annotation.type === 'buy' ? 'belowBar' : 'aboveBar',
                color: annotation.type === 'buy' ? '#10b981' : '#ef4444',
                shape: annotation.type === 'buy' ? 'arrowUp' : 'arrowDown',
                text: `AI: ${annotation.signal} (${annotation.confidence}%)`
            };

            chart.addMarker(marker);
        });
    }

    addPatternHighlights(chart, patterns) {
        patterns.forEach(pattern => {
            // Add pattern highlight overlay
            const highlight = {
                time: pattern.startTime,
                endTime: pattern.endTime,
                color: pattern.color || '#fbbf24',
                opacity: 0.2,
                title: pattern.name
            };

            // This would require custom overlay implementation
            this.createPatternOverlay(chart, highlight);
        });
    }

    setupChartEvents(timeframe, chart) {
        chart.subscribeCrosshairMove((param) => {
            if (this.syncEnabled) {
                this.syncCrosshair(timeframe, param);
            }
        });

        chart.subscribeVisibleTimeRangeChange((newVisibleTimeRange) => {
            if (this.syncEnabled) {
                this.syncTimeRange(timeframe, newVisibleTimeRange);
            }
        });
    }

    setupChartSync() {
        // Implement cross-chart synchronization
        this.charts.forEach((chartObj, timeframe) => {
            const { chart } = chartObj;
            
            chart.subscribeCrosshairMove((param) => {
                if (!this.syncEnabled) return;
                
                // Sync crosshair across all charts
                this.charts.forEach((otherChartObj, otherTimeframe) => {
                    if (timeframe !== otherTimeframe) {
                        // Calculate corresponding time for different timeframes
                        const syncedTime = this.calculateSyncedTime(param.time, timeframe, otherTimeframe);
                        if (syncedTime) {
                            otherChartObj.chart.setCrosshairPosition(syncedTime, param.point?.y);
                        }
                    }
                });
            });
        });
    }

    calculateSyncedTime(time, fromTimeframe, toTimeframe) {
        // Convert time between different timeframes for synchronization
        if (!time) return null;
        
        const timeframeMinutes = {
            '1H': 60,
            '4H': 240,
            '1D': 1440
        };
        
        const fromMinutes = timeframeMinutes[fromTimeframe];
        const toMinutes = timeframeMinutes[toTimeframe];
        
        if (!fromMinutes || !toMinutes) return null;
        
        // Align to the appropriate timeframe boundary
        const timestamp = new Date(time * 1000);
        const alignedTime = new Date(timestamp);
        
        if (toMinutes >= 60) {
            alignedTime.setMinutes(0, 0, 0);
        }
        if (toMinutes >= 240) {
            alignedTime.setHours(Math.floor(alignedTime.getHours() / 4) * 4);
        }
        if (toMinutes >= 1440) {
            alignedTime.setHours(0);
        }
        
        return Math.floor(alignedTime.getTime() / 1000);
    }

    switchTimeframe(timeframe) {
        this.activeTimeframe = timeframe;
        
        // Update active button
        document.querySelectorAll('.timeframe-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.timeframe === timeframe);
        });

        // Highlight active chart
        document.querySelectorAll('.chart-panel').forEach(panel => {
            panel.classList.toggle('active', panel.dataset.timeframe === timeframe);
        });

        console.log(`Switched to ${timeframe} timeframe`);
    }

    toggleSync() {
        this.syncEnabled = !this.syncEnabled;
        
        const syncBtn = document.querySelector('.sync-btn');
        syncBtn.classList.toggle('active', this.syncEnabled);
        
        const statusElement = document.querySelector('.sync-status');
        statusElement.textContent = this.syncEnabled ? 'Synced' : 'Independent';
        
        if (this.syncEnabled) {
            this.setupChartSync();
        }
        
        console.log(`Chart sync ${this.syncEnabled ? 'enabled' : 'disabled'}`);
    }

    toggleIndicatorsPanel() {
        const panel = document.getElementById('indicators-panel');
        panel.classList.toggle('active');
    }

    toggleIndicator(indicatorType, enabled) {
        const indicator = this.indicators.get(indicatorType);
        if (!indicator) return;

        // Toggle indicator visibility across all charts
        this.charts.forEach((chartObj) => {
            if (enabled) {
                this.addIndicatorToChart(chartObj.chart, indicatorType, indicator.data);
            } else {
                this.removeIndicatorFromChart(chartObj.chart, indicatorType);
            }
        });

        indicator.visible = enabled;
        console.log(`${indicatorType} indicator ${enabled ? 'enabled' : 'disabled'}`);
    }

    removeIndicatorFromChart(chart, indicatorType) {
        // Remove indicator series from chart
        // This would require tracking series references
        console.log(`Removing ${indicatorType} from chart`);
    }

    togglePatternHighlights() {
        // Toggle AI pattern highlighting
        const btn = document.querySelector('.patterns-btn');
        const enabled = !btn.classList.contains('active');
        btn.classList.toggle('active', enabled);
        
        this.charts.forEach((chartObj) => {
            // Toggle pattern visibility
            if (enabled) {
                this.showPatterns(chartObj.chart);
            } else {
                this.hidePatterns(chartObj.chart);
            }
        });
    }

    toggleFullscreen() {
        const container = document.getElementById('unified-charts-container');
        if (!document.fullscreenElement) {
            container.requestFullscreen();
        } else {
            document.exitFullscreen();
        }
    }

    handleKeyboardShortcuts(e) {
        if (e.ctrlKey || e.metaKey) {
            switch (e.key) {
                case '1':
                    e.preventDefault();
                    this.switchTimeframe('1H');
                    break;
                case '4':
                    e.preventDefault();
                    this.switchTimeframe('4H');
                    break;
                case 'd':
                    e.preventDefault();
                    this.switchTimeframe('1D');
                    break;
                case 's':
                    e.preventDefault();
                    this.toggleSync();
                    break;
                case 'i':
                    e.preventDefault();
                    this.toggleIndicatorsPanel();
                    break;
            }
        }
    }

    updateChartInfo(timeframe, data) {
        const panel = document.querySelector(`[data-timeframe="${timeframe}"]`);
        if (!panel || !data.current) return;

        const priceDisplay = panel.querySelector('.price-display');
        const changeDisplay = panel.querySelector('.change-display');

        priceDisplay.textContent = `$${data.current.price.toFixed(2)}`;
        
        const change = data.current.change || 0;
        const changePercent = data.current.changePercent || 0;
        
        changeDisplay.textContent = `${change >= 0 ? '+' : ''}${changePercent.toFixed(2)}%`;
        changeDisplay.className = `change-display ${change >= 0 ? 'positive' : 'negative'}`;
    }

    showChartError(timeframe) {
        const container = document.getElementById(`chart-${timeframe}`);
        if (!container) return;

        container.innerHTML = `
            <div class="chart-error">
                <i class="fas fa-exclamation-triangle"></i>
                <p>Failed to load ${timeframe} chart data</p>
                <button onclick="window.unifiedChartSystem.loadChartData()">Retry</button>
            </div>
        `;
    }

    initializeIndicators() {
        // Initialize default indicators
        const defaultIndicators = ['sma20', 'sma50'];
        defaultIndicators.forEach(indicator => {
            this.indicators.set(indicator, { visible: true });
        });
    }

    // Performance optimization methods
    optimizeChartPerformance() {
        this.charts.forEach((chartObj) => {
            // Implement data compression for large datasets
            // Use requestAnimationFrame for smooth updates
            // Implement viewport culling for off-screen charts
        });
    }

    // Memory management
    dispose() {
        this.charts.forEach((chartObj) => {
            chartObj.chart.remove();
        });
        this.charts.clear();
        this.indicators.clear();
        console.log('✅ Unified Chart System disposed');
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.unifiedChartSystem = new UnifiedChartSystem();
});

/**
 * GoldGPT Chart Fix - Clean Chart Implementation
 * Fixes syntax errors and ensures charts load properly
 */

// Clean Chart Manager without syntax errors
class CleanChartManager {
    constructor() {
        this.currentTimeframe = '1h';
        this.activeIndicators = new Set(['volume']);
        this.chart = null;
        this.candlestickSeries = null;
        this.volumeSeries = null;
        this.isInitialized = false;
    }

    init() {
        console.log('🔧 Clean Chart Manager initializing...');
        
        // Wait for DOM and libraries to load
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.initChart());
        } else {
            this.initChart();
        }
    }

    initChart() {
        const container = document.getElementById('tradingview-chart');
        if (!container) {
            console.error('❌ Chart container not found');
            return;
        }

        // Clear any existing content
        container.innerHTML = '';

        // Try LightweightCharts first
        if (typeof LightweightCharts !== 'undefined') {
            console.log('📊 Using LightweightCharts...');
            this.initLightweightCharts(container);
        } else if (typeof TradingView !== 'undefined') {
            console.log('📊 Using TradingView widget...');
            this.initTradingViewWidget(container);
        } else {
            console.log('📊 Using basic chart fallback...');
            this.initBasicChart(container);
        }

        this.setupEventListeners();
    }

    initLightweightCharts(container) {
        try {
            this.chart = LightweightCharts.createChart(container, {
                width: container.clientWidth || 800,
                height: 500,
                layout: {
                    backgroundColor: '#1a1a1a',
                    textColor: '#d1d4dc',
                },
                grid: {
                    vertLines: { color: '#2B2B43' },
                    horzLines: { color: '#2B2B43' },
                },
                crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
                rightPriceScale: { borderColor: '#485c7b' },
                timeScale: { borderColor: '#485c7b' }
            });

            this.candlestickSeries = this.chart.addCandlestickSeries({
                upColor: '#00d084',
                downColor: '#ff4976',
                borderVisible: false,
                wickUpColor: '#00d084',
                wickDownColor: '#ff4976'
            });

            this.volumeSeries = this.chart.addHistogramSeries({
                color: '#26a69a',
                priceFormat: { type: 'volume' },
                priceScaleId: '',
                scaleMargins: { top: 0.8, bottom: 0 }
            });

            this.loadChartData();
            this.isInitialized = true;
            console.log('✅ LightweightCharts initialized successfully');
        } catch (error) {
            console.error('❌ LightweightCharts failed:', error);
            this.initTradingViewWidget(container);
        }
    }

    initTradingViewWidget(container) {
        if (typeof TradingView !== 'undefined') {
            try {
                new TradingView.widget({
                    "width": "100%",
                    "height": "500",
                    "symbol": "OANDA:XAUUSD",
                    "interval": this.convertTimeframeToInterval(this.currentTimeframe),
                    "timezone": "Etc/UTC",
                    "theme": "dark",
                    "style": "1",
                    "locale": "en",
                    "toolbar_bg": "#1a1a1a",
                    "enable_publishing": false,
                    "container_id": "tradingview-chart",
                    "studies": [
                        "RSI@tv-basicstudies",
                        "MACD@tv-basicstudies",
                        "Volume@tv-basicstudies",
                        "BB@tv-basicstudies"
                    ],
                    "overrides": {
                        "mainSeriesProperties.candleStyle.upColor": "#00d084",
                        "mainSeriesProperties.candleStyle.downColor": "#ff4757",
                        "paneProperties.background": "#141414"
                    }
                });
                this.isInitialized = true;
                console.log('✅ TradingView widget initialized');
            } catch (error) {
                console.error('❌ TradingView widget failed:', error);
                this.initBasicChart(container);
            }
        } else {
            this.initBasicChart(container);
        }
    }

    initBasicChart(container) {
        container.innerHTML = `
            <div style="width: 100%; height: 500px; background: #1a1a1a; border-radius: 8px; 
                        display: flex; align-items: center; justify-content: center; color: #d1d4dc;">
                <div style="text-align: center;">
                    <i class="fas fa-chart-line" style="font-size: 48px; color: #FFD700; margin-bottom: 16px;"></i>
                    <h3 style="margin: 0 0 8px 0;">Gold Chart Active</h3>
                    <p style="margin: 0; opacity: 0.7;">Real-time Gold: Loading... | ${this.currentTimeframe} timeframe</p>
                    <button onclick="window.cleanChartManager.loadChartData()" 
                            style="margin-top: 16px; padding: 8px 16px; background: #00d084; color: white; 
                                   border: none; border-radius: 4px; cursor: pointer;">Reload Chart</button>
                </div>
            </div>
        `;
        
        // Load real price
        this.updateBasicChartPrice();
        this.isInitialized = true;
        console.log('✅ Basic chart fallback initialized');
    }

    async updateBasicChartPrice() {
        try {
            const response = await fetch('/api/gold/price');
            const data = await response.json();
            const priceElement = document.querySelector('#tradingview-chart p');
            if (priceElement && data.price) {
                priceElement.innerHTML = `Real-time Gold: $${data.price} | ${this.currentTimeframe} timeframe`;
            }
        } catch (error) {
            console.warn('Could not update basic chart price:', error);
        }
    }

    setupEventListeners() {
        // Timeframe buttons
        document.querySelectorAll('.timeframe-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const timeframe = e.target.dataset.timeframe;
                if (timeframe) {
                    this.changeTimeframe(timeframe);
                }
            });
        });

        // Indicator buttons
        document.querySelectorAll('.indicator-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const indicator = e.target.dataset.indicator;
                if (indicator) {
                    this.toggleIndicator(indicator);
                }
            });
        });
    }

    changeTimeframe(timeframe) {
        console.log(`📅 Changing timeframe to ${timeframe}`);
        this.currentTimeframe = timeframe;

        // Update button states
        document.querySelectorAll('.timeframe-btn').forEach(btn => btn.classList.remove('active'));
        const activeBtn = document.querySelector(`[data-timeframe="${timeframe}"]`);
        if (activeBtn) activeBtn.classList.add('active');

        // Reload chart data
        if (this.isInitialized) {
            this.loadChartData();
        }

        this.showNotification(`Chart switched to ${timeframe} timeframe`, 'success');
    }

    toggleIndicator(indicator) {
        console.log(`📊 Toggling ${indicator} indicator`);
        
        const btn = document.querySelector(`[data-indicator="${indicator}"]`);
        if (btn) {
            btn.classList.toggle('active');
            const isActive = btn.classList.contains('active');
            
            if (isActive) {
                this.activeIndicators.add(indicator);
            } else {
                this.activeIndicators.delete(indicator);
            }

            this.showNotification(
                `${indicator} indicator ${isActive ? 'enabled' : 'disabled'}`,
                'success'
            );
        }
    }

    async loadChartData() {
        if (!this.candlestickSeries) return;

        try {
            const response = await fetch(`/api/chart/data/XAUUSD?timeframe=${this.currentTimeframe}`);
            const data = await response.json();
            
            if (data && data.length > 0) {
                this.candlestickSeries.setData(data);
                console.log(`✅ Loaded ${data.length} data points for ${this.currentTimeframe}`);
            }
        } catch (error) {
            console.error('❌ Failed to load chart data:', error);
            // Generate fallback data
            const fallbackData = this.generateFallbackData();
            if (this.candlestickSeries) {
                this.candlestickSeries.setData(fallbackData);
            }
        }
    }

    async generateFallbackData() {
        const data = [];
        let basePrice = 3350; // Default fallback
        
        // Try to get real current price
        try {
            const response = await fetch('/api/live-gold-price');
            if (response.ok) {
                const priceData = await response.json();
                basePrice = priceData.price;
            }
        } catch (error) {
            console.debug('Using fallback price for chart data');
        }
        
        const now = Math.floor(Date.now() / 1000);
        const interval = this.getTimeframeSeconds();

        for (let i = 100; i >= 0; i--) {
            const time = now - (i * interval);
            const open = basePrice + (Math.random() - 0.5) * 50;
            const close = open + (Math.random() - 0.5) * 20;
            const high = Math.max(open, close) + Math.random() * 10;
            const low = Math.min(open, close) - Math.random() * 10;

            data.push({
                time: time,
                open: parseFloat(open.toFixed(2)),
                high: parseFloat(high.toFixed(2)),
                low: parseFloat(low.toFixed(2)),
                close: parseFloat(close.toFixed(2))
            });
        }

        return data;
    }

    convertTimeframeToInterval(timeframe) {
        const mapping = {
            '1m': '1',
            '5m': '5',
            '15m': '15',
            '1h': '60',
            '4h': '240',
            '1d': 'D',
            '1w': 'W',
            '1M': 'M'
        };
        return mapping[timeframe] || '60';
    }

    getTimeframeSeconds() {
        const mapping = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '1h': 3600,
            '4h': 14400,
            '1d': 86400,
            '1w': 604800,
            '1M': 2592000
        };
        return mapping[this.currentTimeframe] || 3600;
    }

    showNotification(message, type = 'info') {
        console.log(`📢 ${message}`);
        
        // Create notification element if it doesn't exist
        let notification = document.getElementById('chart-notification');
        if (!notification) {
            notification = document.createElement('div');
            notification.id = 'chart-notification';
            notification.style.cssText = `
                position: fixed; top: 20px; right: 20px; z-index: 10000;
                padding: 12px 20px; border-radius: 6px; color: white;
                font-family: Arial, sans-serif; font-size: 14px;
                transition: opacity 0.3s ease;
            `;
            document.body.appendChild(notification);
        }

        // Set notification style based on type
        const colors = {
            success: '#00d084',
            warning: '#ffa500',
            error: '#ff4757',
            info: '#00a8ff'
        };

        notification.style.backgroundColor = colors[type] || colors.info;
        notification.textContent = message;
        notification.style.opacity = '1';

        // Auto-hide after 3 seconds
        setTimeout(() => {
            notification.style.opacity = '0';
        }, 3000);
    }
}

// Initialize the clean chart manager
window.cleanChartManager = new CleanChartManager();

// Auto-initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 DOM ready, initializing Clean Chart Manager...');
    
    // Small delay to ensure all libraries are loaded
    setTimeout(() => {
        window.cleanChartManager.init();
    }, 1000);
});

// Export for global access
window.CleanChartManager = CleanChartManager;

/**
 * Advanced Chart Manager for GoldGPT
 * Handles TradingView widgets and LightweightCharts integration
 */

class AdvancedChartManager {
    constructor() {
        console.log('🚀 Initializing Advanced Chart Manager...');
        this.chart = null;
        this.candlestickSeries = null;
        this.indicators = new Map();
        this.currentTimeframe = '1D';
        this.isInitialized = false;
        this.chartContainer = null;
        
        // Initialize immediately
        this.init();
    }

    async init() {
        try {
            console.log('🔧 Setting up chart manager...');
            
            // Wait for DOM to be ready
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', () => this.setupChart());
            } else {
                this.setupChart();
            }
            
        } catch (error) {
            console.error('❌ Chart Manager initialization error:', error);
        }
    }

    setupChart() {
        try {
            // Find chart container
            this.chartContainer = document.getElementById('chart-container') || 
                                document.getElementById('tradingview-chart') ||
                                document.querySelector('.chart-container');
            
            if (!this.chartContainer) {
                console.warn('⚠️ Chart container not found, chart will be available when container is ready');
                return;
            }

            console.log('✅ Chart container found:', this.chartContainer.id);
            
            // Set up event listeners for chart controls
            this.setupEventListeners();
            
            // Mark as initialized
            this.isInitialized = true;
            console.log('✅ Chart Manager setup complete');
            
        } catch (error) {
            console.error('❌ Chart setup error:', error);
        }
    }

    setupEventListeners() {
        try {
            // Timeframe buttons
            document.querySelectorAll('.timeframe-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const timeframe = e.target.getAttribute('data-timeframe');
                    if (timeframe) {
                        this.changeTimeframe(timeframe);
                    }
                });
            });

            // Indicator toggles
            document.querySelectorAll('.indicator-toggle').forEach(toggle => {
                toggle.addEventListener('change', (e) => {
                    const indicator = e.target.getAttribute('data-indicator');
                    if (indicator) {
                        this.toggleIndicator(indicator, e.target.checked);
                    }
                });
            });

            console.log('✅ Event listeners set up');
        } catch (error) {
            console.error('❌ Event listener setup error:', error);
        }
    }

    changeTimeframe(timeframe) {
        try {
            console.log(`📊 Changing timeframe to: ${timeframe}`);
            this.currentTimeframe = timeframe;
            
            // Update active timeframe button
            document.querySelectorAll('.timeframe-btn').forEach(btn => {
                btn.classList.remove('active');
                if (btn.getAttribute('data-timeframe') === timeframe) {
                    btn.classList.add('active');
                }
            });

            // Emit timeframe change event
            if (typeof window.socket !== 'undefined') {
                window.socket.emit('timeframe_change', { timeframe: timeframe });
            }
            
        } catch (error) {
            console.error('❌ Timeframe change error:', error);
        }
    }

    toggleIndicator(indicator, enabled) {
        try {
            console.log(`📈 Toggling indicator ${indicator}:`, enabled);
            
            if (enabled) {
                this.addIndicator(indicator);
            } else {
                this.removeIndicator(indicator);
            }
            
        } catch (error) {
            console.error('❌ Indicator toggle error:', error);
        }
    }

    addIndicator(indicator) {
        try {
            console.log(`➕ Adding indicator: ${indicator}`);
            
            // Emit indicator add event
            if (typeof window.socket !== 'undefined') {
                window.socket.emit('add_indicator', { indicator: indicator });
            }
            
            this.indicators.set(indicator, { active: true });
            
        } catch (error) {
            console.error(`❌ Error adding indicator ${indicator}:`, error);
        }
    }

    removeIndicator(indicator) {
        try {
            console.log(`➖ Removing indicator: ${indicator}`);
            
            // Emit indicator remove event
            if (typeof window.socket !== 'undefined') {
                window.socket.emit('remove_indicator', { indicator: indicator });
            }
            
            this.indicators.delete(indicator);
            
        } catch (error) {
            console.error(`❌ Error removing indicator ${indicator}:`, error);
        }
    }

    updateChartData(data) {
        try {
            if (!this.isInitialized) {
                console.log('📊 Chart not initialized yet, skipping data update');
                return;
            }

            console.log('📊 Updating chart with new data:', data);
            
            // Emit chart data update event
            if (typeof window.socket !== 'undefined') {
                window.socket.emit('chart_data_update', data);
            }
            
        } catch (error) {
            console.error('❌ Chart data update error:', error);
        }
    }

    getStatus() {
        return {
            initialized: this.isInitialized,
            timeframe: this.currentTimeframe,
            indicators: Array.from(this.indicators.keys()),
            hasContainer: !!this.chartContainer
        };
    }

    destroy() {
        try {
            console.log('🗑️ Destroying chart manager...');
            
            if (this.chart) {
                this.chart.remove();
                this.chart = null;
            }
            
            this.indicators.clear();
            this.isInitialized = false;
            
            console.log('✅ Chart manager destroyed');
        } catch (error) {
            console.error('❌ Chart manager destroy error:', error);
        }
    }
}

// Initialize chart manager when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    if (typeof window.chartManager === 'undefined') {
        window.chartManager = new AdvancedChartManager();
    }
});

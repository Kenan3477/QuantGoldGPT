// Internal Chart Analysis Module
// This module provides chart implementations for technical analysis and data processing
// These charts are NOT displayed in the UI - they're used for internal calculations only

class InternalChartAnalysisEngine {
    constructor() {
        this.chartInstances = new Map();
        this.analysisCache = new Map();
        this.indicators = new Map();
        this.isInitialized = false;
        
        console.log('🔧 Internal Chart Analysis Engine initialized');
    }

    async initialize() {
        if (this.isInitialized) return;
        
        try {
            // Wait for required libraries to load
            await this.waitForLibraries();
            
            // Initialize internal chart instances for analysis
            this.initializeInternalCharts();
            
            this.isInitialized = true;
            console.log('✅ Internal Chart Analysis Engine ready');
        } catch (error) {
            console.error('❌ Failed to initialize Internal Chart Analysis Engine:', error);
        }
    }

    async waitForLibraries() {
        // Wait for Chart.js
        await this.waitForGlobal('Chart', 5000);
        
        // Wait for LightweightCharts (optional - for advanced analysis)
        try {
            await this.waitForGlobal('LightweightCharts', 2000);
            console.log('✅ LightweightCharts available for advanced analysis');
        } catch (error) {
            console.warn('⚠️ LightweightCharts not available - using Chart.js only');
        }
    }

    waitForGlobal(globalName, timeout = 5000) {
        return new Promise((resolve, reject) => {
            const startTime = Date.now();
            
            const checkGlobal = () => {
                if (window[globalName]) {
                    resolve(window[globalName]);
                } else if (Date.now() - startTime > timeout) {
                    reject(new Error(`${globalName} not loaded within ${timeout}ms`));
                } else {
                    setTimeout(checkGlobal, 100);
                }
            };
            
            checkGlobal();
        });
    }

    initializeInternalCharts() {
        // Create hidden container for internal charts
        this.createHiddenContainer();
        
        // Initialize Chart.js instance for technical analysis
        this.initializeChartJSAnalysis();
        
        // Initialize LightweightCharts instance if available
        if (window.LightweightCharts) {
            this.initializeLightweightChartsAnalysis();
        }
    }

    createHiddenContainer() {
        // Create a hidden container for internal chart processing
        const hiddenContainer = document.createElement('div');
        hiddenContainer.id = 'internal-chart-container';
        hiddenContainer.style.cssText = `
            position: absolute;
            top: -9999px;
            left: -9999px;
            width: 800px;
            height: 400px;
            visibility: hidden;
            pointer-events: none;
        `;
        document.body.appendChild(hiddenContainer);

        // Create canvas for Chart.js
        const canvas = document.createElement('canvas');
        canvas.id = 'internal-chartjs-canvas';
        canvas.width = 800;
        canvas.height = 400;
        hiddenContainer.appendChild(canvas);

        // Create div for LightweightCharts
        const lwDiv = document.createElement('div');
        lwDiv.id = 'internal-lightweight-container';
        lwDiv.style.cssText = 'width: 800px; height: 400px;';
        hiddenContainer.appendChild(lwDiv);
    }

    initializeChartJSAnalysis() {
        const canvas = document.getElementById('internal-chartjs-canvas');
        if (!canvas) return;

        this.chartJSInstance = new Chart(canvas, {
            type: 'candlestick',
            data: {
                datasets: [{
                    label: 'OHLC Data',
                    data: [],
                    borderColor: '#00d084',
                    backgroundColor: 'rgba(0, 208, 132, 0.1)'
                }, {
                    label: 'Volume',
                    type: 'bar',
                    data: [],
                    backgroundColor: 'rgba(38, 166, 154, 0.5)',
                    yAxisID: 'y1'
                }]
            },
            options: {
                responsive: false,
                animation: false,
                plugins: {
                    legend: { display: false },
                    tooltip: { enabled: false }
                },
                scales: {
                    x: { display: false },
                    y: { display: false },
                    y1: { display: false, position: 'right' }
                }
            }
        });

        this.chartInstances.set('chartjs', this.chartJSInstance);
        console.log('📊 Internal Chart.js instance created for analysis');
    }

    initializeLightweightChartsAnalysis() {
        const container = document.getElementById('internal-lightweight-container');
        if (!container || !window.LightweightCharts) return;

        this.lightweightInstance = LightweightCharts.createChart(container, {
            width: 800,
            height: 400,
            layout: {
                background: { color: 'transparent' },
                textColor: '#ffffff'
            },
            grid: {
                vertLines: { visible: false },
                horzLines: { visible: false }
            },
            crosshair: { mode: 0 }, // Hidden
            timeScale: { visible: false },
            rightPriceScale: { visible: false },
            leftPriceScale: { visible: false }
        });

        this.candlestickSeries = this.lightweightInstance.addCandlestickSeries({
            upColor: '#00d084',
            downColor: '#ff4757',
            borderUpColor: '#00d084',
            borderDownColor: '#ff4757',
            wickUpColor: '#00d084',
            wickDownColor: '#ff4757'
        });

        this.volumeSeries = this.lightweightInstance.addHistogramSeries({
            color: '#26a69a',
            priceFormat: { type: 'volume' },
            priceScaleId: '',
            scaleMargins: { top: 0.8, bottom: 0 }
        });

        this.chartInstances.set('lightweight', this.lightweightInstance);
        console.log('📊 Internal LightweightCharts instance created for analysis');
    }

    // Data processing methods
    updateAnalysisData(symbol, ohlcData, volumeData) {
        if (!this.isInitialized) {
            console.warn('⚠️ Analysis engine not initialized');
            return;
        }

        try {
            // Update Chart.js instance
            if (this.chartJSInstance) {
                this.chartJSInstance.data.datasets[0].data = ohlcData;
                this.chartJSInstance.data.datasets[1].data = volumeData;
                this.chartJSInstance.update('none');
            }

            // Update LightweightCharts instance
            if (this.candlestickSeries && this.volumeSeries) {
                this.candlestickSeries.setData(ohlcData);
                this.volumeSeries.setData(volumeData);
            }

            // Cache the data for analysis
            this.analysisCache.set(symbol, {
                ohlc: ohlcData,
                volume: volumeData,
                timestamp: Date.now()
            });

            // Trigger technical analysis calculations
            this.performTechnicalAnalysis(symbol, ohlcData);

        } catch (error) {
            console.error('❌ Error updating analysis data:', error);
        }
    }

    performTechnicalAnalysis(symbol, ohlcData) {
        if (!ohlcData || ohlcData.length < 14) return;

        try {
            const analysis = {
                rsi: this.calculateRSI(ohlcData),
                macd: this.calculateMACD(ohlcData),
                bollingerBands: this.calculateBollingerBands(ohlcData),
                sma: this.calculateSMA(ohlcData, 20),
                ema: this.calculateEMA(ohlcData, 20),
                timestamp: Date.now()
            };

            // Cache analysis results
            this.indicators.set(symbol, analysis);

            // Emit analysis update event
            window.dispatchEvent(new CustomEvent('internal-analysis-update', {
                detail: { symbol, analysis }
            }));

            console.log(`📈 Technical analysis updated for ${symbol}`);
        } catch (error) {
            console.error('❌ Error performing technical analysis:', error);
        }
    }

    // Technical indicator calculations
    calculateRSI(data, period = 14) {
        if (data.length < period + 1) return [];

        const rsi = [];
        for (let i = period; i < data.length; i++) {
            let gains = 0;
            let losses = 0;

            for (let j = i - period + 1; j <= i; j++) {
                const change = data[j].c - data[j - 1].c;
                if (change > 0) gains += change;
                else losses -= change;
            }

            const avgGain = gains / period;
            const avgLoss = losses / period;
            const rs = avgGain / avgLoss;
            const rsiValue = 100 - (100 / (1 + rs));

            rsi.push({
                time: data[i].time || data[i].t,
                value: parseFloat(rsiValue.toFixed(2))
            });
        }

        return rsi;
    }

    calculateMACD(data, fastPeriod = 12, slowPeriod = 26, signalPeriod = 9) {
        if (data.length < slowPeriod) return [];

        const macd = [];
        for (let i = slowPeriod; i < data.length; i++) {
            const fastEMA = this.calculateEMA(data.slice(i - fastPeriod + 1, i + 1), fastPeriod);
            const slowEMA = this.calculateEMA(data.slice(i - slowPeriod + 1, i + 1), slowPeriod);
            const macdValue = fastEMA - slowEMA;

            macd.push({
                time: data[i].time || data[i].t,
                value: parseFloat(macdValue.toFixed(4))
            });
        }

        return macd;
    }

    calculateBollingerBands(data, period = 20, stdDev = 2) {
        if (data.length < period) return { upper: [], lower: [], middle: [] };

        const upper = [];
        const lower = [];
        const middle = [];

        for (let i = period - 1; i < data.length; i++) {
            const slice = data.slice(i - period + 1, i + 1);
            const sma = slice.reduce((sum, d) => sum + d.c, 0) / period;
            const variance = slice.reduce((sum, d) => sum + Math.pow(d.c - sma, 2), 0) / period;
            const standardDev = Math.sqrt(variance);

            const time = data[i].time || data[i].t;
            
            upper.push({
                time,
                value: parseFloat((sma + (standardDev * stdDev)).toFixed(2))
            });

            lower.push({
                time,
                value: parseFloat((sma - (standardDev * stdDev)).toFixed(2))
            });

            middle.push({
                time,
                value: parseFloat(sma.toFixed(2))
            });
        }

        return { upper, lower, middle };
    }

    calculateSMA(data, period) {
        if (data.length < period) return [];

        const sma = [];
        for (let i = period - 1; i < data.length; i++) {
            const slice = data.slice(i - period + 1, i + 1);
            const average = slice.reduce((sum, d) => sum + d.c, 0) / period;
            
            sma.push({
                time: data[i].time || data[i].t,
                value: parseFloat(average.toFixed(2))
            });
        }

        return sma;
    }

    calculateEMA(data, period) {
        if (data.length === 0) return 0;
        if (data.length === 1) return data[0].c;

        const multiplier = 2 / (period + 1);
        let ema = data[0].c;

        for (let i = 1; i < data.length; i++) {
            ema = (data[i].c * multiplier) + (ema * (1 - multiplier));
        }

        return ema;
    }

    // Public API methods
    getAnalysisForSymbol(symbol) {
        return this.indicators.get(symbol) || null;
    }

    getCachedData(symbol) {
        return this.analysisCache.get(symbol) || null;
    }

    clearCache() {
        this.analysisCache.clear();
        this.indicators.clear();
        console.log('🧹 Analysis cache cleared');
    }

    destroy() {
        // Clean up chart instances
        if (this.chartJSInstance) {
            this.chartJSInstance.destroy();
        }

        if (this.lightweightInstance) {
            this.lightweightInstance.remove();
        }

        // Remove hidden container
        const container = document.getElementById('internal-chart-container');
        if (container) {
            container.remove();
        }

        this.chartInstances.clear();
        this.analysisCache.clear();
        this.indicators.clear();
        this.isInitialized = false;

        console.log('🗑️ Internal Chart Analysis Engine destroyed');
    }
}

// Create global instance with proper init method
window.internalChartAnalysis = new InternalChartAnalysisEngine();

// Add init method for component loader compatibility
window.internalChartAnalysis.init = function() {
    return this.initialize();
};

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.internalChartAnalysis.initialize();
    });
} else {
    window.internalChartAnalysis.initialize();
}

console.log('📊 Internal Chart Analysis Module loaded');

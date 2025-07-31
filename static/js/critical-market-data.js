/**
 * Critical Market Data Panel for Gold Trading
 * Displays macro economic indicators, ETF flows, positioning data
 */

class CriticalMarketDataPanel {
    constructor() {
        this.marketData = {};
        this.updateInterval = 60000; // 1 minute
        this.updateTimer = null;
        
        this.initializePanel();
        this.startAutoUpdate();
    }
    
    initializePanel() {
        this.createMarketDataPanel();
        this.loadCriticalData();
    }
    
    createMarketDataPanel() {
        const panel = document.createElement('div');
        panel.className = 'dashboard-card critical-data-panel';
        panel.innerHTML = `
            <div class="card-header">
                <div class="card-title">
                    <i class="fas fa-globe-americas"></i>
                    Critical Market Data
                </div>
                <div class="data-controls">
                    <button class="btn-small" onclick="criticalDataPanel.refreshData()">
                        <i class="fas fa-sync-alt"></i>
                    </button>
                    <div class="data-status" id="data-status">
                        <i class="fas fa-circle"></i>
                        <span>Loading...</span>
                    </div>
                </div>
            </div>
            <div class="card-content">
                <!-- Macro Economic Indicators -->
                <div class="macro-section">
                    <div class="section-title">
                        <i class="fas fa-chart-line"></i>
                        Macro Indicators
                    </div>
                    <div class="macro-grid">
                        <div class="macro-item">
                            <div class="macro-label">DXY (Dollar Index)</div>
                            <div class="macro-value" id="dxy-value">--</div>
                            <div class="macro-change" id="dxy-change">--</div>
                            <div class="macro-impact" id="dxy-impact">NEUTRAL</div>
                        </div>
                        <div class="macro-item">
                            <div class="macro-label">Real Interest Rates</div>
                            <div class="macro-value" id="real-rates-value">--</div>
                            <div class="macro-change" id="real-rates-change">--</div>
                            <div class="macro-impact" id="real-rates-impact">NEUTRAL</div>
                        </div>
                        <div class="macro-item">
                            <div class="macro-label">Inflation (CPI)</div>
                            <div class="macro-value" id="cpi-value">--</div>
                            <div class="macro-change" id="cpi-change">--</div>
                            <div class="macro-impact" id="cpi-impact">NEUTRAL</div>
                        </div>
                        <div class="macro-item">
                            <div class="macro-label">Fed Funds Rate</div>
                            <div class="macro-value" id="fed-rate-value">--</div>
                            <div class="macro-change" id="fed-rate-change">--</div>
                            <div class="macro-impact" id="fed-rate-impact">NEUTRAL</div>
                        </div>
                    </div>
                </div>
                
                <!-- ETF Flows -->
                <div class="etf-flows-section">
                    <div class="section-title">
                        <i class="fas fa-exchange-alt"></i>
                        ETF Flows (Gold)
                    </div>
                    <div class="etf-flows-grid">
                        <div class="etf-item">
                            <div class="etf-name">SPDR Gold (GLD)</div>
                            <div class="etf-flow" id="gld-flow">--</div>
                            <div class="etf-volume" id="gld-volume">--</div>
                        </div>
                        <div class="etf-item">
                            <div class="etf-name">iShares Gold (IAU)</div>
                            <div class="etf-flow" id="iau-flow">--</div>
                            <div class="etf-volume" id="iau-volume">--</div>
                        </div>
                        <div class="flow-indicator">
                            <div class="flow-direction" id="overall-flow-direction">NEUTRAL</div>
                            <div class="flow-strength" id="flow-strength">--</div>
                        </div>
                    </div>
                </div>
                
                <!-- CFTC Positioning -->
                <div class="cftc-section">
                    <div class="section-title">
                        <i class="fas fa-users"></i>
                        CFTC Positioning
                    </div>
                    <div class="cftc-grid">
                        <div class="positioning-chart">
                            <canvas id="cftc-chart" width="200" height="100"></canvas>
                        </div>
                        <div class="positioning-data">
                            <div class="position-item">
                                <span class="position-label">Speculative Long</span>
                                <span class="position-value" id="spec-long">--</span>
                            </div>
                            <div class="position-item">
                                <span class="position-label">Speculative Short</span>
                                <span class="position-value" id="spec-short">--</span>
                            </div>
                            <div class="position-item">
                                <span class="position-label">Net Positioning</span>
                                <span class="position-value" id="net-positioning">--</span>
                            </div>
                            <div class="position-item">
                                <span class="position-label">Sentiment</span>
                                <span class="position-value" id="cftc-sentiment">--</span>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Risk Factors -->
                <div class="risk-factors-section">
                    <div class="section-title">
                        <i class="fas fa-exclamation-triangle"></i>
                        Risk Factors
                    </div>
                    <div class="risk-grid">
                        <div class="risk-item">
                            <div class="risk-label">Geopolitical Tension</div>
                            <div class="risk-meter">
                                <div class="risk-fill" id="geo-risk-fill"></div>
                            </div>
                            <div class="risk-value" id="geo-risk-value">--</div>
                        </div>
                        <div class="risk-item">
                            <div class="risk-label">Currency Debasement</div>
                            <div class="risk-meter">
                                <div class="risk-fill" id="currency-risk-fill"></div>
                            </div>
                            <div class="risk-value" id="currency-risk-value">--</div>
                        </div>
                        <div class="risk-item">
                            <div class="risk-label">Central Bank Stress</div>
                            <div class="risk-meter">
                                <div class="risk-fill" id="cb-stress-fill"></div>
                            </div>
                            <div class="risk-value" id="cb-stress-value">--</div>
                        </div>
                    </div>
                </div>
                
                <!-- Market Regime -->
                <div class="regime-section">
                    <div class="section-title">
                        <i class="fas fa-compass"></i>
                        Market Regime
                    </div>
                    <div class="regime-indicator">
                        <div class="regime-circle" id="regime-circle">
                            <div class="regime-text" id="regime-text">--</div>
                        </div>
                        <div class="regime-description" id="regime-description">
                            Analyzing market conditions...
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // Insert into dashboard
        const dashboardGrid = document.querySelector('.dashboard-grid');
        if (dashboardGrid) {
            dashboardGrid.appendChild(panel);
        }
    }
    
    async loadCriticalData() {
        try {
            // Set loading status
            this.updateStatus('loading', 'Loading critical data...');
            
            // Fetch critical market data
            const response = await fetch('/api/critical_market_data');
            const data = await response.json();
            
            if (data.success) {
                this.marketData = data.data;
                this.updateAllPanels();
                this.updateStatus('connected', 'Data updated');
            } else {
                throw new Error(data.error || 'Failed to load data');
            }
            
        } catch (error) {
            console.error('Error loading critical data:', error);
            this.updateStatus('error', 'Data unavailable');
            this.loadFallbackData();
        }
    }
    
    updateAllPanels() {
        this.updateMacroIndicators();
        this.updateETFFlows();
        this.updateCFTCPositioning();
        this.updateRiskFactors();
        this.updateMarketRegime();
    }
    
    updateMacroIndicators() {
        const macro = this.marketData.macro_indicators || {};
        
        // DXY
        this.updateMacroItem('dxy', macro.dxy || {});
        
        // Real Interest Rates
        this.updateMacroItem('real-rates', macro.real_rates || {});
        
        // CPI
        this.updateMacroItem('cpi', macro.cpi || {});
        
        // Fed Funds Rate
        this.updateMacroItem('fed-rate', macro.fed_rate || {});
    }
    
    updateMacroItem(prefix, data) {
        const valueEl = document.getElementById(`${prefix}-value`);
        const changeEl = document.getElementById(`${prefix}-change`);
        const impactEl = document.getElementById(`${prefix}-impact`);
        
        if (valueEl) valueEl.textContent = this.formatValue(data.value);
        if (changeEl) {
            changeEl.textContent = this.formatChange(data.change);
            changeEl.className = `macro-change ${data.change >= 0 ? 'positive' : 'negative'}`;
        }
        if (impactEl) {
            impactEl.textContent = data.impact || 'NEUTRAL';
            impactEl.className = `macro-impact ${(data.impact || 'neutral').toLowerCase()}`;
        }
    }
    
    updateETFFlows() {
        const etfFlows = this.marketData.etf_flows || {};
        
        // GLD
        const gldFlow = document.getElementById('gld-flow');
        const gldVolume = document.getElementById('gld-volume');
        if (gldFlow) gldFlow.textContent = this.formatFlow(etfFlows.gld_flow);
        if (gldVolume) gldVolume.textContent = this.formatVolume(etfFlows.gld_volume);
        
        // IAU
        const iauFlow = document.getElementById('iau-flow');
        const iauVolume = document.getElementById('iau-volume');
        if (iauFlow) iauFlow.textContent = this.formatFlow(etfFlows.iau_flow);
        if (iauVolume) iauVolume.textContent = this.formatVolume(etfFlows.iau_volume);
        
        // Overall flow direction
        const flowDirection = document.getElementById('overall-flow-direction');
        const flowStrength = document.getElementById('flow-strength');
        if (flowDirection) {
            flowDirection.textContent = etfFlows.direction || 'NEUTRAL';
            flowDirection.className = `flow-direction ${(etfFlows.direction || 'neutral').toLowerCase()}`;
        }
        if (flowStrength) flowStrength.textContent = `${etfFlows.strength || 0}%`;
    }
    
    updateCFTCPositioning() {
        const cftc = this.marketData.cftc_positioning || {};
        
        // Update positioning values
        const specLong = document.getElementById('spec-long');
        const specShort = document.getElementById('spec-short');
        const netPositioning = document.getElementById('net-positioning');
        const cftcSentiment = document.getElementById('cftc-sentiment');
        
        if (specLong) specLong.textContent = this.formatContracts(cftc.spec_long);
        if (specShort) specShort.textContent = this.formatContracts(cftc.spec_short);
        if (netPositioning) {
            const net = (cftc.spec_long || 0) - (cftc.spec_short || 0);
            netPositioning.textContent = this.formatContracts(net);
            netPositioning.className = `position-value ${net >= 0 ? 'positive' : 'negative'}`;
        }
        if (cftcSentiment) {
            cftcSentiment.textContent = cftc.sentiment || 'NEUTRAL';
            cftcSentiment.className = `position-value ${(cftc.sentiment || 'neutral').toLowerCase()}`;
        }
        
        // Update CFTC chart
        this.updateCFTCChart(cftc);
    }
    
    updateRiskFactors() {
        const risks = this.marketData.risk_factors || {};
        
        // Geopolitical tension
        this.updateRiskMeter('geo-risk', risks.geopolitical_tension || 0);
        
        // Currency debasement
        this.updateRiskMeter('currency-risk', risks.currency_debasement || 0);
        
        // Central bank stress
        this.updateRiskMeter('cb-stress', risks.central_bank_stress || 0);
    }
    
    updateRiskMeter(prefix, value) {
        const fillEl = document.getElementById(`${prefix}-fill`);
        const valueEl = document.getElementById(`${prefix}-value`);
        
        if (fillEl) {
            fillEl.style.width = `${value}%`;
            fillEl.className = `risk-fill ${this.getRiskLevel(value)}`;
        }
        if (valueEl) valueEl.textContent = `${value}/100`;
    }
    
    updateMarketRegime() {
        const regime = this.marketData.market_regime || {};
        
        const regimeCircle = document.getElementById('regime-circle');
        const regimeText = document.getElementById('regime-text');
        const regimeDescription = document.getElementById('regime-description');
        
        if (regimeText) regimeText.textContent = regime.current || 'UNKNOWN';
        if (regimeCircle) {
            regimeCircle.className = `regime-circle ${(regime.current || 'unknown').toLowerCase()}`;
        }
        if (regimeDescription) {
            regimeDescription.textContent = regime.description || 'Market regime analysis in progress...';
        }
    }
    
    updateCFTCChart(cftc) {
        const canvas = document.getElementById('cftc-chart');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        
        // Clear canvas
        ctx.clearRect(0, 0, width, height);
        
        // Draw positioning chart
        const specLong = cftc.spec_long || 0;
        const specShort = cftc.spec_short || 0;
        const total = specLong + specShort;
        
        if (total > 0) {
            const longWidth = (specLong / total) * width;
            
            // Draw long positions (green)
            ctx.fillStyle = '#00d084';
            ctx.fillRect(0, 0, longWidth, height);
            
            // Draw short positions (red)
            ctx.fillStyle = '#ff4757';
            ctx.fillRect(longWidth, 0, width - longWidth, height);
            
            // Add text labels
            ctx.fillStyle = '#ffffff';
            ctx.font = '12px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('LONG', longWidth / 2, height / 2 + 4);
            ctx.fillText('SHORT', longWidth + (width - longWidth) / 2, height / 2 + 4);
        }
    }
    
    loadFallbackData() {
        // Load mock data when real data is unavailable
        this.marketData = {
            macro_indicators: {
                dxy: { value: 103.45, change: -0.25, impact: 'BULLISH' },
                real_rates: { value: -1.2, change: -0.1, impact: 'BULLISH' },
                cpi: { value: 3.2, change: 0.1, impact: 'NEUTRAL' },
                fed_rate: { value: 5.25, change: 0, impact: 'BEARISH' }
            },
            etf_flows: {
                gld_flow: 150000,
                gld_volume: 8500000,
                iau_flow: 75000,
                iau_volume: 12000000,
                direction: 'INFLOW',
                strength: 65
            },
            cftc_positioning: {
                spec_long: 285000,
                spec_short: 195000,
                sentiment: 'BULLISH'
            },
            risk_factors: {
                geopolitical_tension: 75,
                currency_debasement: 60,
                central_bank_stress: 45
            },
            market_regime: {
                current: 'RISK_ON',
                description: 'Markets showing risk appetite with moderate inflation concerns'
            }
        };
        
        this.updateAllPanels();
    }
    
    // Utility methods
    formatValue(value) {
        if (value === undefined || value === null) return '--';
        return value.toFixed(2);
    }
    
    formatChange(change) {
        if (change === undefined || change === null) return '--';
        const sign = change >= 0 ? '+' : '';
        return `${sign}${change.toFixed(2)}`;
    }
    
    formatFlow(flow) {
        if (!flow) return '--';
        const absFlow = Math.abs(flow);
        const suffix = flow >= 0 ? ' IN' : ' OUT';
        if (absFlow >= 1000000) return `${(absFlow / 1000000).toFixed(1)}M${suffix}`;
        if (absFlow >= 1000) return `${(absFlow / 1000).toFixed(0)}K${suffix}`;
        return `${absFlow}${suffix}`;
    }
    
    formatVolume(volume) {
        if (!volume) return '--';
        if (volume >= 1000000) return `${(volume / 1000000).toFixed(1)}M`;
        if (volume >= 1000) return `${(volume / 1000).toFixed(0)}K`;
        return volume.toString();
    }
    
    formatContracts(contracts) {
        if (!contracts) return '--';
        return `${(contracts / 1000).toFixed(0)}K`;
    }
    
    getRiskLevel(value) {
        if (value >= 80) return 'critical';
        if (value >= 60) return 'high';
        if (value >= 40) return 'medium';
        return 'low';
    }
    
    updateStatus(status, message) {
        const statusEl = document.getElementById('data-status');
        if (statusEl) {
            statusEl.className = `data-status ${status}`;
            statusEl.querySelector('span').textContent = message;
        }
    }
    
    async refreshData() {
        await this.loadCriticalData();
    }
    
    startAutoUpdate() {
        this.updateTimer = setInterval(() => {
            this.loadCriticalData();
        }, this.updateInterval);
    }
    
    stopAutoUpdate() {
        if (this.updateTimer) {
            clearInterval(this.updateTimer);
            this.updateTimer = null;
        }
    }
    
    destroy() {
        this.stopAutoUpdate();
    }
}

// Initialize critical data panel
let criticalDataPanel;
document.addEventListener('DOMContentLoaded', () => {
    criticalDataPanel = new CriticalMarketDataPanel();
});

/**
 * Real-Time Market Context Display
 * Advanced market regime identification and context analysis
 */

class RealTimeMarketContext {
    constructor() {
        this.marketRegime = 'unknown';
        this.volatilityState = 'normal';
        this.supportResistanceLevels = new Map();
        this.economicEvents = [];
        this.volatilityForecast = null;
        this.updateInterval = 15000; // 15 seconds
        
        this.init();
    }

    init() {
        this.createContextInterface();
        this.loadMarketContext();
        this.setupEventListeners();
        this.startRealTimeUpdates();
        
        console.log('✅ Real-Time Market Context initialized');
    }

    createContextInterface() {
        const container = document.getElementById('market-context');
        if (!container) return;

        container.innerHTML = `
            <div class="context-header">
                <div class="header-content">
                    <h3><i class="fas fa-globe-americas"></i> Market Context</h3>
                    <div class="context-controls">
                        <button class="control-btn refresh-context" data-tooltip="Refresh Context">
                            <i class="fas fa-sync-alt"></i>
                        </button>
                        <button class="control-btn alerts-btn" data-tooltip="Setup Alerts">
                            <i class="fas fa-bell"></i>
                        </button>
                        <div class="timeframe-selector">
                            <select id="context-timeframe">
                                <option value="1H">1H Context</option>
                                <option value="4H" selected>4H Context</option>
                                <option value="1D">1D Context</option>
                            </select>
                        </div>
                    </div>
                </div>
                <div class="context-status">
                    <div class="status-indicator">
                        <span class="status-dot live"></span>
                        <span>Live Analysis</span>
                    </div>
                    <div class="market-time">
                        <span id="market-time">--:-- UTC</span>
                    </div>
                </div>
            </div>

            <div class="context-content">
                <!-- Market Regime Panel -->
                <div class="regime-panel">
                    <div class="panel-header">
                        <h4><i class="fas fa-chart-line"></i> Market Regime</h4>
                        <div class="regime-confidence">
                            <span>Confidence: <span id="regime-confidence">--</span></span>
                        </div>
                    </div>
                    <div class="regime-content">
                        <div class="regime-indicator">
                            <div class="regime-icon" id="regime-icon">📊</div>
                            <div class="regime-details">
                                <h5 id="regime-name">Analyzing...</h5>
                                <p id="regime-description">Market regime analysis in progress...</p>
                            </div>
                        </div>
                        <div class="regime-metrics">
                            <div class="metric-item">
                                <label>Trend Strength:</label>
                                <div class="metric-bar">
                                    <div class="metric-fill" id="trend-strength-fill"></div>
                                </div>
                                <span id="trend-strength-value">--</span>
                            </div>
                            <div class="metric-item">
                                <label>Volatility Level:</label>
                                <div class="metric-bar">
                                    <div class="metric-fill" id="volatility-fill"></div>
                                </div>
                                <span id="volatility-value">--</span>
                            </div>
                            <div class="metric-item">
                                <label>Market Efficiency:</label>
                                <div class="metric-bar">
                                    <div class="metric-fill" id="efficiency-fill"></div>
                                </div>
                                <span id="efficiency-value">--</span>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Support/Resistance Levels -->
                <div class="levels-panel">
                    <div class="panel-header">
                        <h4><i class="fas fa-layer-group"></i> Key Levels</h4>
                        <div class="levels-controls">
                            <button class="levels-btn auto-detect active" data-mode="auto">Auto</button>
                            <button class="levels-btn manual-add" data-mode="manual">Manual</button>
                        </div>
                    </div>
                    <div class="levels-content" id="levels-content">
                        <!-- Support/Resistance levels will be rendered here -->
                    </div>
                </div>

                <!-- Economic Events -->
                <div class="events-panel">
                    <div class="panel-header">
                        <h4><i class="fas fa-calendar-alt"></i> Economic Events</h4>
                        <div class="events-filter">
                            <select id="impact-filter">
                                <option value="all">All Impact</option>
                                <option value="high">High Impact</option>
                                <option value="medium">Medium Impact</option>
                                <option value="low">Low Impact</option>
                            </select>
                        </div>
                    </div>
                    <div class="events-content" id="events-content">
                        <!-- Economic events will be rendered here -->
                    </div>
                </div>

                <!-- Volatility Forecast -->
                <div class="volatility-panel">
                    <div class="panel-header">
                        <h4><i class="fas fa-chart-area"></i> Volatility Forecast</h4>
                        <div class="forecast-horizon">
                            <select id="forecast-horizon">
                                <option value="1H">Next Hour</option>
                                <option value="4H">Next 4 Hours</option>
                                <option value="1D" selected>Next Day</option>
                            </select>
                        </div>
                    </div>
                    <div class="volatility-content" id="volatility-content">
                        <!-- Volatility forecast chart will be rendered here -->
                    </div>
                </div>

                <!-- Trading Implications -->
                <div class="implications-panel">
                    <div class="panel-header">
                        <h4><i class="fas fa-lightbulb"></i> Trading Implications</h4>
                    </div>
                    <div class="implications-content" id="implications-content">
                        <!-- Trading implications will be rendered here -->
                    </div>
                </div>
            </div>

            <!-- Manual Level Addition Modal -->
            <div class="modal" id="manual-level-modal">
                <div class="modal-content">
                    <div class="modal-header">
                        <h4>Add Manual Level</h4>
                        <button class="close-modal">&times;</button>
                    </div>
                    <div class="modal-body">
                        <div class="form-group">
                            <label>Level Type:</label>
                            <select id="level-type">
                                <option value="support">Support</option>
                                <option value="resistance">Resistance</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Price Level:</label>
                            <input type="number" id="level-price" step="0.01" placeholder="Enter price level">
                        </div>
                        <div class="form-group">
                            <label>Strength:</label>
                            <select id="level-strength">
                                <option value="weak">Weak</option>
                                <option value="moderate">Moderate</option>
                                <option value="strong">Strong</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Notes:</label>
                            <textarea id="level-notes" placeholder="Optional notes about this level"></textarea>
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button class="btn secondary" onclick="this.closest('.modal').classList.remove('active')">Cancel</button>
                        <button class="btn primary" onclick="window.marketContext.addManualLevel()">Add Level</button>
                    </div>
                </div>
            </div>
        `;
    }

    setupEventListeners() {
        // Refresh button
        document.querySelector('.refresh-context')?.addEventListener('click', () => {
            this.loadMarketContext();
        });

        // Alerts button
        document.querySelector('.alerts-btn')?.addEventListener('click', () => {
            this.showAlertsSetup();
        });

        // Context timeframe
        document.getElementById('context-timeframe')?.addEventListener('change', (e) => {
            this.switchContextTimeframe(e.target.value);
        });

        // Levels controls
        document.querySelectorAll('.levels-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.switchLevelsMode(e.target.dataset.mode);
            });
        });

        // Impact filter
        document.getElementById('impact-filter')?.addEventListener('change', (e) => {
            this.filterEconomicEvents(e.target.value);
        });

        // Forecast horizon
        document.getElementById('forecast-horizon')?.addEventListener('change', (e) => {
            this.updateVolatilityForecast(e.target.value);
        });

        // Modal close
        document.querySelectorAll('.close-modal').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.target.closest('.modal').classList.remove('active');
            });
        });
    }

    async loadMarketContext() {
        try {
            this.showLoadingState();
            
            // Load all context data in parallel
            const [
                regimeData,
                levelsData,
                eventsData,
                volatilityData
            ] = await Promise.allSettled([
                this.fetchMarketRegime(),
                this.fetchSupportResistanceLevels(),
                this.fetchEconomicEvents(),
                this.fetchVolatilityForecast()
            ]);

            // Update each component
            if (regimeData.status === 'fulfilled') {
                this.updateMarketRegime(regimeData.value);
            }

            if (levelsData.status === 'fulfilled') {
                this.updateSupportResistanceLevels(levelsData.value);
            }

            if (eventsData.status === 'fulfilled') {
                this.updateEconomicEvents(eventsData.value);
            }

            if (volatilityData.status === 'fulfilled') {
                this.updateVolatilityForecast(volatilityData.value);
            }

            // Update trading implications
            this.updateTradingImplications();
            
            // Update market time
            this.updateMarketTime();
            
        } catch (error) {
            console.error('Error loading market context:', error);
            this.showErrorState();
        }
    }

    async fetchMarketRegime() {
        const timeframe = document.getElementById('context-timeframe')?.value || '4H';
        const response = await fetch(`/api/market/regime/${timeframe}`);
        if (!response.ok) {
            throw new Error('Failed to fetch market regime');
        }
        return await response.json();
    }

    async fetchSupportResistanceLevels() {
        const timeframe = document.getElementById('context-timeframe')?.value || '4H';
        const response = await fetch(`/api/market/levels/${timeframe}`);
        if (!response.ok) {
            throw new Error('Failed to fetch support/resistance levels');
        }
        return await response.json();
    }

    async fetchEconomicEvents() {
        const response = await fetch('/api/market/economic-events');
        if (!response.ok) {
            throw new Error('Failed to fetch economic events');
        }
        return await response.json();
    }

    async fetchVolatilityForecast() {
        const horizon = document.getElementById('forecast-horizon')?.value || '1D';
        const response = await fetch(`/api/market/volatility-forecast/${horizon}`);
        if (!response.ok) {
            throw new Error('Failed to fetch volatility forecast');
        }
        return await response.json();
    }

    updateMarketRegime(data) {
        if (!data) return;

        this.marketRegime = data.regime;
        
        // Update regime display
        const regimeIcon = document.getElementById('regime-icon');
        const regimeName = document.getElementById('regime-name');
        const regimeDescription = document.getElementById('regime-description');
        const regimeConfidence = document.getElementById('regime-confidence');

        const regimeInfo = this.getRegimeInfo(data.regime);
        
        if (regimeIcon) regimeIcon.textContent = regimeInfo.icon;
        if (regimeName) regimeName.textContent = regimeInfo.name;
        if (regimeDescription) regimeDescription.textContent = regimeInfo.description;
        if (regimeConfidence) regimeConfidence.textContent = `${Math.round(data.confidence * 100)}%`;

        // Update regime metrics
        this.updateRegimeMetrics(data.metrics);
    }

    getRegimeInfo(regime) {
        const regimes = {
            'trending_bullish': {
                icon: '📈',
                name: 'Bullish Trend',
                description: 'Strong upward momentum with higher highs and higher lows'
            },
            'trending_bearish': {
                icon: '📉',
                name: 'Bearish Trend',
                description: 'Strong downward momentum with lower highs and lower lows'
            },
            'ranging': {
                icon: '↔️',
                name: 'Range-bound',
                description: 'Price oscillating between defined support and resistance levels'
            },
            'volatile': {
                icon: '⚡',
                name: 'High Volatility',
                description: 'Increased price volatility with uncertain direction'
            },
            'accumulation': {
                icon: '🔄',
                name: 'Accumulation',
                description: 'Period of consolidation with potential for breakout'
            },
            'distribution': {
                icon: '📊',
                name: 'Distribution',
                description: 'Profit-taking phase with potential for reversal'
            },
            'breakout': {
                icon: '💥',
                name: 'Breakout',
                description: 'Price breaking through key levels with momentum'
            }
        };

        return regimes[regime] || {
            icon: '❓',
            name: 'Unknown',
            description: 'Market regime analysis pending'
        };
    }

    updateRegimeMetrics(metrics) {
        if (!metrics) return;

        // Trend strength
        const trendFill = document.getElementById('trend-strength-fill');
        const trendValue = document.getElementById('trend-strength-value');
        if (trendFill && trendValue) {
            const strength = metrics.trendStrength || 0;
            trendFill.style.width = `${strength * 100}%`;
            trendFill.className = `metric-fill ${this.getStrengthClass(strength)}`;
            trendValue.textContent = `${Math.round(strength * 100)}%`;
        }

        // Volatility
        const volatilityFill = document.getElementById('volatility-fill');
        const volatilityValue = document.getElementById('volatility-value');
        if (volatilityFill && volatilityValue) {
            const volatility = metrics.volatility || 0;
            volatilityFill.style.width = `${volatility * 100}%`;
            volatilityFill.className = `metric-fill ${this.getVolatilityClass(volatility)}`;
            volatilityValue.textContent = `${Math.round(volatility * 100)}%`;
        }

        // Market efficiency
        const efficiencyFill = document.getElementById('efficiency-fill');
        const efficiencyValue = document.getElementById('efficiency-value');
        if (efficiencyFill && efficiencyValue) {
            const efficiency = metrics.efficiency || 0;
            efficiencyFill.style.width = `${efficiency * 100}%`;
            efficiencyFill.className = `metric-fill ${this.getEfficiencyClass(efficiency)}`;
            efficiencyValue.textContent = `${Math.round(efficiency * 100)}%`;
        }
    }

    getStrengthClass(strength) {
        if (strength >= 0.8) return 'very-strong';
        if (strength >= 0.6) return 'strong';
        if (strength >= 0.4) return 'moderate';
        if (strength >= 0.2) return 'weak';
        return 'very-weak';
    }

    getVolatilityClass(volatility) {
        if (volatility >= 0.8) return 'very-high';
        if (volatility >= 0.6) return 'high';
        if (volatility >= 0.4) return 'moderate';
        if (volatility >= 0.2) return 'low';
        return 'very-low';
    }

    getEfficiencyClass(efficiency) {
        if (efficiency >= 0.8) return 'very-efficient';
        if (efficiency >= 0.6) return 'efficient';
        if (efficiency >= 0.4) return 'moderate';
        if (efficiency >= 0.2) return 'inefficient';
        return 'very-inefficient';
    }

    updateSupportResistanceLevels(data) {
        if (!data || !data.levels) return;

        this.supportResistanceLevels = new Map();
        
        const container = document.getElementById('levels-content');
        if (!container) return;

        let html = '<div class="levels-list">';
        
        // Sort levels by price
        const sortedLevels = data.levels.sort((a, b) => b.price - a.price);
        
        sortedLevels.forEach(level => {
            this.supportResistanceLevels.set(level.id, level);
            
            const distance = Math.abs(level.price - data.currentPrice);
            const distancePercent = (distance / data.currentPrice) * 100;
            
            html += `
                <div class="level-item ${level.type} strength-${level.strength}" data-level-id="${level.id}">
                    <div class="level-header">
                        <div class="level-info">
                            <span class="level-type">${level.type.toUpperCase()}</span>
                            <span class="level-price">$${level.price.toFixed(2)}</span>
                            <span class="level-distance">${distancePercent.toFixed(1)}% away</span>
                        </div>
                        <div class="level-strength">
                            <div class="strength-indicator strength-${level.strength}">
                                ${level.strength.toUpperCase()}
                            </div>
                        </div>
                    </div>
                    
                    <div class="level-details">
                        <div class="level-stats">
                            <span>Tests: ${level.tests || 0}</span>
                            <span>Age: ${level.age || '--'}</span>
                            <span>Volume: ${level.volume || '--'}</span>
                        </div>
                        
                        ${level.notes ? `
                            <div class="level-notes">
                                <p>${level.notes}</p>
                            </div>
                        ` : ''}
                    </div>
                    
                    <div class="level-actions">
                        <button class="action-btn watch-btn" onclick="window.marketContext.watchLevel('${level.id}')">
                            <i class="fas fa-eye"></i> Watch
                        </button>
                        <button class="action-btn alert-btn" onclick="window.marketContext.setLevelAlert('${level.id}')">
                            <i class="fas fa-bell"></i> Alert
                        </button>
                        ${level.manual ? `
                            <button class="action-btn remove-btn" onclick="window.marketContext.removeLevel('${level.id}')">
                                <i class="fas fa-trash"></i>
                            </button>
                        ` : ''}
                    </div>
                </div>
            `;
        });
        
        html += '</div>';
        container.innerHTML = html;
    }

    updateEconomicEvents(data) {
        if (!data || !data.events) return;

        this.economicEvents = data.events;
        this.renderEconomicEvents();
    }

    renderEconomicEvents() {
        const container = document.getElementById('events-content');
        if (!container) return;

        const filter = document.getElementById('impact-filter')?.value || 'all';
        const filteredEvents = filter === 'all' ? 
            this.economicEvents : 
            this.economicEvents.filter(event => event.impact === filter);

        if (filteredEvents.length === 0) {
            container.innerHTML = `
                <div class="no-events">
                    <i class="fas fa-calendar-check"></i>
                    <span>No ${filter === 'all' ? '' : filter + ' impact'} events scheduled</span>
                </div>
            `;
            return;
        }

        let html = '<div class="events-list">';
        
        filteredEvents.forEach(event => {
            const timeUntil = this.getTimeUntil(event.datetime);
            const isPast = new Date(event.datetime) < new Date();
            
            html += `
                <div class="event-item impact-${event.impact} ${isPast ? 'past' : 'upcoming'}">
                    <div class="event-header">
                        <div class="event-time">
                            <span class="event-date">${new Date(event.datetime).toLocaleDateString()}</span>
                            <span class="event-time-value">${new Date(event.datetime).toLocaleTimeString()}</span>
                        </div>
                        <div class="event-impact">
                            <span class="impact-badge impact-${event.impact}">${event.impact.toUpperCase()}</span>
                        </div>
                    </div>
                    
                    <div class="event-content">
                        <h5 class="event-title">${event.title}</h5>
                        <div class="event-details">
                            <div class="event-currency">${event.currency || 'USD'}</div>
                            ${event.forecast ? `<div class="event-forecast">Forecast: ${event.forecast}</div>` : ''}
                            ${event.previous ? `<div class="event-previous">Previous: ${event.previous}</div>` : ''}
                        </div>
                    </div>
                    
                    <div class="event-footer">
                        <div class="time-until ${isPast ? 'past' : ''}">
                            ${isPast ? 'Completed' : timeUntil}
                        </div>
                        <div class="event-potential-impact">
                            <span>Potential Impact: ${this.getImpactDescription(event.impact)}</span>
                        </div>
                    </div>
                </div>
            `;
        });
        
        html += '</div>';
        container.innerHTML = html;
    }

    getTimeUntil(datetime) {
        const now = new Date();
        const eventTime = new Date(datetime);
        const diff = eventTime - now;
        
        if (diff < 0) return 'Past';
        
        const hours = Math.floor(diff / (1000 * 60 * 60));
        const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
        
        if (hours > 24) {
            const days = Math.floor(hours / 24);
            return `${days}d ${hours % 24}h`;
        } else if (hours > 0) {
            return `${hours}h ${minutes}m`;
        } else {
            return `${minutes}m`;
        }
    }

    getImpactDescription(impact) {
        const descriptions = {
            'high': 'Significant market movement expected',
            'medium': 'Moderate market movement possible',
            'low': 'Limited market impact expected'
        };
        return descriptions[impact] || 'Unknown impact';
    }

    updateVolatilityForecast(data) {
        if (!data) return;

        this.volatilityForecast = data;
        
        const container = document.getElementById('volatility-content');
        if (!container) return;

        // Create volatility forecast chart
        container.innerHTML = `
            <div class="volatility-chart">
                <div class="chart-header">
                    <h5>Expected Volatility</h5>
                    <div class="current-volatility">
                        <span>Current: ${(data.current * 100).toFixed(1)}%</span>
                    </div>
                </div>
                <div class="volatility-bars" id="volatility-bars">
                    ${this.createVolatilityBars(data.forecast)}
                </div>
                <div class="volatility-summary">
                    <div class="summary-item">
                        <label>Average Expected:</label>
                        <span>${(data.average * 100).toFixed(1)}%</span>
                    </div>
                    <div class="summary-item">
                        <label>Peak Expected:</label>
                        <span>${(data.peak * 100).toFixed(1)}%</span>
                    </div>
                    <div class="summary-item">
                        <label>Trading Recommendation:</label>
                        <span class="recommendation ${data.recommendation?.type}">${data.recommendation?.text}</span>
                    </div>
                </div>
            </div>
        `;
    }

    createVolatilityBars(forecast) {
        if (!forecast || forecast.length === 0) return '';
        
        const maxVolatility = Math.max(...forecast.map(f => f.volatility));
        
        return forecast.map(item => {
            const height = (item.volatility / maxVolatility) * 100;
            const volatilityPercent = (item.volatility * 100).toFixed(1);
            
            return `
                <div class="volatility-bar">
                    <div class="bar-fill" 
                         style="height: ${height}%"
                         data-tooltip="${item.period}: ${volatilityPercent}%">
                    </div>
                    <div class="bar-label">${item.period}</div>
                </div>
            `;
        }).join('');
    }

    updateTradingImplications() {
        const container = document.getElementById('implications-content');
        if (!container) return;

        const implications = this.generateTradingImplications();
        
        container.innerHTML = `
            <div class="implications-list">
                ${implications.map(implication => `
                    <div class="implication-item ${implication.type}">
                        <div class="implication-header">
                            <i class="fas ${implication.icon}"></i>
                            <h5>${implication.title}</h5>
                            <span class="implication-priority ${implication.priority}">${implication.priority.toUpperCase()}</span>
                        </div>
                        <div class="implication-content">
                            <p>${implication.description}</p>
                            ${implication.actions ? `
                                <div class="suggested-actions">
                                    <h6>Suggested Actions:</h6>
                                    <ul>
                                        ${implication.actions.map(action => `<li>${action}</li>`).join('')}
                                    </ul>
                                </div>
                            ` : ''}
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
    }

    generateTradingImplications() {
        const implications = [];
        
        // Market regime implications
        if (this.marketRegime) {
            const regimeImplication = this.getRegimeImplication(this.marketRegime);
            if (regimeImplication) implications.push(regimeImplication);
        }
        
        // Volatility implications
        if (this.volatilityForecast) {
            const volatilityImplication = this.getVolatilityImplication(this.volatilityForecast);
            if (volatilityImplication) implications.push(volatilityImplication);
        }
        
        // Economic events implications
        const upcomingEvents = this.economicEvents.filter(event => 
            new Date(event.datetime) > new Date() && 
            new Date(event.datetime) - new Date() < 24 * 60 * 60 * 1000 // Next 24 hours
        );
        
        if (upcomingEvents.length > 0) {
            const eventsImplication = this.getEventsImplication(upcomingEvents);
            if (eventsImplication) implications.push(eventsImplication);
        }
        
        // Support/resistance implications
        const currentPrice = this.getCurrentPrice();
        if (currentPrice && this.supportResistanceLevels.size > 0) {
            const levelsImplication = this.getLevelsImplication(currentPrice);
            if (levelsImplication) implications.push(levelsImplication);
        }
        
        return implications;
    }

    getRegimeImplication(regime) {
        const implications = {
            'trending_bullish': {
                type: 'bullish',
                icon: 'fa-arrow-up',
                title: 'Bullish Trend Environment',
                description: 'Current market conditions favor long positions with trend-following strategies.',
                priority: 'high',
                actions: [
                    'Look for pullbacks to key support levels for entry',
                    'Use trailing stops to capture extended moves',
                    'Consider momentum-based position sizing'
                ]
            },
            'trending_bearish': {
                type: 'bearish',
                icon: 'fa-arrow-down',
                title: 'Bearish Trend Environment',
                description: 'Current market conditions favor short positions with trend-following strategies.',
                priority: 'high',
                actions: [
                    'Look for rallies to key resistance levels for entry',
                    'Use trailing stops to capture extended moves',
                    'Be cautious with long positions'
                ]
            },
            'ranging': {
                type: 'neutral',
                icon: 'fa-arrows-alt-h',
                title: 'Range-bound Market',
                description: 'Price is oscillating between defined levels. Range trading strategies may be effective.',
                priority: 'medium',
                actions: [
                    'Buy near support, sell near resistance',
                    'Use tight stops outside the range',
                    'Prepare for potential breakout'
                ]
            },
            'volatile': {
                type: 'warning',
                icon: 'fa-exclamation-triangle',
                title: 'High Volatility Environment',
                description: 'Increased volatility requires adjusted risk management and position sizing.',
                priority: 'high',
                actions: [
                    'Reduce position sizes',
                    'Widen stop losses appropriately',
                    'Be prepared for rapid price movements'
                ]
            }
        };
        
        return implications[regime];
    }

    getVolatilityImplication(forecast) {
        if (forecast.average > 0.5) {
            return {
                type: 'warning',
                icon: 'fa-chart-line',
                title: 'Elevated Volatility Expected',
                description: 'Higher than normal volatility forecasted. Adjust risk management accordingly.',
                priority: 'high',
                actions: [
                    'Reduce position sizes by 25-50%',
                    'Consider options strategies for protection',
                    'Monitor positions more closely'
                ]
            };
        } else if (forecast.average < 0.2) {
            return {
                type: 'info',
                icon: 'fa-chart-line',
                title: 'Low Volatility Environment',
                description: 'Lower volatility may lead to range-bound conditions and compressed ranges.',
                priority: 'medium',
                actions: [
                    'Consider range trading strategies',
                    'Prepare for potential volatility expansion',
                    'Look for breakout opportunities'
                ]
            };
        }
        
        return null;
    }

    getEventsImplication(events) {
        const highImpactEvents = events.filter(e => e.impact === 'high');
        
        if (highImpactEvents.length > 0) {
            return {
                type: 'warning',
                icon: 'fa-calendar-alt',
                title: 'High Impact Events Approaching',
                description: `${highImpactEvents.length} high impact economic event(s) scheduled within 24 hours.`,
                priority: 'high',
                actions: [
                    'Consider reducing exposure before events',
                    'Set appropriate stop losses',
                    'Monitor news and reactions closely',
                    'Be prepared for increased volatility'
                ]
            };
        }
        
        return null;
    }

    getLevelsImplication(currentPrice) {
        const nearbyLevels = Array.from(this.supportResistanceLevels.values())
            .filter(level => {
                const distance = Math.abs(level.price - currentPrice) / currentPrice;
                return distance < 0.005; // Within 0.5%
            })
            .sort((a, b) => Math.abs(a.price - currentPrice) - Math.abs(b.price - currentPrice));
        
        if (nearbyLevels.length > 0) {
            const level = nearbyLevels[0];
            const distance = ((level.price - currentPrice) / currentPrice * 100).toFixed(1);
            
            return {
                type: level.type === 'support' ? 'bullish' : 'bearish',
                icon: level.type === 'support' ? 'fa-arrow-up' : 'fa-arrow-down',
                title: `Approaching ${level.type.toUpperCase()} Level`,
                description: `Price is ${Math.abs(distance)}% away from ${level.strength} ${level.type} at $${level.price.toFixed(2)}.`,
                priority: level.strength === 'strong' ? 'high' : 'medium',
                actions: [
                    `Watch for ${level.type === 'support' ? 'bounce' : 'rejection'} at this level`,
                    'Prepare for potential breakout if level fails',
                    'Consider taking profits near the level'
                ]
            };
        }
        
        return null;
    }

    getCurrentPrice() {
        // This would get the current price from the main price display
        return window.goldAPILivePriceFetcher?.currentPrice || null;
    }

    // Event handlers
    switchContextTimeframe(timeframe) {
        this.loadMarketContext();
    }

    switchLevelsMode(mode) {
        document.querySelectorAll('.levels-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.mode === mode);
        });
        
        if (mode === 'manual') {
            document.getElementById('manual-level-modal').classList.add('active');
        }
    }

    filterEconomicEvents(impact) {
        this.renderEconomicEvents();
    }

    addManualLevel() {
        const type = document.getElementById('level-type')?.value;
        const price = parseFloat(document.getElementById('level-price')?.value);
        const strength = document.getElementById('level-strength')?.value;
        const notes = document.getElementById('level-notes')?.value;
        
        if (!type || !price || !strength) {
            alert('Please fill in all required fields');
            return;
        }
        
        const level = {
            id: `manual_${Date.now()}`,
            type,
            price,
            strength,
            notes,
            manual: true,
            tests: 0,
            age: 'New'
        };
        
        this.supportResistanceLevels.set(level.id, level);
        
        // Re-render levels
        this.updateSupportResistanceLevels({
            levels: Array.from(this.supportResistanceLevels.values()),
            currentPrice: this.getCurrentPrice() || 2000
        });
        
        // Close modal
        document.getElementById('manual-level-modal').classList.remove('active');
        
        // Clear form
        document.getElementById('level-price').value = '';
        document.getElementById('level-notes').value = '';
        
        console.log('✅ Manual level added:', level);
    }

    watchLevel(levelId) {
        console.log('Watching level:', levelId);
        // Implement level watching functionality
    }

    setLevelAlert(levelId) {
        console.log('Setting alert for level:', levelId);
        // Implement level alert functionality
    }

    removeLevel(levelId) {
        this.supportResistanceLevels.delete(levelId);
        
        // Re-render levels
        this.updateSupportResistanceLevels({
            levels: Array.from(this.supportResistanceLevels.values()),
            currentPrice: this.getCurrentPrice() || 2000
        });
        
        console.log('✅ Level removed:', levelId);
    }

    showAlertsSetup() {
        console.log('Opening alerts setup');
        // Implement alerts setup modal
    }

    updateMarketTime() {
        const element = document.getElementById('market-time');
        if (element) {
            element.textContent = new Date().toLocaleTimeString('en-US', { timeZone: 'UTC' }) + ' UTC';
        }
    }

    showLoadingState() {
        // Show loading indicators
        const statusElements = document.querySelectorAll('.status-indicator .status-dot');
        statusElements.forEach(el => {
            el.className = 'status-dot loading';
        });
    }

    showErrorState() {
        // Show error indicators
        const statusElements = document.querySelectorAll('.status-indicator .status-dot');
        statusElements.forEach(el => {
            el.className = 'status-dot error';
        });
    }

    startRealTimeUpdates() {
        // Update market time every second
        setInterval(() => {
            this.updateMarketTime();
        }, 1000);
        
        // Update context data every 15 seconds
        setInterval(() => {
            this.loadMarketContext();
        }, this.updateInterval);
    }

    dispose() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
        console.log('✅ Real-Time Market Context disposed');
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.marketContext = new RealTimeMarketContext();
});

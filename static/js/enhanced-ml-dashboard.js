/**
 * Enhanced ML Dashboard Controller
 * Comprehensive Gold Market Analysis Integration
 */

console.log('🚀 Loading Enhanced ML Dashboard Controller...');

// Initialize Enhanced ML Dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    console.log('✨ Initializing Enhanced ML Dashboard...');
    
    // Initialize comprehensive ML analysis
    initializeComprehensiveAnalysis();
    
    // Set up periodic refresh for real-time updates
    setInterval(() => {
        refreshComprehensiveAnalysis();
    }, 60000); // Refresh every minute for comprehensive analysis
});

/**
 * Initialize comprehensive ML analysis
 */
function initializeComprehensiveAnalysis() {
    console.log('🔄 Initializing Comprehensive Gold Analysis...');
    
    // Load enhanced ML predictions
    loadEnhancedMLPredictions();
    
    // Load detailed market analysis
    loadMarketAnalysis();
    
    // Initialize real-time updates
    initializeRealTimeUpdates();
}

/**
 * Load enhanced ML predictions with comprehensive analysis
 */
function loadEnhancedMLPredictions() {
    console.log('🧠 Loading Enhanced ML Predictions...');
    
    const timeframes = ['15m', '1h', '4h', '24h'];
    
    fetch('/api/enhanced-ml-predictions', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            timeframes: timeframes
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        return response.json();
    })
    .then(data => {
        console.log('✅ Enhanced ML Predictions loaded:', data);
        
        if (data.success) {
            // Update prediction cards with comprehensive data
            updateEnhancedPredictionCards(data.predictions);
            
            // Update overall market bias
            updateOverallMarketBias(data.overall_bias);
            
            // Update comprehensive analysis sections
            if (data.comprehensive_analysis) {
                updateTechnicalAnalysisSection(data.comprehensive_analysis.technical);
                updateSentimentAnalysisSection(data.comprehensive_analysis.sentiment);
                updateEconomicAnalysisSection(data.comprehensive_analysis.economic);
                updatePatternAnalysisSection(data.comprehensive_analysis.patterns);
            }
            
            // Update confidence factors
            if (data.confidence_factors) {
                updateConfidenceFactors(data.confidence_factors);
            }
        }
    })
    .catch(error => {
        console.error('❌ Enhanced ML Predictions failed:', error);
        // Fallback to basic ML predictions
        loadBasicMLPredictions();
    });
}

/**
 * Load basic ML predictions as fallback
 */
function loadBasicMLPredictions() {
    console.log('🔄 Loading Basic ML Predictions (Fallback)...');
    
    fetch('/api/ml-predictions')
        .then(response => response.json())
        .then(data => {
            console.log('✅ Basic ML Predictions loaded:', data);
            if (data.success && data.predictions) {
                updateBasicPredictionCards(data.predictions);
            }
        })
        .catch(error => {
            console.error('❌ All ML predictions failed:', error);
            showMLDataUnavailable();
        });
}

/**
 * Update enhanced prediction cards with comprehensive data
 */
function updateEnhancedPredictionCards(predictions) {
    console.log('📊 Updating Enhanced Prediction Cards...');
    
    predictions.forEach(prediction => {
        const timeframe = prediction.timeframe;
        
        // Find prediction card for this timeframe
        const card = document.querySelector(`[data-type="ml-prediction-${timeframe}"]`) ||
                    document.querySelector(`[data-timeframe="${timeframe}"]`) ||
                    document.querySelector(`.prediction-card:contains("${timeframe.toUpperCase()}")`);
        
        if (card) {
            updatePredictionCardContent(card, prediction);
        } else {
            // Create new prediction card if not found
            createPredictionCard(prediction);
        }
    });
}

/**
 * Update prediction card content with comprehensive data
 */
function updatePredictionCardContent(card, prediction) {
    try {
        // Update direction indicator
        const directionEl = card.querySelector('.prediction-direction') || 
                          card.querySelector('.direction') ||
                          card.querySelector('[data-field="direction"]');
        if (directionEl) {
            directionEl.textContent = prediction.direction.toUpperCase();
            directionEl.className = `prediction-direction ${prediction.direction.toLowerCase()}`;
        }
        
        // Update confidence score
        const confidenceEl = card.querySelector('.confidence-score') ||
                           card.querySelector('.confidence') ||
                           card.querySelector('[data-field="confidence"]');
        if (confidenceEl) {
            confidenceEl.textContent = `${prediction.confidence}%`;
        }
        
        // Update target price
        const targetEl = card.querySelector('.target-price') ||
                        card.querySelector('.price-target') ||
                        card.querySelector('[data-field="target"]');
        if (targetEl) {
            targetEl.textContent = `$${prediction.target_price}`;
        }
        
        // Update price change
        const changeEl = card.querySelector('.price-change') ||
                        card.querySelector('[data-field="change"]');
        if (changeEl) {
            const changePercent = prediction.price_change_percent;
            const changeSign = changePercent >= 0 ? '+' : '';
            changeEl.textContent = `${changeSign}${changePercent}%`;
            changeEl.className = `price-change ${changePercent >= 0 ? 'positive' : 'negative'}`;
        }
        
        // Update timeframe
        const timeframeEl = card.querySelector('.timeframe') ||
                          card.querySelector('[data-field="timeframe"]');
        if (timeframeEl) {
            timeframeEl.textContent = prediction.timeframe.toUpperCase();
        }
        
        // Update support/resistance levels
        const supportEl = card.querySelector('.support-level') ||
                         card.querySelector('[data-field="support"]');
        if (supportEl) {
            supportEl.textContent = `$${prediction.support_level}`;
        }
        
        const resistanceEl = card.querySelector('.resistance-level') ||
                           card.querySelector('[data-field="resistance"]');
        if (resistanceEl) {
            resistanceEl.textContent = `$${prediction.resistance_level}`;
        }
        
        // Update stop loss and take profit
        const stopLossEl = card.querySelector('.stop-loss') ||
                          card.querySelector('[data-field="stop-loss"]');
        if (stopLossEl) {
            stopLossEl.textContent = `$${prediction.stop_loss}`;
        }
        
        const takeProfitEl = card.querySelector('.take-profit') ||
                           card.querySelector('[data-field="take-profit"]');
        if (takeProfitEl) {
            takeProfitEl.textContent = `$${prediction.take_profit}`;
        }
        
        // Add visual styling based on direction
        card.classList.remove('bullish', 'bearish', 'neutral');
        card.classList.add(prediction.direction.toLowerCase());
        
        console.log(`✅ Updated ${prediction.timeframe} prediction card`);
        
    } catch (error) {
        console.error(`❌ Error updating prediction card for ${prediction.timeframe}:`, error);
    }
}

/**
 * Update overall market bias display
 */
function updateOverallMarketBias(bias) {
    console.log('📈 Updating Overall Market Bias...', bias);
    
    try {
        // Find bias display elements
        const biasDirection = document.querySelector('.market-bias-direction') ||
                            document.querySelector('[data-field="bias-direction"]');
        if (biasDirection) {
            biasDirection.textContent = bias.direction.toUpperCase();
            biasDirection.className = `market-bias-direction ${bias.direction.toLowerCase()}`;
        }
        
        const biasStrength = document.querySelector('.market-bias-strength') ||
                           document.querySelector('[data-field="bias-strength"]');
        if (biasStrength) {
            biasStrength.textContent = bias.strength.toUpperCase();
        }
        
        const biasScore = document.querySelector('.market-bias-score') ||
                         document.querySelector('[data-field="bias-score"]');
        if (biasScore) {
            biasScore.textContent = bias.score;
        }
        
        // Update factor contributions
        if (bias.factors) {
            const factorEls = {
                technical: document.querySelector('.factor-technical') || document.querySelector('[data-field="factor-technical"]'),
                sentiment: document.querySelector('.factor-sentiment') || document.querySelector('[data-field="factor-sentiment"]'),
                economic: document.querySelector('.factor-economic') || document.querySelector('[data-field="factor-economic"]'),
                pattern: document.querySelector('.factor-pattern') || document.querySelector('[data-field="factor-pattern"]')
            };
            
            Object.keys(factorEls).forEach(factor => {
                if (factorEls[factor] && bias.factors[factor] !== undefined) {
                    factorEls[factor].textContent = bias.factors[factor];
                }
            });
        }
        
    } catch (error) {
        console.error('❌ Error updating market bias:', error);
    }
}

/**
 * Update technical analysis section
 */
function updateTechnicalAnalysisSection(technical) {
    console.log('📊 Updating Technical Analysis...', technical);
    
    try {
        // Update RSI
        const rsiEl = document.querySelector('.rsi-value') || document.querySelector('[data-field="rsi"]');
        if (rsiEl) {
            rsiEl.textContent = technical.rsi;
        }
        
        // Update MACD
        const macdEl = document.querySelector('.macd-value') || document.querySelector('[data-field="macd"]');
        if (macdEl) {
            macdEl.textContent = technical.macd;
        }
        
        // Update trend direction
        const trendEl = document.querySelector('.trend-direction') || document.querySelector('[data-field="trend"]');
        if (trendEl) {
            trendEl.textContent = technical.trend_direction.toUpperCase();
            trendEl.className = `trend-direction ${technical.trend_direction.toLowerCase()}`;
        }
        
        // Update support/resistance
        const supportEl = document.querySelector('.global-support') || document.querySelector('[data-field="global-support"]');
        if (supportEl) {
            supportEl.textContent = `$${technical.support_level}`;
        }
        
        const resistanceEl = document.querySelector('.global-resistance') || document.querySelector('[data-field="global-resistance"]');
        if (resistanceEl) {
            resistanceEl.textContent = `$${technical.resistance_level}`;
        }
        
        // Update Bollinger Bands position
        const bbEl = document.querySelector('.bollinger-position') || document.querySelector('[data-field="bollinger"]');
        if (bbEl) {
            bbEl.textContent = technical.bollinger_position.toUpperCase();
        }
        
        // Update volatility
        const volatilityEl = document.querySelector('.volatility-value') || document.querySelector('[data-field="volatility"]');
        if (volatilityEl) {
            volatilityEl.textContent = `${(technical.volatility * 100).toFixed(1)}%`;
        }
        
    } catch (error) {
        console.error('❌ Error updating technical analysis:', error);
    }
}

/**
 * Update sentiment analysis section
 */
function updateSentimentAnalysisSection(sentiment) {
    console.log('💭 Updating Sentiment Analysis...', sentiment);
    
    try {
        // Update Fear & Greed Index
        const fearGreedEl = document.querySelector('.fear-greed-index') || document.querySelector('[data-field="fear-greed"]');
        if (fearGreedEl) {
            fearGreedEl.textContent = sentiment.fear_greed_index;
        }
        
        // Update news sentiment
        const newsEl = document.querySelector('.news-sentiment') || document.querySelector('[data-field="news-sentiment"]');
        if (newsEl) {
            newsEl.textContent = `${(sentiment.news_sentiment * 100).toFixed(0)}%`;
        }
        
        // Update social sentiment
        const socialEl = document.querySelector('.social-sentiment') || document.querySelector('[data-field="social-sentiment"]');
        if (socialEl) {
            socialEl.textContent = `${(sentiment.social_sentiment * 100).toFixed(0)}%`;
        }
        
        // Update institutional flow
        const institutionalEl = document.querySelector('.institutional-flow') || document.querySelector('[data-field="institutional"]');
        if (institutionalEl) {
            institutionalEl.textContent = sentiment.institutional_flow.replace('_', ' ').toUpperCase();
        }
        
        // Update market mood
        const moodEl = document.querySelector('.market-mood') || document.querySelector('[data-field="market-mood"]');
        if (moodEl) {
            moodEl.textContent = sentiment.market_mood.toUpperCase();
            moodEl.className = `market-mood ${sentiment.market_mood.toLowerCase()}`;
        }
        
    } catch (error) {
        console.error('❌ Error updating sentiment analysis:', error);
    }
}

/**
 * Update economic analysis section
 */
function updateEconomicAnalysisSection(economic) {
    console.log('💰 Updating Economic Analysis...', economic);
    
    try {
        // Update Dollar Index
        const dxyEl = document.querySelector('.dollar-index') || document.querySelector('[data-field="dxy"]');
        if (dxyEl) {
            dxyEl.textContent = economic.dollar_index;
        }
        
        // Update Federal Rate
        const rateEl = document.querySelector('.federal-rate') || document.querySelector('[data-field="fed-rate"]');
        if (rateEl) {
            rateEl.textContent = `${economic.federal_rate}%`;
        }
        
        // Update Inflation
        const inflationEl = document.querySelector('.inflation-rate') || document.querySelector('[data-field="inflation"]');
        if (inflationEl) {
            inflationEl.textContent = `${economic.inflation_cpi}%`;
        }
        
        // Update economic uncertainty
        const uncertaintyEl = document.querySelector('.economic-uncertainty') || document.querySelector('[data-field="uncertainty"]');
        if (uncertaintyEl) {
            uncertaintyEl.textContent = economic.economic_uncertainty;
        }
        
        // Update central bank stance
        const cbEl = document.querySelector('.central-bank-stance') || document.querySelector('[data-field="cb-stance"]');
        if (cbEl) {
            cbEl.textContent = economic.central_bank_stance.toUpperCase();
            cbEl.className = `cb-stance ${economic.central_bank_stance.toLowerCase()}`;
        }
        
    } catch (error) {
        console.error('❌ Error updating economic analysis:', error);
    }
}

/**
 * Update pattern analysis section
 */
function updatePatternAnalysisSection(patterns) {
    console.log('📈 Updating Pattern Analysis...', patterns);
    
    try {
        // Update detected pattern
        const patternEl = document.querySelector('.detected-pattern') || document.querySelector('[data-field="pattern"]');
        if (patternEl) {
            patternEl.textContent = patterns.detected_pattern.replace('_', ' ').toUpperCase();
        }
        
        // Update pattern signal
        const signalEl = document.querySelector('.pattern-signal') || document.querySelector('[data-field="pattern-signal"]');
        if (signalEl) {
            signalEl.textContent = patterns.pattern_signal.toUpperCase();
            signalEl.className = `pattern-signal ${patterns.pattern_signal.toLowerCase()}`;
        }
        
        // Update pattern strength
        const strengthEl = document.querySelector('.pattern-strength') || document.querySelector('[data-field="pattern-strength"]');
        if (strengthEl) {
            strengthEl.textContent = `${(patterns.pattern_strength * 100).toFixed(0)}%`;
        }
        
        // Update reliability score
        const reliabilityEl = document.querySelector('.pattern-reliability') || document.querySelector('[data-field="pattern-reliability"]');
        if (reliabilityEl) {
            reliabilityEl.textContent = `${(patterns.reliability_score * 100).toFixed(0)}%`;
        }
        
    } catch (error) {
        console.error('❌ Error updating pattern analysis:', error);
    }
}

/**
 * Load detailed market analysis
 */
function loadMarketAnalysis() {
    console.log('📊 Loading Detailed Market Analysis...');
    
    fetch('/api/market-analysis')
        .then(response => response.json())
        .then(data => {
            console.log('✅ Market Analysis loaded:', data);
            
            if (data.success) {
                // Update current price display
                updateCurrentPrice(data.current_price);
                
                // Update detailed analysis sections
                if (data.analysis) {
                    updateTechnicalAnalysisSection(data.analysis.technical_indicators);
                    updateSentimentAnalysisSection(data.analysis.market_sentiment);
                    updateEconomicAnalysisSection(data.analysis.economic_factors);
                    updatePatternAnalysisSection(data.analysis.candlestick_patterns);
                }
                
                // Update overall assessment
                if (data.overall_assessment) {
                    updateOverallMarketBias(data.overall_assessment);
                }
            }
        })
        .catch(error => {
            console.error('❌ Market Analysis failed:', error);
        });
}

/**
 * Update current price display
 */
function updateCurrentPrice(price) {
    const priceElements = document.querySelectorAll('.current-price, [data-field="current-price"]');
    priceElements.forEach(el => {
        if (el) {
            el.textContent = `$${price}`;
        }
    });
}

/**
 * Refresh comprehensive analysis
 */
function refreshComprehensiveAnalysis() {
    console.log('🔄 Refreshing Comprehensive Analysis...');
    loadEnhancedMLPredictions();
    loadMarketAnalysis();
}

/**
 * Initialize real-time updates
 */
function initializeRealTimeUpdates() {
    console.log('⚡ Initializing Real-time Updates...');
    
    // Set up WebSocket connection for real-time data if available
    if (window.socket) {
        window.socket.on('ml_prediction_update', (data) => {
            console.log('🔔 Real-time ML prediction update:', data);
            updateEnhancedPredictionCards(data.predictions);
        });
        
        window.socket.on('market_analysis_update', (data) => {
            console.log('🔔 Real-time market analysis update:', data);
            loadMarketAnalysis();
        });
    }
}

/**
 * Show ML data unavailable message
 */
function showMLDataUnavailable() {
    console.log('⚠️ Showing ML data unavailable message');
    
    const mlCards = document.querySelectorAll('.prediction-card, [data-type*="ml-prediction"]');
    mlCards.forEach(card => {
        const loadingEl = card.querySelector('.loading-state') || card;
        if (loadingEl) {
            loadingEl.innerHTML = `
                <div class="ml-unavailable">
                    <span class="warning-icon">⚠️</span>
                    <span>ML Analysis Temporarily Unavailable</span>
                </div>
            `;
        }
    });
}

/**
 * Update basic prediction cards (fallback)
 */
function updateBasicPredictionCards(predictions) {
    console.log('📊 Updating Basic Prediction Cards (Fallback)...');
    
    predictions.forEach(prediction => {
        const card = document.querySelector(`[data-type="ml-prediction-${prediction.timeframe}"]`);
        if (card) {
            updatePredictionCardContent(card, prediction);
        }
    });
}

/**
 * Create new prediction card if not found
 */
function createPredictionCard(prediction) {
    console.log(`📋 Creating new prediction card for ${prediction.timeframe}`);
    // Implementation for creating new card would go here
    // This is a placeholder for dynamic card creation
}

console.log('✅ Enhanced ML Dashboard Controller loaded successfully');

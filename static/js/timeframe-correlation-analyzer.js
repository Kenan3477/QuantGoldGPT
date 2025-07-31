/**
 * Timeframe Correlation Analyzer
 * Analyzes how predictions align/conflict across different timeframes
 */

class TimeframeCorrelationAnalyzer {
    constructor() {
        this.timeframes = ['1M', '5M', '15M', '1H', '4H', '1D'];
        this.correlationMatrix = new Map();
        this.divergenceThreshold = 0.4;
        this.alignmentScores = new Map();
        this.nestedAnalysis = new Map();
        
        this.init();
    }

    init() {
        this.createAnalyzerInterface();
        this.loadCorrelationData();
        this.setupEventListeners();
        this.startRealTimeAnalysis();
        
        console.log('✅ Timeframe Correlation Analyzer initialized');
    }

    createAnalyzerInterface() {
        const container = document.getElementById('correlation-analyzer');
        if (!container) return;

        container.innerHTML = `
            <div class="analyzer-header">
                <div class="header-content">
                    <h3><i class="fas fa-project-diagram"></i> Timeframe Correlation Analysis</h3>
                    <div class="analyzer-controls">
                        <button class="control-btn refresh-correlation" data-tooltip="Refresh Analysis">
                            <i class="fas fa-sync-alt"></i>
                        </button>
                        <button class="control-btn export-analysis" data-tooltip="Export Analysis">
                            <i class="fas fa-download"></i>
                        </button>
                        <div class="sensitivity-control">
                            <label>Sensitivity:</label>
                            <input type="range" id="sensitivity-slider" min="0.1" max="1.0" step="0.1" value="0.4">
                            <span id="sensitivity-value">0.4</span>
                        </div>
                    </div>
                </div>
                <div class="analysis-summary">
                    <div class="summary-metric">
                        <label>Overall Alignment:</label>
                        <span id="overall-alignment" class="alignment-score">--</span>
                    </div>
                    <div class="summary-metric">
                        <label>Divergences Detected:</label>
                        <span id="divergence-count" class="divergence-count">0</span>
                    </div>
                    <div class="summary-metric">
                        <label>Confidence Level:</label>
                        <span id="confidence-level" class="confidence-level">--</span>
                    </div>
                </div>
            </div>

            <div class="analyzer-content">
                <!-- Correlation Matrix -->
                <div class="correlation-section">
                    <h4><i class="fas fa-th"></i> Correlation Matrix</h4>
                    <div class="correlation-matrix" id="correlation-matrix">
                        <!-- Matrix will be rendered here -->
                    </div>
                    <div class="matrix-legend">
                        <div class="legend-item">
                            <div class="legend-color strong-positive"></div>
                            <span>Strong Positive (0.7+)</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color positive"></div>
                            <span>Positive (0.3-0.7)</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color neutral"></div>
                            <span>Neutral (-0.3-0.3)</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color negative"></div>
                            <span>Negative (-0.7--0.3)</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color strong-negative"></div>
                            <span>Strong Negative (-0.7-)</span>
                        </div>
                    </div>
                </div>

                <!-- Divergence Analysis -->
                <div class="divergence-section">
                    <h4><i class="fas fa-exclamation-triangle"></i> Divergence Analysis</h4>
                    <div class="divergences-container" id="divergences-container">
                        <!-- Divergence items will be rendered here -->
                    </div>
                </div>

                <!-- Nested Timeframe Analysis -->
                <div class="nested-analysis-section">
                    <h4><i class="fas fa-layer-group"></i> Nested Timeframe Analysis</h4>
                    <div class="nested-container" id="nested-container">
                        <!-- Nested analysis will be rendered here -->
                    </div>
                </div>

                <!-- Alignment Score Breakdown -->
                <div class="alignment-section">
                    <h4><i class="fas fa-chart-bar"></i> Alignment Score Breakdown</h4>
                    <div class="alignment-charts" id="alignment-charts">
                        <!-- Alignment visualization will be rendered here -->
                    </div>
                </div>

                <!-- Trade Confidence Calculator -->
                <div class="confidence-section">
                    <h4><i class="fas fa-calculator"></i> Trade Confidence Calculator</h4>
                    <div class="confidence-calculator" id="confidence-calculator">
                        <div class="calculator-inputs">
                            <div class="input-group">
                                <label>Primary Timeframe:</label>
                                <select id="primary-timeframe">
                                    ${this.timeframes.map(tf => `<option value="${tf}">${tf}</option>`).join('')}
                                </select>
                            </div>
                            <div class="input-group">
                                <label>Confirmation Timeframes:</label>
                                <div class="timeframe-checkboxes">
                                    ${this.timeframes.map(tf => `
                                        <label>
                                            <input type="checkbox" value="${tf}"> ${tf}
                                        </label>
                                    `).join('')}
                                </div>
                            </div>
                        </div>
                        <div class="calculator-results">
                            <div class="result-item">
                                <label>Trade Confidence:</label>
                                <div class="confidence-meter">
                                    <div class="meter-fill" style="width: 0%"></div>
                                    <span class="meter-value">0%</span>
                                </div>
                            </div>
                            <div class="result-item">
                                <label>Risk Assessment:</label>
                                <span class="risk-level">Low</span>
                            </div>
                            <div class="result-item">
                                <label>Recommended Position Size:</label>
                                <span class="position-size">0%</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    setupEventListeners() {
        // Refresh button
        document.querySelector('.refresh-correlation')?.addEventListener('click', () => {
            this.loadCorrelationData();
        });

        // Export button
        document.querySelector('.export-analysis')?.addEventListener('click', () => {
            this.exportAnalysis();
        });

        // Sensitivity slider
        const sensitivitySlider = document.getElementById('sensitivity-slider');
        const sensitivityValue = document.getElementById('sensitivity-value');
        
        sensitivitySlider?.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            sensitivityValue.textContent = value.toFixed(1);
            this.divergenceThreshold = value;
            this.updateDivergenceAnalysis();
        });

        // Primary timeframe selection
        document.getElementById('primary-timeframe')?.addEventListener('change', (e) => {
            this.calculateTradeConfidence();
        });

        // Confirmation timeframes
        document.querySelectorAll('.timeframe-checkboxes input').forEach(checkbox => {
            checkbox.addEventListener('change', () => {
                this.calculateTradeConfidence();
            });
        });
    }

    async loadCorrelationData() {
        try {
            // Show loading state
            this.showLoadingState();
            
            // Fetch correlation data from backend
            const response = await fetch('/api/analysis/correlation');
            if (!response.ok) {
                throw new Error('Failed to fetch correlation data');
            }
            
            const data = await response.json();
            
            // Update correlation matrix
            this.updateCorrelationMatrix(data.correlations);
            
            // Update alignment scores
            this.updateAlignmentScores(data.alignments);
            
            // Perform divergence analysis
            this.analyzeDivergences(data.predictions);
            
            // Perform nested analysis
            this.performNestedAnalysis(data.predictions);
            
            // Update summary metrics
            this.updateSummaryMetrics();
            
        } catch (error) {
            console.error('Error loading correlation data:', error);
            this.showErrorState();
        }
    }

    updateCorrelationMatrix(correlations) {
        const container = document.getElementById('correlation-matrix');
        if (!container) return;

        let html = '<div class="matrix-grid">';
        
        // Header row
        html += '<div class="matrix-row header-row">';
        html += '<div class="matrix-cell header-cell"></div>'; // Empty corner
        this.timeframes.forEach(tf => {
            html += `<div class="matrix-cell header-cell">${tf}</div>`;
        });
        html += '</div>';
        
        // Data rows
        this.timeframes.forEach(tf1 => {
            html += '<div class="matrix-row">';
            html += `<div class="matrix-cell header-cell">${tf1}</div>`;
            
            this.timeframes.forEach(tf2 => {
                const correlation = this.getCorrelation(correlations, tf1, tf2);
                const cellClass = this.getCorrelationClass(correlation);
                const tooltip = `${tf1} vs ${tf2}: ${correlation.toFixed(3)}`;
                
                html += `
                    <div class="matrix-cell correlation-cell ${cellClass}" 
                         data-tooltip="${tooltip}"
                         data-correlation="${correlation}">
                        ${correlation.toFixed(2)}
                    </div>
                `;
            });
            
            html += '</div>';
        });
        
        html += '</div>';
        container.innerHTML = html;
        
        // Store correlations for further analysis
        this.correlationMatrix = new Map(Object.entries(correlations));
    }

    getCorrelation(correlations, tf1, tf2) {
        if (tf1 === tf2) return 1.0;
        
        const key1 = `${tf1}-${tf2}`;
        const key2 = `${tf2}-${tf1}`;
        
        return correlations[key1] || correlations[key2] || 0.0;
    }

    getCorrelationClass(correlation) {
        if (correlation >= 0.7) return 'strong-positive';
        if (correlation >= 0.3) return 'positive';
        if (correlation >= -0.3) return 'neutral';
        if (correlation >= -0.7) return 'negative';
        return 'strong-negative';
    }

    updateAlignmentScores(alignments) {
        this.alignmentScores = new Map(Object.entries(alignments));
        
        // Render alignment charts
        const container = document.getElementById('alignment-charts');
        if (!container) return;

        let html = '<div class="alignment-grid">';
        
        this.timeframes.forEach(tf => {
            const score = alignments[tf] || 0;
            const scoreClass = this.getAlignmentClass(score);
            
            html += `
                <div class="alignment-item">
                    <div class="timeframe-label">${tf}</div>
                    <div class="alignment-bar">
                        <div class="alignment-fill ${scoreClass}" style="width: ${score * 100}%"></div>
                    </div>
                    <div class="alignment-value">${(score * 100).toFixed(1)}%</div>
                </div>
            `;
        });
        
        html += '</div>';
        container.innerHTML = html;
    }

    getAlignmentClass(score) {
        if (score >= 0.8) return 'excellent';
        if (score >= 0.6) return 'good';
        if (score >= 0.4) return 'moderate';
        if (score >= 0.2) return 'poor';
        return 'very-poor';
    }

    analyzeDivergences(predictions) {
        const divergences = [];
        
        // Compare adjacent timeframes
        for (let i = 0; i < this.timeframes.length - 1; i++) {
            const tf1 = this.timeframes[i];
            const tf2 = this.timeframes[i + 1];
            
            const pred1 = predictions[tf1];
            const pred2 = predictions[tf2];
            
            if (pred1 && pred2) {
                const divergence = this.calculateDivergence(pred1, pred2);
                
                if (divergence.strength >= this.divergenceThreshold) {
                    divergences.push({
                        timeframes: [tf1, tf2],
                        predictions: [pred1, pred2],
                        divergence,
                        impact: this.assessDivergenceImpact(divergence, tf1, tf2)
                    });
                }
            }
        }
        
        this.displayDivergences(divergences);
    }

    calculateDivergence(pred1, pred2) {
        // Direction divergence
        const directionDivergence = (pred1.direction !== pred2.direction) ? 1.0 : 0.0;
        
        // Confidence divergence
        const confidenceDivergence = Math.abs(pred1.confidence - pred2.confidence);
        
        // Price target divergence
        const priceDivergence = pred1.targetPrice && pred2.targetPrice ? 
            Math.abs(pred1.targetPrice - pred2.targetPrice) / Math.max(pred1.targetPrice, pred2.targetPrice) : 0;
        
        // Combined divergence strength
        const strength = (directionDivergence * 0.5) + (confidenceDivergence * 0.3) + (priceDivergence * 0.2);
        
        return {
            strength,
            directionDivergence,
            confidenceDivergence,
            priceDivergence,
            type: this.classifyDivergence(directionDivergence, confidenceDivergence, priceDivergence)
        };
    }

    classifyDivergence(direction, confidence, price) {
        if (direction > 0.5) return 'directional';
        if (confidence > 0.3) return 'confidence';
        if (price > 0.1) return 'price';
        return 'minor';
    }

    assessDivergenceImpact(divergence, tf1, tf2) {
        // Impact assessment based on timeframe relationship and divergence type
        const timeframeWeight = this.getTimeframeWeight(tf1, tf2);
        const typeWeight = this.getDivergenceTypeWeight(divergence.type);
        
        const impact = divergence.strength * timeframeWeight * typeWeight;
        
        return {
            level: this.getImpactLevel(impact),
            score: impact,
            recommendation: this.getImpactRecommendation(impact, divergence.type)
        };
    }

    getTimeframeWeight(tf1, tf2) {
        // Higher weight for adjacent timeframes with significant difference
        const weights = {
            '1M': 1, '5M': 2, '15M': 3, '1H': 4, '4H': 5, '1D': 6
        };
        
        const diff = Math.abs(weights[tf2] - weights[tf1]);
        return Math.min(diff / 3, 1.0);
    }

    getDivergenceTypeWeight(type) {
        const weights = {
            'directional': 1.0,
            'confidence': 0.7,
            'price': 0.5,
            'minor': 0.3
        };
        
        return weights[type] || 0.5;
    }

    getImpactLevel(impact) {
        if (impact >= 0.8) return 'critical';
        if (impact >= 0.6) return 'high';
        if (impact >= 0.4) return 'medium';
        if (impact >= 0.2) return 'low';
        return 'minimal';
    }

    getImpactRecommendation(impact, type) {
        if (impact >= 0.8) {
            return 'Exercise extreme caution. Consider postponing trade until alignment improves.';
        } else if (impact >= 0.6) {
            return 'High divergence detected. Reduce position size and monitor closely.';
        } else if (impact >= 0.4) {
            return 'Moderate divergence. Use higher timeframe bias with smaller position.';
        }
        return 'Minor divergence. Proceed with standard risk management.';
    }

    displayDivergences(divergences) {
        const container = document.getElementById('divergences-container');
        if (!container) return;

        if (divergences.length === 0) {
            container.innerHTML = `
                <div class="no-divergences">
                    <i class="fas fa-check-circle"></i>
                    <span>No significant divergences detected</span>
                </div>
            `;
            return;
        }

        container.innerHTML = divergences.map(div => `
            <div class="divergence-item impact-${div.impact.level}">
                <div class="divergence-header">
                    <div class="divergence-title">
                        <i class="fas fa-exclamation-triangle"></i>
                        <span>${div.timeframes.join(' vs ')} Divergence</span>
                    </div>
                    <div class="divergence-badges">
                        <span class="type-badge ${div.divergence.type}">${div.divergence.type}</span>
                        <span class="impact-badge ${div.impact.level}">${div.impact.level}</span>
                    </div>
                </div>
                
                <div class="divergence-content">
                    <div class="divergence-details">
                        <div class="detail-item">
                            <label>Strength:</label>
                            <div class="strength-bar">
                                <div class="strength-fill" style="width: ${div.divergence.strength * 100}%"></div>
                            </div>
                            <span>${(div.divergence.strength * 100).toFixed(1)}%</span>
                        </div>
                        
                        <div class="predictions-comparison">
                            ${div.predictions.map((pred, i) => `
                                <div class="prediction-summary">
                                    <h6>${div.timeframes[i]}</h6>
                                    <div class="pred-details">
                                        <span class="direction ${pred.direction?.toLowerCase()}">${pred.direction}</span>
                                        <span class="confidence">${Math.round(pred.confidence * 100)}%</span>
                                        <span class="target">$${pred.targetPrice?.toFixed(2)}</span>
                                    </div>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                    
                    <div class="divergence-recommendation">
                        <strong>Recommendation:</strong> ${div.impact.recommendation}
                    </div>
                </div>
            </div>
        `).join('');
    }

    performNestedAnalysis(predictions) {
        const nestedResults = new Map();
        
        // Analyze how shorter timeframes fit into longer trends
        const hierarchies = [
            ['1M', '5M', '15M', '1H'],
            ['5M', '15M', '1H', '4H'],
            ['15M', '1H', '4H', '1D']
        ];
        
        hierarchies.forEach(hierarchy => {
            const analysis = this.analyzeTimeframeHierarchy(hierarchy, predictions);
            nestedResults.set(hierarchy.join('-'), analysis);
        });
        
        this.displayNestedAnalysis(nestedResults);
    }

    analyzeTimeframeHierarchy(hierarchy, predictions) {
        const results = {
            coherence: 0,
            conflicts: [],
            trend_alignment: {},
            confidence_cascade: []
        };
        
        // Calculate coherence across the hierarchy
        let totalCoherence = 0;
        let comparisonCount = 0;
        
        for (let i = 0; i < hierarchy.length - 1; i++) {
            const tf1 = hierarchy[i];
            const tf2 = hierarchy[i + 1];
            
            const pred1 = predictions[tf1];
            const pred2 = predictions[tf2];
            
            if (pred1 && pred2) {
                const coherence = this.calculateCoherence(pred1, pred2);
                totalCoherence += coherence;
                comparisonCount++;
                
                results.trend_alignment[`${tf1}-${tf2}`] = coherence;
                
                if (coherence < 0.5) {
                    results.conflicts.push({
                        timeframes: [tf1, tf2],
                        severity: 1 - coherence
                    });
                }
            }
        }
        
        results.coherence = comparisonCount > 0 ? totalCoherence / comparisonCount : 0;
        
        // Confidence cascade analysis
        hierarchy.forEach(tf => {
            const pred = predictions[tf];
            if (pred) {
                results.confidence_cascade.push({
                    timeframe: tf,
                    confidence: pred.confidence,
                    direction: pred.direction
                });
            }
        });
        
        return results;
    }

    calculateCoherence(pred1, pred2) {
        // Direction alignment
        const directionMatch = (pred1.direction === pred2.direction) ? 1.0 : 0.0;
        
        // Confidence similarity
        const confidenceSimilarity = 1 - Math.abs(pred1.confidence - pred2.confidence);
        
        // Overall coherence
        return (directionMatch * 0.6) + (confidenceSimilarity * 0.4);
    }

    displayNestedAnalysis(nestedResults) {
        const container = document.getElementById('nested-container');
        if (!container) return;

        let html = '<div class="nested-analysis-grid">';
        
        nestedResults.forEach((analysis, hierarchy) => {
            const timeframes = hierarchy.split('-');
            
            html += `
                <div class="nested-analysis-item">
                    <div class="analysis-header">
                        <h5>${timeframes.join(' → ')}</h5>
                        <div class="coherence-score coherence-${this.getCoherenceClass(analysis.coherence)}">
                            ${(analysis.coherence * 100).toFixed(1)}%
                        </div>
                    </div>
                    
                    <div class="analysis-content">
                        <div class="confidence-cascade">
                            <h6>Confidence Cascade</h6>
                            <div class="cascade-bars">
                                ${analysis.confidence_cascade.map(item => `
                                    <div class="cascade-item">
                                        <span class="cascade-tf">${item.timeframe}</span>
                                        <div class="cascade-bar">
                                            <div class="cascade-fill ${item.direction?.toLowerCase()}" 
                                                 style="width: ${item.confidence * 100}%"></div>
                                        </div>
                                        <span class="cascade-value">${Math.round(item.confidence * 100)}%</span>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                        
                        ${analysis.conflicts.length > 0 ? `
                            <div class="hierarchy-conflicts">
                                <h6>Conflicts</h6>
                                ${analysis.conflicts.map(conflict => `
                                    <div class="conflict-item">
                                        <span>${conflict.timeframes.join(' vs ')}</span>
                                        <div class="severity-bar">
                                            <div class="severity-fill" style="width: ${conflict.severity * 100}%"></div>
                                        </div>
                                    </div>
                                `).join('')}
                            </div>
                        ` : ''}
                    </div>
                </div>
            `;
        });
        
        html += '</div>';
        container.innerHTML = html;
        
        this.nestedAnalysis = nestedResults;
    }

    getCoherenceClass(coherence) {
        if (coherence >= 0.8) return 'excellent';
        if (coherence >= 0.6) return 'good';
        if (coherence >= 0.4) return 'moderate';
        if (coherence >= 0.2) return 'poor';
        return 'very-poor';
    }

    calculateTradeConfidence() {
        const primaryTf = document.getElementById('primary-timeframe')?.value;
        const confirmationTfs = Array.from(document.querySelectorAll('.timeframe-checkboxes input:checked'))
            .map(cb => cb.value);
        
        if (!primaryTf || confirmationTfs.length === 0) return;
        
        // Get prediction data for selected timeframes
        const selectedPredictions = [primaryTf, ...confirmationTfs].map(tf => {
            // This would come from the predictions panel
            return window.predictionsPanel?.predictions?.get(tf);
        }).filter(pred => pred !== undefined);
        
        if (selectedPredictions.length === 0) return;
        
        // Calculate alignment score
        const alignmentScore = this.calculateSelectedAlignment(selectedPredictions);
        
        // Calculate confidence based on alignment and individual confidences
        const avgConfidence = selectedPredictions.reduce((sum, pred) => sum + pred.confidence, 0) / selectedPredictions.length;
        const tradeConfidence = (alignmentScore * 0.7) + (avgConfidence * 0.3);
        
        // Risk assessment
        const riskLevel = this.assessRisk(alignmentScore, tradeConfidence);
        
        // Position size recommendation
        const positionSize = this.recommendPositionSize(tradeConfidence, riskLevel);
        
        // Update UI
        this.updateConfidenceCalculator(tradeConfidence, riskLevel, positionSize);
    }

    calculateSelectedAlignment(predictions) {
        if (predictions.length < 2) return predictions[0]?.confidence || 0;
        
        let totalAlignment = 0;
        let comparisonCount = 0;
        
        for (let i = 0; i < predictions.length - 1; i++) {
            for (let j = i + 1; j < predictions.length; j++) {
                const alignment = this.calculateCoherence(predictions[i], predictions[j]);
                totalAlignment += alignment;
                comparisonCount++;
            }
        }
        
        return comparisonCount > 0 ? totalAlignment / comparisonCount : 0;
    }

    assessRisk(alignment, confidence) {
        const riskScore = (1 - alignment) * 0.6 + (1 - confidence) * 0.4;
        
        if (riskScore <= 0.2) return 'Very Low';
        if (riskScore <= 0.4) return 'Low';
        if (riskScore <= 0.6) return 'Medium';
        if (riskScore <= 0.8) return 'High';
        return 'Very High';
    }

    recommendPositionSize(confidence, riskLevel) {
        const baseSize = 2.0; // 2% base position
        
        const confidenceMultiplier = confidence;
        const riskMultipliers = {
            'Very Low': 1.5,
            'Low': 1.2,
            'Medium': 1.0,
            'High': 0.7,
            'Very High': 0.3
        };
        
        const recommendedSize = baseSize * confidenceMultiplier * (riskMultipliers[riskLevel] || 1.0);
        return Math.min(Math.max(recommendedSize, 0.1), 5.0); // Cap between 0.1% and 5%
    }

    updateConfidenceCalculator(confidence, riskLevel, positionSize) {
        // Update confidence meter
        const meterFill = document.querySelector('.confidence-meter .meter-fill');
        const meterValue = document.querySelector('.confidence-meter .meter-value');
        
        if (meterFill && meterValue) {
            const confidencePercent = confidence * 100;
            meterFill.style.width = `${confidencePercent}%`;
            meterFill.className = `meter-fill ${this.getConfidenceClass(confidence)}`;
            meterValue.textContent = `${confidencePercent.toFixed(1)}%`;
        }
        
        // Update risk level
        const riskElement = document.querySelector('.risk-level');
        if (riskElement) {
            riskElement.textContent = riskLevel;
            riskElement.className = `risk-level ${riskLevel.toLowerCase().replace(' ', '-')}`;
        }
        
        // Update position size
        const positionElement = document.querySelector('.position-size');
        if (positionElement) {
            positionElement.textContent = `${positionSize.toFixed(1)}%`;
        }
    }

    getConfidenceClass(confidence) {
        if (confidence >= 0.8) return 'excellent';
        if (confidence >= 0.6) return 'good';
        if (confidence >= 0.4) return 'moderate';
        if (confidence >= 0.2) return 'poor';
        return 'very-poor';
    }

    updateSummaryMetrics() {
        // Overall alignment
        const alignmentValues = Array.from(this.alignmentScores.values());
        const overallAlignment = alignmentValues.length > 0 ? 
            alignmentValues.reduce((sum, val) => sum + val, 0) / alignmentValues.length : 0;
        
        const alignmentElement = document.getElementById('overall-alignment');
        if (alignmentElement) {
            alignmentElement.textContent = `${(overallAlignment * 100).toFixed(1)}%`;
            alignmentElement.className = `alignment-score ${this.getAlignmentClass(overallAlignment)}`;
        }
        
        // Divergence count
        const divergenceCount = document.querySelectorAll('.divergence-item').length;
        const divergenceElement = document.getElementById('divergence-count');
        if (divergenceElement) {
            divergenceElement.textContent = divergenceCount.toString();
        }
        
        // Confidence level
        const confidenceLevel = this.calculateOverallConfidence();
        const confidenceElement = document.getElementById('confidence-level');
        if (confidenceElement) {
            confidenceElement.textContent = `${(confidenceLevel * 100).toFixed(1)}%`;
            confidenceElement.className = `confidence-level ${this.getConfidenceClass(confidenceLevel)}`;
        }
    }

    calculateOverallConfidence() {
        // This would be calculated based on current predictions
        const predictions = window.predictionsPanel?.predictions;
        if (!predictions || predictions.size === 0) return 0;
        
        let totalConfidence = 0;
        predictions.forEach(pred => {
            totalConfidence += pred.confidence;
        });
        
        return totalConfidence / predictions.size;
    }

    updateDivergenceAnalysis() {
        // Re-run divergence analysis with new threshold
        // This would use cached prediction data
        console.log(`Updated divergence threshold to ${this.divergenceThreshold}`);
    }

    exportAnalysis() {
        const analysisData = {
            timestamp: new Date().toISOString(),
            correlationMatrix: Object.fromEntries(this.correlationMatrix),
            alignmentScores: Object.fromEntries(this.alignmentScores),
            nestedAnalysis: Object.fromEntries(this.nestedAnalysis),
            divergenceThreshold: this.divergenceThreshold
        };
        
        const blob = new Blob([JSON.stringify(analysisData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `timeframe-correlation-analysis-${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        
        URL.revokeObjectURL(url);
        
        console.log('✅ Analysis exported successfully');
    }

    showLoadingState() {
        const summaryElements = document.querySelectorAll('.summary-metric span');
        summaryElements.forEach(el => {
            el.textContent = 'Loading...';
        });
    }

    showErrorState() {
        const summaryElements = document.querySelectorAll('.summary-metric span');
        summaryElements.forEach(el => {
            el.textContent = 'Error';
        });
    }

    startRealTimeAnalysis() {
        // Start real-time correlation monitoring
        setInterval(() => {
            this.loadCorrelationData();
        }, 60000); // Update every minute
    }

    dispose() {
        console.log('✅ Timeframe Correlation Analyzer disposed');
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.correlationAnalyzer = new TimeframeCorrelationAnalyzer();
});

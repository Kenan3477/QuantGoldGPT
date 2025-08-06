"""
Enhanced Positions Section JavaScript for GoldGPT Dashboard
Provides complete signal management with history, live P&L, and signal generator
"""

// Enhanced Positions Section with Signal Management
function showEnhancedPositionsSection() {
    const content = document.querySelector('.content');
    content.innerHTML = `
        <div class="section-header">
            <h2><i class="fas fa-layer-group"></i> Positions Management</h2>
            <p>Signal history, live P&L tracking, and signal generator</p>
        </div>
        
        <!-- Portfolio Summary Cards -->
        <div class="portfolio-summary-grid">
            <div class="summary-card">
                <div class="card-icon"><i class="fas fa-wallet"></i></div>
                <div class="card-info">
                    <div class="card-value" id="total-balance">$10,000.00</div>
                    <div class="card-label">Total Balance</div>
                </div>
            </div>
            <div class="summary-card">
                <div class="card-icon"><i class="fas fa-chart-line"></i></div>
                <div class="card-info">
                    <div class="card-value" id="total-pnl">$0.00</div>
                    <div class="card-label">Total P&L</div>
                </div>
            </div>
            <div class="summary-card">
                <div class="card-icon"><i class="fas fa-percentage"></i></div>
                <div class="card-info">
                    <div class="card-value" id="win-rate">0%</div>
                    <div class="card-label">Win Rate</div>
                </div>
            </div>
            <div class="summary-card">
                <div class="card-icon"><i class="fas fa-chart-bar"></i></div>
                <div class="card-info">
                    <div class="card-value" id="open-positions-count">0</div>
                    <div class="card-label">Open Positions</div>
                </div>
            </div>
        </div>
        
        <!-- Main Positions Content -->
        <div class="positions-main-grid">
            <!-- Signal Generator Panel -->
            <div class="positions-panel signal-generator-panel">
                <div class="panel-header">
                    <h3><i class="fas fa-plus-circle"></i> Signal Generator</h3>
                    <button class="collapse-btn" onclick="togglePanel('signal-generator')">
                        <i class="fas fa-chevron-up"></i>
                    </button>
                </div>
                <div class="panel-content" id="signal-generator-content">
                    <form id="signal-form" class="signal-form">
                        <div class="form-row">
                            <div class="form-group">
                                <label>Signal Type</label>
                                <select id="signal-type" required>
                                    <option value="BUY">BUY (Long)</option>
                                    <option value="SELL">SELL (Short)</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label>Symbol</label>
                                <select id="signal-symbol">
                                    <option value="XAUUSD">XAUUSD (Gold)</option>
                                    <option value="EURUSD">EURUSD</option>
                                    <option value="GBPUSD">GBPUSD</option>
                                </select>
                            </div>
                        </div>
                        
                        <div class="form-row">
                            <div class="form-group">
                                <label>Quantity (Lots)</label>
                                <input type="number" id="signal-quantity" step="0.01" min="0.01" value="0.1" required>
                            </div>
                            <div class="form-group">
                                <label>Current Price</label>
                                <input type="number" id="current-price" step="0.01" readonly>
                            </div>
                        </div>
                        
                        <div class="form-row">
                            <div class="form-group">
                                <label>Take Profit</label>
                                <input type="number" id="take-profit" step="0.01" required>
                            </div>
                            <div class="form-group">
                                <label>Stop Loss</label>
                                <input type="number" id="stop-loss" step="0.01" required>
                            </div>
                        </div>
                        
                        <div class="form-row">
                            <div class="form-group">
                                <label>Strategy</label>
                                <select id="signal-strategy">
                                    <option value="Manual">Manual Trade</option>
                                    <option value="AI Analysis">AI Analysis</option>
                                    <option value="Technical">Technical Analysis</option>
                                    <option value="Scalping">Scalping</option>
                                    <option value="Swing">Swing Trading</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label>Confidence (%)</label>
                                <input type="number" id="signal-confidence" min="1" max="100" value="75">
                            </div>
                        </div>
                        
                        <div class="form-group">
                            <label>Notes (Optional)</label>
                            <textarea id="signal-notes" placeholder="Trade reasoning, analysis notes..."></textarea>
                        </div>
                        
                        <button type="submit" class="btn-generate-signal">
                            <i class="fas fa-rocket"></i> Generate Signal
                        </button>
                    </form>
                </div>
            </div>
            
            <!-- Open Positions Panel -->
            <div class="positions-panel open-positions-panel">
                <div class="panel-header">
                    <h3><i class="fas fa-chart-bar"></i> Open Positions</h3>
                    <div class="panel-controls">
                        <button class="refresh-btn" onclick="refreshOpenPositions()">
                            <i class="fas fa-sync-alt"></i>
                        </button>
                        <button class="collapse-btn" onclick="togglePanel('open-positions')">
                            <i class="fas fa-chevron-up"></i>
                        </button>
                    </div>
                </div>
                <div class="panel-content" id="open-positions-content">
                    <div class="loading-state">
                        <i class="fas fa-spinner fa-spin"></i> Loading open positions...
                    </div>
                </div>
            </div>
            
            <!-- Position History Panel -->
            <div class="positions-panel history-panel">
                <div class="panel-header">
                    <h3><i class="fas fa-history"></i> Signal History</h3>
                    <div class="panel-controls">
                        <select id="history-filter" onchange="filterHistory()">
                            <option value="all">All Trades</option>
                            <option value="profitable">Profitable Only</option>
                            <option value="losses">Losses Only</option>
                            <option value="today">Today</option>
                            <option value="week">This Week</option>
                        </select>
                        <button class="collapse-btn" onclick="togglePanel('history')">
                            <i class="fas fa-chevron-up"></i>
                        </button>
                    </div>
                </div>
                <div class="panel-content" id="history-content">
                    <div class="loading-state">
                        <i class="fas fa-spinner fa-spin"></i> Loading trading history...
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Live Price Ticker -->
        <div class="live-price-ticker">
            <div class="ticker-item">
                <span class="symbol">XAUUSD</span>
                <span class="price" id="live-gold-price">$2000.00</span>
                <span class="change" id="price-change">+0.00</span>
            </div>
            <div class="ticker-timestamp">
                Last updated: <span id="price-timestamp">--:--:--</span>
            </div>
        </div>
    `;

    // Add the enhanced styles
    addPositionsStyles();
    
    // Initialize the positions functionality
    initializePositionsSystem();
}

function addPositionsStyles() {
    const styles = `
        <style>
        /* Portfolio Summary Grid */
        .portfolio-summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }
        
        .summary-card {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1.5rem;
            display: flex;
            align-items: center;
            gap: 1rem;
            transition: transform 0.2s ease;
        }
        
        .summary-card:hover {
            transform: translateY(-2px);
        }
        
        .card-icon {
            width: 50px;
            height: 50px;
            background: var(--primary-color);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 1.2rem;
        }
        
        .card-value {
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--text-primary);
        }
        
        .card-label {
            font-size: 0.8rem;
            color: var(--text-secondary);
            margin-top: 0.2rem;
        }
        
        /* Main Positions Grid */
        .positions-main-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
            margin-bottom: 2rem;
        }
        
        @media (max-width: 1200px) {
            .positions-main-grid {
                grid-template-columns: 1fr;
            }
        }
        
        /* Positions Panels */
        .positions-panel {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            overflow: hidden;
        }
        
        .signal-generator-panel {
            grid-column: 1 / -1;
        }
        
        .panel-header {
            background: var(--surface-color);
            padding: 1rem 1.5rem;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .panel-header h3 {
            margin: 0;
            color: var(--text-primary);
            font-size: 1.1rem;
        }
        
        .panel-controls {
            display: flex;
            gap: 0.5rem;
        }
        
        .refresh-btn, .collapse-btn {
            background: none;
            border: none;
            color: var(--text-secondary);
            padding: 0.5rem;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        
        .refresh-btn:hover, .collapse-btn:hover {
            background: var(--primary-color);
            color: white;
        }
        
        .panel-content {
            padding: 1.5rem;
        }
        
        /* Signal Form */
        .signal-form {
            max-width: 100%;
        }
        
        .form-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            margin-bottom: 1rem;
        }
        
        .form-group {
            display: flex;
            flex-direction: column;
        }
        
        .form-group label {
            font-size: 0.9rem;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
        }
        
        .form-group input, .form-group select, .form-group textarea {
            padding: 0.75rem;
            border: 1px solid var(--border-color);
            border-radius: 4px;
            background: var(--surface-color);
            color: var(--text-primary);
            font-size: 0.9rem;
        }
        
        .form-group textarea {
            min-height: 80px;
            resize: vertical;
        }
        
        .btn-generate-signal {
            width: 100%;
            padding: 1rem;
            background: var(--success-color);
            color: white;
            border: none;
            border-radius: 4px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            margin-top: 1rem;
        }
        
        .btn-generate-signal:hover {
            background: var(--success-hover);
            transform: translateY(-1px);
        }
        
        /* Position Tables */
        .position-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }
        
        .position-table th, .position-table td {
            padding: 0.75rem 0.5rem;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
            font-size: 0.85rem;
        }
        
        .position-table th {
            background: var(--surface-color);
            color: var(--text-secondary);
            font-weight: 600;
        }
        
        .position-row {
            transition: background-color 0.2s ease;
        }
        
        .position-row:hover {
            background: var(--surface-color);
        }
        
        .pnl-positive {
            color: var(--success-color);
            font-weight: 600;
        }
        
        .pnl-negative {
            color: var(--danger-color);
            font-weight: 600;
        }
        
        .signal-type-buy {
            color: var(--success-color);
            font-weight: 600;
        }
        
        .signal-type-sell {
            color: var(--danger-color);
            font-weight: 600;
        }
        
        .close-btn {
            padding: 0.3rem 0.6rem;
            background: var(--danger-color);
            color: white;
            border: none;
            border-radius: 3px;
            font-size: 0.8rem;
            cursor: pointer;
            transition: background-color 0.2s ease;
        }
        
        .close-btn:hover {
            background: var(--danger-hover);
        }
        
        /* Live Price Ticker */
        .live-price-ticker {
            background: var(--surface-color);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1rem 1.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .ticker-item {
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        
        .symbol {
            font-weight: 600;
            color: var(--text-primary);
        }
        
        .price {
            font-size: 1.2rem;
            font-weight: 600;
            color: var(--primary-color);
        }
        
        .change {
            font-weight: 600;
        }
        
        .change.positive {
            color: var(--success-color);
        }
        
        .change.negative {
            color: var(--danger-color);
        }
        
        .ticker-timestamp {
            font-size: 0.8rem;
            color: var(--text-secondary);
        }
        
        /* Loading and Empty States */
        .loading-state {
            text-align: center;
            padding: 2rem;
            color: var(--text-secondary);
        }
        
        .empty-state {
            text-align: center;
            padding: 2rem;
            color: var(--text-secondary);
        }
        
        .empty-state i {
            font-size: 2rem;
            margin-bottom: 1rem;
            opacity: 0.5;
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .form-row {
                grid-template-columns: 1fr;
            }
            
            .portfolio-summary-grid {
                grid-template-columns: repeat(2, 1fr);
            }
            
            .live-price-ticker {
                flex-direction: column;
                gap: 1rem;
                text-align: center;
            }
            
            .position-table {
                font-size: 0.8rem;
            }
            
            .position-table th, .position-table td {
                padding: 0.5rem 0.3rem;
            }
        }
        </style>
    `;
    
    if (!document.getElementById('positions-styles')) {
        const styleSheet = document.createElement('style');
        styleSheet.id = 'positions-styles';
        styleSheet.innerHTML = styles;
        document.head.appendChild(styleSheet);
    }
}

function initializePositionsSystem() {
    console.log('🚀 Initializing Enhanced Positions System...');
    
    // Load initial data
    loadPortfolioSummary();
    loadOpenPositions();
    loadPositionHistory();
    updateLivePrice();
    
    // Set up form submission
    const signalForm = document.getElementById('signal-form');
    if (signalForm) {
        signalForm.addEventListener('submit', handleSignalGeneration);
    }
    
    // Update current price when signal type changes
    const signalType = document.getElementById('signal-type');
    if (signalType) {
        signalType.addEventListener('change', updateCurrentPrice);
    }
    
    // Start live updates
    startLiveUpdates();
    
    // Initial price update
    updateCurrentPrice();
    
    console.log('✅ Positions system initialized');
}

async function loadPortfolioSummary() {
    try {
        const response = await fetch('/api/positions/portfolio');
        const data = await response.json();
        
        if (data.success) {
            const portfolio = data.portfolio;
            
            document.getElementById('total-balance').textContent = 
                `$${portfolio.total_balance?.toLocaleString('en-US', {minimumFractionDigits: 2}) || '0.00'}`;
            
            const totalPnlElement = document.getElementById('total-pnl');
            const pnl = portfolio.total_pnl || 0;
            totalPnlElement.textContent = `$${pnl.toLocaleString('en-US', {minimumFractionDigits: 2})}`;
            totalPnlElement.className = pnl >= 0 ? 'card-value pnl-positive' : 'card-value pnl-negative';
            
            document.getElementById('win-rate').textContent = 
                `${(portfolio.win_rate || 0).toFixed(1)}%`;
            
            document.getElementById('open-positions-count').textContent = 
                portfolio.open_positions || 0;
        }
    } catch (error) {
        console.error('Error loading portfolio summary:', error);
    }
}

async function loadOpenPositions() {
    try {
        const response = await fetch('/api/positions/open');
        const data = await response.json();
        
        const content = document.getElementById('open-positions-content');
        
        if (data.success && data.positions.length > 0) {
            content.innerHTML = `
                <table class="position-table">
                    <thead>
                        <tr>
                            <th>Type</th>
                            <th>Symbol</th>
                            <th>Entry</th>
                            <th>Current</th>
                            <th>P&L</th>
                            <th>P&L %</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${data.positions.map(position => `
                            <tr class="position-row">
                                <td class="signal-type-${position.type.toLowerCase()}">${position.type}</td>
                                <td>${position.symbol}</td>
                                <td>$${position.entry_price.toFixed(2)}</td>
                                <td>$${position.current_price.toFixed(2)}</td>
                                <td class="${position.pnl >= 0 ? 'pnl-positive' : 'pnl-negative'}">
                                    $${position.pnl.toFixed(2)}
                                </td>
                                <td class="${position.pnl_percentage >= 0 ? 'pnl-positive' : 'pnl-negative'}">
                                    ${position.pnl_percentage.toFixed(2)}%
                                </td>
                                <td>
                                    <button class="close-btn" onclick="closePosition('${position.id}')">
                                        Close
                                    </button>
                                </td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            `;
        } else {
            content.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-chart-line"></i>
                    <div>No open positions</div>
                    <div style="font-size: 0.9rem; margin-top: 0.5rem; opacity: 0.7;">
                        Generate a signal to start trading
                    </div>
                </div>
            `;
        }
    } catch (error) {
        console.error('Error loading open positions:', error);
        document.getElementById('open-positions-content').innerHTML = `
            <div class="empty-state">
                <i class="fas fa-exclamation-triangle"></i>
                <div>Error loading positions</div>
            </div>
        `;
    }
}

async function loadPositionHistory() {
    try {
        const response = await fetch('/api/positions/history?limit=20');
        const data = await response.json();
        
        const content = document.getElementById('history-content');
        
        if (data.success && data.history.length > 0) {
            content.innerHTML = `
                <table class="position-table">
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th>Type</th>
                            <th>Symbol</th>
                            <th>Entry</th>
                            <th>Exit</th>
                            <th>P&L</th>
                            <th>Strategy</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${data.history.map(trade => `
                            <tr class="position-row">
                                <td>${new Date(trade.created_at).toLocaleDateString()}</td>
                                <td class="signal-type-${trade.type.toLowerCase()}">${trade.type}</td>
                                <td>${trade.symbol}</td>
                                <td>$${trade.entry_price.toFixed(2)}</td>
                                <td>$${trade.current_price ? trade.current_price.toFixed(2) : '--'}</td>
                                <td class="${trade.pnl >= 0 ? 'pnl-positive' : 'pnl-negative'}">
                                    $${trade.pnl ? trade.pnl.toFixed(2) : '0.00'}
                                </td>
                                <td>${trade.strategy || 'Manual'}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            `;
        } else {
            content.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-history"></i>
                    <div>No trading history</div>
                    <div style="font-size: 0.9rem; margin-top: 0.5rem; opacity: 0.7;">
                        Your completed trades will appear here
                    </div>
                </div>
            `;
        }
    } catch (error) {
        console.error('Error loading position history:', error);
        document.getElementById('history-content').innerHTML = `
            <div class="empty-state">
                <i class="fas fa-exclamation-triangle"></i>
                <div>Error loading history</div>
            </div>
        `;
    }
}

async function updateCurrentPrice() {
    try {
        const response = await fetch('/api/gold-price');
        const data = await response.json();
        
        if (data.success) {
            const price = data.price;
            const currentPriceInput = document.getElementById('current-price');
            if (currentPriceInput) {
                currentPriceInput.value = price.toFixed(2);
            }
            
            // Update suggested TP/SL based on signal type
            const signalType = document.getElementById('signal-type').value;
            const tpInput = document.getElementById('take-profit');
            const slInput = document.getElementById('stop-loss');
            
            if (tpInput && slInput) {
                if (signalType === 'BUY') {
                    tpInput.value = (price + 10).toFixed(2); // $10 profit target
                    slInput.value = (price - 5).toFixed(2);  // $5 stop loss
                } else {
                    tpInput.value = (price - 10).toFixed(2); // $10 profit target
                    slInput.value = (price + 5).toFixed(2);  // $5 stop loss
                }
            }
        }
    } catch (error) {
        console.error('Error updating current price:', error);
    }
}

async function updateLivePrice() {
    try {
        const response = await fetch('/api/positions/live-update');
        const data = await response.json();
        
        if (data.success) {
            const priceElement = document.getElementById('live-gold-price');
            const timestampElement = document.getElementById('price-timestamp');
            
            if (priceElement) {
                priceElement.textContent = `$${data.gold_price.toFixed(2)}`;
            }
            
            if (timestampElement) {
                timestampElement.textContent = data.last_updated;
            }
        }
    } catch (error) {
        console.error('Error updating live price:', error);
    }
}

async function handleSignalGeneration(event) {
    event.preventDefault();
    
    const formData = {
        type: document.getElementById('signal-type').value,
        symbol: document.getElementById('signal-symbol').value,
        quantity: parseFloat(document.getElementById('signal-quantity').value),
        take_profit: parseFloat(document.getElementById('take-profit').value),
        stop_loss: parseFloat(document.getElementById('stop-loss').value),
        strategy: document.getElementById('signal-strategy').value,
        confidence: parseInt(document.getElementById('signal-confidence').value),
        notes: document.getElementById('signal-notes').value
    };
    
    const submitButton = document.querySelector('.btn-generate-signal');
    submitButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating...';
    submitButton.disabled = true;
    
    try {
        const response = await fetch('/api/positions/generate-signal', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Reset form
            document.getElementById('signal-form').reset();
            document.getElementById('signal-confidence').value = 75;
            
            // Refresh displays
            loadPortfolioSummary();
            loadOpenPositions();
            updateCurrentPrice();
            
            // Show success message
            showNotification(data.message, 'success');
        } else {
            showNotification(data.error || 'Failed to generate signal', 'error');
        }
    } catch (error) {
        console.error('Error generating signal:', error);
        showNotification('Error generating signal', 'error');
    } finally {
        submitButton.innerHTML = '<i class="fas fa-rocket"></i> Generate Signal';
        submitButton.disabled = false;
    }
}

async function closePosition(signalId) {
    if (!confirm('Are you sure you want to close this position?')) {
        return;
    }
    
    try {
        const response = await fetch(`/api/positions/close/${signalId}`, {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (data.success) {
            showNotification(data.message, 'success');
            
            // Refresh displays
            loadPortfolioSummary();
            loadOpenPositions();
            loadPositionHistory();
        } else {
            showNotification(data.error || 'Failed to close position', 'error');
        }
    } catch (error) {
        console.error('Error closing position:', error);
        showNotification('Error closing position', 'error');
    }
}

function refreshOpenPositions() {
    loadOpenPositions();
    loadPortfolioSummary();
    updateLivePrice();
    showNotification('Positions refreshed', 'info');
}

function togglePanel(panelName) {
    const content = document.getElementById(`${panelName}-content`);
    const button = content.parentElement.querySelector('.collapse-btn i');
    
    if (content.style.display === 'none') {
        content.style.display = 'block';
        button.className = 'fas fa-chevron-up';
    } else {
        content.style.display = 'none';
        button.className = 'fas fa-chevron-down';
    }
}

function startLiveUpdates() {
    // Update live data every 30 seconds
    setInterval(() => {
        updateLivePrice();
        
        // Only refresh positions if we have open positions
        const openPositionsTable = document.querySelector('#open-positions-content table');
        if (openPositionsTable) {
            loadOpenPositions();
            loadPortfolioSummary();
        }
    }, 30000);
}

function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-triangle' : 'info-circle'}"></i>
        <span>${message}</span>
    `;
    
    // Add notification styles if not already added
    if (!document.getElementById('notification-styles')) {
        const notificationStyles = document.createElement('style');
        notificationStyles.id = 'notification-styles';
        notificationStyles.textContent = `
            .notification {
                position: fixed;
                top: 20px;
                right: 20px;
                padding: 1rem 1.5rem;
                border-radius: 8px;
                color: white;
                font-weight: 500;
                z-index: 10000;
                display: flex;
                align-items: center;
                gap: 0.5rem;
                animation: slideIn 0.3s ease;
            }
            .notification-success { background: var(--success-color); }
            .notification-error { background: var(--danger-color); }
            .notification-info { background: var(--primary-color); }
            @keyframes slideIn {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
        `;
        document.head.appendChild(notificationStyles);
    }
    
    document.body.appendChild(notification);
    
    // Remove after 3 seconds
    setTimeout(() => {
        notification.remove();
    }, 3000);
}

// Export for global use
window.showEnhancedPositionsSection = showEnhancedPositionsSection;

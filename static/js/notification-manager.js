/**
 * GoldGPT Enhanced Notification Manager
 * Comprehensive notification system with Trading 212-style design
 */

class NotificationManager {
    constructor() {
        this.notifications = new Map();
        this.config = null;
        this.container = null;
        this.toastCounter = 0;
        this.soundEnabled = true;
        this.persistentNotifications = new Set();
        
        // Enhanced debugging for gold trading
        this.debugMode = true;
        this.notificationMetrics = {
            totalSent: 0,
            totalDismissed: 0,
            averageViewTime: 0,
            categoryStats: {},
            errorCount: 0
        };
        
        // Gold trading specific notification categories
        this.categories = {
            SUCCESS: 'success',
            ERROR: 'error',
            WARNING: 'warning',
            INFO: 'info',
            TRADE: 'trade',
            PRICE: 'price',
            SYSTEM: 'system',
            GOLD_ALERT: 'gold-alert',
            AI_SIGNAL: 'ai-signal',
            NEWS_IMPACT: 'news-impact'
        };
        
        // Trading 212-style user preferences with gold optimization
        this.preferences = {
            enabled: true,
            soundEnabled: true,
            position: 'top-right',
            duration: 6000, // Longer for trading notifications
            maxVisible: 4,  // Less cluttered for trading
            categories: {
                success: { enabled: true, sound: true, persist: false, icon: '✅' },
                error: { enabled: true, sound: true, persist: true, icon: '❌' },
                warning: { enabled: true, sound: true, persist: false, icon: '⚠️' },
                info: { enabled: true, sound: false, persist: false, icon: 'ℹ️' },
                trade: { enabled: true, sound: true, persist: true, icon: '💼' },
                price: { enabled: true, sound: false, persist: false, icon: '💰' },
                system: { enabled: true, sound: true, persist: false, icon: '⚙️' },
                'gold-alert': { enabled: true, sound: true, persist: true, icon: '🥇' },
                'ai-signal': { enabled: true, sound: true, persist: false, icon: '🤖' },
                'news-impact': { enabled: true, sound: false, persist: false, icon: '📰' }
            }
        };
        
        // Gold trading specific thresholds
        this.goldThresholds = {
            priceChange: 0.5,      // 0.5% price change triggers notification
            volumeSpike: 200,      // 200% volume increase
            aiConfidence: 0.8,     // 80% AI confidence threshold
            newsImpact: 'medium'   // Medium or higher news impact
        };
        
        console.log('🔔 Enhanced Notification Manager initialized for Gold Trading');
        this._logDebugInfo('Constructor completed', 'info');
    }

    /**
     * Enhanced debug logging for notification tracking
     * @private
     */
    _logDebugInfo(message, level = 'info', data = null) {
        if (!this.debugMode) return;
        
        const timestamp = new Date().toISOString();
        const logEntry = {
            timestamp,
            level,
            message,
            data,
            metrics: { ...this.notificationMetrics }
        };
        
        const logSymbols = {
            info: '💙',
            warn: '⚠️',
            error: '❌',
            success: '✅',
            notification: '🔔'
        };
        
        console.log(
            `${logSymbols[level] || '🔔'} [NotificationManager] ${timestamp}: ${message}`,
            data ? data : ''
        );
        
        // Store debug info for dashboard display
        if (window.GoldGPTDebugger) {
            window.GoldGPTDebugger.addLog('NotificationManager', logEntry);
        }
    }

    /**
     * Initialize the notification system with gold trading optimizations
     */
    async initialize() {
        try {
            this._logDebugInfo('Starting notification system initialization...', 'info');
            
            // Wait for configuration to be loaded
            await this._waitForConfig();
            
            // Apply gold trading specific configurations
            this._applyGoldTradingConfig();
            
            // Create notification container with Trading 212 styling
            this._createContainer();
            
            // Setup sound system
            await this._setupSoundSystem();
            
            // Load user preferences
            this._loadUserPreferences();
            
            // Setup gold trading specific listeners
            this._setupGoldTradingListeners();
            
            // Initialize metrics tracking
            this._initializeMetrics();
            
            this._logDebugInfo('Notification system initialization completed', 'success');
            
        } catch (error) {
            this._logDebugInfo('Notification system initialization failed', 'error', error);
            console.error('❌ Notification Manager initialization failed:', error);
            
            // Create basic container as fallback
            this._createBasicContainer();
        }
    }

    /**
     * Apply gold trading specific notification configurations
     * @private
     */
    _applyGoldTradingConfig() {
        if (!this.config) return;
        
        try {
            // Apply notification settings from config
            if (this.config.notifications) {
                this.preferences.duration = this.config.notifications.duration || 6000;
                this.preferences.position = this.config.notifications.position || 'top-right';
                this.preferences.maxVisible = this.config.notifications.maxNotifications || 4;
                this.soundEnabled = this.config.notifications.sound?.enabled || true;
                
                // Apply gold threshold settings
                if (this.config.notifications.thresholds) {
                    this.goldThresholds = {
                        ...this.goldThresholds,
                        ...this.config.notifications.thresholds
                    };
                }
            }
            
            // Enable enhanced notifications for gold trading
            if (this.config.goldTrading?.enabled) {
                this.preferences.categories['gold-alert'].enabled = true;
                this.preferences.categories['ai-signal'].enabled = true;
                this.preferences.duration = 7000; // Longer duration for important trading alerts
            }
            
            this._logDebugInfo('Gold trading notification config applied', 'success', {
                duration: this.preferences.duration,
                maxVisible: this.preferences.maxVisible,
                goldThresholds: this.goldThresholds
            });
            
        } catch (error) {
            this._logDebugInfo('Failed to apply gold trading config', 'error', error);
        }
    }

    /**
     * Setup gold trading specific event listeners
     * @private
     */
    _setupGoldTradingListeners() {
        if (!this.config?.goldTrading?.enabled) return;
        
        try {
            // Listen for gold price alerts
            if (window.connectionManager) {
                window.connectionManager.on('gold_price_change', (data) => {
                    this._handleGoldPriceChange(data);
                });
                
                window.connectionManager.on('ai_signal', (data) => {
                    this._handleAISignal(data);
                });
                
                window.connectionManager.on('news_impact', (data) => {
                    this._handleNewsImpact(data);
                });
            }
            
            this._logDebugInfo('Gold trading listeners setup completed', 'success');
            
        } catch (error) {
            this._logDebugInfo('Failed to setup gold trading listeners', 'error', error);
        }
    }

    /**
     * Create notification container in the DOM
     * @private
     */
    _createContainer() {
        try {
            // Remove existing container if present
            const existing = document.getElementById('notification-container');
            if (existing) {
                existing.remove();
            }
            
            // Create main container
            this.container = document.createElement('div');
            this.container.id = 'notification-container';
            this.container.className = `notification-container position-${this.preferences.position}`;
            
            // Add container styles
            this.container.style.cssText = `
                position: fixed;
                z-index: 999999;
                pointer-events: none;
                max-width: 400px;
                display: flex;
                flex-direction: column;
                gap: 10px;
            `;
            
            // Position the container
            this._positionContainer();
            
            // Add to DOM
            document.body.appendChild(this.container);
            
            this._logDebugInfo('Notification container created successfully', 'success');
            
        } catch (error) {
            this._logDebugInfo('Failed to create notification container', 'error', error);
            console.error('❌ Failed to create notification container:', error);
        }
    }

    /**
     * Create basic fallback container
     * @private
     */
    _createBasicContainer() {
        try {
            // Create minimal container for error scenarios
            this.container = document.createElement('div');
            this.container.id = 'notification-container-basic';
            this.container.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                z-index: 999999;
                max-width: 300px;
                pointer-events: none;
            `;
            
            document.body.appendChild(this.container);
            
            this._logDebugInfo('Basic notification container created', 'success');
            
        } catch (error) {
            this._logDebugInfo('Failed to create basic container', 'error', error);
            console.error('❌ Failed to create basic notification container:', error);
        }
    }

    /**
     * Position the notification container based on preferences
     * @private
     */
    _positionContainer() {
        if (!this.container) return;
        
        const position = this.preferences.position;
        
        // Reset all positions
        this.container.style.top = '';
        this.container.style.bottom = '';
        this.container.style.left = '';
        this.container.style.right = '';
        
        // Apply position-specific styles
        switch (position) {
            case 'top-right':
                this.container.style.top = '20px';
                this.container.style.right = '20px';
                break;
            case 'top-left':
                this.container.style.top = '20px';
                this.container.style.left = '20px';
                break;
            case 'bottom-right':
                this.container.style.bottom = '20px';
                this.container.style.right = '20px';
                this.container.style.flexDirection = 'column-reverse';
                break;
            case 'bottom-left':
                this.container.style.bottom = '20px';
                this.container.style.left = '20px';
                this.container.style.flexDirection = 'column-reverse';
                break;
            default:
                this.container.style.top = '20px';
                this.container.style.right = '20px';
        }
    }

    /**
     * Handle gold price change notifications
     * @private
     */
    _handleGoldPriceChange(data) {
        try {
            const { symbol, change, changePercent, price } = data;
            
            if (symbol !== 'XAUUSD') return;
            
            // Check if change exceeds threshold
            if (Math.abs(changePercent) >= this.goldThresholds.priceChange) {
                const direction = changePercent > 0 ? '📈' : '📉';
                const color = changePercent > 0 ? '#00ff88' : '#ff4444';
                
                this.show({
                    title: `${direction} Gold Price Alert`,
                    message: `${symbol}: $${price.toFixed(2)} (${changePercent > 0 ? '+' : ''}${changePercent.toFixed(2)}%)`,
                    type: 'gold-alert',
                    priority: 'high',
                    data: { symbol, price, change, changePercent },
                    customStyle: { borderLeft: `4px solid ${color}` }
                });
                
                this._logDebugInfo('Gold price alert triggered', 'notification', data);
            }
            
        } catch (error) {
            this._logDebugInfo('Failed to handle gold price change', 'error', error);
        }
    }

    /**
     * Wait for configuration manager to be ready
     */
    async _waitForConfig() {
        return new Promise((resolve) => {
            const checkConfig = () => {
                if (window.configManager && window.configManager.isLoaded) {
                    resolve();
                } else {
                    setTimeout(checkConfig, 100);
                }
            };
            checkConfig();
        });
    }

    /**
     * Show success notification
     */
    showSuccess(title, message, options = {}) {
        return this.show(this.categories.SUCCESS, title, message, options);
    }

    /**
     * Show error notification
     */
    showError(title, message, options = {}) {
        return this.show(this.categories.ERROR, title, message, {
            ...options,
            persist: options.persist !== false
        });
    }

    /**
     * Show warning notification
     */
    showWarning(title, message, options = {}) {
        return this.show(this.categories.WARNING, title, message, options);
    }

    /**
     * Show info notification
     */
    showInfo(title, message, options = {}) {
        return this.show(this.categories.INFO, title, message, options);
    }

    /**
     * Show trade notification
     */
    showTrade(title, message, options = {}) {
        return this.show(this.categories.TRADE, title, message, {
            ...options,
            persist: options.persist !== false
        });
    }

    /**
     * Show price alert notification
     */
    showPriceAlert(title, message, options = {}) {
        return this.show(this.categories.PRICE, title, message, options);
    }

    /**
     * Show system notification
     */
    showSystem(title, message, options = {}) {
        return this.show(this.categories.SYSTEM, title, message, options);
    }

    /**
     * Show notification with comprehensive options
     */
    show(category, title, message, options = {}) {
        // Check if notifications are enabled
        if (!this.preferences.enabled || !this.preferences.categories[category]?.enabled) {
            return null;
        }

        const notificationId = this.generateNotificationId();
        const categoryConfig = this.preferences.categories[category];
        
        const notification = {
            id: notificationId,
            category,
            title,
            message,
            timestamp: Date.now(),
            duration: options.duration || (categoryConfig.persist ? 0 : this.preferences.duration),
            persist: options.persist || categoryConfig.persist,
            actions: options.actions || [],
            data: options.data || {},
            icon: options.icon || this.getCategoryIcon(category),
            onClick: options.onClick,
            onClose: options.onClose,
            priority: options.priority || 'normal'
        };

        // Store notification
        this.notifications.set(notificationId, notification);
        
        // Play sound if enabled
        if (this.preferences.soundEnabled && categoryConfig.sound) {
            this.playNotificationSound(category);
        }

        // Show toast
        this.showToast(notification);

        // Add to persistent notifications if needed
        if (notification.persist) {
            this.persistentNotifications.add(notificationId);
        }

        // Auto-hide if not persistent
        if (!notification.persist && notification.duration > 0) {
            setTimeout(() => {
                this.hide(notificationId);
            }, notification.duration);
        }

        // Emit event
        this.emit('notification_shown', notification);

        console.log(`🔔 Notification shown: ${category} - ${title}`);
        
        return notificationId;
    }

    /**
     * Hide notification
     */
    hide(notificationId) {
        const notification = this.notifications.get(notificationId);
        if (!notification) return;

        // Remove from DOM
        const element = document.getElementById(`notification-${notificationId}`);
        if (element) {
            element.classList.add('notification-exit');
            setTimeout(() => {
                element.remove();
                this.cleanupNotificationSpace();
            }, 300);
        }

        // Remove from persistent set
        this.persistentNotifications.delete(notificationId);

        // Call onClose callback
        if (notification.onClose) {
            try {
                notification.onClose(notification);
            } catch (error) {
                console.error('❌ Error in notification close callback:', error);
            }
        }

        // Remove from storage
        this.notifications.delete(notificationId);

        // Emit event
        this.emit('notification_hidden', notification);

        console.log(`🔕 Notification hidden: ${notificationId}`);
    }

    /**
     * Clear all notifications
     */
    clearAll() {
        const notificationIds = Array.from(this.notifications.keys());
        notificationIds.forEach(id => this.hide(id));
        
        this.emit('all_notifications_cleared');
        console.log('🧹 All notifications cleared');
    }

    /**
     * Clear notifications by category
     */
    clearByCategory(category) {
        const notificationIds = Array.from(this.notifications.values())
            .filter(n => n.category === category)
            .map(n => n.id);
        
        notificationIds.forEach(id => this.hide(id));
        
        this.emit('category_notifications_cleared', { category });
        console.log(`🧹 ${category} notifications cleared`);
    }

    /**
     * Show toast notification
     */
    showToast(notification) {
        const toast = document.createElement('div');
        toast.id = `notification-${notification.id}`;
        toast.className = `notification toast ${notification.category} ${notification.priority}`;
        
        toast.innerHTML = `
            <div class="notification-content">
                <div class="notification-icon">
                    ${notification.icon}
                </div>
                <div class="notification-text">
                    <div class="notification-title">${this.escapeHtml(notification.title)}</div>
                    <div class="notification-message">${this.escapeHtml(notification.message)}</div>
                    <div class="notification-timestamp">${this.formatTimestamp(notification.timestamp)}</div>
                </div>
                <div class="notification-actions">
                    ${notification.actions.map(action => 
                        `<button class="notification-action" data-action="${action.id}">${action.label}</button>`
                    ).join('')}
                </div>
                <button class="notification-close" title="Close">×</button>
            </div>
            ${notification.persist ? '<div class="notification-persist-indicator"></div>' : ''}
        `;

        // Add event listeners
        this.addToastEventListeners(toast, notification);

        // Add to container
        this.container.appendChild(toast);

        // Animate in
        setTimeout(() => {
            toast.classList.add('notification-visible');
        }, 50);

        // Manage visible notifications count
        this.manageVisibleNotifications();
    }

    /**
     * Add event listeners to toast
     */
    addToastEventListeners(toast, notification) {
        // Close button
        const closeButton = toast.querySelector('.notification-close');
        closeButton.addEventListener('click', (e) => {
            e.stopPropagation();
            this.hide(notification.id);
        });

        // Toast click
        toast.addEventListener('click', () => {
            if (notification.onClick) {
                try {
                    notification.onClick(notification);
                } catch (error) {
                    console.error('❌ Error in notification click callback:', error);
                }
            }
        });

        // Action buttons
        const actionButtons = toast.querySelectorAll('.notification-action');
        actionButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                e.stopPropagation();
                
                const actionId = button.getAttribute('data-action');
                const action = notification.actions.find(a => a.id === actionId);
                
                if (action && action.callback) {
                    try {
                        action.callback(notification);
                    } catch (error) {
                        console.error('❌ Error in notification action callback:', error);
                    }
                }
                
                // Hide notification unless action specifies otherwise
                if (!action || action.closeAfter !== false) {
                    this.hide(notification.id);
                }
            });
        });

        // Auto-hide hover pause
        if (!notification.persist) {
            toast.addEventListener('mouseenter', () => {
                toast.classList.add('notification-paused');
            });

            toast.addEventListener('mouseleave', () => {
                toast.classList.remove('notification-paused');
            });
        }
    }

    /**
     * Create notification container
     */
    createNotificationContainer() {
        this.container = document.createElement('div');
        this.container.id = 'notification-container';
        this.container.className = `notification-container ${this.preferences.position}`;
        
        document.body.appendChild(this.container);
    }

    /**
     * Manage visible notifications count
     */
    manageVisibleNotifications() {
        const visibleToasts = this.container.querySelectorAll('.notification');
        
        if (visibleToasts.length > this.preferences.maxVisible) {
            // Hide oldest non-persistent notifications
            const oldestNotifications = Array.from(visibleToasts)
                .filter(toast => !toast.classList.contains('persist'))
                .slice(0, visibleToasts.length - this.preferences.maxVisible);
            
            oldestNotifications.forEach(toast => {
                const notificationId = toast.id.replace('notification-', '');
                this.hide(notificationId);
            });
        }
    }

    /**
     * Cleanup notification space
     */
    cleanupNotificationSpace() {
        const toasts = this.container.querySelectorAll('.notification');
        toasts.forEach((toast, index) => {
            toast.style.transform = `translateY(${index * -10}px)`;
        });
    }

    /**
     * Get category icon
     */
    getCategoryIcon(category) {
        const icons = {
            success: '✅',
            error: '❌',
            warning: '⚠️',
            info: 'ℹ️',
            trade: '💰',
            price: '📈',
            system: '⚙️'
        };
        
        return icons[category] || 'ℹ️';
    }

    /**
     * Initialize sound system
     */
    initializeSounds() {
        this.sounds = {
            success: this.createSound('success'),
            error: this.createSound('error'),
            warning: this.createSound('warning'),
            info: this.createSound('info'),
            trade: this.createSound('trade'),
            price: this.createSound('price'),
            system: this.createSound('system')
        };
    }

    /**
     * Create sound for category
     */
    createSound(category) {
        // Create audio context or use simple audio for now
        return {
            play: () => {
                if (!this.preferences.soundEnabled) return;
                
                // Simple beep sound using Web Audio API
                try {
                    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                    const oscillator = audioContext.createOscillator();
                    const gainNode = audioContext.createGain();
                    
                    oscillator.connect(gainNode);
                    gainNode.connect(audioContext.destination);
                    
                    // Different frequencies for different categories
                    const frequencies = {
                        success: 800,
                        error: 400,
                        warning: 600,
                        info: 500,
                        trade: 900,
                        price: 700,
                        system: 550
                    };
                    
                    oscillator.frequency.setValueAtTime(frequencies[category] || 500, audioContext.currentTime);
                    oscillator.type = 'sine';
                    
                    gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
                    gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.2);
                    
                    oscillator.start(audioContext.currentTime);
                    oscillator.stop(audioContext.currentTime + 0.2);
                    
                } catch (error) {
                    console.warn('⚠️ Sound playback failed:', error);
                }
            }
        };
    }

    /**
     * Play notification sound
     */
    playNotificationSound(category) {
        const sound = this.sounds[category];
        if (sound) {
            sound.play();
        }
    }

    /**
     * Load user preferences
     */
    loadPreferences() {
        try {
            const saved = localStorage.getItem('goldgpt_notification_preferences');
            if (saved) {
                const parsed = JSON.parse(saved);
                this.preferences = { ...this.preferences, ...parsed };
            }
        } catch (error) {
            console.error('❌ Failed to load notification preferences:', error);
        }
    }

    /**
     * Save user preferences
     */
    savePreferences() {
        try {
            localStorage.setItem('goldgpt_notification_preferences', JSON.stringify(this.preferences));
            console.log('💾 Notification preferences saved');
        } catch (error) {
            console.error('❌ Failed to save notification preferences:', error);
        }
    }

    /**
     * Update preferences
     */
    updatePreferences(newPreferences) {
        this.preferences = { ...this.preferences, ...newPreferences };
        this.savePreferences();
        this.emit('preferences_updated', this.preferences);
    }

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Connection manager events
        if (window.connectionManager) {
            window.connectionManager.on('error_occurred', (error) => {
                this.showError('Connection Error', error.error.message);
            });
            
            window.connectionManager.on('reconnecting', () => {
                this.showInfo('Reconnecting', 'Attempting to reconnect to server...');
            });
            
            window.connectionManager.on('connected', () => {
                this.showSuccess('Connected', 'Successfully connected to server');
            });
        }

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'n') {
                e.preventDefault();
                this.toggleNotifications();
            }
        });
    }

    /**
     * Toggle notifications on/off
     */
    toggleNotifications() {
        this.preferences.enabled = !this.preferences.enabled;
        this.savePreferences();
        
        if (!this.preferences.enabled) {
            this.clearAll();
        }
        
        this.showInfo('Notifications', `Notifications ${this.preferences.enabled ? 'enabled' : 'disabled'}`);
    }

    /**
     * Show notification preferences modal
     */
    showPreferencesModal() {
        const modal = document.createElement('div');
        modal.className = 'notification-preferences-modal';
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h3>Notification Preferences</h3>
                    <button class="close-button">&times;</button>
                </div>
                <div class="modal-body">
                    <div class="preference-section">
                        <h4>General Settings</h4>
                        <label>
                            <input type="checkbox" id="notifications-enabled" ${this.preferences.enabled ? 'checked' : ''}>
                            Enable Notifications
                        </label>
                        <label>
                            <input type="checkbox" id="sound-enabled" ${this.preferences.soundEnabled ? 'checked' : ''}>
                            Enable Sound
                        </label>
                        <label>
                            Position:
                            <select id="notification-position">
                                <option value="top-right" ${this.preferences.position === 'top-right' ? 'selected' : ''}>Top Right</option>
                                <option value="top-left" ${this.preferences.position === 'top-left' ? 'selected' : ''}>Top Left</option>
                                <option value="bottom-right" ${this.preferences.position === 'bottom-right' ? 'selected' : ''}>Bottom Right</option>
                                <option value="bottom-left" ${this.preferences.position === 'bottom-left' ? 'selected' : ''}>Bottom Left</option>
                            </select>
                        </label>
                    </div>
                    <div class="preference-section">
                        <h4>Category Settings</h4>
                        ${Object.entries(this.preferences.categories).map(([category, config]) => `
                            <div class="category-config">
                                <h5>${category.charAt(0).toUpperCase() + category.slice(1)}</h5>
                                <label>
                                    <input type="checkbox" data-category="${category}" data-setting="enabled" ${config.enabled ? 'checked' : ''}>
                                    Enabled
                                </label>
                                <label>
                                    <input type="checkbox" data-category="${category}" data-setting="sound" ${config.sound ? 'checked' : ''}>
                                    Sound
                                </label>
                                <label>
                                    <input type="checkbox" data-category="${category}" data-setting="persist" ${config.persist ? 'checked' : ''}>
                                    Persistent
                                </label>
                            </div>
                        `).join('')}
                    </div>
                </div>
                <div class="modal-footer">
                    <button class="btn-primary" id="save-preferences">Save</button>
                    <button class="btn-secondary" id="cancel-preferences">Cancel</button>
                </div>
            </div>
        `;

        document.body.appendChild(modal);

        // Event listeners
        modal.querySelector('.close-button').addEventListener('click', () => modal.remove());
        modal.querySelector('#cancel-preferences').addEventListener('click', () => modal.remove());
        modal.querySelector('#save-preferences').addEventListener('click', () => {
            this.savePreferencesFromModal(modal);
            modal.remove();
        });

        // Category checkboxes
        modal.querySelectorAll('input[data-category]').forEach(checkbox => {
            checkbox.addEventListener('change', (e) => {
                const category = e.target.getAttribute('data-category');
                const setting = e.target.getAttribute('data-setting');
                this.preferences.categories[category][setting] = e.target.checked;
            });
        });

        // General settings
        modal.querySelector('#notifications-enabled').addEventListener('change', (e) => {
            this.preferences.enabled = e.target.checked;
        });

        modal.querySelector('#sound-enabled').addEventListener('change', (e) => {
            this.preferences.soundEnabled = e.target.checked;
        });

        modal.querySelector('#notification-position').addEventListener('change', (e) => {
            this.preferences.position = e.target.value;
        });
    }

    /**
     * Save preferences from modal
     */
    savePreferencesFromModal(modal) {
        this.savePreferences();
        
        // Update container position
        this.container.className = `notification-container ${this.preferences.position}`;
        
        this.showSuccess('Preferences', 'Notification preferences saved successfully');
    }

    /**
     * Inject notification styles
     */
    injectNotificationStyles() {
        const style = document.createElement('style');
        style.textContent = `
            /* Notification Container */
            .notification-container {
                position: fixed;
                z-index: 10000;
                pointer-events: none;
                max-width: 400px;
                width: 100%;
            }

            .notification-container.top-right {
                top: 20px;
                right: 20px;
            }

            .notification-container.top-left {
                top: 20px;
                left: 20px;
            }

            .notification-container.bottom-right {
                bottom: 20px;
                right: 20px;
            }

            .notification-container.bottom-left {
                bottom: 20px;
                left: 20px;
            }

            /* Notification Toast */
            .notification {
                background: white;
                border-radius: 8px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
                margin-bottom: 10px;
                opacity: 0;
                transform: translateX(100%);
                transition: all 0.3s ease;
                pointer-events: auto;
                position: relative;
                overflow: hidden;
            }

            .notification.notification-visible {
                opacity: 1;
                transform: translateX(0);
            }

            .notification.notification-exit {
                opacity: 0;
                transform: translateX(100%);
            }

            .notification.notification-paused {
                animation-play-state: paused;
            }

            /* Category Styles */
            .notification.success {
                border-left: 4px solid #4CAF50;
            }

            .notification.error {
                border-left: 4px solid #F44336;
            }

            .notification.warning {
                border-left: 4px solid #FF9800;
            }

            .notification.info {
                border-left: 4px solid #2196F3;
            }

            .notification.trade {
                border-left: 4px solid #9C27B0;
            }

            .notification.price {
                border-left: 4px solid #FF5722;
            }

            .notification.system {
                border-left: 4px solid #607D8B;
            }

            /* Priority Styles */
            .notification.high {
                border-width: 4px;
                box-shadow: 0 6px 16px rgba(0, 0, 0, 0.2);
            }

            /* Notification Content */
            .notification-content {
                display: flex;
                align-items: flex-start;
                padding: 16px;
                gap: 12px;
            }

            .notification-icon {
                font-size: 20px;
                flex-shrink: 0;
                margin-top: 2px;
            }

            .notification-text {
                flex: 1;
                min-width: 0;
            }

            .notification-title {
                font-weight: 600;
                font-size: 14px;
                color: #333;
                margin-bottom: 4px;
            }

            .notification-message {
                font-size: 13px;
                color: #666;
                line-height: 1.4;
                margin-bottom: 8px;
            }

            .notification-timestamp {
                font-size: 11px;
                color: #999;
            }

            .notification-actions {
                display: flex;
                gap: 8px;
                margin-top: 8px;
            }

            .notification-action {
                background: #f0f0f0;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 12px;
                cursor: pointer;
                transition: all 0.2s ease;
            }

            .notification-action:hover {
                background: #e0e0e0;
            }

            .notification-close {
                position: absolute;
                top: 8px;
                right: 8px;
                background: none;
                border: none;
                font-size: 18px;
                cursor: pointer;
                color: #999;
                width: 24px;
                height: 24px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.2s ease;
            }

            .notification-close:hover {
                background: rgba(0, 0, 0, 0.1);
                color: #666;
            }

            /* Persistent Indicator */
            .notification-persist-indicator {
                position: absolute;
                bottom: 0;
                left: 0;
                right: 0;
                height: 2px;
                background: linear-gradient(90deg, #4CAF50, #2196F3);
            }

            /* Preferences Modal */
            .notification-preferences-modal {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.5);
                display: flex;
                justify-content: center;
                align-items: center;
                z-index: 10002;
            }

            .notification-preferences-modal .modal-content {
                background: white;
                border-radius: 8px;
                width: 90%;
                max-width: 600px;
                max-height: 80vh;
                overflow-y: auto;
            }

            .notification-preferences-modal .modal-header {
                padding: 20px;
                border-bottom: 1px solid #eee;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }

            .notification-preferences-modal .modal-body {
                padding: 20px;
            }

            .notification-preferences-modal .modal-footer {
                padding: 20px;
                border-top: 1px solid #eee;
                display: flex;
                justify-content: flex-end;
                gap: 10px;
            }

            .preference-section {
                margin-bottom: 20px;
            }

            .preference-section h4 {
                margin-bottom: 10px;
                color: #333;
            }

            .preference-section label {
                display: block;
                margin-bottom: 8px;
                font-size: 14px;
            }

            .preference-section select {
                margin-left: 8px;
                padding: 4px;
                border: 1px solid #ddd;
                border-radius: 4px;
            }

            .category-config {
                background: #f8f9fa;
                padding: 12px;
                border-radius: 4px;
                margin-bottom: 8px;
            }

            .category-config h5 {
                margin: 0 0 8px 0;
                color: #555;
            }

            .category-config label {
                display: inline-block;
                margin-right: 16px;
                margin-bottom: 4px;
            }

            /* Responsive Design */
            @media (max-width: 768px) {
                .notification-container {
                    max-width: 90%;
                    margin: 0 20px;
                }

                .notification-content {
                    padding: 12px;
                    gap: 8px;
                }

                .notification-title {
                    font-size: 13px;
                }

                .notification-message {
                    font-size: 12px;
                }

                .notification-preferences-modal .modal-content {
                    width: 95%;
                    margin: 20px;
                }
            }

            /* Accessibility */
            .notification:focus {
                outline: 2px solid #2196F3;
                outline-offset: 2px;
            }

            @media (prefers-reduced-motion: reduce) {
                .notification {
                    transition: none;
                }
            }
        `;
        
        document.head.appendChild(style);
    }

    /**
     * Emit event to listeners
     */
    emit(eventName, data) {
        const event = new CustomEvent(`notification_${eventName}`, {
            detail: data
        });
        document.dispatchEvent(event);
    }

    /**
     * Generate unique notification ID
     */
    generateNotificationId() {
        return `notify_${Date.now()}_${++this.toastCounter}`;
    }

    /**
     * Format timestamp
     */
    formatTimestamp(timestamp) {
        const date = new Date(timestamp);
        return date.toLocaleTimeString([], { 
            hour: '2-digit', 
            minute: '2-digit' 
        });
    }

    /**
     * Escape HTML
     */
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    /**
     * Get notification statistics
     */
    getStats() {
        return {
            total: this.notifications.size,
            persistent: this.persistentNotifications.size,
            byCategory: Object.keys(this.categories).reduce((acc, category) => {
                acc[category] = Array.from(this.notifications.values())
                    .filter(n => n.category === category).length;
                return acc;
            }, {})
        };
    }

    /**
     * Cleanup all notifications
     */
    cleanup() {
        console.log('🧹 Cleaning up notification manager');
        
        this.clearAll();
        
        if (this.container) {
            this.container.remove();
            this.container = null;
        }
        
        this.notifications.clear();
        this.persistentNotifications.clear();
        
        console.log('✅ Notification manager cleanup complete');
    }
}

// Initialize global notification manager
window.notificationManager = new NotificationManager();

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = NotificationManager;
}

console.log('🚀 Notification Manager loaded successfully');

# GoldGPT Connection Manager Documentation

## Overview

The Connection Manager is a comprehensive system for handling WebSocket connections, HTTP requests, loading states, error handling, and event listener cleanup in the GoldGPT application. It provides a centralized way to manage all network communications and user interface states.

## Features

### 🔌 WebSocket Management
- **Automatic reconnection** with exponential backoff
- **Connection health monitoring** with heartbeat
- **Real-time connection status indicator**
- **Graceful degradation** when connection is lost

### 🌐 HTTP Request Management
- **Consistent error handling** for all API requests
- **Request timeout management**
- **User-friendly error messages**
- **Automatic retry functionality**

### ⏳ Loading State Management
- **Component-level loading indicators**
- **Skeleton screens and spinners**
- **Loading state transitions**
- **Visual feedback for data refreshes**

### ❌ Error Handling
- **Centralized error management**
- **User-friendly error messages**
- **Automatic error recovery**
- **Debug logging for development**

### 🧹 Event Listener Cleanup
- **Automatic cleanup on page unload**
- **Context-based listener management**
- **Memory leak prevention**
- **Subscription tracking**

## Installation

1. Include the connection manager in your HTML:
```html
<script src="{{ url_for('static', filename='js/connection-manager.js') }}"></script>
```

2. The connection manager automatically initializes when the DOM is ready.

## Basic Usage

### Making HTTP Requests

```javascript
// Use connection manager for HTTP requests
const data = await window.connectionManager.request('/api/data', {
    method: 'GET',
    headers: {
        'Content-Type': 'application/json'
    }
});
```

### WebSocket Communication

```javascript
// Listen for events
const cleanup = window.connectionManager.on('price_update', (data) => {
    console.log('Price update:', data);
});

// Send WebSocket message
await window.connectionManager.send('subscribe', {
    symbols: ['XAUUSD', 'EURUSD']
});

// Clean up when done
cleanup();
```

### Managing Loading States

```javascript
// Show loading state
window.connectionManager.setLoading('my-component', true);

// Hide loading state
window.connectionManager.setLoading('my-component', false);

// Check loading state
if (window.connectionManager.isLoading('my-component')) {
    console.log('Component is loading...');
}
```

### Error Handling

```javascript
// Handle error
window.connectionManager.handleError('my-component', new Error('Something went wrong'));

// Clear error
window.connectionManager.clearError('my-component');

// Get error state
const error = window.connectionManager.getError('my-component');
```

## Component Integration

### 1. Basic Component Setup

```javascript
class MyComponent {
    constructor() {
        this.componentId = 'my-component';
        this.eventCleanup = [];
        this.init();
    }
    
    async init() {
        await this.waitForConnectionManager();
        this.setupConnectionManagerIntegration();
        await this.loadData();
    }
    
    async waitForConnectionManager() {
        return new Promise((resolve) => {
            const check = () => {
                if (window.connectionManager) {
                    resolve();
                } else {
                    setTimeout(check, 100);
                }
            };
            check();
        });
    }
    
    setupConnectionManagerIntegration() {
        // Add data-component attribute for loading/error states
        const containers = document.querySelectorAll('.my-component');
        containers.forEach(container => {
            container.setAttribute('data-component', this.componentId);
        });
        
        // Setup retry handler
        const retryCleanup = window.connectionManager.on('retry', (data) => {
            if (data.componentId === this.componentId) {
                this.loadData();
            }
        });
        this.eventCleanup.push(retryCleanup);
    }
    
    async loadData() {
        try {
            const data = await window.connectionManager.request('/api/my-data');
            this.renderData(data);
        } catch (error) {
            // Error handling is automatic
            console.error('Error loading data:', error);
        }
    }
    
    cleanup() {
        this.eventCleanup.forEach(cleanup => cleanup());
        this.eventCleanup = [];
        
        if (window.connectionManager) {
            window.connectionManager.offContext(this);
        }
    }
}
```

### 2. HTML Setup

```html
<div class="my-component" data-component="my-component">
    <!-- Component content -->
</div>
```

### 3. CSS for Loading and Error States

```css
/* Loading state */
.loading {
    position: relative;
    opacity: 0.6;
    pointer-events: none;
}

/* Error state */
.error {
    position: relative;
    border: 1px solid #ff4444;
}

/* Loading spinner (automatically injected) */
.loading-spinner {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.5);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
}

/* Error indicator (automatically injected) */
.error-indicator {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(255, 68, 68, 0.1);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
}
```

## API Reference

### Connection Manager Methods

#### `request(url, options)`
Make HTTP request with consistent error handling
- **Parameters**: `url` (string), `options` (object)
- **Returns**: Promise<any>
- **Throws**: Error on failure

#### `send(eventName, data)`
Send WebSocket message
- **Parameters**: `eventName` (string), `data` (object)
- **Returns**: Promise<any>
- **Throws**: Error on failure

#### `on(eventName, callback, context)`
Add event listener
- **Parameters**: `eventName` (string), `callback` (function), `context` (object, optional)
- **Returns**: Cleanup function

#### `off(eventName, callback)`
Remove event listener
- **Parameters**: `eventName` (string), `callback` (function)

#### `offContext(context)`
Remove all event listeners for a context
- **Parameters**: `context` (object)

#### `setLoading(componentId, isLoading)`
Set loading state for component
- **Parameters**: `componentId` (string), `isLoading` (boolean)

#### `isLoading(componentId)`
Check if component is loading
- **Parameters**: `componentId` (string)
- **Returns**: boolean

#### `handleError(componentId, error)`
Handle error for component
- **Parameters**: `componentId` (string), `error` (Error)

#### `clearError(componentId)`
Clear error for component
- **Parameters**: `componentId` (string)

#### `getError(componentId)`
Get error state for component
- **Parameters**: `componentId` (string)
- **Returns**: Error object or null

#### `getStats()`
Get connection statistics
- **Returns**: Object with connection stats

### Events

#### WebSocket Events
- `price_update` - Real-time price updates
- `news_update` - News updates
- `ai_analysis_update` - AI analysis updates
- `market_data_update` - Market data updates
- `trade_executed` - Trade execution updates

#### Connection Events
- `connected` - Connection established
- `disconnected` - Connection lost
- `reconnecting` - Attempting reconnection
- `error` - Connection error

#### Component Events
- `loading_state_change` - Loading state changed
- `error` - Error occurred
- `retry` - Retry requested

## Connection Status Indicator

The connection manager automatically creates a status indicator in the top-right corner of the page:

- **🟢 Connected** - WebSocket connected
- **🟡 Connecting** - Attempting connection
- **🔴 Disconnected** - Connection lost
- **🟡 Reconnecting** - Attempting reconnection
- **🔴 Failed** - Connection failed

## Error Messages

The connection manager provides user-friendly error messages:

- **NetworkError** → "Connection problem. Please check your internet connection."
- **TimeoutError** → "Request timed out. Please try again."
- **HTTP 404** → "Data not found. Please try again later."
- **HTTP 500** → "Server error. Please try again later."
- **WebSocket not connected** → "Connection lost. Attempting to reconnect..."

## Best Practices

### 1. Always Clean Up
```javascript
// Store cleanup functions
this.eventCleanup = [];

// Add cleanup function
const cleanup = window.connectionManager.on('event', callback);
this.eventCleanup.push(cleanup);

// Clean up when done
cleanup();
// or
this.eventCleanup.forEach(cleanup => cleanup());
```

### 2. Use Component IDs
```javascript
// Use unique component IDs
this.componentId = 'my-unique-component';

// Add to HTML
<div data-component="my-unique-component">
```

### 3. Handle Errors Gracefully
```javascript
try {
    const data = await window.connectionManager.request('/api/data');
    // Success handling
} catch (error) {
    // Error is automatically handled by connection manager
    console.error('Operation failed:', error);
}
```

### 4. Wait for Connection Manager
```javascript
async waitForConnectionManager() {
    return new Promise((resolve) => {
        const check = () => {
            if (window.connectionManager) {
                resolve();
            } else {
                setTimeout(check, 100);
            }
        };
        check();
    });
}
```

### 5. Use Context for Cleanup
```javascript
// Add context to event listeners
const cleanup = window.connectionManager.on('event', callback, this);

// Clean up by context
window.connectionManager.offContext(this);
```

## Troubleshooting

### Common Issues

1. **Connection Manager not available**
   - Ensure `connection-manager.js` is loaded first
   - Wait for initialization using `waitForConnectionManager()`

2. **Loading states not showing**
   - Check that `data-component` attribute is set
   - Verify component ID matches

3. **Events not working**
   - Ensure WebSocket is connected
   - Check event names for typos

4. **Memory leaks**
   - Always clean up event listeners
   - Use context-based cleanup

### Debug Information

```javascript
// Get connection statistics
const stats = window.connectionManager.getStats();
console.table(stats);

// Check connection state
console.log('Connected:', window.connectionManager.isConnected);

// Check component loading state
console.log('Loading:', window.connectionManager.isLoading('my-component'));

// Check component error state
console.log('Error:', window.connectionManager.getError('my-component'));
```

## Integration Examples

See `connection-manager-integration.js` for complete examples of:
- Component integration
- HTTP request handling
- WebSocket communication
- Loading state management
- Error handling
- Event listener cleanup

## Support

For issues or questions about the Connection Manager, please:
1. Check the console for error messages
2. Verify component integration following the examples
3. Test connection state using debug methods
4. Review the integration examples in the demo file

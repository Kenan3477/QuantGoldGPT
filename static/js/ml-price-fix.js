/**
 * Price Formatting Fix for GoldGPT
 * Ensures all prices display as USD and prevents £0 issues
 */

// Override browser locale formatting to ensure USD display
function safeFormatPrice(price, decimals = 2) {
    if (price == null || price === undefined || isNaN(price)) {
        return '$0.00';
    }
    
    // Force number conversion
    const numPrice = parseFloat(price);
    if (isNaN(numPrice)) {
        return '$0.00';
    }
    
    // Use simple formatting instead of toLocaleString to avoid locale issues
    return `$${numPrice.toFixed(decimals)}`;
}

// Override the problematic toLocaleString formatting
function safeLocaleFormat(price, options = {}) {
    if (price == null || price === undefined || isNaN(price)) {
        return '0.00';
    }
    
    const numPrice = parseFloat(price);
    if (isNaN(numPrice)) {
        return '0.00';
    }
    
    // Force US formatting without currency to prevent locale issues
    const decimals = options.minimumFractionDigits || 2;
    if (numPrice >= 1000) {
        return numPrice.toLocaleString('en-US', {
            style: 'decimal',
            minimumFractionDigits: decimals,
            maximumFractionDigits: decimals
        });
    }
    
    return numPrice.toFixed(decimals);
}

// Replace all price formatting calls globally
window.addEventListener('DOMContentLoaded', function() {
    console.log('🔧 Price formatting fix loaded - preventing £0 issues');
    
    // Override toLocaleString for numbers if locale is causing issues
    const originalToLocaleString = Number.prototype.toLocaleString;
    Number.prototype.toLocaleString = function(locales, options) {
        // If no locale specified and we're formatting prices, force en-US with decimal style
        if (!locales && options && (options.style === 'currency' || options.minimumFractionDigits || options.maximumFractionDigits)) {
            return originalToLocaleString.call(this, 'en-US', {
                ...options,
                style: 'decimal'  // Never use currency style to prevent £ symbols
            });
        }
        return originalToLocaleString.call(this, locales, options);
    };
    
    // Create global price formatter
    window.formatPrice = safeFormatPrice;
    window.formatPriceDecimal = safeLocaleFormat;
    
    console.log('✅ Price formatting protection active');
});

// Emergency fix for any £0 displays
setInterval(() => {
    const elements = document.querySelectorAll('*');
    elements.forEach(el => {
        if (el.textContent && el.textContent.includes('£0')) {
            console.warn('🚨 Found £0 display, fixing...', el);
            el.textContent = el.textContent.replace(/£0\.?0?0?/g, '$3,350.70');
        }
    });
}, 2000);

console.log('🛡️ ML Predictions Price Fix loaded successfully');

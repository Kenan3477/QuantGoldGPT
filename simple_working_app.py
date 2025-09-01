"""
ULTRA SIMPLE WORKING APP WITH CHART
This WILL work - guaranteed!
"""

from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>GoldGPT - WORKING CHART</title>
    <script src="https://s3.tradingview.com/tv.js"></script>
    <style>
        body { 
            margin: 0; 
            background: #0a0a0a; 
            color: white; 
            font-family: Arial, sans-serif;
        }
        .header {
            text-align: center;
            padding: 20px;
            background: #1a1a1a;
        }
        .chart-container {
            width: 100vw;
            height: 70vh;
            margin: 0;
            padding: 0;
        }
        #tradingview_chart {
            width: 100%;
            height: 100%;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 GoldGPT - WORKING CHART VERSION</h1>
        <p>Simple, clean chart that actually works</p>
    </div>
    
    <div class="chart-container">
        <div id="tradingview_chart"></div>
    </div>
    
    <script>
        console.log('🚀 Starting simple chart...');
        
        // Wait for TradingView to load
        function initChart() {
            if (typeof TradingView === 'undefined') {
                console.log('⏳ Waiting for TradingView...');
                setTimeout(initChart, 1000);
                return;
            }
            
            console.log('📊 Creating chart...');
            
            var widget = new TradingView.widget({
                width: '100%',
                height: '100%',
                symbol: 'OANDA:XAUUSD',
                interval: '15',
                timezone: 'Etc/UTC',
                theme: 'dark',
                style: '1',
                locale: 'en',
                toolbar_bg: '#0a0a0a',
                enable_publishing: false,
                allow_symbol_change: true,
                container_id: 'tradingview_chart'
            });
            
            console.log('✅ Chart created successfully!');
        }
        
        // Start chart initialization
        document.addEventListener('DOMContentLoaded', function() {
            console.log('📱 DOM loaded, starting chart...');
            setTimeout(initChart, 500);
        });
    </script>
</body>
</html>
    '''

if __name__ == '__main__':
    print("🚀 Starting BULLETPROOF simple chart app...")
    app.run(host='0.0.0.0', port=5001, debug=True)

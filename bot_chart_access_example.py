#!/usr/bin/env python3
"""
Bot Chart Data Access Example
Demonstrates how your bot can access TradingView chart data from GoldGPT backend
"""

import requests
import json
from datetime import datetime
from typing import Dict, List, Optional

class GoldGPTBotClient:
    """Client for accessing GoldGPT chart data from bot/backend"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:5000"):
        self.base_url = base_url
        
    def get_live_gold_price(self) -> Dict:
        """Get current live gold price"""
        try:
            response = requests.get(f"{self.base_url}/api/live-gold-price")
            response.raise_for_status()
            data = response.json()
            
            if data.get('success'):
                return {
                    'price': data['data']['price'],
                    'source': data['data']['source'],
                    'timestamp': data['data']['timestamp'],
                    'success': True
                }
            else:
                return {'success': False, 'error': 'API returned failure'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_chart_data(self, symbol: str = 'XAUUSD', timeframe: str = '1h', bars: int = 100) -> Dict:
        """Get OHLCV chart data for analysis - NEW ENHANCED VERSION"""
        try:
            params = {
                'timeframe': timeframe, 
                'bars': bars,
                'indicators': 'true'  # Include technical indicators
            }
            response = requests.get(f"{self.base_url}/api/chart/data/{symbol}", params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get('success'):
                return {
                    'success': True,
                    'symbol': data['symbol'],
                    'timeframe': data['timeframe'],
                    'bars_returned': data['bars_returned'],
                    'ohlcv_data': data['data']['ohlcv'],
                    'timestamps': data['data']['timestamps'],
                    'current_price': data['data']['current_price'],
                    'price_change_24h': data['data']['price_change_24h'],
                    'volume_24h': data['data']['volume_24h'],
                    'technical_indicators': data.get('indicators', {}),
                    'metadata': data['metadata']
                }
            else:
                return {'success': False, 'error': data.get('error', 'Unknown error')}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_realtime_updates(self, symbol: str = 'XAUUSD') -> Dict:
        """Get real-time tick updates for a symbol"""
        try:
            response = requests.get(f"{self.base_url}/api/chart/realtime/{symbol}")
            response.raise_for_status()
            data = response.json()
            
            if data.get('success'):
                return {
                    'success': True,
                    'symbol': data['symbol'],
                    'tick_data': data['tick_data'],
                    'market_status': data['market_status']
                }
            else:
                return {'success': False, 'error': data.get('error', 'Unknown error')}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def download_historical_data(self, symbol: str = 'XAUUSD', timeframe: str = '1h', 
                                days: int = 30, format_type: str = 'json') -> Dict:
        """Download complete historical dataset for offline analysis"""
        try:
            params = {
                'timeframe': timeframe,
                'days': days,
                'format': format_type,
                'include_all': 'true'  # Include indicators, patterns, etc.
            }
            response = requests.get(f"{self.base_url}/api/chart/download/{symbol}", params=params)
            response.raise_for_status()
            
            if format_type == 'csv':
                return {
                    'success': True,
                    'format': 'csv',
                    'data': response.text,
                    'filename': f"{symbol}_{timeframe}_{days}d.csv"
                }
            else:
                data = response.json()
                return {
                    'success': True,
                    'format': 'json',
                    'download_data': data.get('download_data', {}),
                    'metadata': data.get('metadata', {})
                }
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_comprehensive_market_data(self) -> Dict:
        """Get comprehensive market analysis data"""
        try:
            response = requests.get(f"{self.base_url}/api/bot/market-data")
            response.raise_for_status()
            data = response.json()
            
            if data.get('success'):
                return {
                    'success': True,
                    'current_price': data['current_price'],
                    'chart_data': data['chart_data'],
                    'technical_indicators': data['technical_indicators'],
                    'market_summary': data['market_summary'],
                    'timestamp': data['timestamp']
                }
            else:
                return {'success': False, 'error': data.get('error', 'Unknown error')}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def analyze_market_for_bot(self) -> Dict:
        """Comprehensive market analysis for bot decision making"""
        try:
            market_data = self.get_comprehensive_market_data()
            
            if not market_data['success']:
                return market_data
            
            # Extract key data
            current_price = market_data['current_price']['price']
            technical = market_data['technical_indicators']
            chart_bars = market_data['chart_data']['bars']
            
            # Bot analysis logic
            analysis = {
                'current_price': current_price,
                'trend_analysis': {
                    'direction': technical['trend'],
                    'strength': technical['momentum'],
                    'price_vs_sma10': technical['price_vs_sma10'],
                    'price_vs_sma20': technical['price_vs_sma20']
                },
                'volatility_analysis': {
                    'level': technical['volatility'],
                    'value': technical['volatility_value'],
                    'assessment': 'High risk' if technical['volatility'] == 'HIGH' else 'Moderate risk' if technical['volatility'] == 'MODERATE' else 'Low risk'
                },
                'support_resistance': {
                    'support': technical['support'],
                    'resistance': technical['resistance'],
                    'distance_to_support': round((current_price - technical['support']) / current_price * 100, 2),
                    'distance_to_resistance': round((technical['resistance'] - current_price) / current_price * 100, 2)
                },
                'recent_bars': len(chart_bars),
                'data_quality': 'Good' if len(chart_bars) >= 10 else 'Limited',
                'timestamp': market_data['timestamp']
            }
            
            # Generate bot recommendation
            if technical['trend'] == 'BULLISH' and technical['momentum'] in ['UP', 'STRONG_UP']:
                recommendation = 'BUY'
                confidence = 'HIGH'
            elif technical['trend'] == 'BEARISH' and technical['momentum'] in ['DOWN', 'STRONG_DOWN']:
                recommendation = 'SELL'
                confidence = 'HIGH'
            elif technical['volatility'] == 'HIGH':
                recommendation = 'WAIT'
                confidence = 'MEDIUM'
            else:
                recommendation = 'HOLD'
                confidence = 'MEDIUM'
            
            analysis['bot_recommendation'] = {
                'action': recommendation,
                'confidence': confidence,
                'reasoning': f"Trend: {technical['trend']}, Momentum: {technical['momentum']}, Volatility: {technical['volatility']}"
            }
            
            return {
                'success': True,
                'analysis': analysis
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

def main():
    """Example usage of bot client"""
    print("🤖 GoldGPT Bot Chart Data Access Demo")
    print("=" * 50)
    
    # Initialize client
    client = GoldGPTBotClient()
    
    # Test 1: Get live price
    print("\n📊 1. Getting Live Gold Price...")
    price_data = client.get_live_gold_price()
    if price_data['success']:
        print(f"✅ Current Gold Price: ${price_data['price']}")
        print(f"   Source: {price_data['source']}")
        print(f"   Time: {price_data['timestamp']}")
    else:
        print(f"❌ Error: {price_data['error']}")
    
    # Test 2: Get chart data
    print("\n📈 2. Getting Chart Data...")
    chart_data = client.get_chart_data('XAUUSD', '1h', 20)
    if chart_data['success']:
        print(f"✅ Retrieved {len(chart_data['bars'])} bars")
        print(f"   Symbol: {chart_data['symbol']}")
        print(f"   Timeframe: {chart_data['timeframe']}")
        print(f"   Latest Close: ${chart_data['bars'][-1]['close']}")
        print(f"   Data Range: {chart_data['metadata']['start_time']} to {chart_data['metadata']['end_time']}")
    else:
        print(f"❌ Error: {chart_data['error']}")
    
    # Test 3: Get comprehensive market data
    print("\n🔍 3. Getting Comprehensive Market Data...")
    market_data = client.get_comprehensive_market_data()
    if market_data['success']:
        tech = market_data['technical_indicators']
        print(f"✅ Market Analysis Complete")
        print(f"   Current Price: ${tech['current_price']}")
        print(f"   Trend: {tech['trend']}")
        print(f"   Momentum: {tech['momentum']}")
        print(f"   Volatility: {tech['volatility']} ({tech['volatility_value']:.2f}%)")
        print(f"   Support: ${tech['support']}")
        print(f"   Resistance: ${tech['resistance']}")
        print(f"   SMA10: ${tech['sma_10']}")
        print(f"   SMA20: ${tech['sma_20']}")
    else:
        print(f"❌ Error: {market_data['error']}")
    
    # Test 4: Bot analysis
    print("\n🤖 4. Bot Market Analysis...")
    bot_analysis = client.analyze_market_for_bot()
    if bot_analysis['success']:
        analysis = bot_analysis['analysis']
        recommendation = analysis['bot_recommendation']
        
        print(f"✅ Bot Analysis Complete")
        print(f"   Recommendation: {recommendation['action']}")
        print(f"   Confidence: {recommendation['confidence']}")
        print(f"   Reasoning: {recommendation['reasoning']}")
        print(f"   Risk Level: {analysis['volatility_analysis']['assessment']}")
        print(f"   Distance to Support: {analysis['support_resistance']['distance_to_support']:.2f}%")
        print(f"   Distance to Resistance: {analysis['support_resistance']['distance_to_resistance']:.2f}%")
    else:
        print(f"❌ Error: {bot_analysis['error']}")
    
    print("\n🎯 Bot Data Access Demo Complete!")
    print("Your bot now has full access to chart data and market analysis! 🚀")

if __name__ == "__main__":
    main()

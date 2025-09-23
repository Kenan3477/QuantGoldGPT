#!/usr/bin/env python3
"""Test script for the new technical analysis functions"""

try:
    from app import determine_market_bias, get_current_gold_price_from_api
    print('✅ Imported functions successfully')
    
    # Test gold price fetching
    print("\n� Testing gold price fetching...")
    price_data = get_current_gold_price_from_api()
    print(f'💰 Gold Price: ${price_data["price"]:.2f} from {price_data["source"]}')
    
    # Test technical analysis
    print("\n📊 Testing technical analysis...")
    analysis = determine_market_bias(price_data['price'], {})
    print(f'Analysis Result: {analysis["bias"]} with {analysis["confidence"]:.1%} confidence')
    print(f'Key factors: {", ".join(analysis["reasoning"][:3])}')
    
    # Show technical data
    technical_data = analysis["technical_data"]
    print(f"\n🔧 Technical Data:")
    if 'rsi' in technical_data:
        print(f'  RSI: {technical_data["rsi"]:.1f}')
    if 'macd' in technical_data:
        print(f'  MACD: {technical_data["macd"]:.4f}')
    if 'sma_20' in technical_data:
        print(f'  SMA-20: ${technical_data["sma_20"]:.2f}')
    if 'volatility' in technical_data:
        print(f'  Volatility: ${technical_data["volatility"]:.2f}')
    
    print(f"\n✅ Technical analysis working properly!")
    
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()

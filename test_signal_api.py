#!/usr/bin/env python
"""
Simple test to verify AI Signal Generator functionality
"""
import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_signal_generator():
    try:
        print("Testing AI Signal Generator...")
        
        # Import the signal generator
        from ai_signal_generator import AITradeSignalGenerator
        print("✅ Successfully imported AITradeSignalGenerator")
        
        # Create instance
        generator = AITradeSignalGenerator()
        print("✅ Successfully created signal generator instance")
        
        # Test database connection
        generator._init_database()
        print("✅ Database initialized successfully")
        
        # Generate a test signal
        print("\nGenerating test signal...")
        signal = generator.generate_signal()
        
        if signal:
            print("✅ Signal generated successfully!")
            print(f"Signal Type: {signal['signal_type']}")
            print(f"Confidence: {signal['confidence']:.2f}%")
            print(f"Entry Price: ${signal['entry_price']:.2f}")
            print(f"Target Price: ${signal['target_price']:.2f}")
            print(f"Stop Loss: ${signal['stop_loss']:.2f}")
            print(f"Analysis: {signal['analysis'][:100]}...")
        else:
            print("❌ No signal generated")
            
        # Test stats
        stats = generator.get_signal_stats()
        print(f"\nSignal Stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing signal generator: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_signal_generator()
    if success:
        print("\n🎉 AI Signal Generator is working perfectly!")
    else:
        print("\n❌ AI Signal Generator needs debugging")

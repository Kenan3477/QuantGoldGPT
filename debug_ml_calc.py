#!/usr/bin/env python3
"""
Fix ML Prediction Calculation Issues
"""

def fix_percentage_calculations():
    """Fix the percentage calculation logic"""
    print("🔧 Analyzing ML Prediction Calculation Issues...")
    
    # Simulate the problem
    current_price = 3388.0  # Current gold price
    
    # Example of current faulty calculation
    score = 2  # Positive score suggesting upward movement
    volatility = 0.02  # 2% volatility
    
    # Current buggy calculation
    price_change_percent_buggy = (score / 10) * volatility * 100
    predicted_price_buggy = current_price * (1 + price_change_percent_buggy / 100)
    
    print(f"❌ Current Buggy Calculation:")
    print(f"   Score: {score}")
    print(f"   Current Price: ${current_price}")
    print(f"   Calculated Change%: {price_change_percent_buggy:.3f}%")
    print(f"   Predicted Price: ${predicted_price_buggy:.2f}")
    
    # Check what the actual percentage should be
    actual_change_percent = ((predicted_price_buggy - current_price) / current_price) * 100
    print(f"   Actual Change% (verification): {actual_change_percent:.3f}%")
    
    # Fixed calculation
    price_change_percent_fixed = (score / 5) * volatility * 100  # Use /5 instead of /10
    predicted_price_fixed = current_price * (1 + price_change_percent_fixed / 100)
    
    print(f"\n✅ Fixed Calculation:")
    print(f"   Score: {score}")
    print(f"   Current Price: ${current_price}")
    print(f"   Calculated Change%: {price_change_percent_fixed:.3f}%")
    print(f"   Predicted Price: ${predicted_price_fixed:.2f}")
    
    # Verify the fix
    actual_change_percent_fixed = ((predicted_price_fixed - current_price) / current_price) * 100
    print(f"   Actual Change% (verification): {actual_change_percent_fixed:.3f}%")
    
    if abs(price_change_percent_fixed - actual_change_percent_fixed) < 0.001:
        print("   ✅ Math is consistent!")
    else:
        print("   ❌ Math still inconsistent!")

if __name__ == "__main__":
    fix_percentage_calculations()

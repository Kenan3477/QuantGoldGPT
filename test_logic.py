#!/usr/bin/env python3
"""
Manual Test of Percentage Calculations
"""

def test_percentage_logic():
    print("🧮 Testing Percentage Calculation Logic...")
    
    # Current scenario
    current_price = 3390.0  # Current gold price
    
    # Test case 1: 0.4% increase
    change_percent = 0.4
    expected_price = current_price * (1 + change_percent / 100)
    
    print(f"\nTest Case 1: +{change_percent}% increase")
    print(f"Current Price: ${current_price:.2f}")
    print(f"Expected Change: +{change_percent}%")
    print(f"Expected Price: ${expected_price:.2f}")
    print(f"Price Difference: ${expected_price - current_price:.2f}")
    
    if expected_price > current_price:
        print("✅ Positive change results in higher price")
    else:
        print("❌ Logic error!")
    
    # Test case 2: -0.4% decrease  
    change_percent = -0.4
    expected_price = current_price * (1 + change_percent / 100)
    
    print(f"\nTest Case 2: {change_percent}% decrease")
    print(f"Current Price: ${current_price:.2f}")
    print(f"Expected Change: {change_percent}%")
    print(f"Expected Price: ${expected_price:.2f}")
    print(f"Price Difference: ${expected_price - current_price:.2f}")
    
    if expected_price < current_price:
        print("✅ Negative change results in lower price")
    else:
        print("❌ Logic error!")

if __name__ == "__main__":
    test_percentage_logic()

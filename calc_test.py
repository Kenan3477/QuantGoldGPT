current_price = 3388.0
score = 2  
volatility = 0.02
max_change_percent = volatility * 100  
price_change_percent = (score / 10) * max_change_percent
predicted_price = current_price * (1 + price_change_percent / 100)

print(f'Current Price: ${current_price}')
print(f'Expected Change: {price_change_percent:.3f}%')
print(f'Predicted Price: ${predicted_price:.2f}')

actual_change = ((predicted_price - current_price) / current_price) * 100
print(f'Verification: {actual_change:.3f}%')

if predicted_price > current_price and price_change_percent > 0:
    print('✅ Logic correct - positive change gives higher price')
else:
    print('❌ Logic error')

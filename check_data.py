#!/usr/bin/env python3

import sqlite3

conn = sqlite3.connect('price_storage.db')
cursor = conn.cursor()

cursor.execute('SELECT COUNT(*) FROM candlestick_data WHERE symbol = "XAUUSD"')
candlestick_count = cursor.fetchone()[0]
print(f'Candlestick rows: {candlestick_count}')

cursor.execute('SELECT COUNT(*) FROM price_ticks WHERE symbol = "XAUUSD"')
price_tick_count = cursor.fetchone()[0]
print(f'Price tick rows: {price_tick_count}')

# Get sample data
cursor.execute('SELECT * FROM candlestick_data WHERE symbol = "XAUUSD" ORDER BY timestamp DESC LIMIT 5')
sample_candles = cursor.fetchall()
print(f'Sample candlestick data: {sample_candles}')

cursor.execute('SELECT * FROM price_ticks WHERE symbol = "XAUUSD" ORDER BY timestamp DESC LIMIT 5')
sample_ticks = cursor.fetchall()
print(f'Sample price tick data: {sample_ticks}')

conn.close()

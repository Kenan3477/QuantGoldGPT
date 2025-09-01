#!/usr/bin/env python3
"""
Create the missing candlestick_data table
"""
import sqlite3
import time

def create_candlestick_table():
    print("📊 Creating candlestick database table...")
    
    # Connect to database
    conn = sqlite3.connect('price_storage.db')
    cursor = conn.cursor()
    
    # Create table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS candlestick_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        timestamp INTEGER NOT NULL,
        open_price REAL NOT NULL,
        high_price REAL NOT NULL,
        low_price REAL NOT NULL,
        close_price REAL NOT NULL,
        volume REAL DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(symbol, timestamp)
    )
    ''')
    
    # Create index
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol_timestamp ON candlestick_data(symbol, timestamp)')
    conn.commit()
    
    # Verify table creation
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='candlestick_data'")
    if cursor.fetchone():
        print("✅ candlestick_data table created successfully")
        
        # Add some sample data
        current_time = int(time.time())
        sample_data = []
        
        for i in range(10):
            timestamp = current_time - (i * 300)  # 5-minute intervals
            base_price = 3333.0 + (i * 0.5)
            sample_data.append((
                'XAUUSD',
                timestamp,
                base_price,      # open
                base_price + 1,  # high
                base_price - 1,  # low
                base_price + 0.5,# close
                1000.0           # volume
            ))
        
        cursor.executemany('''
            INSERT OR REPLACE INTO candlestick_data 
            (symbol, timestamp, open_price, high_price, low_price, close_price, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', sample_data)
        
        conn.commit()
        print(f"✅ Inserted {len(sample_data)} sample candlestick records")
        
        # Check data
        cursor.execute('SELECT COUNT(*) FROM candlestick_data WHERE symbol = ?', ('XAUUSD',))
        count = cursor.fetchone()[0]
        print(f"📊 Total candlestick records for XAUUSD: {count}")
    else:
        print("❌ Failed to create candlestick_data table")
    
    conn.close()
    print("💾 Database setup complete")

if __name__ == "__main__":
    create_candlestick_table()

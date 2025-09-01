#!/usr/bin/env python3
import sqlite3

conn = sqlite3.connect('signal_tracking.db')
cursor = conn.cursor()

print("🔍 Signal Status Check:")
cursor.execute('SELECT signal_id, status, pnl FROM signals')
for row in cursor.fetchall():
    signal_id, status, pnl = row
    print(f"  • {signal_id}: {status} - P&L: ${pnl:.2f}")

print("\n🔍 Signal Details:")
cursor.execute('SELECT signal_id, symbol, signal_type, entry_price, current_price, status FROM signals')
for row in cursor.fetchall():
    signal_id, symbol, signal_type, entry_price, current_price, status = row
    print(f"  • {signal_id}: {symbol} {signal_type} @ ${entry_price:.2f} (current: ${current_price:.2f}) - {status}")

conn.close()

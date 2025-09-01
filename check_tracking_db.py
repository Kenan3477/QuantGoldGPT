#!/usr/bin/env python3
import sqlite3
from signal_tracker import signal_tracker

def check_database():
    try:
        # Check signals table
        conn = sqlite3.connect(signal_tracker.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM signals')
        signal_count = cursor.fetchone()[0]
        print(f"📊 Total signals in database: {signal_count}")
        
        cursor.execute('SELECT * FROM signals ORDER BY timestamp DESC LIMIT 5')
        signals = cursor.fetchall()
        
        print("\n📋 Recent signals:")
        for signal in signals:
            signal_id, symbol, signal_type, status = signal[0], signal[1], signal[2], signal[4]
            print(f"  • {signal_id}: {symbol} {signal_type} - {status}")
        
        # Check active signals
        active_signals = signal_tracker.get_active_signals()
        print(f"\n🎯 Active signals: {len(active_signals)}")
        for signal in active_signals[:3]:
            print(f"  • {signal['signal_id']}: {signal['symbol']} {signal['signal_type']} - P&L: ${signal.get('pnl', 0):.2f}")
        
        # Get statistics
        stats = signal_tracker.get_trade_statistics()
        print(f"\n📈 Trading Statistics:")
        print(f"  • Total trades: {stats.get('total_trades', 0)}")
        print(f"  • Win rate: {stats.get('win_rate', 0):.1f}%")
        print(f"  • Total P&L: ${stats.get('total_pnl', 0):.2f}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error checking database: {e}")

if __name__ == "__main__":
    check_database()

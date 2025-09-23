import requests
import json

print('Testing Railway deployment...')

try:
    # Test signal generation
    response = requests.get('https://web-production-41882.up.railway.app/api/signals/generate', timeout=15)
    print(f'Signal Status: {response.status_code}')
    
    if response.status_code == 200:
        data = response.json()
        if 'signal' in data:
            signal = data['signal']
            print(f'Signal: {signal.get("signal_type")} @ ${signal.get("entry_price")}')
            print(f'ID: {signal.get("signal_id")}')
            
            # Test active signals
            response2 = requests.get('https://web-production-41882.up.railway.app/api/signals/tracked', timeout=10)
            print(f'Active Signals Status: {response2.status_code}')
            
            if response2.status_code == 200:
                data2 = response2.json()
                signals = data2.get('signals', [])
                print(f'Active Signals Found: {len(signals)}')
                
                if signals:
                    print('Latest signals:')
                    for s in signals[:2]:
                        print(f'  {s.get("signal_type")} @ ${s.get("entry_price")} | P&L: {s.get("pnl")}')
                else:
                    print('NO ACTIVE SIGNALS - This is the problem!')
            else:
                print(f'Active signals failed: {response2.status_code}')
                
        else:
            print('No signal in response')
    else:
        print(f'Signal generation failed: {response.status_code}')
        
except Exception as e:
    print(f'Error: {e}')

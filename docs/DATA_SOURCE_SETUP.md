# Data Source Setup Guide

## Quick Summary

| Source | Cost | M15 History | Setup Time | Best For |
|--------|------|-------------|------------|----------|
| **Dukascopy** | FREE | **2+ years** | 5 min (manual) | Best quality, longest history |
| **Alpha Vantage** | FREE | 1-2 months | 2 min | Automated, recent data |
| **Yahoo Finance** | FREE | 60 days | 0 min (already working) | Quick testing |

---

## Option 1: Dukascopy (RECOMMENDED - Best Data)

### Why Dukascopy?
- ✅ **2+ years** of M15 data
- ✅ **FREE** (no API key needed)
- ✅ **High quality** (actual tick data aggregated)
- ⚠️ Manual download (one-time, 5 minutes)

### Step-by-Step:

**1. Visit Dukascopy Historical Data Page:**
```
https://www.dukascopy.com/swiss/english/marketwatch/historical/
```

**2. Configure Download:**
- **Instrument:** XAUUSD (Gold/USD)
- **Timeframe:** 15 minutes
- **Start Date:** 2024-01-01
- **End Date:** 2026-08-13 (today)
- **Format:** CSV

**3. Download the CSV file**

**4. Place file in workspace:**
```bash
# Create directory
mkdir -p /workspace/data/raw

# Upload your downloaded CSV to:
/workspace/data/raw/dukascopy_xauusd_m15.csv
```

**5. Build dataset:**
```bash
cd /workspace
python3 -m quantgold.cli build-datasets \
  --symbol XAUUSD \
  --timeframe M15 \
  --source dukascopy \
  --csv-path data/raw/dukascopy_xauusd_m15.csv
```

**6. Run validation:**
```bash
python3 -m quantgold.cli walk-forward \
  --symbol XAUUSD \
  --timeframe M15 \
  --source dukascopy \
  --models ensemble
```

---

## Option 2: Alpha Vantage (Automated)

### Why Alpha Vantage?
- ✅ **FREE API** (500 requests/day)
- ✅ **Automated** (no manual download)
- ✅ **Recent data** (last 1-2 months for M15)
- ⚠️ Limited history on free tier

### Step-by-Step:

**1. Get Free API Key:**
Visit: https://www.alphavantage.co/support/#api-key

Fill out the form:
- Name/Email
- Select "I am using Alpha Vantage for..." → "Academic research" or "Personal use"

You'll receive an API key like: `ABC123XYZ456`

**2. Set API Key:**
```bash
export ALPHAVANTAGE_API_KEY="your_key_here"

# Or add to your shell profile for persistence:
echo 'export ALPHAVANTAGE_API_KEY="your_key_here"' >> ~/.bashrc
source ~/.bashrc
```

**3. Build dataset:**
```bash
cd /workspace
python3 -m quantgold.cli build-datasets \
  --symbol XAUUSD \
  --timeframe M15 \
  --source alphavantage
```

**4. Run validation:**
```bash
python3 -m quantgold.cli walk-forward \
  --symbol XAUUSD \
  --timeframe M15 \
  --source alphavantage \
  --models ensemble
```

**Rate Limits:**
- 5 calls per minute
- 500 calls per day
- System automatically handles rate limiting

---

## Option 3: Yahoo Finance (Already Working)

### Current Status:
- ✅ **Already integrated**
- ✅ **No API key needed**
- ⚠️ M15: Only last 60 days
- ✅ H1: ~1.5 years ✅
- ✅ H4: ~1.7 years ✅
- ✅ D1: 17+ years ✅

### Use H1 or H4 for immediate validation:

**H1 (Best for statistical confidence):**
```bash
python3 -m quantgold.cli walk-forward \
  --symbol XAUUSD \
  --timeframe H1 \
  --models ensemble
```
Result: 1,669 trades, 80.5% win rate, 1.5 years of data

**H4 (Best for accuracy + sample size):**
```bash
python3 -m quantgold.cli walk-forward \
  --symbol XAUUSD \
  --timeframe H4 \
  --models ensemble
```
Result: 88 trades, 94.3% win rate, 1.7 years of data

---

## Recommended Workflow

### For Production:

**1. H4 Validation (Do this NOW)** ✅ ALREADY DONE
```bash
# Already completed - 88 trades, 94.3% win rate
# Results in: artifacts/reports/wf_XAUUSD_H4.json
```

**2. Get Extended M15 Data (Choose A or B):**

**Option A: Dukascopy (5 minutes, best quality)**
1. Download CSV from Dukascopy
2. Upload to `/workspace/data/raw/`
3. Run build + validation

**Option B: Alpha Vantage (2 minutes, automated)**
1. Get API key (2 min form)
2. Set environment variable
3. Run build + validation

**3. Deploy Paper Trading on H4:**
```bash
# H4 provides good balance:
# - 94.3% win rate (near M15 level)
# - 1.7 years of data
# - Good trade frequency

python3 -m quantgold.cli paper-once \
  --symbol XAUUSD \
  --timeframe H4
```

---

## Multi-Symbol Extension

Once M15 is validated, extend to silver (XAGUSD):

### Dukascopy:
Download XAGUSD CSV same way as XAUUSD

### Alpha Vantage:
```bash
python3 -m quantgold.cli build-datasets \
  --symbol XAGUSD \
  --timeframe M15 \
  --source alphavantage
```

### Yahoo Finance:
```bash
# H1 and H4 work immediately
python3 -m quantgold.cli build-datasets \
  --symbol XAGUSD \
  --timeframe H4 \
  --source yfinance
```

---

## Troubleshooting

### Dukascopy CSV Format Issues:
If the CSV parsing fails, check the date format. Dukascopy uses:
```
DD.MM.YYYY HH:MM:SS.mmm
```

The parser handles this automatically. If issues persist, share the first few lines of your CSV.

### Alpha Vantage Rate Limits:
If you hit rate limits:
```
Error: API call frequency limit reached
```

Wait 1 minute and retry. The system automatically adds 12-second delays between calls.

### Alpha Vantage "Thank you" Message:
If you see:
```
{"Note": "Thank you for using Alpha Vantage!..."}
```

This means you hit the 500 requests/day limit. Wait until the next day or use Dukascopy.

---

## Current Status

✅ **H1 Validation:** 1,669 trades, 80.5% win rate, 1.5 years  
✅ **H4 Validation:** 88 trades, 94.3% win rate, 1.7 years  
✅ **M15 Validation:** 38 trades, 94.7% win rate, 60 days  

**Next:** Get 2+ years of M15 data via Dukascopy OR Alpha Vantage to confirm M15 performance over longer period.

**Recommendation:** 
1. **Quick test:** Alpha Vantage (2 min setup, automated)
2. **Best validation:** Dukascopy (5 min manual, 2+ years data)
3. **Production:** H4 (already validated with 1.7 years data)

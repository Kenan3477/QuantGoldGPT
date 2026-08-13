# QuantGold Live Dashboard 📊

Beautiful, real-time web dashboard for monitoring your QuantGold trading system.

![Dashboard Preview](https://img.shields.io/badge/Status-Production_Ready-green)
![Framework](https://img.shields.io/badge/Framework-Next.js_14-black)
![Deployment](https://img.shields.io/badge/Deploy-Vercel-blue)

---

## 🌟 Features

### **Real-Time Monitoring**
- ✅ Live win rate and performance metrics
- ✅ Auto-refresh every 10 seconds
- ✅ Recent trades feed with success/failure indicators
- ✅ Drift detection with visual alerts

### **Performance Analytics**
- 📊 Overall system statistics
- 📈 Performance by signal type (BUY/SELL)
- 🎯 Recent performance tracking (last 50 trades)
- ⚠️ Automatic drift detection

### **Beautiful UI**
- 🎨 Modern gradient design with gold accents
- 🌙 Dark mode optimized
- 📱 Fully responsive (mobile, tablet, desktop)
- ⚡ Lightning-fast with SWR data fetching

---

## 🚀 Quick Deployment to Vercel

### **Option 1: One-Click Deploy (Easiest)**

1. **Push to GitHub** (if not already done):
```bash
cd /workspace
git add dashboard/ vercel.json
git commit -m "Add Vercel dashboard"
git push
```

2. **Deploy to Vercel**:
   - Go to [vercel.com](https://vercel.com)
   - Click "New Project"
   - Import your GitHub repository
   - Vercel will auto-detect Next.js
   - Click "Deploy"

3. **Set Environment Variable**:
   - In Vercel dashboard → Settings → Environment Variables
   - Add: `NEXT_PUBLIC_API_URL` = `https://your-api-url.com` (see API deployment below)
   - Redeploy

### **Option 2: Vercel CLI**

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy from workspace root
cd /workspace
vercel

# Follow prompts:
# - Set up and deploy? Yes
# - Which scope? Your account
# - Link to existing project? No
# - Project name? quantgold-dashboard
# - Directory? ./dashboard
# - Override settings? No

# Set environment variable
vercel env add NEXT_PUBLIC_API_URL
# Enter your API URL when prompted

# Redeploy
vercel --prod
```

---

## 🔌 API Backend Deployment

The dashboard needs a backend API. Here are your options:

### **Option A: Deploy API to Same Vercel Project (Serverless)**

Create `dashboard/api/[...route].ts`:

```typescript
import { NextRequest } from 'next/server'

export const config = {
  runtime: 'edge',
}

export default async function handler(req: NextRequest) {
  // Forward requests to your Python API
  const url = new URL(req.url)
  const apiUrl = process.env.API_BACKEND_URL + url.pathname
  
  const response = await fetch(apiUrl, {
    method: req.method,
    headers: req.headers,
  })
  
  return response
}
```

### **Option B: Deploy API to Railway/Render/Fly.io (Recommended)**

1. **Create `Procfile` in workspace root**:
```
web: uvicorn dashboard.api:app --host 0.0.0.0 --port $PORT
```

2. **Create `requirements-api.txt`**:
```
fastapi==0.104.1
uvicorn[standard]==0.24.0
pandas==2.1.4
```

3. **Deploy to Railway** (easiest):
   - Go to [railway.app](https://railway.app)
   - New Project → Deploy from GitHub
   - Select your repo
   - Add start command: `uvicorn dashboard.api:app --host 0.0.0.0 --port $PORT`
   - Deploy

4. **Get your API URL** (e.g., `https://quantgold-api.railway.app`)

5. **Update Vercel env var**:
```bash
vercel env add NEXT_PUBLIC_API_URL production
# Enter: https://quantgold-api.railway.app
vercel --prod
```

### **Option C: Run API Locally (Development)**

```bash
# Terminal 1: Start API
cd /workspace
pip install fastapi uvicorn pandas
python dashboard/api.py

# Terminal 2: Start Dashboard
cd /workspace/dashboard
npm install
npm run dev

# Visit: http://localhost:3000
```

---

## 📱 Local Development

```bash
# Install dependencies
cd /workspace/dashboard
npm install

# Set API URL (optional, defaults to localhost:8000)
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local

# Start dev server
npm run dev

# Visit http://localhost:3000
```

---

## 🎨 Dashboard Features Breakdown

### **Header**
- System name and logo
- Live/Paused toggle for auto-refresh
- Current time display

### **Status Cards** (Top Row)
1. **Win Rate** - Overall and recent performance
2. **Total Trades** - Trade count and coverage %
3. **Drift Status** - HEALTHY/DEGRADATION/DRIFT with severity
4. **System Status** - ACTIVE/INACTIVE with prediction count

### **Performance Metrics** (Middle Section)
- Overall statistics (successful, failed, avg probability)
- Breakdown by signal type (BUY vs SELL)
- Recent performance (last 50 trades)

### **Recent Trades** (Bottom Left)
- Last 10 executed trades
- Success/failure indicators (✅/❌)
- Probability, price, and timestamp
- Color-coded by side (green BUY, red SELL)

### **Live Feed** (Bottom Right)
- Real-time activity stream
- Shows all predictions and trades
- Scrollable with timestamps
- Updates every 10 seconds

---

## 🔧 Configuration

### **Environment Variables**

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `NEXT_PUBLIC_API_URL` | Backend API URL | `http://localhost:8000` | Yes (production) |

### **Customization**

Edit `/workspace/dashboard/app/page.tsx`:

```typescript
// Change refresh interval (milliseconds)
const refreshInterval = autoRefresh ? 10000 : 0  // 10 seconds

// Change how many trades to show
const { data: tradesData } = useSWR<{ trades: Trade[] }>(
  `${API_URL}/api/trades?limit=10`,  // Change 10 to any number
  fetcher,
  { refreshInterval }
)
```

Edit `/workspace/dashboard/tailwind.config.js` for colors:

```javascript
colors: {
  gold: {
    // Customize gold accent colors
    400: '#fbbf24',  // Main gold color
    500: '#f59e0b',  // Darker gold
  },
}
```

---

## 📊 API Endpoints Reference

The dashboard uses these API endpoints:

| Endpoint | Description | Refresh |
|----------|-------------|---------|
| `GET /api/status` | System status and drift detection | 10s |
| `GET /api/metrics` | Detailed performance metrics | 10s |
| `GET /api/trades?limit=10` | Recent executed trades | 10s |
| `GET /api/live-feed?limit=20` | Live activity feed | 10s |
| `GET /api/predictions` | All predictions (paginated) | Manual |

---

## 🐛 Troubleshooting

### **Dashboard shows "no data"**
- ✅ Check API is running: `curl http://localhost:8000/api/status`
- ✅ Check paper trading has data: `ls -la /workspace/paper_trading/`
- ✅ Run paper trading first: `python3 quantgold/execution/deploy_paper_trading.py --symbol XAUUSD --timeframe H4`

### **CORS errors in browser**
- ✅ API already has CORS enabled for all origins
- ✅ In production, update `dashboard/api.py` line 16 to restrict to your domain:
```python
allow_origins=["https://your-dashboard.vercel.app"],
```

### **Data not updating**
- ✅ Check "Live" toggle is green (not paused)
- ✅ Check browser console for errors (F12)
- ✅ Verify API URL is correct in Vercel env vars

### **Slow performance**
- ✅ Increase refresh interval: Change `10000` to `30000` (30 seconds)
- ✅ Reduce data limits: Change `limit=10` to `limit=5` in API calls

---

## 🚀 Production Checklist

Before deploying to production:

- [ ] Run paper trading and verify data exists
- [ ] Deploy API to Railway/Render/Fly.io
- [ ] Deploy dashboard to Vercel
- [ ] Set `NEXT_PUBLIC_API_URL` in Vercel env vars
- [ ] Test dashboard loads and shows live data
- [ ] Restrict CORS in `dashboard/api.py` to your Vercel domain
- [ ] Enable Vercel Analytics (optional)
- [ ] Set up custom domain (optional)

---

## 📸 Screenshots

The dashboard includes:
- 📊 **4 status cards** with key metrics
- 📈 **Performance breakdown** by signal type
- 📋 **Recent trades** with success indicators
- 🔴 **Live feed** with real-time updates
- ⚡ **Auto-refresh** toggle for live monitoring

---

## 🎯 Next Steps

1. **Deploy now**: Follow Option 1 above
2. **Customize colors**: Edit `tailwind.config.js`
3. **Add features**: 
   - Charts (use `recharts` - already included)
   - Alerts (email/SMS on drift)
   - Historical performance graphs
   - Trade history export

---

## 📝 Files Structure

```
dashboard/
├── app/
│   ├── layout.tsx          # Root layout
│   ├── page.tsx            # Main dashboard
│   └── globals.css         # Global styles
├── package.json            # Dependencies
├── tsconfig.json           # TypeScript config
├── tailwind.config.js      # Tailwind config
├── next.config.js          # Next.js config
└── postcss.config.js       # PostCSS config

api.py                      # FastAPI backend
vercel.json                 # Vercel deployment config
```

---

## 🏆 Credits

Built with:
- ⚡ **Next.js 14** - React framework
- 🎨 **Tailwind CSS** - Styling
- 📊 **SWR** - Data fetching
- 🚀 **FastAPI** - Backend API
- 🐼 **Pandas** - Data processing

---

**Deploy your dashboard now:** [vercel.com](https://vercel.com) → Import your repo → Deploy! 🚀

# 🚀 Deploy QuantGold Dashboard to Vercel

Quick guide to get your live dashboard online in under 5 minutes.

---

## ✅ What You're Deploying

- **Frontend:** Beautiful Next.js dashboard (deployed to Vercel)
- **Backend:** FastAPI serving paper trading data (deployed to Railway/Render)
- **Result:** Live monitoring accessible from anywhere via https://your-name.vercel.app

**Current API Status:** ✅ Running at http://localhost:8000

---

## 🚀 Quick Deploy (5 minutes)

### **Step 1: Deploy API Backend (2 min)**

**Option A: Railway (Easiest)**

1. Go to [railway.app](https://railway.app)
2. Click "Start a New Project"
3. Select "Deploy from GitHub repo"
4. Choose your `QuantGoldGPT` repository
5. Railway auto-detects the `Procfile` and deploys
6. Copy your API URL (e.g., `https://quantgold-api.up.railway.app`)

**Option B: Render**

1. Go to [render.com](https://render.com)
2. New → Web Service
3. Connect your GitHub repo
4. Build Command: `pip install -r requirements-dashboard.txt`
5. Start Command: `uvicorn dashboard.api:app --host 0.0.0.0 --port $PORT`
6. Create Web Service
7. Copy your API URL

### **Step 2: Deploy Dashboard to Vercel (2 min)**

1. Go to [vercel.com](https://vercel.com)
2. Click "Add New..." → "Project"
3. Import your `QuantGoldGPT` GitHub repository
4. Vercel auto-detects Next.js in `/dashboard`
5. Add environment variable:
   - Key: `NEXT_PUBLIC_API_URL`
   - Value: Your Railway/Render API URL (from Step 1)
6. Click "Deploy"
7. Visit your dashboard at `https://your-project.vercel.app`

### **Step 3: Test It Works**

1. Open your Vercel URL
2. You should see:
   - ✅ Win Rate: 86.7%
   - ✅ Total Trades: 278
   - ✅ Recent trades list
   - ✅ Live feed updating
3. Toggle "Live" button to pause/resume auto-refresh

---

## 🖥️ Local Testing (Optional)

### **Quick Start (One Command)**

```bash
cd /workspace
./start-dashboard.sh
```

This will:
1. Check if paper trading data exists (run deployment if not)
2. Start FastAPI backend on http://localhost:8000
3. Show instructions for starting frontend

### **Manual Start**

```bash
# Terminal 1: API Backend
cd /workspace
pip install -r requirements-dashboard.txt
python3 -m uvicorn dashboard.api:app --host 0.0.0.0 --port 8000

# Terminal 2: Frontend (in a new terminal)
cd /workspace/dashboard
npm install
npm run dev

# Visit: http://localhost:3000
```

---

## 🔧 Configuration

### **Environment Variables (Vercel)**

After deploying, go to your Vercel project → Settings → Environment Variables:

| Variable | Value | Required |
|----------|-------|----------|
| `NEXT_PUBLIC_API_URL` | `https://your-api.railway.app` | ✅ Yes |

### **Update API URL Later**

```bash
# Using Vercel CLI
vercel env add NEXT_PUBLIC_API_URL production
# Enter your new API URL
vercel --prod
```

---

## 📊 What the Dashboard Shows

### **Status Cards** (Top)
- 📈 **Win Rate:** Overall and recent performance
- 🎯 **Total Trades:** Count and coverage percentage
- ⚠️ **Drift Status:** HEALTHY/DEGRADATION/DRIFT
- 🟢 **System Status:** ACTIVE/INACTIVE

### **Performance Metrics** (Middle)
- Overall stats (successful, failed, avg probability)
- By signal type (BUY vs SELL with individual win rates)
- Recent performance (last 50 trades)

### **Recent Trades** (Bottom Left)
- Last 10 executed trades
- ✅/❌ Success indicators
- Probability, price, timestamp
- Color-coded: 🟢 BUY, 🔴 SELL

### **Live Feed** (Bottom Right)
- Real-time activity stream
- All predictions and trades
- Auto-updates every 10 seconds

---

## 🎨 Customization

### **Change Refresh Rate**

Edit `/workspace/dashboard/app/page.tsx`:

```typescript
// Line ~60: Change 10000 to desired milliseconds
const refreshInterval = autoRefresh ? 30000 : 0  // 30 seconds instead of 10
```

### **Change Colors**

Edit `/workspace/dashboard/tailwind.config.js`:

```javascript
colors: {
  gold: {
    400: '#your-color-here',  // Main accent
  },
}
```

### **Show More Trades**

Edit `/workspace/dashboard/app/page.tsx`:

```typescript
// Line ~85: Change limit
const { data: tradesData } = useSWR<{ trades: Trade[] }>(
  `${API_URL}/api/trades?limit=20`,  // Show 20 instead of 10
  fetcher,
  { refreshInterval }
)
```

---

## 🐛 Troubleshooting

### **Dashboard shows "no data"**

✅ **Solution:** Run paper trading first:
```bash
python3 /workspace/quantgold/execution/deploy_paper_trading.py \
  --symbol XAUUSD \
  --timeframe H4 \
  --model xgboost
```

### **API not responding**

✅ **Check API is running:**
```bash
curl http://localhost:8000/api/status
# Should return JSON with status: "active"
```

✅ **Check paper trading files exist:**
```bash
ls -la /workspace/paper_trading/
# Should show predictions_*.parquet files
```

### **CORS errors in browser**

✅ **API has CORS enabled by default**

For production, restrict to your domain in `/workspace/dashboard/api.py`:
```python
# Line 16
allow_origins=["https://your-app.vercel.app"],  # Instead of "*"
```

### **Vercel deployment fails**

✅ **Check Vercel detected Next.js:**
- Framework Preset should be "Next.js"
- Root Directory should be `dashboard`

✅ **Check build logs:**
- Go to Vercel → Deployments → Click failed deployment
- Check logs for errors

---

## 📱 Mobile Access

Dashboard is fully responsive! Access from:
- 📱 Phone
- 📱 Tablet
- 💻 Desktop

All features work on mobile including:
- ✅ Live auto-refresh
- ✅ Scrollable feeds
- ✅ Tap to pause/resume

---

## 🎯 Production Checklist

Before going live:

- [ ] Paper trading data exists (`ls /workspace/paper_trading/`)
- [ ] API deployed to Railway/Render
- [ ] Dashboard deployed to Vercel
- [ ] Environment variable `NEXT_PUBLIC_API_URL` set in Vercel
- [ ] Test dashboard loads and shows data
- [ ] Verify auto-refresh works (toggle Live button)
- [ ] Check mobile responsiveness
- [ ] (Optional) Set up custom domain in Vercel
- [ ] (Optional) Restrict CORS to your domain in `dashboard/api.py`

---

## 🔗 Useful Links

- **Vercel Dashboard:** https://vercel.com/dashboard
- **Railway Dashboard:** https://railway.app/dashboard
- **Render Dashboard:** https://dashboard.render.com
- **API Docs (local):** http://localhost:8000/docs
- **Dashboard Repo:** https://github.com/Kenan3477/QuantGoldGPT

---

## 🆘 Need Help?

1. Check `dashboard/README.md` for detailed docs
2. Test API locally: `curl http://localhost:8000/api/status`
3. Check Vercel deployment logs
4. Verify environment variables are set

---

**Your dashboard is ready to deploy!** 🚀

Just follow Step 1 (Railway) → Step 2 (Vercel) → Done!

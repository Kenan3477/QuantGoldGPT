# 🚨 Vercel Deployment Fix

## Issue
Vercel tried to install Python dependencies from the root directory. The dashboard is pure Next.js and doesn't need Python.

## ✅ Solution: Set Root Directory

When deploying to Vercel, you need to tell it where the Next.js app is:

### **In Vercel Dashboard:**

1. Go to your project settings
2. **General** tab
3. Find **"Root Directory"** section
4. Click **"Edit"**
5. Set to: **`dashboard`**
6. Click **"Save"**
7. **Redeploy** from Deployments tab

### **Or Using Vercel CLI:**

```bash
cd /workspace/dashboard
vercel

# When prompted:
# "Set up and deploy?" → Yes
# "Which scope?" → Your account
# "Link to existing project?" → No (or Yes if you already created one)
# "What's your project's name?" → quantgold-dashboard
# "In which directory is your code located?" → ./ (current directory)
```

This deploys just the dashboard directory, avoiding the Python dependency conflict.

---

## 🔧 Alternative: Manual Deployment

If you prefer to deploy the entire repo:

1. In Vercel project settings → **Root Directory** → Set to `dashboard`
2. This tells Vercel to only look in the `dashboard/` folder
3. The Python code in the root will be ignored

---

## ✅ Verification

After deployment, your dashboard should:
- ✅ Build successfully
- ✅ Show at `https://your-project.vercel.app`
- ✅ Display real-time metrics (if API is deployed to Railway)

---

## 📝 Environment Variable

Don't forget to set the API URL:

1. Vercel project → **Settings** → **Environment Variables**
2. Add:
   - **Name:** `NEXT_PUBLIC_API_URL`
   - **Value:** `https://your-railway-api-url.railway.app`
3. **Redeploy**

---

## 🐛 Still Having Issues?

Try deploying from the dashboard directory directly:

```bash
cd /workspace/dashboard
npx vercel --prod

# Follow the prompts
# This ensures Vercel only sees the Next.js app
```

---

Your dashboard will be live at: `https://your-project.vercel.app` 🚀

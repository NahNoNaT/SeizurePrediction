# Render Deployment Guide

## 🚀 Quick Setup

### Step 1: Connect Your GitHub Repository
1. Go to [render.com](https://render.com)
2. Click "New +" → "Web Service"
3. Select your GitHub repository
4. Render will auto-detect the `render.yaml` configuration

### Step 2: Set Environment Variables on Render
⚠️ **IMPORTANT**: These variables must be set BEFORE deployment

1. In Render dashboard, go to your service → **Environment**
2. Add these variables:

| Key | Value | Notes |
|-----|-------|-------|
| `DATABASE_URL` | `postgresql://user:password@host:port/dbname?sslmode=require` | **REQUIRED** - Get from Supabase |
| `APP_LOG_LEVEL` | `INFO` | Optional, default is INFO |
| `SEIZURE_MODEL_DEVICE` | `cpu` | GPU not available on free tier |

### Step 3: Get DATABASE_URL from Supabase
1. Go to [supabase.com](https://supabase.com) → Your Project
2. Click **Settings** → **Database**
3. Copy the **Connection string** (Connection pooler version)
4. It should look like:
   ```
   postgresql://postgres.xxxxx:password@aws-1-ap-northeast-2.pooler.supabase.com:5432/postgres?sslmode=require
   ```

### Step 4: Deploy
1. Paste `DATABASE_URL` into Render environment
2. Click "Deploy"
3. Watch the logs - deployment takes 5-10 minutes

## ✅ Troubleshooting

### Error: `psycopg.ProgrammingError: invalid connection option "DATABASE_URL"`
**Cause**: DATABASE_URL environment variable is not set on Render  
**Fix**: 
- Go to Render Dashboard → Your Service → Environment
- Make sure `DATABASE_URL` is set with the actual connection string (not empty)
- Click "Deploy" again

### Error: Connection refused / timeout
**Cause**: Supabase server not reachable from Render  
**Fix**:
- Verify Supabase is running and not paused
- Check if you need to add Render's IP to Supabase firewall (usually not needed for cloud)
- Test connection locally first: `python test_database_connection.py`

### Error: Authentication failed / invalid password
**Cause**: Wrong credentials in DATABASE_URL  
**Fix**:
- Go to Supabase → Settings → Database
- Use the **Connection pooler** (not direct connection)
- Double-check password - watch for special characters that need escaping
- Copy fresh from Supabase dashboard

### Error: Cold start timeout (free tier)
**Cause**: Initial startup takes too long  
**Fix**:
- Free tier has limited resources
- Upgrade to a paid plan for faster startup
- Model loading is optimized but still takes ~30-60s first time

## 📊 Monitoring

Check logs in Render:
1. Go to your service
2. Click **Logs** tab
3. Look for these messages:
   - ✅ `Application startup complete` = Ready to serve
   - ❌ `Application startup failed` = Check error details above
   - 🔄 `Started server process` = FastAPI running

## 🔗 Useful Links
- [Render Docs](https://render.com/docs)
- [Supabase Connection Strings](https://supabase.com/docs/guides/database/connecting-to-postgres)
- [psycopg PostgreSQL Driver](https://www.psycopg.org/)

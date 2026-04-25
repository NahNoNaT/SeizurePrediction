# Seizure Prediction Web App (Checkpoint + Legacy Models)

This project supports two local prediction backends:

- `checkpoint/*.pkl` pipeline (`global_model_smart.pkl` + scaler/top_idx + feature extractor)
- legacy `.joblib` ensembles discovered from:

- `UNIVERSAL_LOPO_MODELS`
- `LOSO_Models_Final`

Hugging Face model wrappers were removed from active usage.

## Main routes

- `GET /dashboard`
- `GET /cases`
- `GET /reports`
- `GET /api/health`

## Feature extractor integration

Legacy prediction expects feature extraction from:

- `step1_extract_features.py`
- `chb_mit_preprocess.py`

Default extractor path:

- `app/services/legacy_feature_extractor.py`

Custom override is supported via env:

- `SEIZURE_LEGACY_FEATURE_EXTRACTOR=<module>:<attribute>`

Contract:

```python
extract(prepared_segment=..., feature_set=..., edf_path=...) -> np.ndarray
```

## Benchmark fallback tuning

When torch checkpoints are not configured, case analysis uses legacy scan mode
from `UNIVERSAL_LOPO_MODELS` / `LOSO_Models_Final`.

Tuning variables (used by legacy fallback scan):

- `SEIZURE_BENCHMARK_SCAN_WINDOW_SEC` (default: `10`)
- `SEIZURE_BENCHMARK_SCAN_HOP_SEC` (default: `10`)
- `SEIZURE_BENCHMARK_SCAN_MAX_WINDOWS` (default: `2000`)
- `SEIZURE_BENCHMARK_FLAG_THRESHOLD` (default: `0.60`)
- `SEIZURE_BENCHMARK_MIN_CONSECUTIVE_WINDOWS` (default: `2`)
- `SEIZURE_BENCHMARK_MIN_INTERVAL_SEC` (default: `20`)

Checkpoint PKL artifacts expected in `checkpoint/` by default:

- `global_model_smart.pkl`
- `global_scaler.pkl`
- `global_top_idx.pkl`
- `feature_extractor.py`

Optional env overrides:

- `SEIZURE_PKL_CHECKPOINT_DIR`
- `SEIZURE_PKL_MODEL_FILE`
- `SEIZURE_PKL_SCALER_FILE`
- `SEIZURE_PKL_TOP_IDX_FILE`
- `SEIZURE_PKL_FEATURE_EXTRACTOR_FILE`
- `SEIZURE_PKL_MAX_MISSING_CHANNELS` (default: `4`)

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Run:

```powershell
uvicorn app.main:app --reload
```

Open:

- `http://127.0.0.1:8000/dashboard`
- `http://127.0.0.1:8000/auth/login`

## Authentication and roles

Authentication is enabled by default.

1. Open `http://127.0.0.1:8000/auth/register` to create the first account.
2. The very first account is created with role `admin`.
3. Sign in at `http://127.0.0.1:8000/auth/login`.

Role model:

- `viewer`: read-only access to dashboard/cases/reports and read APIs.
- `clinician`: all viewer permissions + create/upload/analyze/generate report.
- `admin`: all clinician permissions + admin page/checkpoint updates + case deletion + user creation.

Environment variables:

- `SEIZURE_AUTH_ENABLED=true|false` (default: `true`)
- `SEIZURE_ALLOW_PUBLIC_REGISTRATION=true|false` (default: `false`)
- `SEIZURE_SESSION_SECRET=<strong-random-secret>`
- `SEIZURE_SESSION_HTTPS_ONLY=true|false` (default: `false`, set `true` behind HTTPS)

Registration policy:

- When `SEIZURE_ALLOW_PUBLIC_REGISTRATION=false`, only the first account can self-register, then only admins can create users.
- When `SEIZURE_ALLOW_PUBLIC_REGISTRATION=true`, new visitors can self-register and are assigned role `viewer` by default.

Limitations:

- No password reset flow yet.
- No email verification flow.
- Session-based login; if the secret changes, all active sessions are invalidated.

## Supabase + Vercel deployment

Use this production path:

1. Deploy backend API to Render with Docker (`render.yaml` + `Dockerfile` are included).
2. Set `DATABASE_URL` on Render to your Supabase Postgres connection string.
3. Wait for Render to finish and copy backend URL (example: `https://your-backend-service.onrender.com`).
4. Edit `vercel.json` and replace `https://your-backend-service.onrender.com` with your real Render URL.
5. Deploy this repo on Vercel (Vercel acts as reverse proxy/domain only).

Required env:

- Render: `DATABASE_URL` (required), optional `SEIZURE_*` overrides.
- Vercel: no Python runtime env needed for this proxy setup.

When `DATABASE_URL` is present, the app uses Supabase Postgres automatically.
Without `DATABASE_URL`, it falls back to local SQLite.

## Notes

- This app is for benchmarking/research workflow support, not clinical deployment.
- Model probabilities are estimator outputs and are not jointly calibrated clinical risk.

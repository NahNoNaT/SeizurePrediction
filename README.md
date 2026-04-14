# Seizure Prediction Web App (Checkpoint + Legacy Models)

This project supports two local prediction backends:

- `checkpoint/*.pkl` pipeline (`global_model_smart.pkl` + scaler/top_idx + feature extractor)
- legacy `.joblib` ensembles discovered from:

- `UNIVERSAL_LOPO_MODELS`
- `LOSO_Models_Final`

Hugging Face model wrappers were removed from active usage.

## Main routes

- `GET /demo`
- `GET /health`
- `POST /predict`
- `GET /legacy/health`
- `POST /legacy/predict`
- `GET /replay`

## Model source and selection

The backend discovers models recursively:

- `UNIVERSAL_LOPO_MODELS/<FEATURE_SET>/*.joblib`
- `LOSO_Models_Final/<SUBJECT_ID>/<FEATURE_SET>/*.joblib`

Supported feature sets:

- `LBP`
- `GLCM`
- `COMBINED`

For exact `model_id`, call `GET /legacy/health`.

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

## Full-file scan mode

`POST /legacy/predict` supports scanning the full recording in one request:

- `scan_full_file=true`
- `scan_window_sec`
- `scan_hop_sec`
- `scan_start_sec`
- `scan_end_sec`
- `scan_max_windows`

Response includes:

- `scan.timeline` (window-by-window risk)
- `scan.peak_window` (highest-risk window + model details)

## Replay mode

`/replay` uploads one EDF and streams sliding-window risk from a configured legacy model.

Optional env:

- `SEIZURE_REPLAY_LEGACY_MODEL_ID` (default: `universal:dt:combined`)

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

- `http://127.0.0.1:8000/demo`
- `http://127.0.0.1:8000/replay`

## Notes

- This app is for benchmarking/research workflow support, not clinical deployment.
- Model probabilities are estimator outputs and are not jointly calibrated clinical risk.

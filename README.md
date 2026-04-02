# Clinical EEG Seizure Risk Assessment Platform

Doctor-facing FastAPI application for AI-assisted seizure prediction from scalp EEG recordings.

The platform now uses an EDF-first clinical workflow:

Create Case  
→ Upload EDF recording  
→ Validate EEG metadata  
→ Review channel mapping  
→ Preprocess and segment signal  
→ Run seizure-risk inference  
→ Apply temporal post-processing  
→ Detect flagged EEG intervals  
→ Generate clinical summary and report

The clinician-facing interface is case-centered and avoids exposing machine-learning controls in the main workflow.

## Current architecture

### Core backend layers

- `app/main.py`
  - HTML pages for Dashboard, Cases, New Analysis, Reports, and Admin
  - case-based API routes
- `app/services/eeg_intake.py`
  - EDF intake
  - EDF header metadata parsing
  - EDF waveform decoding
  - channel mapping
  - resampling and signal preparation for downstream analysis
- `app/services/pipeline.py`
  - preprocessing
  - segmentation
  - PyTorch inference
  - probability timeline generation
- `app/services/clinical_workflow.py`
  - temporal smoothing
  - consecutive high-risk decision rules
  - segment-level risk labeling
  - clinical interpretation and report scaffolding
- `app/services/clinical_store.py`
  - SQLite persistence for cases, recordings, analyses, segment results, and reports

## Data model

### Case

- `id`
- `patient_id`
- `clinician_name`
- `recording_date`
- `notes`
- `created_at`

### Recording

- `id`
- `case_id`
- `file_name`
- `file_path`
- `duration_sec`
- `sampling_rate`
- `channel_count`
- `channel_names`
- `mapped_channels`
- `uploaded_at`

### Analysis

- `id`
- `case_id`
- `recording_id`
- `model_version`
- `overall_risk`
- `review_priority`
- `max_risk_score`
- `mean_risk_score`
- `flagged_segments_count`
- `clinical_summary`
- `interpretation`
- `created_at`

### SegmentResult

- `id`
- `analysis_id`
- `segment_index`
- `start_sec`
- `end_sec`
- `risk_score`
- `risk_label`
- `is_flagged`

### Report

- `id`
- `analysis_id`
- `case_id`
- `report_path`
- `generated_at`

## Primary clinician workflow

- EDF is the primary recording format in the doctor-facing workflow
- `.npy`, `.npz`, and `.csv` remain available internally for backend/demo compatibility, but are not presented as the main clinical upload path

## API structure

- `POST /api/cases`
- `GET /api/cases`
- `GET /api/cases/{case_id}`
- `POST /api/cases/{case_id}/recordings`
- `POST /api/recordings/{recording_id}/analyze`
- `GET /api/analyses/{analysis_id}`
- `POST /api/analyses/{analysis_id}/report`

Additional utility endpoints:

- `GET /api/health`
- `GET /api/model/info`
- `POST /api/model/checkpoint`

## Frontend page structure

- `/dashboard`
  - total analyses
  - high-risk cases
  - pending reviews
  - recent analyses
- `/cases`
  - case history
  - risk/date filters
  - access to case detail
- `/cases/new`
  - patient metadata
  - EDF upload
  - run analysis
- `/cases/{case_id}`
  - patient info
  - recording metadata
  - estimated seizure risk
  - review priority
  - risk timeline
  - flagged EEG segments
  - interpretation panel
- `/reports`
  - generated analysis reports
- `/admin`
  - technical configuration hidden from clinicians

## EDF intake behavior

The current implementation includes an EDF intake layer that:

- parses EDF header metadata
- decodes waveform signals from EDF
- extracts channel labels, signal count, estimated sampling rate, and duration
- applies channel mapping
- resamples signals to the configured analysis rate
- prepares internal continuous EEG input for downstream segmentation and inference

Current limitation:

- if an EDF recording is missing required montage channels, missing mapped channels are zero-filled and reported in metadata status
- the surrounding model checkpoint must still be present for full inference

## Model checkpoint

By default the app looks for a trained PyTorch checkpoint at:

```text
models/checkpoints/seizure_prediction_model.pt
```

Override with:

```powershell
$env:SEIZURE_MODEL_CHECKPOINT="D:\path\to\checkpoint.pt"
```

If no checkpoint is available:

- the web app still runs
- EDF upload and metadata intake still work
- analysis endpoints return a clear `model_unavailable` response

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Run locally

```powershell
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/dashboard
```

## Medical disclaimer

This platform is for research and educational purposes only and does not replace professional medical diagnosis.

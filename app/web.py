from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar
from urllib.parse import urlencode
from uuid import uuid4

from fastapi import HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse

from app.config import RuntimeConfig
from app.schemas import AnalysisResponse, CaseDetail, CaseSummary, TimelinePoint

UPLOAD_CHUNK_SIZE_BYTES = 8 * 1024 * 1024
CASE_DETAIL_MAX_CHART_POINTS = 600
CASE_DETAIL_MAX_SEGMENT_ROWS = 200
T = TypeVar("T")


def app_http_exception(status_code: int, code: str, detail: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"code": code, "detail": detail})


def sanitize_filename(filename: str) -> str:
    safe_name = Path(filename or "eeg_recording").name
    stem = Path(safe_name).stem or "eeg_recording"
    allowed = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in stem)
    compact = "-".join(part for part in allowed.split("-") if part)
    return compact[:64] or "eeg-recording"


def build_redirect(url: str, notice: str, tone: str = "info") -> RedirectResponse:
    query = urlencode({"notice": notice, "tone": tone})
    separator = "&" if "?" in url else "?"
    return RedirectResponse(url=f"{url}{separator}{query}", status_code=303)


def page_context(
    request: Request,
    *,
    page_title: str,
    page_subtitle: str,
    active_page: str,
    extra: dict | None = None,
) -> dict:
    config: RuntimeConfig = request.app.state.runtime_config
    context = {
        "request": request,
        "app_title": config.app_title,
        "platform_name": config.platform_name,
        "research_disclaimer": config.research_disclaimer,
        "page_title": page_title,
        "page_subtitle": page_subtitle,
        "active_page": active_page,
        "notice": request.query_params.get("notice"),
        "tone": request.query_params.get("tone", "info"),
        "current_year": datetime.now().year,
    }
    if extra:
        context.update(extra)
    return context


def build_case_summary(case_detail: CaseDetail) -> CaseSummary:
    return CaseSummary(
        case_id=case_detail.case_id,
        patient_id=case_detail.patient_id,
        clinician_name=case_detail.clinician_name,
        recording_date=case_detail.recording_date,
        status=case_detail.status,
        created_at=case_detail.created_at,
        updated_at=case_detail.updated_at,
        overall_risk=case_detail.analysis.overall_risk if case_detail.analysis else None,
        review_priority=case_detail.analysis.review_priority if case_detail.analysis else None,
        recording_id=case_detail.recording.recording_id if case_detail.recording else None,
        latest_analysis_id=case_detail.analysis.analysis_id if case_detail.analysis else None,
        recording_file_name=case_detail.recording.file_name if case_detail.recording else None,
        report_available=case_detail.report is not None,
    )


def format_duration_label(total_seconds: float) -> str:
    seconds = max(int(round(total_seconds)), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def format_time_label(seconds: float) -> str:
    total_seconds = max(int(round(seconds)), 0)
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def downsample_items(items: list[T], max_items: int) -> list[T]:
    if max_items <= 0 or not items:
        return []
    if len(items) <= max_items:
        return items
    if max_items == 1:
        return [items[-1]]

    last_index = len(items) - 1
    sampled: list[T] = []
    previous_index = -1
    for index in range(max_items):
        resolved_index = round(index * last_index / (max_items - 1))
        if resolved_index == previous_index:
            continue
        sampled.append(items[resolved_index])
        previous_index = resolved_index
    return sampled


def timeline_from_segments(segment_results) -> list[TimelinePoint]:
    return [
        TimelinePoint(
            segment_index=segment.segment_index,
            start_sec=segment.start_sec,
            end_sec=segment.end_sec,
            risk_score=segment.risk_score,
            risk_label=segment.risk_label,
            is_flagged=segment.is_flagged,
        )
        for segment in segment_results
    ]


def chart_points_from_timeline(
    timeline: list[TimelinePoint],
    *,
    max_points: int | None = None,
) -> list[dict[str, str | float | bool]]:
    points = downsample_items(timeline, max_points) if max_points is not None else timeline
    return [
        {
            "label": format_time_label(point.start_sec),
            "value": point.risk_score,
            "highlight": point.is_flagged,
        }
        for point in points
    ]


async def save_upload(
    upload_file: UploadFile,
    *,
    uploads_dir: Path,
    config: RuntimeConfig,
    clinician_mode: bool = True,
) -> tuple[dict, Path]:
    original_name = Path(upload_file.filename or "").name
    if not original_name:
        raise app_http_exception(400, "missing_file", "Please select an EEG recording before starting analysis.")

    extension = Path(original_name).suffix.lower()
    allowed_extensions = config.clinician_upload_extensions if clinician_mode else config.supported_upload_extensions
    if extension not in allowed_extensions:
        allowed = ", ".join(allowed_extensions)
        raise app_http_exception(400, "unsupported_file_type", f"Supported EEG recording types are: {allowed}.")

    upload_id = str(uuid4())
    stored_filename = f"{sanitize_filename(original_name)}-{upload_id[:8]}{extension}"
    upload_path = uploads_dir / stored_filename

    size_bytes = 0
    try:
        with upload_path.open("wb") as output_file:
            while True:
                chunk = await upload_file.read(UPLOAD_CHUNK_SIZE_BYTES)
                if not chunk:
                    break
                size_bytes += len(chunk)
                if size_bytes > config.max_upload_size_bytes:
                    raise app_http_exception(
                        413,
                        "file_too_large",
                        f"EEG file exceeds the {config.max_upload_size_mb} MB upload limit.",
                    )
                output_file.write(chunk)
    except HTTPException:
        upload_path.unlink(missing_ok=True)
        raise
    except Exception:
        upload_path.unlink(missing_ok=True)
        raise
    finally:
        await upload_file.close()

    if size_bytes == 0:
        upload_path.unlink(missing_ok=True)
        raise app_http_exception(400, "empty_file", "The uploaded EEG recording is empty.")

    upload = {
        "upload_id": upload_id,
        "filename": original_name,
        "stored_filename": stored_filename,
        "extension": extension,
        "size_bytes": size_bytes,
        "upload_path": str(upload_path),
        "created_at": datetime.now(timezone.utc),
    }
    return upload, upload_path


def build_analysis_response(detail: CaseDetail, message: str) -> AnalysisResponse:
    assert detail.recording is not None
    assert detail.analysis is not None
    timeline = timeline_from_segments(detail.segment_results)
    return AnalysisResponse(
        case=build_case_summary(detail),
        recording=detail.recording,
        analysis=detail.analysis,
        high_risk_intervals=detail.high_risk_intervals,
        segment_results=detail.segment_results,
        timeline=timeline,
        report=detail.report,
        message=message,
    )

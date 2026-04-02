from __future__ import annotations

import logging
from datetime import datetime, timezone
from html import escape
from pathlib import Path

from app.config import RuntimeConfig, runtime_config
from app.schemas import AnalysisOverview, CaseDetail, ReportSummary

logger = logging.getLogger(__name__)


class ClinicalReportService:
    def __init__(self, project_root: Path, config: RuntimeConfig = runtime_config):
        self.project_root = project_root
        self.config = config
        self.report_directory = config.reports_directory(project_root)
        self.report_directory.mkdir(parents=True, exist_ok=True)

    def generate(self, *, report_id: str, case_detail: CaseDetail) -> ReportSummary:
        if case_detail.analysis is None or case_detail.recording is None:
            raise ValueError("A completed analysis and linked recording are required before generating a report.")
        if case_detail.analysis.status not in {"COMPLETED", "REPORT_READY"}:
            raise ValueError("A completed analysis is required before generating a clinician-facing report.")

        report_path = self.report_directory / f"{report_id}.html"
        report_path.write_text(self._render_document(case_detail), encoding="utf-8")
        logger.info(
            "Report generated",
            extra={"event": "report_generated", "case_id": case_detail.case_id, "analysis_id": case_detail.analysis.analysis_id, "status": "REPORT_READY"},
        )
        return ReportSummary(
            report_id=report_id,
            analysis_id=case_detail.analysis.analysis_id,
            case_id=case_detail.case_id,
            patient_id=case_detail.patient_id,
            clinician_name=case_detail.clinician_name,
            report_title=f"Clinical EEG Analysis Report - {case_detail.patient_id}",
            report_path=str(report_path),
            report_url=f"/reports/{report_id}",
            report_status="Generated",
            generated_at=datetime.now(timezone.utc),
        )

    def _render_document(self, case_detail: CaseDetail) -> str:
        assert case_detail.analysis is not None
        assert case_detail.recording is not None

        interval_rows = "".join(
            (
                "<tr>"
                f"<td>{interval.interval_index}</td>"
                f"<td>{interval.start_sec:.2f}s</td>"
                f"<td>{interval.end_sec:.2f}s</td>"
                f"<td>{interval.mean_risk * 100:.0f}%</td>"
                f"<td>{escape(interval.review_status)}</td>"
                "</tr>"
            )
            for interval in case_detail.high_risk_intervals
        )
        if not interval_rows:
            interval_rows = '<tr><td colspan="5">No high-risk intervals were identified in this analysis.</td></tr>'

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Clinical EEG Analysis Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; color: #11263d; margin: 32px; }}
    h1, h2 {{ color: #123b64; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; margin: 20px 0; }}
    .card {{ border: 1px solid #d6e3ee; border-radius: 12px; padding: 14px; background: #f9fcfe; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
    th, td {{ padding: 10px; border-bottom: 1px solid #e2ebf3; text-align: left; }}
    th {{ color: #4f647a; font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }}
    .disclaimer {{ margin-top: 24px; padding: 14px; border-radius: 10px; background: #fff6e8; color: #79541d; }}
  </style>
</head>
<body>
  <h1>Clinical EEG Analysis Report</h1>
  <p>AI-assisted review summary for scalp EEG seizure-risk analysis.</p>

  <div class="grid">
    <div class="card"><strong>Patient ID</strong><br>{escape(case_detail.patient_id)}</div>
    <div class="card"><strong>Clinician</strong><br>{escape(case_detail.clinician_name)}</div>
    <div class="card"><strong>Recording Date</strong><br>{case_detail.recording_date.isoformat()}</div>
    <div class="card"><strong>Analysis Date</strong><br>{case_detail.analysis.created_at.isoformat()}</div>
    <div class="card"><strong>Recording File</strong><br>{escape(case_detail.recording.file_name)}</div>
    <div class="card"><strong>Recording Duration</strong><br>{case_detail.recording.duration_sec / 60.0:.1f} minutes</div>
    <div class="card"><strong>Overall Risk</strong><br>{escape(case_detail.analysis.overall_risk)}</div>
    <div class="card"><strong>Review Priority</strong><br>{escape(case_detail.analysis.review_priority)}</div>
  </div>

  <h2>Clinical Summary</h2>
  <p>{escape(case_detail.analysis.clinical_summary)}</p>

  <h2>Clinical Interpretation</h2>
  <p>{escape(case_detail.analysis.interpretation)}</p>

  <h2>High-Risk Intervals</h2>
  <table>
    <thead>
      <tr>
        <th>Interval</th>
        <th>Start</th>
        <th>End</th>
        <th>Mean Risk</th>
        <th>Review Status</th>
      </tr>
    </thead>
    <tbody>
      {interval_rows}
    </tbody>
  </table>

  <div class="disclaimer">{escape(self.config.research_disclaimer)}</div>
</body>
</html>"""

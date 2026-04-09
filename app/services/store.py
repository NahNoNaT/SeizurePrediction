from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Any

from app.schemas import (
    Analysis,
    AnalysisOverview,
    Case,
    CaseDetail,
    CaseSummary,
    DashboardStats,
    HighRiskInterval,
    ModelComparison,
    Recording,
    RecordingOverview,
    ReportSummary,
    SegmentResult,
    SegmentResultRecord,
)


class ClinicalCaseStore:
    def __init__(self, database_path: Path):
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS clinical_cases (
                    id TEXT PRIMARY KEY,
                    patient_id TEXT NOT NULL,
                    clinician_name TEXT NOT NULL,
                    recording_date TEXT NOT NULL,
                    notes TEXT,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS recordings (
                    id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    duration_sec REAL NOT NULL,
                    sampling_rate REAL NOT NULL,
                    channel_count INTEGER NOT NULL,
                    channel_names_json TEXT NOT NULL,
                    input_montage_type TEXT NOT NULL DEFAULT 'unsupported',
                    conversion_status TEXT NOT NULL DEFAULT 'blocked',
                    conversion_messages_json TEXT NOT NULL DEFAULT '[]',
                    mapped_channels_json TEXT NOT NULL,
                    derived_channels_json TEXT NOT NULL DEFAULT '[]',
                    approximated_channels_json TEXT NOT NULL DEFAULT '[]',
                    missing_channels_json TEXT NOT NULL DEFAULT '[]',
                    metadata_status TEXT NOT NULL DEFAULT 'PENDING',
                    mapped_channel_count INTEGER NOT NULL DEFAULT 0,
                    validation_messages_json TEXT NOT NULL DEFAULT '[]',
                    uploaded_at TEXT NOT NULL,
                    FOREIGN KEY(case_id) REFERENCES clinical_cases(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS clinical_analyses (
                    id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    recording_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'ANALYZING',
                    model_version TEXT,
                    overall_risk TEXT,
                    review_priority TEXT,
                    max_risk_score REAL,
                    mean_risk_score REAL,
                    estimated_seizure_risk REAL,
                    flagged_segments_count INTEGER NOT NULL DEFAULT 0,
                    total_segments INTEGER NOT NULL DEFAULT 0,
                    high_risk_intervals_count INTEGER NOT NULL DEFAULT 0,
                    recording_duration_sec REAL NOT NULL DEFAULT 0,
                    clinical_summary TEXT,
                    recommendation TEXT,
                    interpretation TEXT,
                    failure_code TEXT,
                    failure_message TEXT,
                    trace_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    report_generated INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY(case_id) REFERENCES clinical_cases(id) ON DELETE CASCADE,
                    FOREIGN KEY(recording_id) REFERENCES recordings(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS segment_results (
                    id TEXT PRIMARY KEY,
                    analysis_id TEXT NOT NULL,
                    segment_index INTEGER NOT NULL,
                    start_sec REAL NOT NULL,
                    end_sec REAL NOT NULL,
                    risk_score REAL NOT NULL,
                    risk_label TEXT NOT NULL,
                    is_flagged INTEGER NOT NULL,
                    FOREIGN KEY(analysis_id) REFERENCES clinical_analyses(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS interval_results (
                    id TEXT PRIMARY KEY,
                    analysis_id TEXT NOT NULL,
                    interval_index INTEGER NOT NULL,
                    start_sec REAL NOT NULL,
                    end_sec REAL NOT NULL,
                    mean_risk REAL NOT NULL,
                    max_risk REAL NOT NULL,
                    flagged_segments_count INTEGER NOT NULL,
                    review_status TEXT NOT NULL,
                    FOREIGN KEY(analysis_id) REFERENCES clinical_analyses(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS analysis_reports (
                    id TEXT PRIMARY KEY,
                    analysis_id TEXT NOT NULL UNIQUE,
                    case_id TEXT NOT NULL,
                    report_title TEXT NOT NULL,
                    report_path TEXT NOT NULL,
                    report_url TEXT NOT NULL DEFAULT '',
                    report_status TEXT NOT NULL,
                    generated_at TEXT NOT NULL,
                    FOREIGN KEY(analysis_id) REFERENCES clinical_analyses(id) ON DELETE CASCADE,
                    FOREIGN KEY(case_id) REFERENCES clinical_cases(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS analysis_model_runs (
                    id TEXT PRIMARY KEY,
                    analysis_id TEXT NOT NULL,
                    model_key TEXT NOT NULL,
                    model_label TEXT NOT NULL,
                    model_version TEXT,
                    checkpoint_path TEXT,
                    status TEXT NOT NULL,
                    backend_status TEXT NOT NULL,
                    overall_risk TEXT,
                    review_priority TEXT,
                    estimated_seizure_risk REAL,
                    max_risk_score REAL,
                    mean_risk_score REAL,
                    flagged_segments_count INTEGER NOT NULL DEFAULT 0,
                    high_risk_intervals_count INTEGER NOT NULL DEFAULT 0,
                    confidence_score REAL,
                    confidence_label TEXT,
                    agreement_score REAL,
                    inference_time_seconds REAL,
                    failure_code TEXT,
                    failure_message TEXT,
                    is_primary INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY(analysis_id) REFERENCES clinical_analyses(id) ON DELETE CASCADE
                );
                """
            )
            self._ensure_column(connection, "recordings", "missing_channels_json", "TEXT NOT NULL DEFAULT '[]'")
            self._ensure_column(connection, "recordings", "input_montage_type", "TEXT NOT NULL DEFAULT 'unsupported'")
            self._ensure_column(connection, "recordings", "conversion_status", "TEXT NOT NULL DEFAULT 'blocked'")
            self._ensure_column(connection, "recordings", "conversion_messages_json", "TEXT NOT NULL DEFAULT '[]'")
            self._ensure_column(connection, "recordings", "derived_channels_json", "TEXT NOT NULL DEFAULT '[]'")
            self._ensure_column(connection, "recordings", "approximated_channels_json", "TEXT NOT NULL DEFAULT '[]'")
            self._ensure_column(connection, "clinical_analyses", "status", "TEXT NOT NULL DEFAULT 'ANALYZING'")
            self._ensure_column(connection, "clinical_analyses", "failure_code", "TEXT")
            self._ensure_column(connection, "clinical_analyses", "failure_message", "TEXT")
            self._ensure_column(connection, "clinical_analyses", "recommendation", "TEXT")
            self._ensure_column(connection, "clinical_analyses", "trace_json", "TEXT NOT NULL DEFAULT '{}'")
            self._ensure_column(connection, "analysis_reports", "report_url", "TEXT NOT NULL DEFAULT ''")
            connection.execute("UPDATE clinical_cases SET status = 'COMPLETED' WHERE status = 'REVIEW_REQUIRED'")
            connection.execute("UPDATE clinical_cases SET status = 'REPORT_READY' WHERE status = 'HIGH_RISK'")
            connection.execute("UPDATE clinical_analyses SET status = 'FAILED' WHERE status IS NULL")

    def create_case(
        self,
        *,
        case_id: str,
        patient_id: str,
        clinician_name: str,
        recording_date: date,
        notes: str | None,
        status: str,
        created_at: datetime,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO clinical_cases (
                    id, patient_id, clinician_name, recording_date, notes, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    case_id,
                    patient_id,
                    clinician_name,
                    recording_date.isoformat(),
                    notes,
                    status,
                    created_at.isoformat(),
                    created_at.isoformat(),
                ),
            )

    def create_recording(self, recording: RecordingOverview) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO recordings (
                    id, case_id, file_name, file_path, file_type, duration_sec, sampling_rate,
                    channel_count, channel_names_json, input_montage_type, conversion_status,
                    conversion_messages_json, mapped_channels_json, derived_channels_json, approximated_channels_json,
                    missing_channels_json, metadata_status, mapped_channel_count, validation_messages_json, uploaded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    recording.recording_id,
                    recording.case_id,
                    recording.file_name,
                    recording.file_path,
                    recording.file_type,
                    recording.duration_sec,
                    recording.sampling_rate,
                    recording.channel_count,
                    json.dumps(recording.channel_names),
                    recording.input_montage_type,
                    recording.conversion_status,
                    json.dumps(recording.conversion_messages),
                    json.dumps(recording.mapped_channels),
                    json.dumps(recording.derived_channels),
                    json.dumps(recording.approximated_channels),
                    json.dumps(recording.missing_channels),
                    recording.validation_status,
                    recording.mapped_channel_count,
                    json.dumps(recording.validation_messages),
                    recording.uploaded_at.isoformat(),
                ),
            )

    def save_analysis(
        self,
        analysis: AnalysisOverview,
        segment_results: list[SegmentResultRecord],
        high_risk_intervals: list[HighRiskInterval],
        *,
        model_comparisons: list[ModelComparison] | None = None,
        trace_json: dict[str, Any] | None = None,
        case_status: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO clinical_analyses (
                    id, case_id, recording_id, status, model_version, overall_risk, review_priority, max_risk_score,
                    mean_risk_score, estimated_seizure_risk, flagged_segments_count, total_segments,
                    high_risk_intervals_count, recording_duration_sec, clinical_summary, recommendation, interpretation,
                    failure_code, failure_message, trace_json, created_at, report_generated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    analysis.analysis_id,
                    analysis.case_id,
                    analysis.recording_id,
                    analysis.status,
                    analysis.model_version,
                    analysis.overall_risk,
                    analysis.review_priority,
                    analysis.max_risk_score,
                    analysis.mean_risk_score,
                    analysis.estimated_seizure_risk,
                    analysis.flagged_segments_count,
                    analysis.total_segments,
                    analysis.high_risk_intervals_count,
                    analysis.recording_duration_sec,
                    analysis.clinical_summary,
                    analysis.recommendation,
                    analysis.interpretation,
                    analysis.failure_code,
                    analysis.failure_message,
                    json.dumps(trace_json or {}),
                    analysis.created_at.isoformat(),
                    int(analysis.report_generated),
                ),
            )
            if segment_results:
                connection.executemany(
                    """
                    INSERT INTO segment_results (
                        id, analysis_id, segment_index, start_sec, end_sec, risk_score, risk_label, is_flagged
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            row.id,
                            analysis.analysis_id,
                            row.segment_index,
                            row.start_sec,
                            row.end_sec,
                            row.risk_score,
                            row.risk_label,
                            int(row.is_flagged),
                        )
                        for row in segment_results
                    ],
                )
            if high_risk_intervals:
                connection.executemany(
                    """
                    INSERT INTO interval_results (
                        id, analysis_id, interval_index, start_sec, end_sec, mean_risk, max_risk,
                        flagged_segments_count, review_status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            interval.id,
                            analysis.analysis_id,
                            interval.interval_index,
                            interval.start_sec,
                            interval.end_sec,
                            interval.mean_risk,
                            interval.max_risk,
                            interval.flagged_segments_count,
                            interval.review_status,
                        )
                        for interval in high_risk_intervals
                    ],
                )
            if model_comparisons:
                connection.executemany(
                    """
                    INSERT INTO analysis_model_runs (
                        id, analysis_id, model_key, model_label, model_version, checkpoint_path,
                        status, backend_status, overall_risk, review_priority, estimated_seizure_risk,
                        max_risk_score, mean_risk_score, flagged_segments_count, high_risk_intervals_count,
                        confidence_score, confidence_label, agreement_score, inference_time_seconds,
                        failure_code, failure_message, is_primary
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            item.model_run_id,
                            analysis.analysis_id,
                            item.model_key,
                            item.model_label,
                            item.model_version,
                            item.checkpoint_path,
                            item.status,
                            item.backend_status,
                            item.overall_risk,
                            item.review_priority,
                            item.estimated_seizure_risk,
                            item.max_risk_score,
                            item.mean_risk_score,
                            item.flagged_segments_count,
                            item.high_risk_intervals_count,
                            item.confidence_score,
                            item.confidence_label,
                            item.agreement_score,
                            item.inference_time_seconds,
                            item.failure_code,
                            item.failure_message,
                            int(item.is_primary),
                        )
                        for item in model_comparisons
                    ],
                )
            connection.execute(
                "UPDATE clinical_cases SET status = ?, updated_at = ? WHERE id = ?",
                (case_status, analysis.created_at.isoformat(), analysis.case_id),
            )

    def save_report(self, report: ReportSummary) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO analysis_reports (
                    id, analysis_id, case_id, report_title, report_path, report_url, report_status, generated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(analysis_id) DO UPDATE SET
                    id=excluded.id,
                    report_title=excluded.report_title,
                    report_path=excluded.report_path,
                    report_url=excluded.report_url,
                    report_status=excluded.report_status,
                    generated_at=excluded.generated_at
                """,
                (
                    report.report_id,
                    report.analysis_id,
                    report.case_id,
                    report.report_title,
                    report.report_path,
                    report.report_url,
                    report.report_status,
                    report.generated_at.isoformat(),
                ),
            )
            connection.execute(
                "UPDATE clinical_analyses SET report_generated = 1, status = 'REPORT_READY' WHERE id = ?",
                (report.analysis_id,),
            )
            connection.execute(
                "UPDATE clinical_cases SET status = 'REPORT_READY', updated_at = ? WHERE id = ?",
                (report.generated_at.isoformat(), report.case_id),
            )

    def update_case_status(self, case_id: str, status: str, updated_at: datetime) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE clinical_cases SET status = ?, updated_at = ? WHERE id = ?",
                (status, updated_at.isoformat(), case_id),
            )

    def delete_case(self, case_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM clinical_cases WHERE id = ?", (case_id,))
            return cursor.rowcount > 0

    def dashboard_stats(self) -> DashboardStats:
        with self._connect() as connection:
            total_analyses = connection.execute("SELECT COUNT(*) FROM clinical_analyses").fetchone()[0]
            high_risk_cases = connection.execute(
                """
                SELECT COUNT(*)
                FROM clinical_cases c
                WHERE EXISTS (
                    SELECT 1
                    FROM clinical_analyses a
                    WHERE a.case_id = c.id
                      AND a.id = (
                          SELECT id FROM clinical_analyses latest
                          WHERE latest.case_id = c.id
                          ORDER BY created_at DESC LIMIT 1
                      )
                      AND a.overall_risk = 'High'
                )
                """
            ).fetchone()[0]
            pending_review_cases = connection.execute(
                "SELECT COUNT(*) FROM clinical_cases WHERE status IN ('UPLOADED', 'VALIDATED', 'ANALYZING', 'FAILED')"
            ).fetchone()[0]
            recent_reports = connection.execute("SELECT COUNT(*) FROM analysis_reports").fetchone()[0]
        return DashboardStats(
            total_analyses=total_analyses,
            high_risk_cases=high_risk_cases,
            pending_review_cases=pending_review_cases,
            recent_reports=recent_reports,
        )

    def list_cases(
        self,
        *,
        search: str = "",
        risk: str = "",
        date_from: str = "",
        date_to: str = "",
        limit: int | None = None,
    ) -> list[CaseSummary]:
        conditions = []
        parameters: list[Any] = []
        if search:
            conditions.append("(c.patient_id LIKE ? OR c.clinician_name LIKE ?)")
            wildcard = f"%{search.strip()}%"
            parameters.extend([wildcard, wildcard])
        if risk:
            conditions.append("a.overall_risk = ?")
            parameters.append(risk)
        if date_from:
            conditions.append("date(c.recording_date) >= date(?)")
            parameters.append(date_from)
        if date_to:
            conditions.append("date(c.recording_date) <= date(?)")
            parameters.append(date_to)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        limit_clause = "LIMIT ?" if limit is not None else ""
        if limit is not None:
            parameters.append(limit)

        query = f"""
            SELECT
                c.id AS case_id,
                c.patient_id,
                c.clinician_name,
                c.recording_date,
                c.status,
                c.created_at,
                c.updated_at,
                r.id AS recording_id,
                r.file_name AS recording_file_name,
                a.id AS analysis_id,
                a.overall_risk,
                a.review_priority,
                CASE WHEN rep.id IS NOT NULL THEN 1 ELSE 0 END AS report_available
            FROM clinical_cases c
            LEFT JOIN recordings r ON r.id = (
                SELECT id FROM recordings WHERE case_id = c.id ORDER BY uploaded_at DESC LIMIT 1
            )
            LEFT JOIN clinical_analyses a ON a.id = (
                SELECT id FROM clinical_analyses WHERE case_id = c.id ORDER BY created_at DESC LIMIT 1
            )
            LEFT JOIN analysis_reports rep ON rep.analysis_id = a.id
            {where_clause}
            ORDER BY c.updated_at DESC
            {limit_clause}
        """
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [
            CaseSummary(
                case_id=row["case_id"],
                patient_id=row["patient_id"],
                clinician_name=row["clinician_name"],
                recording_date=date.fromisoformat(row["recording_date"]),
                status=row["status"],
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
                overall_risk=row["overall_risk"],
                review_priority=row["review_priority"],
                recording_id=row["recording_id"],
                latest_analysis_id=row["analysis_id"],
                recording_file_name=row["recording_file_name"],
                report_available=bool(row["report_available"]),
            )
            for row in rows
        ]

    def list_reports(self) -> list[ReportSummary]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    rep.id AS report_id,
                    rep.analysis_id,
                    rep.case_id,
                    rep.report_title,
                    rep.report_path,
                    rep.report_url,
                    rep.report_status,
                    rep.generated_at,
                    c.patient_id,
                    c.clinician_name
                FROM analysis_reports rep
                JOIN clinical_cases c ON c.id = rep.case_id
                ORDER BY rep.generated_at DESC
                """
            ).fetchall()
        return [self._row_to_report(row) for row in rows]

    def get_case(self, case_id: str) -> Case | None:
        row = self._get_case_row(case_id)
        if row is None:
            return None
        recordings = self.list_recordings(case_id)
        analyses = self.list_analyses(case_id)
        active_recording = recordings[0] if recordings else None
        active_analysis = analyses[0] if analyses else None
        return self._build_case(
            row=row,
            recording=active_recording,
            recordings=recordings,
            analysis=active_analysis,
            analyses=analyses,
            model_comparisons=self.list_model_comparisons(active_analysis.analysis_id) if active_analysis else [],
            segment_results=self.list_segment_results(active_analysis.analysis_id) if active_analysis else [],
            intervals=self.list_high_risk_intervals(active_analysis.analysis_id) if active_analysis else [],
            report=self.get_report_for_analysis(active_analysis.analysis_id) if active_analysis else None,
        )

    def get_case_detail(self, case_id: str) -> CaseDetail | None:
        return self.get_case(case_id)

    def get_recording(self, recording_id: str) -> Recording | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM recordings WHERE id = ?", (recording_id,)).fetchone()
        return self._row_to_recording(row) if row else None

    def list_recordings(self, case_id: str) -> list[Recording]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM recordings WHERE case_id = ? ORDER BY uploaded_at DESC",
                (case_id,),
            ).fetchall()
        return [self._row_to_recording(row) for row in rows]

    def get_analysis(self, analysis_id: str) -> Analysis | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM clinical_analyses WHERE id = ?", (analysis_id,)).fetchone()
        return self._row_to_analysis(row) if row else None

    def get_analysis_detail(self, analysis_id: str) -> CaseDetail | None:
        analysis = self.get_analysis(analysis_id)
        if analysis is None:
            return None
        row = self._get_case_row(analysis.case_id)
        if row is None:
            return None
        recordings = self.list_recordings(analysis.case_id)
        linked_recording = self.get_recording(analysis.recording_id)
        analyses = self.list_analyses(analysis.case_id)
        return self._build_case(
            row=row,
            recording=linked_recording,
            recordings=recordings,
            analysis=analysis,
            analyses=analyses,
            model_comparisons=self.list_model_comparisons(analysis_id),
            segment_results=self.list_segment_results(analysis_id),
            intervals=self.list_high_risk_intervals(analysis_id),
            report=self.get_report_for_analysis(analysis_id),
        )

    def list_analyses(self, case_id: str) -> list[Analysis]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM clinical_analyses WHERE case_id = ? ORDER BY created_at DESC",
                (case_id,),
            ).fetchall()
        return [self._row_to_analysis(row) for row in rows]

    def list_segment_results(self, analysis_id: str) -> list[SegmentResult]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM segment_results WHERE analysis_id = ? ORDER BY segment_index ASC",
                (analysis_id,),
            ).fetchall()
        return [
            SegmentResult(
                id=row["id"],
                analysis_id=row["analysis_id"],
                segment_index=row["segment_index"],
                start_sec=row["start_sec"],
                end_sec=row["end_sec"],
                risk_score=row["risk_score"],
                risk_label=row["risk_label"],
                is_flagged=bool(row["is_flagged"]),
            )
            for row in rows
        ]

    def list_high_risk_intervals(self, analysis_id: str) -> list[HighRiskInterval]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM interval_results WHERE analysis_id = ? ORDER BY interval_index ASC",
                (analysis_id,),
            ).fetchall()
        return [
            HighRiskInterval(
                id=row["id"],
                analysis_id=row["analysis_id"],
                interval_index=row["interval_index"],
                start_sec=row["start_sec"],
                end_sec=row["end_sec"],
                mean_risk=row["mean_risk"],
                max_risk=row["max_risk"],
                flagged_segments_count=row["flagged_segments_count"],
                review_status=row["review_status"],
            )
            for row in rows
        ]

    def list_model_comparisons(self, analysis_id: str) -> list[ModelComparison]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM analysis_model_runs
                WHERE analysis_id = ?
                ORDER BY is_primary DESC, model_label ASC
                """,
                (analysis_id,),
            ).fetchall()
        return [
            ModelComparison(
                model_run_id=row["id"],
                analysis_id=row["analysis_id"],
                model_key=row["model_key"],
                model_label=row["model_label"],
                model_version=row["model_version"],
                checkpoint_path=row["checkpoint_path"],
                status=row["status"],
                backend_status=row["backend_status"],
                overall_risk=row["overall_risk"],
                review_priority=row["review_priority"],
                estimated_seizure_risk=row["estimated_seizure_risk"],
                max_risk_score=row["max_risk_score"],
                mean_risk_score=row["mean_risk_score"],
                flagged_segments_count=row["flagged_segments_count"] or 0,
                high_risk_intervals_count=row["high_risk_intervals_count"] or 0,
                confidence_score=row["confidence_score"],
                confidence_label=row["confidence_label"],
                agreement_score=row["agreement_score"],
                inference_time_seconds=row["inference_time_seconds"],
                failure_code=row["failure_code"],
                failure_message=row["failure_message"],
                is_primary=bool(row["is_primary"]),
            )
            for row in rows
        ]

    def get_report(self, report_id: str) -> ReportSummary | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    rep.id AS report_id,
                    rep.analysis_id,
                    rep.case_id,
                    rep.report_title,
                    rep.report_path,
                    rep.report_url,
                    rep.report_status,
                    rep.generated_at,
                    c.patient_id,
                    c.clinician_name
                FROM analysis_reports rep
                JOIN clinical_cases c ON c.id = rep.case_id
                WHERE rep.id = ?
                """,
                (report_id,),
            ).fetchone()
        return self._row_to_report(row) if row else None

    def get_report_for_analysis(self, analysis_id: str) -> ReportSummary | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    rep.id AS report_id,
                    rep.analysis_id,
                    rep.case_id,
                    rep.report_title,
                    rep.report_path,
                    rep.report_url,
                    rep.report_status,
                    rep.generated_at,
                    c.patient_id,
                    c.clinician_name
                FROM analysis_reports rep
                JOIN clinical_cases c ON c.id = rep.case_id
                WHERE rep.analysis_id = ?
                """,
                (analysis_id,),
            ).fetchone()
        return self._row_to_report(row) if row else None

    def _build_case(
        self,
        *,
        row: sqlite3.Row,
        recording: Recording | None,
        recordings: list[Recording],
        analysis: Analysis | None,
        analyses: list[Analysis],
        model_comparisons: list[ModelComparison],
        segment_results: list[SegmentResult],
        intervals: list[HighRiskInterval],
        report: ReportSummary | None,
    ) -> Case:
        return Case(
            case_id=row["id"],
            patient_id=row["patient_id"],
            clinician_name=row["clinician_name"],
            recording_date=date.fromisoformat(row["recording_date"]),
            notes=row["notes"],
            status=row["status"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            recording=recording,
            recordings=recordings,
            analysis=analysis,
            analyses=analyses,
            model_comparisons=model_comparisons,
            high_risk_intervals=intervals,
            segment_results=segment_results,
            report=report,
        )

    def _get_case_row(self, case_id: str) -> sqlite3.Row | None:
        with self._connect() as connection:
            return connection.execute("SELECT * FROM clinical_cases WHERE id = ?", (case_id,)).fetchone()

    def _row_to_recording(self, row: sqlite3.Row) -> Recording:
        mapped_channels = json.loads(row["mapped_channels_json"])
        conversion_messages = json.loads(row["conversion_messages_json"]) if "conversion_messages_json" in row.keys() else []
        derived_channels = json.loads(row["derived_channels_json"]) if "derived_channels_json" in row.keys() else []
        approximated_channels = json.loads(row["approximated_channels_json"]) if "approximated_channels_json" in row.keys() else []
        missing_channels = json.loads(row["missing_channels_json"]) if "missing_channels_json" in row.keys() else []
        validation_messages = json.loads(row["validation_messages_json"])
        return Recording(
            recording_id=row["id"],
            case_id=row["case_id"],
            file_name=row["file_name"],
            file_path=row["file_path"],
            file_type=row["file_type"],
            duration_sec=row["duration_sec"],
            sampling_rate=row["sampling_rate"],
            channel_count=row["channel_count"],
            channel_names=json.loads(row["channel_names_json"]),
            input_montage_type=row["input_montage_type"] if "input_montage_type" in row.keys() else "unsupported",
            conversion_status=row["conversion_status"] if "conversion_status" in row.keys() else "blocked",
            conversion_messages=conversion_messages,
            mapped_channels=mapped_channels,
            derived_channels=derived_channels,
            approximated_channels=approximated_channels,
            missing_channels=missing_channels,
            mapped_channel_count=row["mapped_channel_count"],
            validation_status=row["metadata_status"],
            validation_messages=validation_messages,
            uploaded_at=datetime.fromisoformat(row["uploaded_at"]),
        )

    def _row_to_analysis(self, row: sqlite3.Row) -> Analysis:
        return Analysis(
            analysis_id=row["id"],
            case_id=row["case_id"],
            recording_id=row["recording_id"],
            status=row["status"],
            model_version=row["model_version"],
            overall_risk=row["overall_risk"],
            review_priority=row["review_priority"],
            max_risk_score=row["max_risk_score"],
            mean_risk_score=row["mean_risk_score"],
            estimated_seizure_risk=row["estimated_seizure_risk"],
            flagged_segments_count=row["flagged_segments_count"] or 0,
            total_segments=row["total_segments"] or 0,
            high_risk_intervals_count=row["high_risk_intervals_count"] or 0,
            recording_duration_sec=row["recording_duration_sec"] or 0.0,
            clinical_summary=row["clinical_summary"],
            recommendation=row["recommendation"] if "recommendation" in row.keys() else None,
            interpretation=row["interpretation"],
            failure_code=row["failure_code"],
            failure_message=row["failure_message"],
            created_at=datetime.fromisoformat(row["created_at"]),
            report_generated=bool(row["report_generated"]),
        )

    def _row_to_report(self, row: sqlite3.Row) -> ReportSummary:
        return ReportSummary(
            report_id=row["report_id"],
            analysis_id=row["analysis_id"],
            case_id=row["case_id"],
            patient_id=row["patient_id"],
            clinician_name=row["clinician_name"],
            report_title=row["report_title"],
            report_path=row["report_path"],
            report_url=row["report_url"] or f"/reports/{row['report_id']}",
            report_status=row["report_status"],
            generated_at=datetime.fromisoformat(row["generated_at"]),
        )

    def _ensure_column(self, connection: sqlite3.Connection, table_name: str, column_name: str, definition: str) -> None:
        columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()}
        if column_name not in columns:
            connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

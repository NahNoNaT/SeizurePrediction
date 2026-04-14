from app.services.legacy_joblib import LegacyFeatureExtractorUnavailableError


def test_demo_health_reports_legacy_catalog(client):
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["task_type"] == "legacy_joblib_prediction"
    assert payload["model_count"] >= 1
    assert len(payload["models"]) >= 1


def test_legacy_health_reports_catalog_shape(client):
    response = client.get("/legacy/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["task_type"] == "legacy_joblib_prediction"
    assert "model_count" in payload
    assert "subjects" in payload
    assert "algorithms" in payload
    assert "feature_sets" in payload
    assert "counts_by_source" in payload
    assert "counts_by_subject" in payload
    assert "models" in payload
    assert payload["counts_by_source"]["universal_lopo"] >= 1
    assert payload["counts_by_source"]["loso_final"] >= 1
    assert len(payload["models"]) >= 1
    first_model = payload["models"][0]
    assert "model_id" in first_model
    assert "display_name" in first_model


def test_legacy_predict_returns_503_when_feature_extractor_missing(client):
    class MissingExtractorLegacyService:
        def catalog_summary(self):
            return {
                "model_count": 1,
                "subjects": [],
                "algorithms": ["LR"],
                "feature_sets": ["COMBINED"],
                "counts_by_source": {"universal_lopo": 1, "loso_final": 0},
                "counts_by_subject": {},
                "models": [],
            }

        def predict(self, *args, **kwargs):
            raise LegacyFeatureExtractorUnavailableError("feature extractor missing")

    client.app.state.legacy_joblib_service = MissingExtractorLegacyService()
    response = client.post(
        "/legacy/predict",
        data={"source": "universal_lopo", "feature_set": "COMBINED"},
        files={"file": ("sample.edf", b"dummy-edf-bytes", "application/octet-stream")},
    )

    assert response.status_code == 503
    payload = response.json()
    assert payload["code"] == "feature_extractor_unavailable"

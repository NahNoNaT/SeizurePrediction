def test_analysis_success_saves_retrievable_result(client, case_payload):
    case_response = client.post("/api/cases", json=case_payload)
    case_id = case_response.json()["case"]["case_id"]

    recording_response = client.post(
        f"/api/cases/{case_id}/recordings",
        files={"eeg_file": ("recording.edf", b"fake-edf-payload", "application/octet-stream")},
    )
    recording_id = recording_response.json()["recording"]["recording_id"]

    pending_response = client.get(f"/api/cases/{case_id}/analysis")
    assert pending_response.status_code == 200
    pending_payload = pending_response.json()
    assert pending_payload["status"] == "pending"
    assert pending_payload["timeline"] == []
    assert pending_payload["segments"] == []

    analysis_response = client.post(f"/api/recordings/{recording_id}/analyze")
    assert analysis_response.status_code == 200
    payload = analysis_response.json()
    assert payload["status"] == "completed"
    assert payload["clinical_summary"]["risk_level"] in {"low", "moderate", "high"}
    assert len(payload["model_comparisons"]) == 4
    assert len(payload["timeline"]) >= 1

    analyses_response = client.get(f"/api/cases/{case_id}/analyses")
    analysis_id = analyses_response.json()["analyses"][0]["analysis_id"]
    detail_response = client.get(f"/api/analyses/{analysis_id}")
    assert detail_response.status_code == 200
    assert detail_response.json()["status"] == "completed"


def test_analysis_failure_returns_clear_status_and_message(failing_client, case_payload):
    case_response = failing_client.post("/api/cases", json=case_payload)
    case_id = case_response.json()["case"]["case_id"]

    recording_response = failing_client.post(
        f"/api/cases/{case_id}/recordings",
        files={"eeg_file": ("recording.edf", b"fake-edf-payload", "application/octet-stream")},
    )
    recording_id = recording_response.json()["recording"]["recording_id"]

    analysis_response = failing_client.post(f"/api/recordings/{recording_id}/analyze")
    assert analysis_response.status_code == 200
    payload = analysis_response.json()
    assert payload["status"] == "failed"
    assert payload["error"] == "Analysis could not be completed because the model is unavailable."
    assert payload["timeline"] == []
    assert payload["segments"] == []


def test_bipolar_analysis_success_runs_after_conversion(bipolar_client, case_payload):
    case_response = bipolar_client.post("/api/cases", json=case_payload)
    case_id = case_response.json()["case"]["case_id"]

    recording_response = bipolar_client.post(
        f"/api/cases/{case_id}/recordings",
        files={"eeg_file": ("recording.edf", b"fake-edf-payload", "application/octet-stream")},
    )
    recording = recording_response.json()["recording"]
    assert recording["input_montage_type"] == "bipolar"
    assert recording["conversion_status"] == "converted"

    analysis_response = bipolar_client.post(f"/api/recordings/{recording['recording_id']}/analyze")
    assert analysis_response.status_code == 200
    payload = analysis_response.json()
    assert payload["status"] == "completed"
    assert len(payload["model_comparisons"]) == 4


def test_bipolar_analysis_blocked_returns_clear_validation_message(blocked_bipolar_client, case_payload):
    case_response = blocked_bipolar_client.post("/api/cases", json=case_payload)
    case_id = case_response.json()["case"]["case_id"]

    recording_response = blocked_bipolar_client.post(
        f"/api/cases/{case_id}/recordings",
        files={"eeg_file": ("recording.edf", b"fake-edf-payload", "application/octet-stream")},
    )
    recording_id = recording_response.json()["recording"]["recording_id"]

    analysis_response = blocked_bipolar_client.post(f"/api/recordings/{recording_id}/analyze")
    assert analysis_response.status_code == 409
    payload = analysis_response.json()
    assert payload["code"] == "validation_failed"

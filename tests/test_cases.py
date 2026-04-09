def test_create_and_list_cases(client, case_payload):
    create_response = client.post("/api/cases", json=case_payload)

    assert create_response.status_code == 201
    created = create_response.json()
    assert created["case"]["status"] == "NEW"

    list_response = client.get("/api/cases")
    assert list_response.status_code == 200
    cases = list_response.json()
    assert len(cases) == 1
    assert cases[0]["patient_id"] == case_payload["patient_id"]


def test_case_detail_page_renders_pending_state(client, case_payload):
    case_response = client.post("/api/cases", json=case_payload)
    case_id = case_response.json()["case"]["case_id"]

    client.post(
        f"/api/cases/{case_id}/recordings",
        files={"eeg_file": ("recording.edf", b"fake-edf-payload", "application/octet-stream")},
    )

    detail_response = client.get(f"/cases/{case_id}")
    assert detail_response.status_code == 200
    assert "Clinical findings will appear here once the analysis is complete." in detail_response.text
    assert "Timeline pending" in detail_response.text
    assert "EEG recording preview" in detail_response.text


def test_case_detail_page_renders_failed_state(failing_client, case_payload):
    case_response = failing_client.post("/api/cases", json=case_payload)
    case_id = case_response.json()["case"]["case_id"]

    recording_response = failing_client.post(
        f"/api/cases/{case_id}/recordings",
        files={"eeg_file": ("recording.edf", b"fake-edf-payload", "application/octet-stream")},
    )
    recording_id = recording_response.json()["recording"]["recording_id"]

    failing_client.post(f"/api/recordings/{recording_id}/analyze")
    detail_response = failing_client.get(f"/cases/{case_id}")
    assert detail_response.status_code == 200
    assert "Analysis could not be completed." in detail_response.text
    assert "Please review the technical issue before retrying." in detail_response.text


def test_case_detail_page_renders_model_comparison_table(client, case_payload):
    case_response = client.post("/api/cases", json=case_payload)
    case_id = case_response.json()["case"]["case_id"]

    recording_response = client.post(
        f"/api/cases/{case_id}/recordings",
        files={"eeg_file": ("recording.edf", b"fake-edf-payload", "application/octet-stream")},
    )
    recording_id = recording_response.json()["recording"]["recording_id"]

    client.post(f"/api/recordings/{recording_id}/analyze")
    detail_response = client.get(f"/cases/{case_id}")
    assert detail_response.status_code == 200
    assert "Per-model analysis comparison" in detail_response.text
    assert "Model 4" in detail_response.text


def test_case_detail_page_renders_bipolar_conversion_state(bipolar_client, case_payload):
    case_response = bipolar_client.post("/api/cases", json=case_payload)
    case_id = case_response.json()["case"]["case_id"]

    recording_response = bipolar_client.post(
        f"/api/cases/{case_id}/recordings",
        files={"eeg_file": ("recording.edf", b"fake-edf-payload", "application/octet-stream")},
    )
    recording_id = recording_response.json()["recording"]["recording_id"]
    bipolar_client.post(f"/api/recordings/{recording_id}/analyze")

    detail_response = bipolar_client.get(f"/cases/{case_id}")
    assert detail_response.status_code == 200
    assert "Montage and conversion status" in detail_response.text
    assert "Bipolar" in detail_response.text
    assert "Converted" in detail_response.text


def test_case_detail_page_renders_blocked_bipolar_state_with_preview_section(blocked_bipolar_client, case_payload):
    case_response = blocked_bipolar_client.post("/api/cases", json=case_payload)
    case_id = case_response.json()["case"]["case_id"]

    blocked_bipolar_client.post(
        f"/api/cases/{case_id}/recordings",
        files={"eeg_file": ("recording.edf", b"fake-edf-payload", "application/octet-stream")},
    )

    detail_response = blocked_bipolar_client.get(f"/cases/{case_id}")
    assert detail_response.status_code == 200
    assert "EEG recording preview" in detail_response.text
    assert "Analysis blocked before model comparison" in detail_response.text

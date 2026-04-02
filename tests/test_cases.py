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

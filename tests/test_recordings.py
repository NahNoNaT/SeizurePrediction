def test_upload_recording_moves_case_to_validated(client, case_payload):
    case_response = client.post("/api/cases", json=case_payload)
    case_id = case_response.json()["case"]["case_id"]

    upload_response = client.post(
        f"/api/cases/{case_id}/recordings",
        files={"eeg_file": ("recording.edf", b"fake-edf-payload", "application/octet-stream")},
    )

    assert upload_response.status_code == 201
    recording = upload_response.json()["recording"]
    assert recording["validation_status"] == "VALIDATED"

    detail_response = client.get(f"/api/cases/{case_id}")
    assert detail_response.status_code == 200
    assert detail_response.json()["status"] == "VALIDATED"


def test_recording_preview_returns_limited_waveform_window(client, case_payload):
    case_response = client.post("/api/cases", json=case_payload)
    case_id = case_response.json()["case"]["case_id"]

    upload_response = client.post(
        f"/api/cases/{case_id}/recordings",
        files={"eeg_file": ("recording.edf", b"fake-edf-payload", "application/octet-stream")},
    )
    recording_id = upload_response.json()["recording"]["recording_id"]

    preview_response = client.get(
        f"/api/recordings/{recording_id}/preview",
        params={"start_sec": 120, "duration_sec": 30, "channels": "Fp1-F7,F7-T3,Missing"},
    )

    assert preview_response.status_code == 200
    payload = preview_response.json()
    assert payload["recording_id"] == recording_id
    assert payload["channels"] == ["Fp1-F7", "F7-T3"]
    assert payload["missing_channels"] == ["Missing"]
    assert payload["duration_sec"] == 30
    assert len(payload["times"]) > 0
    assert len(payload["signals"]) == 2


def test_bipolar_recording_upload_is_saved_and_reports_conversion_metadata(blocked_bipolar_client, case_payload):
    case_response = blocked_bipolar_client.post("/api/cases", json=case_payload)
    case_id = case_response.json()["case"]["case_id"]

    upload_response = blocked_bipolar_client.post(
        f"/api/cases/{case_id}/recordings",
        files={"eeg_file": ("recording.edf", b"fake-edf-payload", "application/octet-stream")},
    )

    assert upload_response.status_code == 201
    recording = upload_response.json()["recording"]
    assert recording["input_montage_type"] == "bipolar"
    assert recording["conversion_status"] == "blocked"
    assert recording["validation_status"] == "BLOCKED"

    preview_response = blocked_bipolar_client.get(f"/api/recordings/{recording['recording_id']}/preview")
    assert preview_response.status_code == 200
    payload = preview_response.json()
    assert len(payload["channels"]) > 0
    assert len(payload["signals"]) > 0

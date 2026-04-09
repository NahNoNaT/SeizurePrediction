def test_report_detail_is_available_after_successful_analysis(client, case_payload):
    case_response = client.post("/api/cases", json=case_payload)
    case_id = case_response.json()["case"]["case_id"]

    recording_response = client.post(
        f"/api/cases/{case_id}/recordings",
        files={"eeg_file": ("recording.edf", b"fake-edf-payload", "application/octet-stream")},
    )
    recording_id = recording_response.json()["recording"]["recording_id"]

    client.post(f"/api/recordings/{recording_id}/analyze")
    case_detail_response = client.get(f"/api/cases/{case_id}")
    report_id = case_detail_response.json()["report"]["report_id"]

    report_response = client.get(f"/api/reports/{report_id}")
    assert report_response.status_code == 200
    report_payload = report_response.json()
    assert report_payload["report"]["report_id"] == report_id
    assert len(report_payload["model_comparisons"]) == 4

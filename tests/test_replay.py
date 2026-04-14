def test_replay_page_renders_controls(client):
    response = client.get("/replay")

    assert response.status_code == 200
    assert "EDF replay through legacy models" in response.text
    assert "Start Replay" in response.text
    assert "replayTimelineChart" in response.text


def test_replay_api_upload_start_and_stop(client):
    upload_response = client.post(
        "/api/replay/upload",
        files={"eeg_file": ("recording.edf", b"fake-edf-payload", "application/octet-stream")},
    )

    assert upload_response.status_code == 201
    upload_payload = upload_response.json()
    assert upload_payload["status"] == "uploaded"
    assert len(upload_payload["available_channels"]) == 3

    session_id = upload_payload["session_id"]
    start_response = client.post(
        f"/api/replay/{session_id}/start",
        json={"window_sec": 12, "hop_sec": 3, "replay_speed": 4},
    )
    assert start_response.status_code == 200
    start_payload = start_response.json()
    assert start_payload["status"] == "running"
    assert start_payload["processed_windows"] == 1
    assert len(start_payload["timeline"]) == 1

    state_response = client.get(f"/api/replay/{session_id}")
    assert state_response.status_code == 200
    state_payload = state_response.json()
    assert state_payload["latest_top_channel"] == "Fp1-F7"
    assert state_payload["latest_risk_score"] == 0.83

    stop_response = client.post(f"/api/replay/{session_id}/stop")
    assert stop_response.status_code == 200
    assert stop_response.json()["status"] == "stopped"

def test_health_endpoint_returns_backend_status(client):
    response = client.get("/api/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "backend_status" in payload

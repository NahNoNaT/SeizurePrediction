from fastapi.testclient import TestClient

from conftest import build_test_app


def test_first_registration_creates_admin_and_allows_dashboard(tmp_path):
    app = build_test_app(tmp_path)
    app.state.runtime_config.auth_enabled = True
    try:
        with TestClient(app) as client:
            register_response = client.post(
                "/auth/register",
                data={
                    "username": "admin",
                    "full_name": "Main Admin",
                    "password": "StrongPass123",
                    "role": "viewer",
                },
                follow_redirects=False,
            )
            assert register_response.status_code == 303
            assert register_response.headers["location"].startswith("/dashboard")

            dashboard_response = client.get("/dashboard")
            assert dashboard_response.status_code == 200
            assert "Signed in as Main Admin (admin)" in dashboard_response.text
    finally:
        app.state.runtime_config.auth_enabled = False


def test_viewer_cannot_create_case_via_api(tmp_path, case_payload):
    app = build_test_app(tmp_path)
    app.state.runtime_config.auth_enabled = True
    try:
        with TestClient(app) as client:
            client.post(
                "/auth/register",
                data={
                    "username": "admin",
                    "full_name": "Main Admin",
                    "password": "StrongPass123",
                    "role": "admin",
                },
            )
            client.post(
                "/auth/register",
                data={
                    "username": "viewer1",
                    "full_name": "Viewer One",
                    "password": "StrongPass123",
                    "role": "viewer",
                },
            )
            client.post("/auth/logout")

            client.post(
                "/auth/login",
                data={
                    "username": "viewer1",
                    "password": "StrongPass123",
                    "next_path": "/dashboard",
                },
            )
            response = client.post("/api/cases", json=case_payload)
            assert response.status_code == 403
            assert response.json()["code"] == "forbidden"
    finally:
        app.state.runtime_config.auth_enabled = False


def test_public_registration_allows_new_visitor_signup_as_viewer(tmp_path):
    app = build_test_app(tmp_path)
    app.state.runtime_config.auth_enabled = True
    app.state.runtime_config.allow_public_registration = True
    try:
        with TestClient(app) as client:
            client.post(
                "/auth/register",
                data={
                    "username": "admin",
                    "full_name": "Main Admin",
                    "password": "StrongPass123",
                    "role": "admin",
                },
            )
            client.post("/auth/logout")

            register_response = client.post(
                "/auth/register",
                data={
                    "username": "newviewer",
                    "full_name": "New Viewer",
                    "password": "StrongPass123",
                    "role": "admin",
                },
                follow_redirects=False,
            )
            assert register_response.status_code == 303
            assert register_response.headers["location"].startswith("/auth/register")

            login_response = client.post(
                "/auth/login",
                data={
                    "username": "newviewer",
                    "password": "StrongPass123",
                    "next_path": "/dashboard",
                },
                follow_redirects=False,
            )
            assert login_response.status_code == 303
            assert login_response.headers["location"] == "/dashboard"

            dashboard_response = client.get("/dashboard")
            assert "Signed in as New Viewer (viewer)" in dashboard_response.text
    finally:
        app.state.runtime_config.allow_public_registration = False
        app.state.runtime_config.auth_enabled = False

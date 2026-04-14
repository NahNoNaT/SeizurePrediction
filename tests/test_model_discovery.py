from __future__ import annotations

from pathlib import Path

from app.config import RuntimeConfig, runtime_config
from conftest import build_test_app


def test_runtime_config_auto_discovers_checkpoints_from_default_directory(tmp_path: Path):
    checkpoint_dir = tmp_path / "models" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "alpha.pt").write_bytes(b"")
    (checkpoint_dir / "beta.ckpt").write_bytes(b"")
    (checkpoint_dir / "ignore.txt").write_text("not a model", encoding="utf-8")

    config = RuntimeConfig(
        checkpoint_path=None,
        checkpoint_paths=(),
        checkpoint_directory_name="models/checkpoints",
        auto_discover_checkpoints=True,
    )

    assert config.configured_checkpoint_paths(tmp_path) == (
        "models/checkpoints/alpha.pt",
        "models/checkpoints/beta.ckpt",
    )
    assert config.resolved_checkpoint_paths(tmp_path) == (
        checkpoint_dir / "alpha.pt",
        checkpoint_dir / "beta.ckpt",
    )


def test_model_info_reports_auto_discovered_checkpoint_count(tmp_path: Path):
    checkpoint_dir = tmp_path / "models" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "fold-1.pt").write_bytes(b"")
    (checkpoint_dir / "fold-2.pth").write_bytes(b"")

    original_checkpoint_path = runtime_config.checkpoint_path
    original_checkpoint_paths = runtime_config.checkpoint_paths
    original_directory_name = runtime_config.checkpoint_directory_name
    original_auto_discovery = runtime_config.auto_discover_checkpoints

    try:
        runtime_config.checkpoint_path = None
        runtime_config.checkpoint_paths = ()
        runtime_config.checkpoint_directory_name = "models/checkpoints"
        runtime_config.auto_discover_checkpoints = True

        app = build_test_app(tmp_path)
        app.state.project_root = tmp_path

        from fastapi.testclient import TestClient

        with TestClient(app) as client:
            response = client.get("/api/model/info")

        assert response.status_code == 200
        payload = response.json()
        assert payload["configured_model_count"] == 2
    finally:
        runtime_config.checkpoint_path = original_checkpoint_path
        runtime_config.checkpoint_paths = original_checkpoint_paths
        runtime_config.checkpoint_directory_name = original_directory_name
        runtime_config.auto_discover_checkpoints = original_auto_discovery

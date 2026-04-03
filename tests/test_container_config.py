"""Tests for Docker and compose configuration validity."""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestDockerInfra:
    """Verify all Docker serving files exist and have expected content."""

    def test_dockerfile_serve_exists(self):
        assert (PROJECT_ROOT / "Dockerfile.serve").is_file()

    def test_docker_compose_serve_exists(self):
        assert (PROJECT_ROOT / "docker-compose.serve.yml").is_file()

    def test_env_serve_exists(self):
        assert (PROJECT_ROOT / ".env.serve").is_file()

    def test_dockerfile_cuda_exists(self):
        assert (PROJECT_ROOT / "docker" / "Dockerfile.cuda").is_file()

    def test_dockerfile_mlx_exists(self):
        assert (PROJECT_ROOT / "docker" / "Dockerfile.mlx").is_file()

    def test_docker_compose_yaml_exists(self):
        assert (PROJECT_ROOT / "docker" / "docker-compose.yaml").is_file()

    def test_dockerfile_serve_uses_anima_base(self):
        content = (PROJECT_ROOT / "Dockerfile.serve").read_text()
        assert "anima-serve" in content

    def test_dockerfile_serve_has_healthcheck(self):
        content = (PROJECT_ROOT / "Dockerfile.serve").read_text()
        assert "HEALTHCHECK" in content

    def test_env_serve_has_module_name(self):
        content = (PROJECT_ROOT / ".env.serve").read_text()
        assert "ANIMA_MODULE_NAME=def-uavdetr" in content

    def test_compose_serve_has_profiles(self):
        content = (PROJECT_ROOT / "docker-compose.serve.yml").read_text()
        for profile in ("serve", "ros2", "api", "test"):
            assert profile in content, f"Missing profile: {profile}"

    def test_dockerfile_cuda_uses_cu128(self):
        content = (PROJECT_ROOT / "docker" / "Dockerfile.cuda").read_text()
        assert "cu128" in content

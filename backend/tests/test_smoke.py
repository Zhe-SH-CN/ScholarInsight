"""冒烟测试：确认 FastAPI app 能启动且健康检查通过。"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from cg.main import create_app, lifespan
from cg.settings import Settings, get_settings


def _isolated_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("CG_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    return create_app()


@pytest.mark.asyncio
async def test_health(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    test_app = _isolated_app(monkeypatch, tmp_path)
    async with lifespan(test_app):
        transport = httpx.ASGITransport(app=test_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            r = await client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    test_app = _isolated_app(monkeypatch, tmp_path)
    async with lifespan(test_app):
        transport = httpx.ASGITransport(app=test_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            r = await client.get("/")
        assert r.status_code == 200
        body = r.json()
        assert body["name"] == "ScholarInsight API"
        assert "version" in body
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_runs_require_login(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    test_app = _isolated_app(monkeypatch, tmp_path)
    async with lifespan(test_app):
        transport = httpx.ASGITransport(app=test_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            r = await client.get("/api/runs")
        assert r.status_code == 401
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_login_allows_runs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CG_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CG_AUTH_USERNAME", "test-user")
    monkeypatch.setenv("CG_AUTH_PASSWORD", "test-password")
    monkeypatch.setenv("CG_AUTH_SECRET", "test-session-secret")
    get_settings.cache_clear()
    test_app = create_app()
    try:
        async with lifespan(test_app):
            transport = httpx.ASGITransport(app=test_app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                r = await client.post("/api/login", json={"username": "test-user", "password": "test-password"})
                assert r.status_code == 200
                assert r.json()["authenticated"] is True

                runs = await client.get("/api/runs")
                assert runs.status_code == 200
    finally:
        get_settings.cache_clear()


def test_ark_llm_settings_select_endpoint_and_key() -> None:
    settings = Settings(
        cg_llm_provider="ark",
        cg_llm_model="ep-example",
        ark_api_key="ark-test-key",
    )

    assert settings.active_llm_api_key == "ark-test-key"
    assert settings.active_llm_base_url == "https://ark.cn-beijing.volces.com/api/v3"

from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from app.api.agent import graph
from app.main import app

client = TestClient(app)


def test_run_agent_uses_registered_route_and_returns_frontend_shape(
    monkeypatch,
):
    fake_result = {
        "diagnosis": {"summary": "检测正常"},
        "trace": [{"event": "diagnosis", "data": {}}],
    }
    invoke = AsyncMock(return_value=fake_result)
    monkeypatch.setattr(graph, "ainvoke", invoke)

    response = client.post(
        "/agent/run",
        json={
            "message": "检查设备",
            "device_id": "device-001",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["session_id"]
    assert data["diagnosis"] == fake_result["diagnosis"]
    assert data["trace"] == fake_result["trace"]
    assert data["pending_tool"] is None

    assert invoke.await_args is not None
    initial_state = invoke.await_args.args[0]
    assert initial_state["device_id"] == "device-001"


def test_resume_agent_returns_pending_approval_shape(monkeypatch):
    pending_tool = {
        "tool_name": "restart_device",
        "risk_level": "high",
    }
    invoke = AsyncMock(
        return_value={
            "__interrupt__": [
                SimpleNamespace(value=pending_tool),
            ],
            "trace": [],
        }
    )
    monkeypatch.setattr(graph, "ainvoke", invoke)

    response = client.post(
        "/agent/thread-001/resume",
        json={"approved": True},
    )

    assert response.status_code == 200
    assert response.json() == {
        "session_id": "thread-001",
        "status": "waiting_for_approval",
        "pending_tool": pending_tool,
        "diagnosis": None,
        "trace": [],
    }

from __future__ import annotations

from fastapi.testclient import TestClient

from server import app as root_app


def test_root_reset_and_grader_accept_missing_or_null_body() -> None:
    client = TestClient(root_app)

    reset_without_body = client.post("/reset")
    assert reset_without_body.status_code == 200
    assert "observation" in reset_without_body.json()

    reset_with_null_body = client.post(
        "/reset",
        content="null",
        headers={"content-type": "application/json"},
    )
    assert reset_with_null_body.status_code == 200
    assert "observation" in reset_with_null_body.json()

    grader_without_body = client.post("/grader")
    assert grader_without_body.status_code == 200
    grader_payload = grader_without_body.json()
    assert set(grader_payload) == {"score", "reward"}
    assert 0.0 < float(grader_payload["score"]) < 1.0
    assert 0.0 < float(grader_payload["reward"]) < 1.0

    grader_with_null_body = client.post(
        "/grader",
        content="null",
        headers={"content-type": "application/json"},
    )
    assert grader_with_null_body.status_code == 200
    grader_null_payload = grader_with_null_body.json()
    assert set(grader_null_payload) == {"score", "reward"}
    assert 0.0 < float(grader_null_payload["score"]) < 1.0
    assert 0.0 < float(grader_null_payload["reward"]) < 1.0


def test_root_tasks_and_state_follow_openenv_style_ids() -> None:
    client = TestClient(root_app)

    tasks_response = client.get("/tasks")
    assert tasks_response.status_code == 200
    payload = tasks_response.json()
    assert "tasks" in payload

    task_ids = [task["id"] for task in payload["tasks"]]
    assert task_ids == ["alert_prioritization", "threat_detection", "incident_response"]

    for task in payload["tasks"]:
        assert task["grade_endpoint"].startswith("/grade/")
        assert task["grade_endpoint"].endswith(task["id"])
        assert task["grader"]["endpoint"] == task["grade_endpoint"]
        assert task["internal_task_id"] in {"task_1", "task_2", "task_3"}

    reset_response = client.post("/reset", json={"task_id": "alert_prioritization", "scenario_index": 0})
    assert reset_response.status_code == 200
    reset_payload = reset_response.json()
    assert reset_payload["task_id"] == "alert_prioritization"
    assert reset_payload["internal_task_id"] == "task_1"

    state_response = client.get("/state", params={"task_id": "threat_detection", "scenario_index": 0})
    assert state_response.status_code == 200
    state_payload = state_response.json()
    assert state_payload["task_id"] == "threat_detection"
    assert state_payload["internal_task_id"] == "task_2"

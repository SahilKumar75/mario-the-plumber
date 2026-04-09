"""Local validator-facing smoke test for Mario task and grader discovery."""

from __future__ import annotations

from graders import grade_episode
from tasks.definitions import list_tasks


def main() -> None:
    tasks = list_tasks()
    assert len(tasks) >= 3, "Need at least 3 tasks"

    live_successes = 0
    for task in tasks:
        payload = grade_episode(task.id, split="eval", seed=42)
        assert 0.0 < float(payload["score"]) < 1.0, f"{task.id} score must be strictly inside (0, 1)"
        assert 0.0 < float(payload["reward"]) < 1.0, f"{task.id} reward must be strictly inside (0, 1)"
        if payload.get("success"):
            live_successes += 1

    assert live_successes >= 3, "Need at least 3 tasks whose live graders reach success"
    print("Validation checks passed.")


if __name__ == "__main__":
    main()

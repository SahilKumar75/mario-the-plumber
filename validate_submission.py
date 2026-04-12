"""Local validator-facing smoke test for Mario task and grader discovery."""

from __future__ import annotations

from grader import grade_episode
from tasks.definitions import list_tasks


MIN_SCORE_BOUND = 0.01
MAX_SCORE_BOUND = 0.99


def main() -> None:
    tasks = list_tasks()
    assert len(tasks) >= 3, "Need at least 3 tasks"

    live_successes = 0
    for task in tasks:
        payload = grade_episode(task.id, split="eval", seed=42)
        assert MIN_SCORE_BOUND <= float(payload["score"]) <= MAX_SCORE_BOUND, (
            f"{task.id} score must be inside [{MIN_SCORE_BOUND}, {MAX_SCORE_BOUND}]"
        )
        assert MIN_SCORE_BOUND <= float(payload["reward"]) <= MAX_SCORE_BOUND, (
            f"{task.id} reward must be inside [{MIN_SCORE_BOUND}, {MAX_SCORE_BOUND}]"
        )
        if payload.get("success"):
            live_successes += 1

    assert live_successes >= 3, "Need at least 3 tasks whose live graders reach success"
    print("Validation checks passed.")


if __name__ == "__main__":
    main()

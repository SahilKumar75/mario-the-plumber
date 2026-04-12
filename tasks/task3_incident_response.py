"""Compatibility wrapper for Task 3 under openenv naming.

Mario logic remains ETL-based; this file only bridges naming/layout.
"""

from __future__ import annotations

from grader import grade_episode
from tasks.definitions import build_task_definition


COMPAT_TASK_ID = "incident_response"
MARIO_TASK_ID = "task_3"


def get_task_definition():
    return build_task_definition(MARIO_TASK_ID)


def grade(*, episode_id: str | None = None, seed: int = 42, split: str = "eval") -> dict[str, object]:
    return grade_episode(MARIO_TASK_ID, episode_id=episode_id, seed=seed, split=split)


__all__ = ["COMPAT_TASK_ID", "MARIO_TASK_ID", "get_task_definition", "grade"]

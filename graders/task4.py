"""Task 4 grader export."""

from graders.runtime import grade_episode


def grade(*, episode_id: str | None = None, seed: int = 42, split: str = "eval") -> dict[str, object]:
    return grade_episode("task_4", episode_id=episode_id, seed=seed, split=split)

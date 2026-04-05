from __future__ import annotations

import numpy as np

from server.scenarios import (
    Scenario,
    generate_task1,
    generate_task2,
    generate_task3,
    generate_task4,
    generate_task5,
)


def generate_scenario(
    task_id: int,
    seed: int | None = None,
    split: str = "train",
) -> Scenario:
    rng = np.random.default_rng(seed)
    if task_id == 1:
        return generate_task1(rng, seed, split)
    if task_id == 2:
        return generate_task2(rng, seed, split)
    if task_id == 3:
        return generate_task3(rng, seed, split)
    if task_id == 4:
        return generate_task4(rng, seed, split)
    if task_id == 5:
        return generate_task5(rng, seed, split)
    raise ValueError(f"Unsupported task_id: {task_id}")

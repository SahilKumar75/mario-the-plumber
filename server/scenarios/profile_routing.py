from __future__ import annotations


def _family_selector(seed: int, task_id: int) -> int:
    """Deterministic selector that avoids direct odd/even seed routing."""

    return (seed * seed + 3 * seed + task_id) % 7


def _profile_index(seed: int, task_id: int, profile_count: int) -> int:
    if profile_count <= 0:
        raise ValueError("profile_count must be positive")
    mixed = (seed * 17 + task_id * 31 + seed * seed * 7) % (profile_count * 11)
    return mixed % profile_count


def select_eval_profile(
    *,
    seed: int,
    task_id: int,
    familiar_profiles: list[str],
    heldout_profiles: list[str],
) -> str:
    """Route eval seeds across familiar and held-out profile families deterministically."""

    use_heldout = _family_selector(seed, task_id) in {0, 2, 5}
    profile_pool = heldout_profiles if use_heldout else familiar_profiles
    return profile_pool[_profile_index(seed, task_id, len(profile_pool))]

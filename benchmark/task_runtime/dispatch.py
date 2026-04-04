from __future__ import annotations

try:
    from . import task3, task4, task5
except ImportError:
    from benchmark.task_runtime import task3, task4, task5


def subgoal_progress_map(env) -> dict[str, bool]:
    if env._task_id == 3:
        return task3.subgoal_progress_map(env)
    if env._task_id == 4:
        return task4.subgoal_progress_map(env)
    if env._task_id == 5:
        return task5.subgoal_progress_map(env)
    return {}


def dependency_health_summary(env, trace_summary: dict[str, str]) -> dict[str, str]:
    if env._task_id == 3:
        return task3.dependency_health_summary(env, trace_summary)
    if env._task_id == 4:
        return task4.dependency_health_summary(env, trace_summary)
    if env._task_id == 5:
        return task5.dependency_health_summary(env, trace_summary)
    return {"single_table_state": "stable" if not env._recent_errors else "repair_required"}


def runtime_errors(env) -> list[str]:
    if env._task_id == 3:
        return task3.runtime_errors(env)
    if env._task_id == 4:
        return task4.runtime_errors(env)
    if env._task_id == 5:
        return task5.runtime_errors(env)
    return []

from __future__ import annotations

from benchmark.inspection import structural_mismatch_errors
from benchmark.observation_support import dependency_alerts, orchestration_alerts
from benchmark.task_runtime.dispatch import runtime_errors


def refresh_errors(env) -> None:
    errors = structural_mismatch_errors(env)
    errors.extend(runtime_errors(env))
    errors.extend(dependency_alerts(env))
    errors.extend(orchestration_alerts(env))
    env._recent_errors = errors[:6]

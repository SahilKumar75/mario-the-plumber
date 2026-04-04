from __future__ import annotations

try:
    from ..grading import calculation_mismatch_count
    from ..observation_support import dependency_alerts
    from ..actions.validation import table_has_structural_issues
except ImportError:
    from benchmark.grading import calculation_mismatch_count
    from benchmark.observation_support import dependency_alerts
    from benchmark.actions.validation import table_has_structural_issues


def subgoal_progress_map(env) -> dict[str, bool]:
    return {
        "repair_customers": not table_has_structural_issues(env, "customers"),
        "repair_products": not table_has_structural_issues(env, "products"),
        "repair_orders": not table_has_structural_issues(env, "orders"),
        "restore_dependency_consistency": calculation_mismatch_count(env._tables["orders"], env._tables["products"]) == 0,
        "commit_pipeline": bool(env._state.done and env._state.success),
    }


def dependency_health_summary(env, trace_summary: dict[str, str]) -> dict[str, str]:
    return {
        **trace_summary,
        "customer_contract": "stable" if not table_has_structural_issues(env, "customers") else "repair_required",
        "product_contract": "stable" if not table_has_structural_issues(env, "products") else "repair_required",
        "order_dependency": "consistent" if not dependency_alerts(env) else "cascading_breakage",
    }


def runtime_errors(env) -> list[str]:
    return []

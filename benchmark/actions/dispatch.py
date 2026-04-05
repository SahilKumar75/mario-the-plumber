from __future__ import annotations

import json

from benchmark.action_metadata import ACTION_NAMES, PARAMETER_ACTIONS
from benchmark.actions.orchestration import (
    commit_changes,
    prioritize_incremental_batch,
    refresh_downstream_summary,
    scale_resources,
)
from benchmark.actions.transforms import (
    cast_column,
    deduplicate_current_table,
    drop_outliers,
    fill_with_statistic,
    handle_inspect_schema,
    normalize_string_value,
    validate_parameter_action,
)
from benchmark.runtime_state import current_frame, current_table


def apply_action(env, action) -> str:
    if action.action_id not in ACTION_NAMES:
        raise ValueError("unknown action_id")

    if action.action_id == 0:
        return handle_inspect_schema(env, action)
    if action.action_id == 1:
        return "\n".join(env._recent_errors) if env._recent_errors else "No errors detected."
    if action.action_id == 2:
        rows = current_table(env).head(5).to_dict(orient="records")
        return json.dumps(rows, default=str)
    if action.action_id in PARAMETER_ACTIONS and not action.target_column:
        raise ValueError("missing required parameter target_column")
    if (
        action.action_id in PARAMETER_ACTIONS
        and action.action_id not in {12, 13}
        and action.target_column not in current_frame(env).columns
    ):
        raise ValueError(f"target column '{action.target_column}' is not present in the active table")
    if action.action_id in {12, 13}:
        validate_parameter_action(action, list(current_frame(env).columns))

    if action.action_id == 3:
        fill_with_statistic(env, action.target_column, "mean")
    elif action.action_id == 4:
        fill_with_statistic(env, action.target_column, "median")
    elif action.action_id == 5:
        current_frame(env)[action.target_column] = (
            current_frame(env)[action.target_column].ffill().bfill()
        )
    elif action.action_id == 6:
        current = current_frame(env)
        env._tables[env._state.active_table] = current[current[action.target_column].notna()].reset_index(
            drop=True
        )
    elif action.action_id == 7:
        cast_column(env, action.target_column, "int64")
    elif action.action_id == 8:
        cast_column(env, action.target_column, "float64")
    elif action.action_id == 9:
        current_frame(env)[action.target_column] = current_frame(env)[action.target_column].map(
            lambda value: normalize_string_value(env, value, action.target_column)
        )
    elif action.action_id == 10:
        env._tables[env._state.active_table] = deduplicate_current_table(env)
    elif action.action_id == 11:
        drop_outliers(env, action.target_column)
    elif action.action_id == 12:
        env._tables[env._state.active_table] = current_frame(env).rename(
            columns={action.target_column: action.new_name}
        )
        pending_orders = env._scenario_meta.get("pending_orders")
        if hasattr(pending_orders, "rename") and action.target_column in pending_orders.columns:
            env._scenario_meta["pending_orders"] = pending_orders.rename(
                columns={action.target_column: action.new_name}
            )
    elif action.action_id == 13:
        env._tables[env._state.active_table] = current_frame(env)[action.column_order].copy()
    elif action.action_id == 14:
        return "\n".join(env._recent_errors) if env._recent_errors else "Schema validation passed."
    elif action.action_id == 15:
        commit_changes(env)
        return "Changes committed."
    elif action.action_id == 16:
        return scale_resources(env, up=True)
    elif action.action_id == 17:
        return scale_resources(env, up=False)
    elif action.action_id == 18:
        return prioritize_incremental_batch(env)
    elif action.action_id == 19:
        return refresh_downstream_summary(env)

    return ""

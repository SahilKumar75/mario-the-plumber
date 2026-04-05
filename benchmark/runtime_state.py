from __future__ import annotations


def current_frame(env):
    table_name = env._state.active_table
    if table_name not in env._tables:
        if env._tables:
            table_name = next(iter(env._tables))
        else:
            import pandas as pd

            return pd.DataFrame()
    return env._tables[table_name]


def current_table(env):
    return current_frame(env)


def commit_ready(env) -> bool:
    commit_gate = getattr(env, "_commit_ready", None)
    if callable(commit_gate):
        return bool(commit_gate())
    return True

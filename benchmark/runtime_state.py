from __future__ import annotations


def current_frame(env):
    return env._tables[env._state.active_table]


def current_table(env):
    return current_frame(env)


def commit_ready(env) -> bool:
    commit_gate = getattr(env, "_commit_ready", None)
    if callable(commit_gate):
        return bool(commit_gate())
    return True

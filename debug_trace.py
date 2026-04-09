"""Structured stderr debug logging for validator and deployment diagnostics."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import Any


def _enabled() -> bool:
    value = os.getenv("MARIO_DEBUG_LOGS", "0").strip().lower()
    return value not in {"0", "false", "off", "no"}


def _coerce(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _coerce(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coerce(item) for item in value]
    return str(value)


def debug_log(event: str, /, **fields: Any) -> None:
    """Emit one JSON diagnostic line to stderr when debug logging is enabled."""

    if not _enabled():
        return
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "source": "mario-debug",
        "event": event,
        **{str(key): _coerce(value) for key, value in fields.items()},
    }
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")), file=sys.stderr, flush=True)

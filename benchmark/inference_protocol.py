"""Structured stdout protocol helpers for inference CLI runs."""

from __future__ import annotations

import json
from typing import Any, Iterable

PROTOCOL_VERSION = "mario-inference-v1"
ALLOWED_TAGS = {"START", "STEP", "END"}


def format_protocol_event(tag: str, payload: dict[str, Any]) -> str:
    """Format one protocol event line for stdout."""

    if tag not in ALLOWED_TAGS:
        raise ValueError(f"Unsupported protocol tag: {tag}")
    return f"{tag} {json.dumps(payload, sort_keys=True)}"


def emit_protocol_event(tag: str, payload: dict[str, Any]) -> None:
    """Print one protocol event line to stdout."""

    print(format_protocol_event(tag, payload))


def parse_protocol_lines(lines: Iterable[str]) -> dict[str, Any]:
    """Parse strict START/STEP/END protocol lines.

    Returns a dict with `start`, `steps`, and `end` payloads.
    Raises ValueError on malformed protocol streams.
    """

    start_payload: dict[str, Any] | None = None
    end_payload: dict[str, Any] | None = None
    step_payloads: list[dict[str, Any]] = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if " " not in line:
            raise ValueError(f"Malformed protocol line (missing separator): {line}")
        tag, payload_blob = line.split(" ", 1)
        if tag not in ALLOWED_TAGS:
            raise ValueError(f"Malformed protocol line (unknown tag): {line}")

        try:
            payload = json.loads(payload_blob)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed protocol JSON payload for tag {tag}: {payload_blob}") from exc

        if not isinstance(payload, dict):
            raise ValueError(f"Protocol payload must be a JSON object for tag {tag}")

        if tag == "START":
            if start_payload is not None:
                raise ValueError("Protocol stream contains multiple START lines")
            if end_payload is not None:
                raise ValueError("START cannot appear after END")
            start_payload = payload
        elif tag == "STEP":
            if start_payload is None:
                raise ValueError("STEP cannot appear before START")
            if end_payload is not None:
                raise ValueError("STEP cannot appear after END")
            step_payloads.append(payload)
        elif tag == "END":
            if start_payload is None:
                raise ValueError("END cannot appear before START")
            if end_payload is not None:
                raise ValueError("Protocol stream contains multiple END lines")
            end_payload = payload

    if start_payload is None:
        raise ValueError("Protocol stream is missing START")
    if end_payload is None:
        raise ValueError("Protocol stream is missing END")

    return {
        "start": start_payload,
        "steps": step_payloads,
        "end": end_payload,
    }

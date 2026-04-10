"""Gradio UI for the Mario compatibility API.

This UI follows the openenv_hackthon root layout but drives Mario ETL tasks.
"""

from __future__ import annotations

import json
import os

import gradio as gr
import requests


ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860").rstrip("/")

TASK_OPTIONS = {
    "Task 1 - Ingestion Contract Repair": "alert_prioritization",
    "Task 2 - Validation and Event Stabilization": "threat_detection",
    "Task 3 - Referential Repair": "incident_response",
}

ACTION_HINTS = {
    "alert_prioritization": (
        "Provide ETL action JSON. Example: {\"action_id\": 0} for inspect or "
        "{\"action_id\": 3, \"target_column\": \"age\"}"
    ),
    "threat_detection": (
        "Provide ETL action JSON. Example: {\"action_id\": 10} for deduplicate "
        "or {\"action_id\": 8, \"target_column\": \"amount\"}"
    ),
    "incident_response": (
        "Provide ETL action JSON. Example: {\"action_id\": 0, "
        "\"target_column\": \"products\"} for table switch."
    ),
}


def _post(path: str, payload: dict) -> dict:
    try:
        response = requests.post(f"{ENV_BASE_URL}{path}", json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        return {"error": str(exc)}


def _get(path: str, params: dict | None = None) -> dict:
    try:
        response = requests.get(f"{ENV_BASE_URL}{path}", params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        return {"error": str(exc)}


def _fmt(payload: dict | list) -> str:
    return json.dumps(payload, indent=2)


def check_health() -> str:
    return _fmt(_get("/health"))


def update_hint(task_label: str) -> str:
    task_id = TASK_OPTIONS[task_label]
    return ACTION_HINTS[task_id]


def run_episode(task_label: str, scenario_index: int, action_json: str, seed: int) -> tuple[str, str, str]:
    task_id = TASK_OPTIONS[task_label]
    reset_payload = {
        "task_id": task_id,
        "scenario_index": int(scenario_index),
        "seed": int(seed),
        "split": "train",
    }
    reset_result = _post("/reset", reset_payload)
    if "error" in reset_result:
        return _fmt(reset_result), "{}", "Reset failed"

    try:
        action_payload = json.loads(action_json)
    except json.JSONDecodeError as exc:
        return _fmt(reset_result), _fmt({"error": f"Invalid action JSON: {exc}"}), "Step failed"

    step_payload = {
        "task_id": task_id,
        "scenario_index": int(scenario_index),
        "action": action_payload,
    }
    step_result = _post("/step", step_payload)
    if "error" in step_result:
        return _fmt(reset_result), _fmt(step_result), "Step failed"

    reward = step_result.get("reward", 0.0)
    done = step_result.get("done", False)
    summary = f"reward={reward:.4f}, done={done}"
    return _fmt(reset_result), _fmt(step_result), summary


def fetch_state(task_label: str, scenario_index: int) -> str:
    task_id = TASK_OPTIONS[task_label]
    result = _get("/state", {"task_id": task_id, "scenario_index": int(scenario_index)})
    return _fmt(result)


with gr.Blocks(title="Mario ETL Compatibility UI") as demo:
    gr.Markdown("# Mario ETL Compatibility UI")
    gr.Markdown(
        "This UI uses openenv-style root endpoints while preserving Mario ETL repair logic."
    )

    with gr.Row():
        health_btn = gr.Button("Check health")
        health_out = gr.Code(label="Health", language="json")
    health_btn.click(check_health, outputs=health_out)

    with gr.Row():
        task = gr.Dropdown(
            choices=list(TASK_OPTIONS.keys()),
            value=list(TASK_OPTIONS.keys())[0],
            label="Task",
        )
        scenario = gr.Slider(minimum=0, maximum=2, value=0, step=1, label="Scenario index")
        seed = gr.Number(value=42, precision=0, label="Seed")

    hint = gr.Textbox(label="Action hint", lines=3, interactive=False, value=ACTION_HINTS["alert_prioritization"])
    task.change(update_hint, inputs=task, outputs=hint)

    action_input = gr.Code(
        label="Action JSON",
        language="json",
        value='{"action_id": 0}',
        lines=8,
    )

    with gr.Row():
        run_btn = gr.Button("Reset + Step")
        state_btn = gr.Button("Get state")

    reset_out = gr.Code(label="Reset response", language="json", lines=14)
    step_out = gr.Code(label="Step response", language="json", lines=18)
    summary_out = gr.Textbox(label="Summary", lines=1)
    state_out = gr.Code(label="State", language="json", lines=14)

    run_btn.click(
        run_episode,
        inputs=[task, scenario, action_input, seed],
        outputs=[reset_out, step_out, summary_out],
    )
    state_btn.click(fetch_state, inputs=[task, scenario], outputs=state_out)


if __name__ == "__main__":
    port = int(os.environ.get("UI_PORT", "7861"))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)

"""Custom Gradio benchmark demo for the Mario OpenEnv Space."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr

from benchmark.catalog import TASK_CARDS, benchmark_metadata
from benchmark.runtime import (
    adaptation_payload,
    benchmark_profiles_payload,
    benchmark_runs_payload,
    benchmark_tasks_payload,
    runtime_summary,
)
from models import PipelineDoctorAction

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
ACTION_EXAMPLES = [
    [14, "", "", ""],
    [16, "", "", ""],
    [18, "", "", ""],
    [19, "", "", ""],
    [12, "user_identifier", "user_id", ""],
    [13, "", "", "user_id,event_ts,amount,status"],
]
ACTION_REFERENCE = [
    ("0", "Inspect schema or switch table", "Use when you need table context before editing."),
    ("3-5", "Fill missing values", "Recover null-heavy columns before validating or committing."),
    ("6", "Drop null rows", "Use sparingly when rows are genuinely unrecoverable."),
    ("7-9", "Cast or normalize values", "Fix schema drift, malformed types, and formatting issues."),
    ("10", "Remove duplicates", "Apply after replay or retry issues create duplicate events."),
    ("11", "Drop outliers", "Use when abnormal rows are poisoning downstream checks."),
    ("12", "Rename column", "Repair contract drift or alias mismatches."),
    ("13", "Reorder columns", "Use when schema order matters for validation."),
    ("14", "Validate schema", "Safe first move when you want a read-like diagnostic action."),
    ("15", "Commit changes", "Only when commit-ready signals are clean."),
    ("16-19", "Orchestration controls", "Scale workers, replay batches, or refresh downstream assets."),
]


class _WebManager:
    def __init__(self, env):
        self.env = env


def build_benchmark_demo(
    web_manager,
    action_fields,
    metadata,
    is_chat_env,
    title,
    quick_start_md,
):
    """Build a concise operator-first visualization tab for the Space web UI."""

    del action_fields, is_chat_env, quick_start_md

    benchmark_meta = benchmark_metadata()
    runtime_meta = runtime_summary()
    benchmark_runs = benchmark_runs_payload()
    adaptation = adaptation_payload()

    def task_card_markdown(task_id: int) -> str:
        card = TASK_CARDS[int(task_id)]
        return "\n".join(
            [
                f"### Task {task_id}",
                f"**Incident type:** {card['incident_type']}",
                f"**Objective:** {card['objective']}",
                f"**What is broken:** {card['broken_state']}",
                f"**What to watch:** {', '.join(card['diagnosis_signals'])}",
                f"**Recovery target:** {', '.join(card['recovery_requirements'])}",
                f"**Do not commit if:** {', '.join(card['unsafe_commit_conditions'])}",
                f"**Success threshold:** `{card['success_threshold']}`",
            ]
        )

    def profile_markdown() -> str:
        payload = benchmark_profiles_payload()
        profiles = payload["scenario_profiles"]
        descriptions = payload.get("profile_descriptions", {})
        lines = ["### Scenario Profile Families"]
        for task_id in sorted(profiles):
            lines.append(f"**Task {task_id}**")
            lines.append(f"- train: {', '.join(profiles[task_id]['train'])}")
            lines.append(f"- eval: {', '.join(profiles[task_id]['eval'])}")
            example_profile = profiles[task_id]["eval"][0]
            if example_profile in descriptions:
                lines.append(f"- example incident: {descriptions[example_profile]}")
            lines.append("")
        return "\n".join(lines)

    def benchmark_summary_markdown() -> str:
        lines = [
            f"# {title}",
            "",
            "Mario is an ETL incident-recovery environment built on OpenEnv.",
            "Use the `Run Environment` tab to reset an incident, apply actions, and inspect the resulting state.",
        ]
        return "\n".join(lines)

    def quick_start_markdown() -> str:
        return "\n".join(
            [
                "## Start Here",
                "1. Pick a task, split, and seed.",
                "2. Click `Reset environment` to load a broken pipeline incident.",
                "3. Start with action `14` to validate, then use repair or orchestration actions.",
                "4. Watch `commit_ready`, `current_score`, backlog, freshness, and dependency signals.",
                "5. Use action `15` only when the incident is actually recovered.",
            ]
        )

    def action_reference_markdown() -> str:
        lines = ["## Action Guide"]
        for action_id, label, usage in ACTION_REFERENCE:
            lines.append(f"- `{action_id}`: **{label}**. {usage}")
        return "\n".join(lines)

    def live_status_markdown(state: dict[str, Any] | None) -> str:
        if not state:
            return "\n".join(
                [
                    "## Current Episode",
                    "No episode loaded yet.",
                    "Reset the environment to populate state and observation panels.",
                ]
            )

        commit_ready = state.get("commit_ready")
        success = state.get("success")
        score = state.get("current_score")
        steps = state.get("step_count")
        backlog = state.get("backlog_rows")
        freshness = state.get("freshness_lag_minutes")
        reason = state.get("done_reason") or "in_progress"
        return "\n".join(
            [
                "## Current Episode",
                f"- score: `{score}`",
                f"- steps: `{steps}`",
                f"- commit_ready: `{commit_ready}`",
                f"- success: `{success}`",
                f"- backlog_rows: `{backlog}`",
                f"- freshness_lag_minutes: `{freshness}`",
                f"- status: `{reason}`",
            ]
        )

    def workspace_summary_markdown() -> str:
        return "\n".join(
            [
                "## Environment Snapshot",
                f"- benchmark version: `{runtime_meta['benchmark_version']}`",
                f"- runtime mode: `{runtime_meta['runtime_mode']}`",
                f"- tasks: `{len(benchmark_meta['task_names'])}`",
                "- action space: `0-19` discrete operations",
                "- splits: `train`, `eval`",
            ]
        )

    def reset_episode(task_id: int, split: str, seed: int):
        observation = web_manager.env.reset(task_id=int(task_id), split=split, seed=int(seed))
        state = web_manager.env.state.model_dump()
        return (
            observation.model_dump(),
            state,
            task_card_markdown(int(task_id)),
            live_status_markdown(state),
        )

    def step_episode(
        action_id: int,
        target_column: str,
        new_name: str,
        column_order: str,
    ):
        parsed_order = [item.strip() for item in column_order.split(",") if item.strip()] or None
        action = PipelineDoctorAction(
            action_id=int(action_id),
            target_column=target_column or None,
            new_name=new_name or None,
            column_order=parsed_order,
        )
        observation = web_manager.env.step(action)
        state = web_manager.env.state.model_dump()
        return observation.model_dump(), state, live_status_markdown(state)

    def latest_runs_json() -> dict[str, Any]:
        return benchmark_runs

    def latest_adaptation_json() -> dict[str, Any]:
        return adaptation

    with gr.Blocks(title=title) as blocks:
        gr.Markdown(benchmark_summary_markdown())
        with gr.Row():
            with gr.Column():
                gr.Markdown(workspace_summary_markdown())
            with gr.Column():
                gr.Markdown(
                    "\n".join(
                        [
                            "## What You Can Do",
                            "- Manually inspect ETL recovery incidents.",
                            "- Step through the same action contract used by the benchmark.",
                            "- Review task cards, score structure, and held-out profiles if needed.",
                        ]
                    )
                )
            with gr.Column():
                gr.Markdown(
                    "\n".join(
                        [
                            "## API",
                            "- `POST /reset`",
                            "- `POST /step`",
                            "- `GET /state`",
                            "- `GET /tasks`",
                            "- `GET /grade/task_1` to `task_3`",
                        ]
                    )
                )

        with gr.Tabs():
            with gr.Tab("Run Environment"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown(quick_start_markdown())
                        with gr.Row():
                            task_id = gr.Dropdown(
                                choices=[(f"Task {task_id}", task_id) for task_id in sorted(TASK_CARDS)],
                                value=3,
                                label="Task",
                            )
                            split = gr.Dropdown(["train", "eval"], value="eval", label="Split")
                            seed = gr.Number(value=42, precision=0, label="Seed")
                        reset_btn = gr.Button("Reset environment", variant="primary")
                        status_md = gr.Markdown(live_status_markdown(None))
                        inspector_task_card = gr.Markdown(task_card_markdown(3))
                    with gr.Column(scale=3):
                        with gr.Accordion("Observation", open=True):
                            observation_json = gr.JSON(label="Observation payload")
                        with gr.Accordion("State", open=False):
                            state_json = gr.JSON(label="Internal state")

                task_id.change(task_card_markdown, inputs=task_id, outputs=inspector_task_card)
                reset_btn.click(
                    reset_episode,
                    inputs=[task_id, split, seed],
                    outputs=[observation_json, state_json, inspector_task_card, status_md],
                )

                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown(action_reference_markdown())
                    with gr.Column(scale=3):
                        gr.Markdown("## Take an Action")
                        with gr.Row():
                            action_id = gr.Number(value=14, precision=0, label="Action id")
                            target_column = gr.Textbox(label="Target column or table")
                            new_name = gr.Textbox(label="New name")
                            column_order = gr.Textbox(label="Column order (comma separated)")
                        step_btn = gr.Button("Apply action", variant="secondary")
                        gr.Examples(
                            examples=ACTION_EXAMPLES,
                            inputs=[action_id, target_column, new_name, column_order],
                            label="Quick action presets",
                        )

                step_btn.click(
                    step_episode,
                    inputs=[action_id, target_column, new_name, column_order],
                    outputs=[observation_json, state_json, status_md],
                )

            with gr.Tab("Tasks"):
                with gr.Row():
                    with gr.Column(scale=2):
                        task_picker = gr.Dropdown(
                            choices=[(f"Task {task_id}", task_id) for task_id in sorted(TASK_CARDS)],
                            value=1,
                            label="Task card",
                        )
                        task_card = gr.Markdown(task_card_markdown(1))
                        task_picker.change(task_card_markdown, inputs=task_picker, outputs=task_card)
                    with gr.Column(scale=3):
                        gr.Markdown(profile_markdown())

            with gr.Tab("Benchmark"):
                gr.Markdown("## Benchmark Details")
                with gr.Row():
                    _image_component("benchmark_overview.png", "Benchmark ladder")
                    _image_component("difficulty_gap.png", "Difficulty gap")
                    _image_component("objective_weights.png", "Task scoring weights")
                with gr.Accordion("Latest benchmark runs", open=False):
                    gr.JSON(value=latest_runs_json(), label="Runs")
                with gr.Accordion("Adaptation report", open=False):
                    gr.JSON(value=latest_adaptation_json(), label="Adaptation")
                with gr.Accordion("Task metadata and scoring weights", open=False):
                    gr.JSON(value=benchmark_tasks_payload(), label="Task metadata")
                with gr.Accordion("Scenario profile catalog", open=False):
                    gr.JSON(value=benchmark_profiles_payload(), label="Profiles")

            with gr.Tab("API and Architecture"):
                gr.Markdown(
                    "\n".join(
                        [
                            "## API Endpoints",
                            "- `GET /health`",
                            "- `GET /metadata`",
                            "- `GET /schema`",
                            "- `POST /reset`",
                            "- `POST /step`",
                            "- `GET /state`",
                            "- `GET /tasks`",
                            "- `GET /grader` and `GET /grade/task_1` to `task_3`",
                            "",
                            "## Architecture Notes",
                            "- Each task models a broken ETL incident rather than a toy game.",
                            "- The action space mixes data repair and orchestration controls.",
                            "- The observation carries quality, dependency, backlog, freshness, and reward-machine signals.",
                            "- Task grading is deterministic and validator-facing scores stay strictly inside the public range.",
                        ]
                    )
                )

    return blocks


def create_space_demo(env):
    return build_benchmark_demo(
        web_manager=_WebManager(env),
        action_fields=[],
        metadata={},
        is_chat_env=False,
        title="Mario the Plumber",
        quick_start_md="",
    )


def _image_component(name: str, label: str):
    path = ASSETS / name
    if path.exists():
        return gr.Image(value=str(path), label=label)
    return gr.Markdown(f"_{label} not generated yet._")

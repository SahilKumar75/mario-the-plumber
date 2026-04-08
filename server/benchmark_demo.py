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
    """Build a benchmark-focused visualization tab for the Space web UI."""

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
                f"**Incident summary:** {card['incident_description']}",
                f"**Broken state:** {card['broken_state']}",
                f"**Diagnosis signals:** {', '.join(card['diagnosis_signals'])}",
                f"**Recovery requirements:** {', '.join(card['recovery_requirements'])}",
                f"**Unsafe commit conditions:** {', '.join(card['unsafe_commit_conditions'])}",
                f"**Success threshold:** `{card['success_threshold']}`",
                f"**Threshold rationale:** {card['threshold_rationale']}",
                f"**Target policy:** {card['target_policy']}",
                f"**Failure / truncation:** {', '.join(card['failure_conditions'])}",
                f"**Key subgoals:** {', '.join(card['key_subgoals'])}",
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
            f"**Benchmark version:** `{runtime_meta['benchmark_version']}`",
            f"**Runtime mode:** `{runtime_meta['runtime_mode']}`",
            "",
            runtime_meta["runtime_mode_card"]["summary"],
            "",
            f"Tasks: `{len(benchmark_meta['task_names'])}` | Actions: `20` | Splits: `train`, `eval`",
            "",
            "Mario is an ELT/ETL incident fixer delivered through OpenEnv. Agents diagnose broken ingestion and recovery states, repair upstream tables, restore downstream freshness, and decide when a pipeline is safe to commit.",
        ]
        return "\n".join(lines)

    def reset_episode(task_id: int, split: str, seed: int):
        observation = web_manager.env.reset(task_id=int(task_id), split=split, seed=int(seed))
        return (
            observation.model_dump(),
            web_manager.env.state.model_dump(),
            task_card_markdown(int(task_id)),
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
        return observation.model_dump(), web_manager.env.state.model_dump()

    def latest_runs_json() -> dict[str, Any]:
        return benchmark_runs

    def latest_adaptation_json() -> dict[str, Any]:
        return adaptation

    with gr.Blocks(title=title) as blocks:
        gr.Markdown(benchmark_summary_markdown())

        with gr.Tabs():
            with gr.Tab("Overview"):
                gr.Markdown("## Benchmark Overview")
                with gr.Row():
                    with gr.Column(scale=3):
                        gr.Markdown(profile_markdown())
                    with gr.Column(scale=2):
                        gr.Markdown("## Incident Explorer")
                        task_picker = gr.Dropdown(
                            choices=[(f"Task {task_id}", task_id) for task_id in sorted(TASK_CARDS)],
                            value=1,
                            label="Incident card",
                        )
                        task_card = gr.Markdown(task_card_markdown(1))
                        task_picker.change(task_card_markdown, inputs=task_picker, outputs=task_card)

                with gr.Row():
                    _image_component("benchmark_overview.png", "Benchmark ladder")
                    _image_component("difficulty_gap.png", "Difficulty gap")
                    _image_component("objective_weights.png", "Task scoring weights")
                gr.Markdown(
                    "Tasks 1-2 show single-table stabilization. Tasks 3-5 show pipeline-level recovery objectives for the harder ETL incidents."
                )

            with gr.Tab("Episode Inspector"):
                gr.Markdown(
                    "## Live Episode Inspector\nReset an incident, inspect the diagnosis/recovery state bundle, and step manually with the same discrete action contract the benchmark uses."
                )
                with gr.Row():
                    task_id = gr.Dropdown(
                        choices=[(f"Task {task_id}", task_id) for task_id in sorted(TASK_CARDS)],
                        value=3,
                        label="Task",
                    )
                    split = gr.Dropdown(["train", "eval"], value="eval", label="Split")
                    seed = gr.Number(value=42, precision=0, label="Seed")
                    reset_btn = gr.Button("Reset episode", variant="primary")
                inspector_task_card = gr.Markdown(task_card_markdown(3))
                with gr.Row():
                    observation_json = gr.JSON(label="Observation")
                    state_json = gr.JSON(label="State")

                task_id.change(task_card_markdown, inputs=task_id, outputs=inspector_task_card)
                reset_btn.click(
                    reset_episode,
                    inputs=[task_id, split, seed],
                    outputs=[observation_json, state_json, inspector_task_card],
                )

                with gr.Row():
                    action_id = gr.Number(value=14, precision=0, label="Action id")
                    target_column = gr.Textbox(label="Target column / table")
                    new_name = gr.Textbox(label="New name")
                    column_order = gr.Textbox(label="Column order (comma separated)")
                    step_btn = gr.Button("Step", variant="secondary")

                step_btn.click(
                    step_episode,
                    inputs=[action_id, target_column, new_name, column_order],
                    outputs=[observation_json, state_json],
                )

            with gr.Tab("Results"):
                gr.Markdown("## Benchmark Results")
                with gr.Row():
                    gr.JSON(value=latest_runs_json(), label="Latest benchmark runs")
                    gr.JSON(value=latest_adaptation_json(), label="Latest adaptation report")
                with gr.Row():
                    gr.JSON(value=benchmark_tasks_payload(), label="Task cards and objective weights")
                    gr.JSON(value=benchmark_profiles_payload(), label="Scenario profile catalog")

            with gr.Tab("Architecture"):
                gr.Markdown(
                    "\n".join(
                        [
                            "## Benchmark Architecture",
                            "",
                            "- **Incident framing:** each task is a broken ETL incident with diagnosis signals, recovery requirements, and unsafe-commit conditions.",
                            "- **Observation signals:** data quality, dependency consistency, backlog age, freshness severity, resource pressure, and reward-machine state.",
                            "- **Action space:** discrete repair plus orchestration actions under a stable 0-19 contract.",
                            "- **Scoring:** per-task deterministic grader with explicit incident-recovery and objective-weight reporting for the hard tasks.",
                            "- **Scoring weights:** Tasks 1-2 expose the single-table score mix; Tasks 3-5 expose multi-objective pipeline recovery weights.",
                            "- **Runtime modes:** benchmark, incident, and hybrid. These change the framing/reporting surface, not the underlying benchmark contract.",
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

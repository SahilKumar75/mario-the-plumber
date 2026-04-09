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
CUSTOM_CSS = """
.gradio-container {
  background:
    radial-gradient(circle at top right, rgba(209, 227, 255, 0.45), transparent 28%),
    linear-gradient(180deg, #f5f7fb 0%, #eef2f8 100%);
}
.mario-shell {
  max-width: 1280px;
  margin: 0 auto;
}
.hero-panel,
.info-panel,
.task-panel,
.control-panel,
.status-panel {
  background: rgba(255, 255, 255, 0.92);
  border: 1px solid #d8e0eb;
  border-radius: 18px;
  box-shadow: 0 14px 40px rgba(15, 23, 42, 0.08);
}
.hero-panel {
  padding: 28px 30px 22px;
  background:
    linear-gradient(135deg, rgba(17, 24, 39, 0.98), rgba(27, 44, 84, 0.96)),
    linear-gradient(180deg, #111827, #1f2937);
  color: #f8fafc;
}
.hero-panel h1 {
  margin: 0 0 10px;
  font-size: 2.2rem;
  letter-spacing: -0.03em;
}
.hero-panel p {
  margin: 0;
  color: #dbe5f0;
  line-height: 1.6;
}
.hero-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px;
  margin-top: 20px;
}
.hero-stat {
  padding: 14px 16px;
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.08);
  border: 1px solid rgba(255, 255, 255, 0.1);
}
.hero-stat .label {
  display: block;
  font-size: 0.8rem;
  color: #c8d6e5;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}
.hero-stat .value {
  display: block;
  margin-top: 8px;
  font-size: 1.2rem;
  font-weight: 600;
  color: #ffffff;
}
.panel-body {
  padding: 20px 22px;
}
.panel-body h2,
.panel-body h3 {
  margin-top: 0;
  letter-spacing: -0.02em;
}
.status-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}
.status-card {
  padding: 14px 16px;
  border-radius: 14px;
  border: 1px solid #dbe3ef;
  background: #f8fbff;
}
.status-card .label {
  display: block;
  color: #5f6f86;
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}
.status-card .value {
  display: block;
  margin-top: 8px;
  color: #162033;
  font-size: 1.15rem;
  font-weight: 600;
}
.status-banner {
  margin-bottom: 12px;
  padding: 12px 14px;
  border-radius: 14px;
  background: #e8f1ff;
  color: #163056;
  border: 1px solid #cadeff;
  font-weight: 600;
}
.status-banner.complete {
  background: #e7f7ed;
  border-color: #b6dfc2;
  color: #165a2d;
}
.status-banner.failed {
  background: #fff0f0;
  border-color: #f1c8c8;
  color: #8b1f1f;
}
.mini-list {
  margin: 0;
  padding-left: 18px;
  color: #334155;
}
.mini-list li {
  margin: 6px 0;
}
"""


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

    def action_reference_markdown() -> str:
        lines = ["### Action Guide"]
        for action_id, label, usage in ACTION_REFERENCE:
            lines.append(f"- `{action_id}`: **{label}**. {usage}")
        return "\n".join(lines)

    def _hero_html() -> str:
        return f"""
        <section class="hero-panel">
          <h1>{title}</h1>
          <p>
            A production-style ETL incident recovery environment for OpenEnv.
            Reset a scenario, inspect the observation contract, apply repair actions,
            and decide when the pipeline is safe to commit.
          </p>
          <div class="hero-grid">
            <div class="hero-stat">
              <span class="label">Runtime</span>
              <span class="value">{runtime_meta['runtime_mode'].title()}</span>
            </div>
            <div class="hero-stat">
              <span class="label">Validator Tasks</span>
              <span class="value">3 Public Tasks</span>
            </div>
            <div class="hero-stat">
              <span class="label">Action Space</span>
              <span class="value">20 Discrete Operations</span>
            </div>
          </div>
        </section>
        """

    def _overview_html() -> str:
        return """
        <section class="info-panel panel-body">
          <h3>Operator Workflow</h3>
          <ol class="mini-list">
            <li>Select a task, split, and seed.</li>
            <li>Reset the environment to load the broken pipeline.</li>
            <li>Validate first, then repair data quality and orchestration issues.</li>
            <li>Track score, readiness, backlog, freshness, and dependency health.</li>
            <li>Commit only when the episode is actually recovered.</li>
          </ol>
        </section>
        """

    def live_status_html(state: dict[str, Any] | None) -> str:
        if not state:
            return """
            <section class="status-panel panel-body">
              <div class="status-banner">No episode loaded. Reset the environment to begin.</div>
              <div class="status-grid">
                <div class="status-card"><span class="label">Score</span><span class="value">-</span></div>
                <div class="status-card"><span class="label">Steps</span><span class="value">-</span></div>
                <div class="status-card"><span class="label">Commit Ready</span><span class="value">-</span></div>
                <div class="status-card"><span class="label">Status</span><span class="value">Idle</span></div>
              </div>
            </section>
            """

        commit_ready = state.get("commit_ready")
        success = state.get("success")
        score = state.get("current_score")
        steps = state.get("step_count")
        backlog = state.get("backlog_rows")
        freshness = state.get("freshness_lag_minutes")
        reason = state.get("done_reason") or "in_progress"
        banner_class = "status-banner"
        banner_text = f"Episode status: {reason}"
        if success is True:
            banner_class += " complete"
            banner_text = f"Episode complete: {reason}"
        elif state.get("done"):
            banner_class += " failed"
            banner_text = f"Episode ended: {reason}"

        return f"""
        <section class="status-panel panel-body">
          <div class="{banner_class}">{banner_text}</div>
          <div class="status-grid">
            <div class="status-card"><span class="label">Score</span><span class="value">{score}</span></div>
            <div class="status-card"><span class="label">Steps Used</span><span class="value">{steps}</span></div>
            <div class="status-card"><span class="label">Commit Ready</span><span class="value">{commit_ready}</span></div>
            <div class="status-card"><span class="label">Success</span><span class="value">{success}</span></div>
            <div class="status-card"><span class="label">Backlog Rows</span><span class="value">{backlog}</span></div>
            <div class="status-card"><span class="label">Freshness Lag</span><span class="value">{freshness}</span></div>
          </div>
        </section>
        """

    def workspace_summary_html() -> str:
        return f"""
        <section class="info-panel panel-body">
          <h3>Environment Snapshot</h3>
          <ul class="mini-list">
            <li>Benchmark version: <strong>{runtime_meta['benchmark_version']}</strong></li>
            <li>Runtime mode: <strong>{runtime_meta['runtime_mode']}</strong></li>
            <li>Total benchmark tasks: <strong>{len(benchmark_meta['task_names'])}</strong></li>
            <li>Available splits: <strong>train</strong> and <strong>eval</strong></li>
            <li>Validator-facing tasks: <strong>task_1</strong> to <strong>task_3</strong></li>
          </ul>
        </section>
        """

    def reset_episode(task_id: int, split: str, seed: int):
        observation = web_manager.env.reset(task_id=int(task_id), split=split, seed=int(seed))
        state = web_manager.env.state.model_dump()
        return (
            observation.model_dump(),
            state,
            task_card_markdown(int(task_id)),
            live_status_html(state),
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
        return observation.model_dump(), state, live_status_html(state)

    def latest_runs_json() -> dict[str, Any]:
        return benchmark_runs

    def latest_adaptation_json() -> dict[str, Any]:
        return adaptation

    with gr.Blocks(title=title, css=CUSTOM_CSS, theme=gr.themes.Soft()) as blocks:
        with gr.Column(elem_classes="mario-shell"):
            gr.HTML(_hero_html())
            with gr.Row():
                with gr.Column():
                    gr.HTML(workspace_summary_html())
                with gr.Column():
                    gr.HTML(_overview_html())
                with gr.Column():
                    gr.HTML(
                        """
                        <section class="info-panel panel-body">
                          <h3>Public API</h3>
                          <ul class="mini-list">
                            <li><code>POST /reset</code></li>
                            <li><code>POST /step</code></li>
                            <li><code>GET /state</code></li>
                            <li><code>GET /tasks</code></li>
                            <li><code>GET /grade/task_1</code> to <code>task_3</code></li>
                          </ul>
                        </section>
                        """
                    )

            with gr.Tabs():
                with gr.Tab("Run Environment"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            gr.HTML('<section class="control-panel panel-body"><h3>Scenario Setup</h3><p>Select a task profile and reset the environment to load a broken pipeline incident.</p></section>')
                            with gr.Row():
                                task_id = gr.Dropdown(
                                    choices=[(f"Task {task_id}", task_id) for task_id in sorted(TASK_CARDS)],
                                    value=3,
                                    label="Task",
                                )
                                split = gr.Dropdown(["train", "eval"], value="eval", label="Split")
                                seed = gr.Number(value=42, precision=0, label="Seed")
                            reset_btn = gr.Button("Reset environment", variant="primary")
                            status_md = gr.HTML(live_status_html(None))
                            inspector_task_card = gr.Markdown(task_card_markdown(3), elem_classes="task-panel")
                        with gr.Column(scale=3):
                            with gr.Accordion("Observation payload", open=True):
                                observation_json = gr.JSON(label="Observation payload")
                            with gr.Accordion("Internal state", open=False):
                                state_json = gr.JSON(label="Internal state")

                    task_id.change(task_card_markdown, inputs=task_id, outputs=inspector_task_card)
                    reset_btn.click(
                        reset_episode,
                        inputs=[task_id, split, seed],
                        outputs=[observation_json, state_json, inspector_task_card, status_md],
                    )

                    with gr.Row():
                        with gr.Column(scale=2):
                            gr.Markdown(action_reference_markdown(), elem_classes="info-panel")
                        with gr.Column(scale=3):
                            gr.HTML(
                                """
                                <section class="control-panel panel-body">
                                  <h3>Action Console</h3>
                                  <p>Use presets for common recovery moves, or enter parameters manually for table-specific operations.</p>
                                </section>
                                """
                            )
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
                    gr.HTML('<section class="info-panel panel-body"><h3>Benchmark Details</h3><p>Visual summaries of task difficulty, objective weighting, and held-out adaptation behavior.</p></section>')
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

from __future__ import annotations

from fastapi.testclient import TestClient

from benchmark.api_payloads import benchmark_metadata_payload
from benchmark.api_payloads import tasks_payload
from benchmark.grading import score_task4
from benchmark.policies.heuristics import heuristic_action_for
from graders import grade_episode
from graders.runtime import validator_grade_payload
from models import PipelineDoctorAction
from server.app import app
from server.data_generator import generate_scenario
from server.pipeline_doctor_environment import PipelineDoctorEnvironment
from tasks.definitions import list_task_ids


def _assert_nested_scores_strictly_inside_open_interval(value) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _assert_nested_scores_strictly_inside_open_interval(item)
    elif isinstance(value, list):
        for item in value:
            _assert_nested_scores_strictly_inside_open_interval(item)
    elif isinstance(value, float):
        assert 0.0 < value < 1.0


def _assert_minimal_validator_grade_payload(payload: dict[str, object]) -> None:
    assert set(payload) == {"score", "reward"}
    assert 0.0 < float(payload["score"]) < 1.0
    assert 0.0 < float(payload["reward"]) < 1.0


def test_reset_and_step_lifecycle_invariants() -> None:
    fresh_env = PipelineDoctorEnvironment()
    fresh_observation = fresh_env.reset(task_id=4, split="eval", seed=7)

    assert fresh_env.state.step_count == 0
    assert fresh_observation.steps_taken == 0
    assert fresh_observation.time_budget_remaining == fresh_env.state.max_steps
    assert fresh_observation.current_score == fresh_env.state.current_score
    assert fresh_env.state.initial_score == fresh_observation.current_score

    step_observation = fresh_env.step(PipelineDoctorAction(action_id=2))

    assert fresh_env.state.step_count == 1
    assert step_observation.steps_taken == 1
    assert step_observation.time_budget_remaining == fresh_env.state.max_steps - 1
    assert step_observation.done is False
    assert step_observation.done_reason == ""
    assert step_observation.current_score == fresh_observation.current_score

    dirty_env = PipelineDoctorEnvironment()
    dirty_env.reset(task_id=5, split="eval", seed=7)
    dirty_env.step(PipelineDoctorAction(action_id=16))
    dirty_observation = dirty_env.reset(task_id=4, split="eval", seed=7)

    assert dirty_observation.current_score == fresh_observation.current_score
    assert dirty_env.state.initial_score == fresh_observation.current_score


def test_task3_commit_requires_dependency_repair_and_action_19_restores_it() -> None:
    env = PipelineDoctorEnvironment()
    env.reset(task_id=3, split="eval", seed=7)

    env._tables = {
        name: frame.copy(deep=True)
        for name, frame in env._ground_truth.items()
    }
    env._tables["orders"]["total_price"] = env._tables["orders"]["total_price"] + 5.0
    env._refresh_errors()
    env._update_task_progress_state()

    assert env._task3_commit_ready() is False
    repair_observation = env.step(PipelineDoctorAction(action_id=19))
    assert repair_observation.dependency_alerts == []
    assert env._task3_commit_ready() is True

    commit_observation = env.step(PipelineDoctorAction(action_id=15))
    assert commit_observation.done is True
    assert commit_observation.current_score >= 0.75
    assert env.state.done is True
    assert env.state.success is True


def test_task4_commit_stays_blocked_while_incident_pressure_remains() -> None:
    env = PipelineDoctorEnvironment()
    env.reset(task_id=4, split="eval", seed=2)

    assert env._task4_commit_ready() is False

    env.step(PipelineDoctorAction(action_id=16))
    env.step(PipelineDoctorAction(action_id=16))
    while env.state.backlog_rows > 0:
        env.step(PipelineDoctorAction(action_id=18))
    env.step(PipelineDoctorAction(action_id=19))

    assert env.state.backlog_rows == 0
    assert env.state.pending_batches == 0
    assert env.state.freshness_lag_minutes == 0
    assert env.state.resource_level == 3
    assert env._recent_errors
    assert env._task4_commit_ready() is False

    commit_observation = env.step(PipelineDoctorAction(action_id=15))
    assert commit_observation.done is True
    assert commit_observation.done_reason == "commit_failure"
    assert env.state.success is False


def test_task5_commit_stays_blocked_until_temporal_recovery_is_complete() -> None:
    env = PipelineDoctorEnvironment()
    observation = env.reset(task_id=5, split="eval", seed=1)

    assert observation.scenario_profile.startswith("heldout_temporal_")
    assert env._task5_commit_ready() is False

    env.step(PipelineDoctorAction(action_id=16))
    env.step(PipelineDoctorAction(action_id=16))
    while env.state.backlog_rows > 0:
        env.step(PipelineDoctorAction(action_id=18))
    refresh = env.step(PipelineDoctorAction(action_id=19))

    assert env.state.backlog_rows == 0
    assert env.state.pending_batches == 0
    assert env.state.freshness_lag_minutes == 0
    assert env.state.resource_level == 3
    assert refresh.action_result.startswith("invalid:")
    assert env._scenario_meta.get("downstream_stale") is True
    assert refresh.dependency_alerts
    assert env._task5_commit_ready() is False

    commit_observation = env.step(PipelineDoctorAction(action_id=15))
    assert commit_observation.done is True
    assert commit_observation.done_reason == "commit_failure"
    assert env.state.success is False


def test_task5_train_temporal_refresh_allows_commit_after_full_recovery() -> None:
    env = PipelineDoctorEnvironment()
    observation = env.reset(task_id=5, split="train", seed=42)

    env.step(PipelineDoctorAction(action_id=16))
    env.step(PipelineDoctorAction(action_id=16))
    while env.state.backlog_rows > 0:
        env.step(PipelineDoctorAction(action_id=18))

    observation = env.step(PipelineDoctorAction(action_id=9, target_column="event_ts"))
    observation = env.step(PipelineDoctorAction(action_id=4, target_column="product_id"))
    observation = env.step(PipelineDoctorAction(action_id=4, target_column="quantity"))
    observation = env.step(PipelineDoctorAction(action_id=3, target_column="gross_revenue"))
    observation = env.step(PipelineDoctorAction(action_id=0, target_column="catalog"))
    observation = env.step(PipelineDoctorAction(action_id=3, target_column="unit_price"))
    observation = env.step(PipelineDoctorAction(action_id=0, target_column="hourly_rollup"))
    observation = env.step(PipelineDoctorAction(action_id=19))

    assert observation.commit_ready is True
    commit_observation = env.step(PipelineDoctorAction(action_id=15))

    assert commit_observation.done is True
    assert commit_observation.done_reason == "commit_success"
    assert env.state.success is True


def test_task5_eval_heldout_alias_family_is_recoverable_with_heuristic_policy() -> None:
    env = PipelineDoctorEnvironment()
    observation = env.reset(task_id=5, split="eval", seed=42)

    assert observation.scenario_profile == "heldout_temporal_correction_replay_family"

    while not env.state.done:
        observation = env.step(heuristic_action_for(5, observation))

    assert env.state.done_reason == "commit_success"
    assert env.state.success is True
    assert observation.current_score >= 0.82


def test_observation_contract_includes_incident_and_recovery_fields() -> None:
    observation = PipelineDoctorEnvironment().reset(task_id=5, split="eval", seed=2)

    assert observation.incident_type
    assert observation.incident_summary
    assert observation.diagnosis_signals
    assert observation.recovery_requirements
    assert observation.unsafe_commit_conditions
    assert observation.recent_errors
    assert observation.commit_ready is False
    assert observation.available_actions == list(range(20))
    assert isinstance(observation.reward_breakdown, dict)
    assert observation.objective_breakdown
    assert observation.tradeoff_weights
    assert observation.subgoal_progress
    assert observation.reward_machine_state
    assert observation.active_subgoal
    assert observation.adaptation_target
    assert observation.dependency_alerts
    assert observation.dependency_health_summary
    assert observation.recent_failure_counters
    assert observation.drift_markers
    assert observation.queue_backlog_age_minutes > 0
    assert observation.freshness_lag_minutes > 0
    assert observation.sla_severity in {"high", "critical"}
    assert observation.time_budget_remaining == 35
    assert observation.heldout_profile_family is False


def test_task4_scoring_matches_published_weights() -> None:
    scenario = generate_scenario(task_id=4, split="eval", seed=7)
    perfect_tables = {
        name: frame.copy(deep=True)
        for name, frame in scenario.ground_truth_tables.items()
    }
    score, breakdown = score_task4(
        perfect_tables,
        scenario.ground_truth_tables,
        scenario.expected_types,
        backlog_rows=0,
        freshness_lag_minutes=0,
        resource_level=1,
        required_resource_level=1,
        downstream_stale=False,
    )

    assert score == 1.0
    assert breakdown["pipeline"] == {
        "data_quality": 1.0,
        "backlog": 1.0,
        "freshness": 1.0,
        "summary_consistency": 1.0,
        "resource_efficiency": 1.0,
    }


def test_benchmark_metadata_and_tasks_payload_smoke() -> None:
    metadata = benchmark_metadata_payload()
    payload = tasks_payload()

    assert metadata["benchmark_version"] == "2.1"
    assert metadata["runtime_mode"] == "benchmark"
    assert metadata["runtime_mode_card"]["summary"]
    assert metadata["task_names"][5] == "Temporal Rollup Recovery"
    assert metadata["formal_task_specs"][5]["reward_machine_order"]
    assert metadata["objective_weights"][5]["rollup_consistency"] == 0.15

    assert len(payload["tasks"]) == 3
    assert "incident_signals" in payload["action_schema"]
    assert "reward_machine_signals" in payload["action_schema"]
    assert "incident_type" in payload["action_schema"]["incident_signals"]
    assert "reward_machine_state" in payload["action_schema"]["reward_machine_signals"]


def test_validator_facing_task_and_grade_endpoints_match_hackathon_pattern() -> None:
    client = TestClient(app)

    tasks_response = client.get("/tasks")
    assert tasks_response.status_code == 200
    tasks = tasks_response.json()

    assert {"task_1", "task_2", "task_3"}.issubset(tasks)
    assert "tasks" in tasks
    assert len(tasks["tasks"]) == 3
    assert sum(1 for task in tasks["tasks"] if bool(task["grader"])) == 3
    assert tasks["task_1"]["grader"] == "/grade/task_1"
    assert "description" in tasks["task_1"]
    assert tasks["task_1"]["grade_endpoint"] == "/grade/task_1"
    assert tasks["task_1"]["difficulty"] == "easy"

    for task_id in ("task_1", "task_2", "task_3"):
        grade_response = client.get(f"/grade/{task_id}")
        assert grade_response.status_code == 200
        grade_payload = grade_response.json()
        _assert_minimal_validator_grade_payload(grade_payload)

    grader_get_response = client.get("/grader", params={"task_id": "task_2"})
    assert grader_get_response.status_code == 200
    _assert_minimal_validator_grade_payload(grader_get_response.json())

def test_root_task_registry_and_grader_modules_expose_validator_tasks_and_live_grades() -> None:
    client = TestClient(app)
    task_ids = list_task_ids()

    assert task_ids == ["task_1", "task_2", "task_3"]

    for task_id in ["task_1", "task_2", "task_3"]:
        payload = grade_episode(task_id, split="eval", seed=42)
        assert 0.0 < payload["score"] < 1.0
        assert 0.0 < payload["reward"] < 1.0
        _assert_nested_scores_strictly_inside_open_interval(payload["breakdown"])
        _assert_nested_scores_strictly_inside_open_interval(payload["objective_breakdown"])
        assert payload["grader_mode"] == "live"

    for task_id in ["task_4", "task_5"]:
        payload = grade_episode(task_id, split="eval", seed=42)
        assert 0.0 < payload["score"] < 1.0
        assert 0.0 < payload["reward"] < 1.0
        _assert_nested_scores_strictly_inside_open_interval(payload["breakdown"])
        _assert_nested_scores_strictly_inside_open_interval(payload["objective_breakdown"])
        assert payload["grader_mode"] in {"live", "live-fallback"}
        assert payload["success"] is True

    reset_response = client.post("/reset", json={"task_id": "task_2", "seed": 42})
    assert reset_response.status_code == 200

    grader_response = client.post("/grader", json={"task_id": "task_1"})
    assert grader_response.status_code == 200
    _assert_minimal_validator_grade_payload(grader_response.json())

    validate_response = client.get("/validate")
    assert validate_response.status_code == 200
    validate_payload = validate_response.json()
    assert validate_payload["valid"] is True
    assert validate_payload["checks"]["min_3_tasks"] is True
    assert validate_payload["checks"]["all_tasks_have_graders"] is True

    openapi_schema = client.get("/openapi.json").json()
    reset_props = openapi_schema["components"]["schemas"]["ResetRequest"]["properties"]
    assert "task_id" in reset_props
    assert "split" in reset_props


def test_validator_grade_payloads_stay_strictly_inside_declared_score_bounds() -> None:
    for split in ("train", "eval"):
        for seed in range(1, 11):
            for task_id in ("task_1", "task_2", "task_3"):
                payload = validator_grade_payload(task_id, split=split, seed=seed)
                assert 0.02 < float(payload["score"]) < 0.98
                assert 0.02 < float(payload["reward"]) < 0.98

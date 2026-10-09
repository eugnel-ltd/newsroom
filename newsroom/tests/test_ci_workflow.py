from __future__ import annotations

from pathlib import Path
import tomllib
from typing import Any, Mapping

import yaml


REPO_ROOT = Path(__file__).parents[2]
WORKFLOW_ROOT = REPO_ROOT / ".github" / "workflows"
FOCUS_PATH = WORKFLOW_ROOT / "focus-gates.yml"
RESEARCH_PATH = WORKFLOW_ROOT / "ci.yml"
HEALTH_PATH = WORKFLOW_ROOT / "evidence.yml"
LIFECYCLE_PATH = WORKFLOW_ROOT / "pr-lifecycle.yml"
GATES_PATH = REPO_ROOT / ".sdlc" / "gates.toml"
LEGACY_DIAGNOSTICS = (
    "authority-a2a.yml",
    "authority-a2b.yml",
    "projection-b1.yml",
    "projection-b2-neo4j.yml",
)
PINNED_ACTIONS = {
    "actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd",
    "actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1",
    "astral-sh/setup-uv@08807647e7069bb48b6ef5acd8ec9567f424441b",
    "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a",
}


def _load(path: Path) -> dict[str, Any]:
    value = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert isinstance(value, dict)
    return value


def _steps(job: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    value = job["steps"]
    assert isinstance(value, list)
    assert all(isinstance(step, dict) for step in value)
    return value


def test_every_retained_workflow_job_has_a_positive_explicit_timeout() -> None:
    for path in sorted(WORKFLOW_ROOT.glob("*.yml")):
        jobs = _load(path)["jobs"]
        assert isinstance(jobs, dict)
        for job_id, job in jobs.items():
            assert isinstance(job, dict)
            timeout = job.get("timeout-minutes")
            assert timeout is not None, (path.name, job_id)
            assert int(timeout) > 0, (path.name, job_id, timeout)


def test_sdlc_workflows_check_out_exact_head_without_credentials() -> None:
    checkout = "actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd"
    expected_refs = {
        FOCUS_PATH: "${{ github.event.pull_request.head.sha || github.sha }}",
        RESEARCH_PATH: "${{ github.event.pull_request.head.sha || github.sha }}",
        HEALTH_PATH: "${{ github.sha }}",
    }
    for path, expected_ref in expected_refs.items():
        for job in _load(path)["jobs"].values():
            steps = _steps(job)
            checkouts = [step for step in steps if step.get("uses") == checkout]
            assert len(checkouts) == 1
            assert steps[0] == checkouts[0]
            with_block = checkouts[0]["with"]
            assert with_block["ref"] == expected_ref
            assert with_block["fetch-depth"] == "0"
            assert with_block["persist-credentials"] == "false"
            assert with_block["show-progress"] == "false"
            python = [
                step
                for step in steps
                if step.get("uses")
                == "actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1"
            ]
            assert len(python) == 1
            assert python[0]["with"] == {"python-version": "3.12"}


def test_focus_gate_is_one_job_and_one_conditional_bootstrap() -> None:
    workflow = _load(FOCUS_PATH)
    assert workflow["name"] == "Focus Gates"
    assert set(workflow["on"]) == {"pull_request", "workflow_dispatch"}
    assert set(workflow["jobs"]) == {"focus"}
    job = workflow["jobs"]["focus"]
    assert job["name"] == "focus-gates"
    assert workflow["concurrency"]["cancel-in-progress"] == "true"

    steps = _steps(job)
    rendered = FOCUS_PATH.read_text(encoding="utf-8")
    assert rendered.count("astral-sh/setup-uv@") == 1
    assert rendered.count("uv sync --dev --locked") == 1
    assert "scripts.sdlc.focus_gate_v2" in rendered
    assert "--output .focus/route.json" in rendered
    assert "owner-authorised" in rendered
    assert "graphiti-core" not in rendered
    assert "test_graphiti_combined_temporal" not in rendered
    assert "matrix:" not in rendered
    assert "poll" not in rendered.casefold()

    names = [str(step.get("name", "")) for step in steps]
    assert names.index("Prepare locked environment once") < names.index(
        "F0 exact change integrity"
    )
    setup_uv = next(step for step in steps if step.get("name") == "Set up uv once")
    assert setup_uv["if"] == "steps.route.outputs.bootstrap_required == 'true'"
    service = next(
        step for step in steps if step.get("name") == "Start bounded Neo4j service"
    )
    assert service["if"] == "steps.route.outputs.service_required == 'true'"
    for name in (
        "NEWSROOM_NEO4J_COMPLETE_SERVICE_REQUIRED",
        "NEWSROOM_NEO4J_INCREMENT_2D_SERVICE_REQUIRED",
        "NEWSROOM_NEO4J_RETRIEVAL_SERVICE_REQUIRED",
        "NEWSROOM_NEO4J_SERVICE_REQUIRED",
    ):
        assert name not in job["env"]
    execute = next(
        step for step in steps if step.get("name") == "Execute selected evidence"
    )
    assert all(
        "steps.route.outputs.service_required" in value
        for value in execute["env"].values()
    )
    upload = next(
        step
        for step in steps
        if step.get("name") == "Upload canonical Focus Gate evidence"
    )
    assert upload["if"] == "always()"
    assert upload["with"]["path"] == ".focus"
    assert upload["with"]["include-hidden-files"] == "true"


def test_graphiti_research_is_path_scoped_and_provider_free() -> None:
    workflow = _load(RESEARCH_PATH)
    assert workflow["name"] == "Graphiti Research"
    assert set(workflow["on"]) == {"pull_request", "schedule", "workflow_dispatch"}
    assert "push" not in workflow["on"]
    paths = workflow["on"]["pull_request"]["paths"]
    assert "pyproject.toml" in paths
    assert "uv.lock" in paths
    assert "docs/research/**" not in paths
    assert "docs/research/**/*.json" in paths
    assert "docs/research/**/*.csv" in paths
    rendered = RESEARCH_PATH.read_text(encoding="utf-8")
    assert "uv sync --dev --extra graphiti --locked" in rendered
    assert "graphiti-core" in rendered
    assert "test_graphiti_combined_temporal_runtime.py" in rendered
    assert "${{ secrets." not in rendered
    assert "CURSOR_API_KEY" not in rendered
    assert "OPENROUTER" not in rendered
    assert "provider-free" in rendered


def test_full_health_runs_after_main_merge_not_on_pull_request() -> None:
    workflow = _load(HEALTH_PATH)
    assert workflow["name"] == "Full Repository Health"
    assert set(workflow["on"]) == {"push", "schedule", "workflow_dispatch"}
    assert workflow["on"]["push"]["branches"] == ["main"]
    assert "pull_request" not in workflow["on"]
    assert "merge_group" not in workflow["on"]
    assert set(workflow["jobs"]) == {"full-health"}
    assert workflow["concurrency"] == {
        "group": "full-health-${{ github.ref }}",
        "cancel-in-progress": "true",
    }
    rendered = HEALTH_PATH.read_text(encoding="utf-8")
    assert rendered.count("astral-sh/setup-uv@") == 1
    assert rendered.count("uv sync --dev --locked") == 1
    assert "scripts.sdlc.focus_gate_v2" in rendered
    assert "--repo-root . full-health" in rendered
    assert "python -m pytest" not in rendered
    assert "matrix:" not in rendered


def test_full_health_job_timeout_exceeds_the_cancelled_twenty_minute_budget() -> None:
    job = _load(HEALTH_PATH)["jobs"]["full-health"]
    timeout = int(job["timeout-minutes"])
    assert timeout > 20


def test_new_sdlc_workflows_use_only_exactly_pinned_actions() -> None:
    observed: set[str] = set()
    for path in (FOCUS_PATH, RESEARCH_PATH, HEALTH_PATH, LIFECYCLE_PATH):
        jobs = _load(path)["jobs"]
        for job in jobs.values():
            for step in _steps(job):
                selected = step.get("uses")
                if selected is not None:
                    observed.add(selected)
                    assert selected in PINNED_ACTIONS
    assert observed == PINNED_ACTIONS


def test_pr_lifecycle_is_separate_trusted_policy_without_project_bootstrap() -> None:
    workflow = _load(LIFECYCLE_PATH)
    assert "pull_request_target" in workflow["on"]
    validate = workflow["jobs"]["validate"]
    assert validate["if"] == "github.event_name == 'pull_request_target'"
    rendered = LIFECYCLE_PATH.read_text(encoding="utf-8")
    assert "uv sync" not in rendered
    assert "pip install" not in rendered
    assert "scripts.sdlc.pr_lifecycle validate-event" in rendered
    assert "pull-requests: read" in rendered


def test_legacy_authority_and_projection_workflows_remain_manual_only() -> None:
    for filename in LEGACY_DIAGNOSTICS:
        workflow = _load(WORKFLOW_ROOT / filename)
        assert workflow["on"] == {"workflow_dispatch": ""}
        assert "push" not in workflow["on"]
        assert "pull_request" not in workflow["on"]


def test_retired_clustering_path_group_selects_no_evaluator_dependencies() -> None:
    contract = tomllib.loads(GATES_PATH.read_text(encoding="utf-8"))
    clustering_paths = set(contract["classification"]["paths"]["clustering"])

    assert clustering_paths == {"newsroom/legacy_operational_stack_retired.py"}


def test_focus_service_uses_content_pinned_official_mirror_without_hub_login():
    image=_load(FOCUS_PATH)['jobs']['focus']['env']['NEO4J_IMAGE']
    name,digest=image.split('@sha256:')
    assert name.startswith('public.ecr.aws/docker/library/neo4j:')
    assert len(digest)==64 and set(digest)<=set('0123456789abcdef')

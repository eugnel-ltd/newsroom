from __future__ import annotations

import sys
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.sdlc.focus_gate_v2 as focus_gate
import scripts.sdlc.focus_selector as selector


class _Graph:
    def __init__(self, dependents: dict[str, tuple[str, ...]] | None = None) -> None:
        self._dependents = dependents or {}

    def dependent_paths(self, path: str) -> tuple[str, ...]:
        return self._dependents.get(path, ())


def _write(root: Path, relative: str, content: str = "") -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _commit(root: Path, message: str) -> str:
    subprocess.run(("git", "add", "."), cwd=root, check=True)
    subprocess.run(
        ("git", "-c", "user.name=Focus", "-c", "user.email=focus@example.invalid",
         "commit", "-m", message),
        cwd=root,
        check=True,
        capture_output=True,
    )
    return subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def test_symbol_sensitive_route_keeps_changed_class_consumers_only(tmp_path: Path) -> None:
    subprocess.run(("git", "init", "-q"), cwd=tmp_path, check=True)
    _write(
        tmp_path,
        "newsroom/feature.py",
        "class Changed:\n    def value(self) -> int:\n        return 1\n\nclass Other:\n    pass\n\n"
        "def _wrapper():\n    return Changed\n\ndef public_factory():\n    return _wrapper()\n",
    )
    _write(
        tmp_path, "newsroom/consumer.py",
        "from .feature import Changed\n\ndef use() -> int:\n    return Changed().value()\n",
    )
    _write(tmp_path, "newsroom/downstream.py", "from newsroom.consumer import use\n")
    _write(
        tmp_path, "newsroom/dynamic.py",
        "from importlib import import_module\n\nfeature = import_module('newsroom.feature')\n",
    )
    _write(tmp_path, "newsroom/api/__init__.py", "from ..feature import Changed\n")
    _write(tmp_path, "newsroom/tests/test_direct.py", "from newsroom.feature import Changed\n")
    _write(tmp_path, "newsroom/tests/test_api.py", "from newsroom.api import Changed\n")
    _write(tmp_path, "newsroom/tests/test_relative.py", "from ..api import Changed\n")
    _write(tmp_path, "newsroom/tests/test_dynamic.py", "from newsroom.dynamic import feature\n")
    _write(tmp_path, "newsroom/tests/test_downstream.py", "from newsroom.downstream import use\n")
    _write(tmp_path, "newsroom/tests/test_module_style.py", "import newsroom.feature as feature\n\nVALUE = feature.Changed\n")
    _write(tmp_path, "newsroom/tests/test_local_reference.py", "from newsroom.feature import public_factory\n")
    _write(tmp_path, "newsroom/tests/test_private_reference.py", "from newsroom.feature import _wrapper\n")
    _write(
        tmp_path, "newsroom/tests/test_package_module.py",
        "from newsroom import feature as selected\n\nVALUE = getattr(selected, 'Other')\n",
    )
    _write(tmp_path, "newsroom/tests/test_other_neo4j_service.py", "from newsroom.feature import Other\n")
    base = _commit(tmp_path, "base")
    _write(
        tmp_path,
        "newsroom/feature.py",
        "class Changed:\n    def value(self) -> int:\n        return 2\n\nclass Other:\n    pass\n\n"
        "def _wrapper():\n    return Changed\n\ndef public_factory():\n    return _wrapper()\n",
    )
    head = _commit(tmp_path, "head")

    route = selector.select_focus(
        ("newsroom/feature.py",), repo_root=tmp_path, base_sha=base, head_sha=head
    )

    assert route["selected_tests"] == [
        "newsroom/tests/test_api.py",
        "newsroom/tests/test_direct.py",
        "newsroom/tests/test_downstream.py",
        "newsroom/tests/test_dynamic.py",
        "newsroom/tests/test_local_reference.py",
        "newsroom/tests/test_module_style.py",
        "newsroom/tests/test_package_module.py",
        "newsroom/tests/test_private_reference.py",
        "newsroom/tests/test_relative.py",
    ]
    assert route["selected_service_tests"] == []
    assert route["gates"] == ["F0", "F1", "F2"]


def test_symbol_alias_falls_back_to_broad_consumer_routing(tmp_path: Path) -> None:
    subprocess.run(("git", "init", "-q"), cwd=tmp_path, check=True)
    _write(
        tmp_path, "newsroom/feature.py",
        "class Changed:\n    def value(self):\n        return 1\n\nclass Other:\n    pass\n\nAlias = Changed\n",
    )
    _write(tmp_path, "newsroom/tests/test_alias.py", "from newsroom.feature import Alias\n")
    _write(tmp_path, "newsroom/tests/test_other_neo4j_service.py", "from newsroom.feature import Other\n")
    base = _commit(tmp_path, "base")
    _write(
        tmp_path, "newsroom/feature.py",
        "class Changed:\n    def value(self):\n        return 2\n\nclass Other:\n    pass\n\nAlias = Changed\n",
    )
    head = _commit(tmp_path, "head")

    route = selector.select_focus(
        ("newsroom/feature.py",), repo_root=tmp_path, base_sha=base, head_sha=head
    )

    assert route["selected_tests"] == ["newsroom/tests/test_alias.py"]
    assert route["selected_service_tests"] == [
        "newsroom/tests/test_other_neo4j_service.py"
    ]


def test_private_helper_keeps_direct_tests_and_local_callers_only(tmp_path: Path) -> None:
    subprocess.run(("git", "init", "-q"), cwd=tmp_path, check=True)
    source = "def _render():\n    return 1\n\ndef render():\n    return _render()\n\ndef other():\n    return 0\n"
    _write(tmp_path, "newsroom/feature.py", source)
    _write(tmp_path, "newsroom/tests/test_private.py", "from newsroom.feature import _render\n")
    _write(tmp_path, "newsroom/tests/test_public.py", "from newsroom.feature import render\n")
    _write(tmp_path, "newsroom/tests/test_other_neo4j_service.py", "from newsroom.feature import other\n")
    base = _commit(tmp_path, "base")
    _write(tmp_path, "newsroom/feature.py", source.replace("return 1", "return 2"))
    head = _commit(tmp_path, "head")

    route = selector.select_focus(("newsroom/feature.py",), repo_root=tmp_path, base_sha=base, head_sha=head)

    assert route["selected_tests"] == ["newsroom/tests/test_private.py", "newsroom/tests/test_public.py"]
    assert route["selected_service_tests"] == []
    assert route["full_health_required"] is False


@pytest.mark.parametrize("import_statement, selected_call, other_call", (
    ("from .feature import render as selected", "selected()", "0"),
    ("import newsroom.feature as selected", "selected.render()", "selected.other()"),
    ("from newsroom import feature as selected", "selected.render()", "selected.other()"),
    ("import newsroom.feature", "newsroom.feature.render()", "newsroom.feature.other()"),
))
def test_symbol_closure_filters_each_importer_hop_and_keeps_reexport_aliases(
    tmp_path: Path, import_statement: str, selected_call: str, other_call: str,
) -> None:
    subprocess.run(("git", "init", "-q"), cwd=tmp_path, check=True)
    _write(tmp_path, "newsroom/__init__.py")
    source = "def render():\n    return 1\n\ndef other():\n    return 0\n"
    _write(tmp_path, "newsroom/feature.py", source)
    _write(tmp_path, "newsroom/middle.py", f"{import_statement}\n\ndef affected():\n    return {selected_call}\n\ndef other():\n    return {other_call}\n")
    _write(tmp_path, "newsroom/downstream.py", "from .middle import affected as alias\n\ndef consume():\n    return alias()\n\ndef other():\n    return 0\n")
    _write(tmp_path, "newsroom/api/__init__.py", "from ..middle import affected as Export\n")
    _write(tmp_path, "newsroom/tests/test_direct.py", "from newsroom.feature import render\n")
    _write(tmp_path, "newsroom/tests/test_middle.py", "from newsroom.middle import affected\n")
    _write(tmp_path, "newsroom/tests/test_downstream.py", "from newsroom.downstream import consume\n")
    _write(tmp_path, "newsroom/tests/test_api.py", "from newsroom.api import Export\n")
    _write(tmp_path, "newsroom/tests/test_other_neo4j_service.py", "from newsroom.downstream import other\n")
    base = _commit(tmp_path, "base")
    _write(tmp_path, "newsroom/feature.py", source.replace("return 1", "return 2"))
    head = _commit(tmp_path, "head")

    route = selector.select_focus(("newsroom/feature.py",), repo_root=tmp_path, base_sha=base, head_sha=head)

    assert route["selected_tests"] == ["newsroom/tests/test_api.py", "newsroom/tests/test_direct.py",
                                        "newsroom/tests/test_downstream.py", "newsroom/tests/test_middle.py"]
    assert route["selected_service_tests"] == []
    assert route["gates"] == ["F0", "F1", "F2"]


@pytest.mark.parametrize("middle", (
    "from .feature import *\n",
    "from .feature import render as local\nAlias = local\n",
    "from importlib import import_module\nloaded = import_module('newsroom.feature')\n",
    "from . import feature\n\ndef reflect():\n    return getattr(feature, 'render')\n",
))
def test_uncertain_importer_surface_keeps_broad_descendant_fallback(tmp_path: Path, middle: str) -> None:
    subprocess.run(("git", "init", "-q"), cwd=tmp_path, check=True)
    _write(tmp_path, "newsroom/__init__.py")
    source = "def render():\n    return 1\n"
    _write(tmp_path, "newsroom/feature.py", source)
    _write(tmp_path, "newsroom/middle.py", middle + "\ndef other():\n    return 0\n")
    _write(tmp_path, "newsroom/tests/test_other_neo4j_service.py", "from newsroom.middle import other\n")
    base = _commit(tmp_path, "base")
    _write(tmp_path, "newsroom/feature.py", source.replace("return 1", "return 2"))
    head = _commit(tmp_path, "head")
    route = selector.select_focus(("newsroom/feature.py",), repo_root=tmp_path, base_sha=base, head_sha=head)
    assert route["selected_service_tests"] == ["newsroom/tests/test_other_neo4j_service.py"]
    assert "F3" in route["gates"]


@pytest.mark.parametrize("source", (
    "value = 0\n\ndef _update():\n    global value\n    value = 1\n\ndef read():\n    return value\n",
    "state = {'value': 0}\n\ndef _update():\n    state['value'] = 1\n\ndef read():\n    return state['value']\n",
))
def test_private_global_write_is_not_a_closed_local_surface(tmp_path: Path, source: str) -> None:
    subprocess.run(("git", "init", "-q"), cwd=tmp_path, check=True)
    _write(tmp_path, "newsroom/feature.py", source)
    _write(tmp_path, "newsroom/tests/test_reader.py", "from newsroom.feature import read\n")
    base = _commit(tmp_path, "base")
    _write(tmp_path, "newsroom/feature.py", source.replace("= 1", "= 2"))
    head = _commit(tmp_path, "head")
    route = selector.select_focus(("newsroom/feature.py",), repo_root=tmp_path, base_sha=base, head_sha=head)
    assert route["selected_tests"] == ["newsroom/tests/test_reader.py"]


@pytest.mark.parametrize("lookup", (
    "getattr(sys.modules[__name__], '_render')()",
    "sys.modules[__name__].__dict__['_render']()",
    "vars(sys.modules[__name__])['_render']()",
))
def test_self_module_reflection_keeps_conservative_consumer_route(tmp_path: Path, lookup: str) -> None:
    subprocess.run(("git", "init", "-q"), cwd=tmp_path, check=True)
    source = f"import sys\n\ndef _render():\n    return 1\n\ndef public():\n    return {lookup}\n"
    _write(tmp_path, "newsroom/feature.py", source)
    _write(tmp_path, "newsroom/tests/test_public.py", "from newsroom.feature import public\n")
    base = _commit(tmp_path, "base")
    _write(tmp_path, "newsroom/feature.py", source.replace("return 1", "return 2"))
    head = _commit(tmp_path, "head")
    route = selector.select_focus(("newsroom/feature.py",), repo_root=tmp_path, base_sha=base, head_sha=head)
    assert route["selected_tests"] == ["newsroom/tests/test_public.py"]


def test_cyclic_reexport_closure_retains_consumer_without_unrelated_service(tmp_path: Path) -> None:
    subprocess.run(("git", "init", "-q"), cwd=tmp_path, check=True)
    _write(tmp_path, "newsroom/__init__.py")
    source = "def render():\n    return 1\n"
    _write(tmp_path, "newsroom/feature.py", source)
    _write(tmp_path, "newsroom/left.py", "from .feature import render\n\ndef use():\n    return render()\n\nfrom .right import alias\n")
    _write(tmp_path, "newsroom/right.py", "from .left import use as alias\n\ndef other():\n    return 0\n")
    _write(tmp_path, "newsroom/tests/test_consumer.py", "from newsroom.right import alias\n")
    _write(tmp_path, "newsroom/tests/test_other_neo4j_service.py", "from newsroom.right import other\n")
    base = _commit(tmp_path, "base")
    _write(tmp_path, "newsroom/feature.py", source.replace("return 1", "return 2"))
    head = _commit(tmp_path, "head")
    route = selector.select_focus(("newsroom/feature.py",), repo_root=tmp_path, base_sha=base, head_sha=head)
    assert route["selected_tests"] == ["newsroom/tests/test_consumer.py"]
    assert route["selected_service_tests"] == []


def test_short_constant_reexport_selects_exact_consumer_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write(tmp_path, "newsroom/example/models.py", "X = 7\n")
    _write(
        tmp_path,
        "newsroom/example/__init__.py",
        "from .models import X\n\nOther = 9\n",
    )
    _write(
        tmp_path,
        "newsroom/tests/test_constant.py",
        "from newsroom.example import X\n\n"
        "def test_constant() -> None:\n    assert X == 7\n",
    )
    _write(
        tmp_path,
        "newsroom/tests/test_unrelated_package_import.py",
        "from newsroom.example import Other\n\n"
        "def test_other() -> None:\n    assert Other == 9\n",
    )
    monkeypatch.setattr(
        selector,
        "build_dependency_graph",
        lambda _root: _Graph(
            {
                "newsroom/example/models.py": (
                    "newsroom/example/__init__.py",
                )
            }
        ),
    )

    route = selector.select_focus(
        ("newsroom/example/models.py",),
        repo_root=tmp_path,
    )

    assert route["selected_tests"] == ["newsroom/tests/test_constant.py"]
    assert route["full_health_required"] is False


def test_stateful_route_uses_direct_tests_and_two_bounded_sentinels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write(tmp_path, "newsroom/authority/example.py", "def commit() -> int:\n    return 1\n")
    _write(
        tmp_path,
        "newsroom/tests/test_example_authority_consumer.py",
        "from newsroom.authority.example import commit\n\n"
        "def test_commit() -> None:\n    assert commit() == 1\n",
    )
    _write(tmp_path, "newsroom/tests/test_authority_migration_compatibility.py")
    _write(tmp_path, "newsroom/tests/test_authority_store_conformance.py")
    for index in range(5):
        _write(tmp_path, f"newsroom/tests/test_authority_unrelated_{index}.py")
    monkeypatch.setattr(
        selector,
        "build_dependency_graph",
        lambda _root: _Graph(),
    )

    route = selector.select_focus(
        ("newsroom/authority/example.py",),
        repo_root=tmp_path,
    )

    assert set(route["selected_tests"]) == {
        "newsroom/tests/test_example_authority_consumer.py",
        "newsroom/tests/test_authority_migration_compatibility.py",
        "newsroom/tests/test_authority_store_conformance.py",
    }
    assert not any("unrelated" in path for path in route["selected_tests"])
    assert route["full_health_required"] is False
    assert "bounded_stateful_sentinels:F2" in route["reasons"]


def test_stateful_route_without_direct_evidence_escalates_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write(tmp_path, "newsroom/authority/uncovered.py", "VALUE = 1\n")
    _write(tmp_path, "newsroom/tests/test_authority_migration_compatibility.py")
    _write(tmp_path, "newsroom/tests/test_authority_store_conformance.py")
    for index in range(5):
        _write(tmp_path, f"newsroom/tests/test_authority_unrelated_{index}.py")
    monkeypatch.setattr(selector, "build_dependency_graph", lambda _root: _Graph())

    route = selector.select_focus(
        ("newsroom/authority/uncovered.py",),
        repo_root=tmp_path,
    )

    assert route["full_health_required"] is True
    assert set(route["selected_tests"]) == {
        "newsroom/tests/test_authority_migration_compatibility.py",
        "newsroom/tests/test_authority_store_conformance.py",
    }
    assert not any("unrelated" in path for path in route["selected_tests"])
    assert "stateful_without_direct_evidence:full_health" in route["reasons"]


def test_discovered_actual_service_consumer_promotes_route_to_f3(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write(tmp_path, "newsroom/feature.py", "def value() -> int:\n    return 1\n")
    _write(
        tmp_path,
        "newsroom/tests/test_feature_neo4j_service.py",
        "from newsroom.feature import value\n\n"
        "def test_service() -> None:\n    assert value() == 1\n",
    )
    monkeypatch.setattr(
        selector,
        "build_dependency_graph",
        lambda _root: _Graph(),
    )

    route = selector.select_focus(
        ("newsroom/feature.py",),
        repo_root=tmp_path,
    )

    assert "F3" in route["gates"]
    assert route["selected_tests"] == []
    assert route["selected_service_tests"] == [
        "newsroom/tests/test_feature_neo4j_service.py"
    ]
    assert "actual_service_consumer:F3" in route["reasons"]


def test_shared_dependency_change_truthfully_selects_research_and_full_health() -> None:
    route = selector.select_focus(("pyproject.toml",))

    assert route["research_required"] is True
    assert route["full_health_required"] is True
    assert route["bootstrap_required"] is True
    assert {"F0", "F1", "F2"} <= set(route["gates"])


def test_research_markdown_remains_documentation_only() -> None:
    route = selector.select_focus(("docs/research/notes.md",))

    assert route["research_required"] is False
    assert route["gates"] == ["F0"]
    assert route["bootstrap_required"] is False


def test_machine_research_evidence_matches_workflow_trigger() -> None:
    route = selector.select_focus(("docs/research/result.json",))

    assert route["research_required"] is True
    assert route["gates"] == ["F0"]
    assert route["selected_tests"] == []
    assert route["bootstrap_required"] is False


def test_f0_uses_locked_interpreter_only_when_bootstrap_is_required() -> None:
    workflow = (
        Path(__file__).parents[2] / ".github/workflows/focus-gates.yml"
    ).read_text(encoding="utf-8")

    assert "BOOTSTRAP_REQUIRED: ${{ steps.route.outputs.bootstrap_required }}" in workflow
    assert 'if [[ "${BOOTSTRAP_REQUIRED}" == "true" ]]' in workflow
    assert "uv run --no-sync python -m scripts.sdlc.focus_gate_v2" in workflow
    assert "python -m scripts.sdlc.focus_gate_v2" in workflow


@pytest.mark.parametrize(
    "changed",
    (
        "newsroom/control_plane/issue_790_disposition.py",
        "newsroom/control_plane/cycle.py",
        "newsroom/control_plane/graphiti.py",
    ),
)
def test_prepared_canary_parity_is_required_for_pre_provider_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed: str,
) -> None:
    _write(tmp_path, changed, "VALUE = 1\n")
    _write(tmp_path, "newsroom/tests/test_issue_790_prepared_canary.py")
    _write(tmp_path, "newsroom/tests/test_issue_790_retry_forbidden_safety_state.py")
    monkeypatch.setattr(selector, "build_dependency_graph", lambda _root: _Graph())

    route = selector.select_focus((changed,), repo_root=tmp_path)

    assert "newsroom/tests/test_issue_790_prepared_canary.py" in route["selected_tests"]
    assert (
        "newsroom/tests/test_issue_790_retry_forbidden_safety_state.py"
        in route["selected_tests"]
    )
    assert "prepared_canary_parity:F1" in route["reasons"]


@pytest.mark.parametrize(
    "changed",
    (
        "newsroom/control_plane/issue_790_disposition.py",
        "newsroom/control_plane/issue_790_canary.py",
        "newsroom/control_plane/issue_790_prepared_canary.py",
        "newsroom/control_plane/issue_790_rehearsal.py",
        "scripts/issue_790_live_canary_preflight.py",
        "scripts/issue_790_prepared_canary_rehearsal.py",
    ),
)
def test_prepared_canary_route_selects_model_usage_receipt_consumer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed: str,
) -> None:
    _write(tmp_path, changed, "VALUE = 1\n")
    _write(tmp_path, "newsroom/tests/test_model_usage_receipts.py")
    monkeypatch.setattr(selector, "build_dependency_graph", lambda _root: _Graph())

    route = selector.select_focus((changed,), repo_root=tmp_path)

    assert "F2" in route["gates"]
    assert "newsroom/tests/test_model_usage_receipts.py" in route["selected_tests"]
    assert "prepared_canary_consumers:F2" in route["reasons"]


def test_f0_validates_yaml_and_shell_syntax(tmp_path: Path) -> None:
    valid_yaml = tmp_path / "valid.yml"
    invalid_yaml = tmp_path / "invalid.yml"
    valid_shell = tmp_path / "valid.sh"
    invalid_shell = tmp_path / "invalid.sh"

    valid_yaml.write_text("jobs:\n  test:\n    runs-on: ubuntu-latest\n", encoding="utf-8")
    invalid_yaml.write_text("jobs: [\n", encoding="utf-8")
    valid_shell.write_text("#!/usr/bin/env bash\nset -euo pipefail\necho ok\n", encoding="utf-8")
    invalid_shell.write_text("#!/usr/bin/env bash\nif then\n", encoding="utf-8")

    focus_gate._validate_yaml(valid_yaml)
    focus_gate._validate_shell(valid_shell)
    with pytest.raises(focus_gate.FocusGateError, match="invalid YAML"):
        focus_gate._validate_yaml(invalid_yaml)
    with pytest.raises(focus_gate.FocusGateError, match="invalid shell syntax"):
        focus_gate._validate_shell(invalid_shell)


def test_selected_evidence_uses_fixed_parallel_then_serial_commands(
    tmp_path: Path,
) -> None:
    commands = focus_gate.build_selected_test_commands(
        tmp_path,
        selected_tests=("newsroom/tests/test_b.py", "newsroom/tests/test_a.py"),
        selected_service_tests=("newsroom/tests/test_neo4j_service.py",),
        junit=".focus/pytest.xml",
    )

    assert commands == (
        (
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--assert=plain",
            "-p",
            "no:cacheprovider",
            "-n",
            "4",
            "--dist",
            "worksteal",
            "--max-worker-restart=0",
            "newsroom/tests/test_a.py",
            "newsroom/tests/test_b.py",
            f"--junitxml={tmp_path / '.focus/pytest-provider-free.xml'}",
        ),
        (
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--assert=plain",
            "-p",
            "no:cacheprovider",
            "newsroom/tests/test_neo4j_service.py",
            f"--junitxml={tmp_path / '.focus/pytest-service.xml'}",
        ),
    )


def test_focus_workflow_keeps_finite_budget_for_two_phase_execution() -> None:
    workflow = (
        Path(__file__).parents[2] / ".github/workflows/focus-gates.yml"
    ).read_text(encoding="utf-8")

    assert "timeout-minutes: 45" in workflow
    assert "--junit .focus/pytest.xml" in workflow


def test_two_phase_junit_preserves_requested_report_path(tmp_path: Path) -> None:
    provider_free = tmp_path / "pytest-provider-free.xml"
    service = tmp_path / "pytest-service.xml"
    requested = tmp_path / "pytest.xml"
    provider_free.write_text(
        '<testsuites><testsuite name="provider-free" tests="2" failures="0" '
        'errors="0" skipped="1" time="1.25" /></testsuites>',
        encoding="utf-8",
    )
    service.write_text(
        '<testsuites><testsuite name="service" tests="1" failures="0" '
        'errors="0" skipped="0" time="2.5" /></testsuites>',
        encoding="utf-8",
    )

    focus_gate._merge_junit_reports(requested, (provider_free, service))

    root = ET.parse(requested).getroot()
    assert root.attrib == {
        "tests": "3",
        "failures": "0",
        "errors": "0",
        "skipped": "1",
        "time": "3.750000",
    }
    assert [suite.attrib["name"] for suite in root] == [
        "provider-free",
        "service",
    ]


def test_failed_phase_cannot_publish_stale_junit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested = tmp_path / "pytest.xml"
    stale = tmp_path / "pytest-provider-free.xml"
    requested.write_text("stale requested", encoding="utf-8")
    stale.write_text("stale phase", encoding="utf-8")
    command = ("pytest", f"--junitxml={stale}")

    monkeypatch.setattr(
        focus_gate,
        "verify_route",
        lambda _root, _route: {
            "research_required": False,
            "bootstrap_required": True,
            "full_health_required": False,
            "selected_tests": ["newsroom/tests/test_a.py"],
            "selected_service_tests": [],
        },
    )
    monkeypatch.setattr(
        focus_gate,
        "build_selected_test_commands",
        lambda *_args, **_kwargs: (command,),
    )
    monkeypatch.setattr(
        focus_gate.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=3),
    )

    assert focus_gate.execute_route(tmp_path, "route.json", junit=requested) == 3
    assert not requested.exists()
    assert not stale.exists()


@pytest.mark.parametrize("route", [selector.select_focus, selector.legacy.select_focus])
def test_internal_publication_is_not_a_public_effect_gate(route) -> None:
    paths = (
        "newsroom/increment10/publication.py",
        "newsroom/tests/test_increment10_publication.py",
    )
    internal = route(paths)
    assert internal["owner_authority_required"] is False
    assert "F4" not in internal["gates"]
    for effect in (
        "deploy/hermes.plist",
        "newsroom/control_plane/keychain.py",
        "scripts/production_operational_admission.py",
    ):
        mixed = route((*paths, effect))
        assert mixed["owner_authority_required"] is True
        assert "F4" in mixed["gates"]


@pytest.mark.parametrize("symbol_sensitive", (False, True))
def test_changed_test_helper_routes_exact_direct_and_transitive_consumers(
    tmp_path: Path, symbol_sensitive: bool,
) -> None:
    subprocess.run(("git", "init", "-q"), cwd=tmp_path, check=True)
    _write(tmp_path, "newsroom/__init__.py")
    _write(tmp_path, "newsroom/tests/__init__.py")
    helper = "newsroom/tests/fixture.py"
    original = "def changed():\n    return 1\n\ndef other():\n    return 9\n"
    _write(tmp_path, helper, original)
    _write(tmp_path, "newsroom/tests/bridge.py", "from .fixture import changed as retained\n\ndef build():\n    return retained()\n")
    _write(tmp_path, "newsroom/tests/relay.py", "from newsroom.tests import bridge as selected\n\ndef wrapped():\n    return selected.build()\n")
    _write(tmp_path, "newsroom/tests/other_fixture.py", "def unrelated(): return 9\n")
    _write(tmp_path, "newsroom/tests/test_direct.py", "from .fixture import changed as selected\n")
    _write(tmp_path, "newsroom/tests/test_module.py", "import newsroom.tests.fixture as selected\n")
    _write(tmp_path, "newsroom/tests/test_package_child.py", "from . import fixture as selected\n")
    _write(tmp_path, "newsroom/tests/test_dynamic.py", "import importlib\nselected = importlib.import_module('newsroom.tests.fixture')\n")
    _write(tmp_path, "newsroom/tests/test_transitive.py", "from .relay import wrapped\n")
    _write(tmp_path, "newsroom/tests/test_helper_neo4j_service.py", "from .bridge import build\n")
    _write(tmp_path, "newsroom/tests/test_other_symbol.py", "from .fixture import other\n")
    _write(tmp_path, "newsroom/tests/test_unrelated.py", "from .other_fixture import unrelated\n")
    base = _commit(tmp_path, "base helper")
    _write(tmp_path, helper, original.replace("return 1", "return 2"))
    head = _commit(tmp_path, "changed helper")

    route = selector.select_focus(
        (helper,), repo_root=tmp_path,
        **({"base_sha": base, "head_sha": head} if symbol_sensitive else {}),
    )
    assert route["full_health_required"] is False
    assert "unresolved_dependency_analysis:full_health" not in route["reasons"]
    assert route["selected_tests"] == sorted([
        "newsroom/tests/test_direct.py", "newsroom/tests/test_dynamic.py",
        "newsroom/tests/test_module.py", "newsroom/tests/test_package_child.py",
        "newsroom/tests/test_transitive.py",
        *([] if symbol_sensitive else ["newsroom/tests/test_other_symbol.py"]),
    ])
    assert route["selected_service_tests"] == ["newsroom/tests/test_helper_neo4j_service.py"]
    assert route["gates"] == ["F0", "F1", "F2", "F3"]
    assert "actual_service_consumer:F3" in route["reasons"]


@pytest.mark.parametrize("fault", ("syntax", "missing_module", "deleted_helper"))
def test_test_helper_dependency_fault_keeps_full_health_fallback(tmp_path: Path, fault: str) -> None:
    _write(tmp_path, "newsroom/__init__.py")
    _write(tmp_path, "newsroom/tests/__init__.py")
    helper = "newsroom/tests/fixture.py"
    if fault != "deleted_helper":
        _write(tmp_path, helper, "def broken(\n" if fault == "syntax" else "import newsroom.missing\n")
    _write(tmp_path, "newsroom/tests/test_consumer.py", "from .fixture import value\n")
    route = selector.select_focus((helper,), repo_root=tmp_path)
    assert route["full_health_required"] is True
    assert "unresolved_dependency_analysis:full_health" in route["reasons"]


def test_broad_package_change_retains_absolute_and_relative_submodule_consumers(tmp_path: Path) -> None:
    _write(tmp_path, "newsroom/__init__.py")
    _write(tmp_path, "newsroom/package/__init__.py", "SETTING = 1\n")
    _write(tmp_path, "newsroom/package/child.py", "VALUE = 1\n")
    _write(tmp_path, "newsroom/tests/test_absolute.py", "import newsroom.package.child as selected\n")
    _write(tmp_path, "newsroom/tests/test_relative.py", "from ..package.child import VALUE\n")
    route = selector.select_focus(("newsroom/package/__init__.py",), repo_root=tmp_path)
    assert route["selected_tests"] == [
        "newsroom/tests/test_absolute.py", "newsroom/tests/test_relative.py",
    ]
    assert route["full_health_required"] is False


@pytest.mark.parametrize("dependent_count", (1, 100))
def test_broad_test_import_walk_is_once_per_file_not_per_dependent(
    tmp_path: Path, monkeypatch, dependent_count: int,
) -> None:
    import ast

    helper = "newsroom/tests/fixture.py"
    _write(tmp_path, helper, "VALUE = 1\n")
    body = "import unrelated\n\ndef test_counted_tree():\n" + "    value = 1\n" * 500
    for name in ("test_first.py", "test_second.py"):
        _write(tmp_path, "newsroom/tests/" + name, body)
    monkeypatch.setattr(selector, "build_dependency_graph", lambda _: _Graph({
        helper: tuple(f"newsroom/dependent_{number}.py" for number in range(dependent_count)),
    }))
    monkeypatch.setattr(selector, "_changed_public_symbols", lambda *_: None)
    original = ast.walk
    walks = []

    def counted(tree):
        if isinstance(tree, ast.Module) and any(
            isinstance(node, ast.FunctionDef) and node.name == "test_counted_tree"
            for node in tree.body
        ):
            walks.append(tree)
        yield from original(tree)

    monkeypatch.setattr(ast, "walk", counted)
    selected, unresolved = selector._discover_tests(tmp_path, (helper,))
    assert selected == set() and unresolved is False
    assert len(walks) == 2


def test_invalid_test_relative_import_keeps_unresolved_fallback(tmp_path: Path) -> None:
    _write(tmp_path, "newsroom/tests/fixture.py", "VALUE = 1\n")
    _write(tmp_path, "newsroom/tests/test_consumer.py", "from ....fixture import VALUE\n")
    route = selector.select_focus(("newsroom/tests/fixture.py",), repo_root=tmp_path)
    assert route["full_health_required"] is True
    assert "unresolved_dependency_analysis:full_health" in route["reasons"]


def _previous_public_symbol_match(tree, package, symbols, importer):
    """Frozen single-package semantics, before combining the AST traversals."""
    import ast

    if not symbols:
        return False
    aliases = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and selector._imported_from(node, importer) == package:
            if any(alias.name == "*" or alias.name in symbols for alias in node.names):
                return True
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == package:
                    aliases.add(alias.asname or alias.name.split(".")[0])
    package_parts = tuple(package.split("."))
    for node in ast.walk(tree):
        chain = selector._attribute_chain(node)
        if chain and chain[-1] in symbols:
            if chain[:-1] == package_parts or (len(chain) == 2 and chain[0] in aliases):
                return True
    return False


@pytest.mark.parametrize("source,importer,symbols,expected", (
    ("from newsroom.api import Changed", None, {"Changed"}, True),
    ("from newsroom.other_api import Changed as Selected", None, {"Changed"}, True),
    ("from newsroom.api import *", None, {"Changed"}, True),
    ("import newsroom.api as selected\nvalue = selected.Changed", None, {"Changed"}, True),
    ("import newsroom.other_api\nvalue = newsroom.other_api.Changed", None, {"Changed"}, True),
    ("value = newsroom.api.Changed", None, {"Changed"}, True),
    ("import newsroom.api\nvalue = newsroom.Changed", None, {"Changed"}, True),
    ("from ..api import Changed", "newsroom.tests.test_example", {"Changed"}, True),
    ("from ..other_api import Changed as Selected", "newsroom.tests.test_example", {"Changed"}, True),
    ("from ..api import Changed", None, {"Changed"}, False),
    ("from newsroom.api import Other", None, {"Changed"}, False),
    ("from newsroom.api.child import Changed", None, {"Changed"}, False),
    ("import newsroom.unrelated as selected\nvalue = selected.Changed", None, {"Changed"}, False),
    ("import newsroom.api as selected\nvalue = selected.Other", None, {"Changed"}, False),
    ("import newsroom.api as selected\nvalue = selected.child.Changed", None, {"Changed"}, False),
    ("from newsroom.api import *", None, set(), False),
))
def test_reexport_union_matches_previous_package_disjunction(source, importer, symbols, expected):
    import ast

    tree = ast.parse(source)
    packages = {"newsroom.api", "newsroom.other_api"}
    previous = any(
        _previous_public_symbol_match(tree, package, symbols, importer)
        for package in packages
    )
    assert previous is expected
    assert selector._imports_any_public_symbol(tree, packages, symbols, importer) is previous


def test_empty_reexport_packages_do_not_match_or_walk(monkeypatch):
    import ast

    tree = ast.parse("from newsroom.api import Changed")

    def forbidden(_tree):
        raise AssertionError("empty package set needs no AST traversal")

    with monkeypatch.context() as patch:
        patch.setattr(ast, "walk", forbidden)
        result = selector._imports_any_public_symbol(tree, set(), {"Changed"})
    assert result is False


@pytest.mark.parametrize("package_count", (0, 1, 100))
def test_reexport_import_walks_are_constant_per_test_file(tmp_path, monkeypatch, package_count):
    import ast

    source = "newsroom/feature.py"
    _write(tmp_path, source, "class Changed:\n    pass\n")
    _write(tmp_path, "newsroom/tests/test_consumer.py", "import unrelated\n\ndef test_counted_tree():\n    pass\n")
    monkeypatch.setattr(selector, "build_dependency_graph", lambda _: _Graph({
        source: tuple(f"newsroom/public_{number}/__init__.py" for number in range(package_count)),
    }))
    monkeypatch.setattr(selector, "_changed_public_symbols", lambda *_: None)
    original = ast.walk
    walks = []

    def counted(tree):
        if isinstance(tree, ast.Module) and any(
            isinstance(node, ast.FunctionDef) and node.name == "test_counted_tree"
            for node in tree.body
        ):
            walks.append(tree)
        yield from original(tree)

    monkeypatch.setattr(ast, "walk", counted)
    selected, unresolved = selector._discover_tests(tmp_path, (source,))
    assert selected == set() and unresolved is False
    # One existing import inventory pass, then at most two reexport passes.
    assert len(walks) == (3 if package_count else 1)


def test_added_private_declaration_keeps_unchanged_land_importers_out_of_route(tmp_path):
    subprocess.run(('git', 'init', '-q'), cwd=tmp_path, check=True)
    source = ('LAND = "land"\n'
              'def _landed_units():\n    return ()\n'
              'class Journal:\n    def read(self):\n        return 1\n')
    _write(tmp_path, 'newsroom/progress.py', source)
    _write(tmp_path, 'newsroom/accounting.py',
           'from .progress import LAND, _landed_units\n\ndef unrelated():\n    return LAND, _landed_units()\n')
    _write(tmp_path, 'newsroom/consumer.py',
           'from .progress import Journal\n\ndef selected():\n    return Journal().read()\n')
    _write(tmp_path, 'newsroom/tests/test_direct.py', 'from newsroom.progress import Journal\n')
    _write(tmp_path, 'newsroom/tests/test_caller.py', 'from newsroom.consumer import selected\n')
    _write(tmp_path, 'newsroom/tests/test_other_neo4j_service.py', 'from newsroom.accounting import unrelated\n')
    base = _commit(tmp_path, 'base')
    _write(tmp_path, 'newsroom/progress.py', source.replace('return 1', 'return _selected()')
           + '\ndef _selected():\n    return 2\n')
    _write(tmp_path, 'newsroom/tests/test_private.py', 'from newsroom.progress import _selected\n')
    head = _commit(tmp_path, 'added private body reader')
    route = selector.select_focus(('newsroom/progress.py',), repo_root=tmp_path,
                                  base_sha=base, head_sha=head)
    assert route['selected_tests'] == ['newsroom/tests/test_caller.py', 'newsroom/tests/test_direct.py',
                                       'newsroom/tests/test_private.py']
    assert route['selected_service_tests'] == []


@pytest.mark.parametrize('change', ('literal', 'import'))
def test_declaration_change_retains_bound_callers_not_unrelated_consumers(tmp_path, change):
    subprocess.run(('git', 'init', '-q'), cwd=tmp_path, check=True)
    _write(tmp_path, 'newsroom/headers.py', 'def headers():\n    return ()\n\ndef landed():\n    return ()\n')
    source = ('from .headers import landed\nSCHEMA = ("CREATE TABLE current",)\n'
              'def ensure_schema():\n    return SCHEMA\n'
              'class Intake:\n    def namespace(self):\n        return landed()\n'
              'def verified_native_observation():\n    return 1\n')
    _write(tmp_path, 'newsroom/state.py', source)
    _write(tmp_path, 'newsroom/store.py', 'from .state import ensure_schema\n\ndef connect():\n    return ensure_schema()\n')
    _write(tmp_path, 'newsroom/weather.py', 'from .state import verified_native_observation\n\ndef unchanged():\n    return verified_native_observation()\n')
    _write(tmp_path, 'newsroom/tests/test_state.py', 'from newsroom.state import ensure_schema\n')
    _write(tmp_path, 'newsroom/tests/test_intake.py', 'from newsroom.state import Intake\n')
    _write(tmp_path, 'newsroom/tests/test_store.py', 'from newsroom.store import connect\n')
    _write(tmp_path, 'newsroom/tests/test_other_neo4j_service.py', 'from newsroom.weather import unchanged\n')
    base = _commit(tmp_path, 'base')
    changed = (source.replace('"CREATE TABLE current",', '"CREATE TABLE current", "CREATE INDEX revision",')
               if change == 'literal' else source.replace('import landed', 'import landed, headers')
               .replace('return landed()', 'return headers()'))
    _write(tmp_path, 'newsroom/state.py', changed)
    head = _commit(tmp_path, 'closed declaration')
    route = selector.select_focus(('newsroom/state.py',), repo_root=tmp_path, base_sha=base, head_sha=head)
    assert route['selected_tests'] == (['newsroom/tests/test_state.py', 'newsroom/tests/test_store.py']
                                       if change == 'literal' else ['newsroom/tests/test_intake.py'])
    assert route['selected_service_tests'] == []


@pytest.mark.parametrize('binding,decorator', (
    ('from dataclasses import dataclass', 'dataclass'),
    ('from dataclasses import dataclass as dc', 'dc(frozen=True, slots=True)'),
    ('import dataclasses as dc', 'dc.dataclass(frozen=True)'),
))
def test_added_static_stdlib_dataclass_is_a_changed_binding_not_whole_module(tmp_path, binding, decorator):
    subprocess.run(('git', 'init', '-q'), cwd=tmp_path, check=True)
    source = binding + '\nLAND = "land"\n\ndef unchanged():\n    return LAND\n'
    _write(tmp_path, 'newsroom/progress.py', source)
    _write(tmp_path, 'newsroom/accounting.py', 'from .progress import unchanged\n')
    _write(tmp_path, 'newsroom/tests/test_other_neo4j_service.py', 'from newsroom.accounting import unchanged\n')
    base = _commit(tmp_path, 'base')
    _write(tmp_path, 'newsroom/progress.py', source + '\n@' + decorator
           + '\nclass Header:\n    name: str\n    identity: tuple[str, str]\n    latest: str | None = None\n')
    _write(tmp_path, 'newsroom/tests/test_header.py', 'from newsroom.progress import Header\n')
    head = _commit(tmp_path, 'added static header')
    route = selector.select_focus(('newsroom/progress.py',), repo_root=tmp_path, base_sha=base, head_sha=head)
    assert route['selected_tests'] == ['newsroom/tests/test_header.py']
    assert route['selected_service_tests'] == []


@pytest.mark.parametrize('prefix,declaration', (
    ('def custom(cls):\n    return cls\n', '@custom\nclass Header:\n    name: str\n'),
    ('from dataclasses import dataclass\ndef custom(cls):\n    return cls\ndataclass = custom\n', '@dataclass\nclass Header:\n    name: str\n'),
    ('import dataclasses as dc\ndef custom(cls):\n    return cls\ndc = custom\n', '@dc.dataclass\nclass Header:\n    name: str\n'),
    ('import dataclasses as dc\ndef custom(cls):\n    return cls\ndc.dataclass = custom\n', '@dc.dataclass\nclass Header:\n    name: str\n'),
    ('from dataclasses import dataclass\ndef custom(cls):\n    return cls\nif True:\n    dataclass = custom\n', '@dataclass\nclass Header:\n    name: str\n'),
    ('from dataclasses import dataclass\nif True:\n    from custom import dataclass\n', '@dataclass\nclass Header:\n    name: str\n'),
    ('from dataclasses import dataclass, field\n', '@dataclass\nclass Header:\n    value: str = field(default_factory=str)\n'),
    ('from dataclasses import dataclass\n', '@dataclass(slots=bool(1))\nclass Header:\n    name: str\n'),
    ('from dataclasses import dataclass\n', '@dataclass\nclass Header(object):\n    name: str\n'),
    ('from dataclasses import dataclass\n', '@dataclass\nclass Header:\n    name: custom()\n'),
    ('from dataclasses import dataclass\n', '@dataclass\nclass Header:\n    name: str\n    register()\n'),
    ('from dataclasses import dataclass\nstr = 7\n', '@dataclass\nclass Header:\n    name: str\n'),
))
def test_unknown_dataclass_transform_keeps_conservative_fanout(tmp_path, prefix, declaration):
    subprocess.run(('git', 'init', '-q'), cwd=tmp_path, check=True)
    source = prefix + '\ndef unchanged():\n    return 1\n'
    _write(tmp_path, 'newsroom/progress.py', source)
    _write(tmp_path, 'newsroom/tests/test_other_neo4j_service.py', 'from newsroom.progress import unchanged\n')
    base = _commit(tmp_path, 'base')
    _write(tmp_path, 'newsroom/progress.py', source + '\n' + declaration)
    head = _commit(tmp_path, 'uncertain transform')
    route = selector.select_focus(('newsroom/progress.py',), repo_root=tmp_path, base_sha=base, head_sha=head)
    assert route['selected_service_tests'] == ['newsroom/tests/test_other_neo4j_service.py']


@pytest.mark.parametrize('declaration', (
    'def added(value=register()):\n    return value\n',
    'class Added(make_base()):\n    pass\n',
    'class Added:\n    register()\n',
))
def test_added_executable_definition_is_not_a_closed_declaration(tmp_path, declaration):
    subprocess.run(('git', 'init', '-q'), cwd=tmp_path, check=True)
    source = 'def unchanged():\n    return 1\n'
    _write(tmp_path, 'newsroom/progress.py', source)
    _write(tmp_path, 'newsroom/tests/test_other_neo4j_service.py', 'from newsroom.progress import unchanged\n')
    base = _commit(tmp_path, 'base')
    _write(tmp_path, 'newsroom/progress.py', source + '\n' + declaration)
    head = _commit(tmp_path, 'executable definition')
    route = selector.select_focus(('newsroom/progress.py',), repo_root=tmp_path, base_sha=base, head_sha=head)
    assert route['selected_service_tests'] == ['newsroom/tests/test_other_neo4j_service.py']


def test_exact_diff_first_adopter_retains_focused_graphiti_model_consumers(tmp_path):
    subprocess.run(('git', 'init', '-q'), cwd=tmp_path, check=True)
    source = 'class Changed:\n    def value(self):\n        return 1\n\nclass Other:\n    pass\n'
    _write(tmp_path, 'newsroom/graphiti_adapter/models.py', source)
    _write(tmp_path, 'newsroom/control_plane/graphiti.py',
           'from ..graphiti_adapter.models import Changed\n\ndef ingest():\n    return Changed().value()\n')
    _write(tmp_path, 'newsroom/tests/test_graphiti_models.py', 'from newsroom.graphiti_adapter.models import Changed\n')
    _write(tmp_path, 'newsroom/tests/test_graphiti_consumer.py', 'from newsroom.control_plane.graphiti import ingest\n')
    _write(tmp_path, 'newsroom/tests/test_other_neo4j_service.py', 'from newsroom.graphiti_adapter.models import Other\n')
    base = _commit(tmp_path, 'base')
    _write(tmp_path, 'newsroom/graphiti_adapter/models.py', source.replace('return 1', 'return 2'))
    head = _commit(tmp_path, 'exact model change')
    route = selector.select_focus(('newsroom/graphiti_adapter/models.py',), repo_root=tmp_path,
                                  base_sha=base, head_sha=head)
    assert route['selected_tests'] == ['newsroom/tests/test_graphiti_consumer.py', 'newsroom/tests/test_graphiti_models.py']
    assert route['selected_service_tests'] == []
    assert route['full_health_required'] is False


_BODY_CONTRACT_SOURCES = (
    'newsroom/control_plane/native_story_entities.py',
    'newsroom/control_plane/native_story_dates.py',
)
_BODY_CONTRACT_TESTS = (
    'test_native_story_entities.py', 'test_native_story_writer.py',
    'test_native_story_model.py', 'test_native_story_dates.py',
    'test_native_story_editorial.py', 'test_native_brief_prompt_identity.py',
    'test_native_publication.py', 'test_native_publication_continuation.py',
    'test_native_pipeline.py', 'test_native_composition.py',
    'test_native_source_context_ranges.py', 'test_native_vertical.py',
)


def _body_contract_repo(tmp_path, source_override=None):
    """Real Git identities; actual consumer source is parsed, never executed."""
    subprocess.run(('git', 'init', '-q'), cwd=tmp_path, check=True)
    root = Path(__file__).parents[2]
    sources = {path: (root / path).read_text() for path in _BODY_CONTRACT_SOURCES}
    sources.update(source_override or {})
    for path, source in sources.items():
        _write(tmp_path, path, source)
    for name in _BODY_CONTRACT_TESTS:
        _write(tmp_path, 'newsroom/tests/' + name)
    _write(tmp_path, 'newsroom/tests/test_other_neo4j_service.py',
           'from newsroom.control_plane.native_story_entities import story_entity_names_are_bound\n'
           'from newsroom.control_plane.native_story_dates import _anchor\n')
    return sources, _commit(tmp_path, 'actual source-name/date consumer baseline')


def _body_contract_change(sources, kind):
    names, dates = _BODY_CONTRACT_SOURCES
    changed = dict(sources)
    if kind in {'role', 'both'}:
        changed[names] = sources[names].replace("if kind == 'PLACE'", "if kind in ('PLACE',)")
    if kind in {'date', 'both'}:
        changed[dates] = sources[dates].replace("    _require(publication == update, 'ANCHOR_AMBIGUOUS')\n", '')
    return changed


@pytest.mark.parametrize('kind', ['role', 'date', 'both'])
def test_closed_source_name_date_bodies_use_direct_contract_without_service(tmp_path, kind):
    sources, base = _body_contract_repo(tmp_path)
    changed = _body_contract_change(sources, kind)
    paths = [path for path in sources if changed[path] != sources[path]]
    for path in paths:
        _write(tmp_path, path, changed[path])
    changed_test = 'newsroom/tests/test_native_story_entities.py' if kind == 'role' else 'newsroom/tests/test_native_story_dates.py'
    _write(tmp_path, changed_test, '# The corresponding direct regression changed.\n')
    head = _commit(tmp_path, 'closed existing-function body delta')
    route = selector.select_focus((*paths, changed_test), repo_root=tmp_path, base_sha=base, head_sha=head)
    assert route['selected_tests'] == sorted('newsroom/tests/' + name for name in _BODY_CONTRACT_TESTS)
    assert route['selected_service_tests'] == []
    assert route['gates'] == ['F0', 'F1', 'F2']
    assert route['full_health_required'] is False
    assert 'explicit_source_name_date_body_contract:F2' in route['reasons']


def test_actual_natural_copy_date_fix_keeps_existing_data_callee_contract(tmp_path):
    root = Path(__file__).parents[2]
    path = _BODY_CONTRACT_SOURCES[1]
    after = (root / path).read_text(encoding='utf-8')
    anchor = "             and original_draft['body'].count(selected[0][1]['rendered_assertion']) == 1, 'COPY_ALIGNMENT')"
    literal = "             and selected[0][1]['rendered_assertion'] == claim.rendered_assertion_zh_hant_hk\n"
    assert after.count(anchor) == 1 and after.count(literal) == 0
    # Recreate the observed predicate removal without requiring private commits
    # to survive squash or to exist in the CI checkout's Git history.
    before = after.replace(anchor, literal + anchor, 1)
    assert before.count(anchor) == 1 and before.count(literal) == 1
    _sources, base = _body_contract_repo(tmp_path, {path: before})
    _write(tmp_path, path, after)
    head = _commit(tmp_path, 'observed natural-copy literal predicate removal')
    route = selector.select_focus((path,), repo_root=tmp_path, base_sha=base, head_sha=head)
    assert route['selected_tests'] == sorted('newsroom/tests/' + name for name in _BODY_CONTRACT_TESTS)
    assert route['selected_service_tests'] == [] and route['gates'] == ['F0', 'F1', 'F2']
    assert 'explicit_source_name_date_body_contract:F2' in route['reasons']


@pytest.mark.parametrize('kind', ['open', 'import', 'map', 'signature', 'callee', 'version',
    'protected-path', 'helper', 'getattr', 'shadow', 'reference', 'other-test', 'control', 'deploy',
    'missing-test', 'reflected-field', 'dunder-key', 'existing-reference-call'])
def test_unproved_source_name_date_delta_keeps_normal_service_discovery(tmp_path, kind):
    sources, base = _body_contract_repo(tmp_path)
    names, _dates = _BODY_CONTRACT_SOURCES
    source = _body_contract_change(sources, 'role')[names]
    marker = 'def story_entity_names_are_bound(text, claims):\n'
    if kind == 'open':
        source = source.replace(marker, marker + "    open('local-fixture')\n")
    elif kind == 'import':
        source = 'import socket\n' + source
    elif kind == 'map':
        source = source.replace("frozenset({'UK', '英國'})", "frozenset({'UK', '英國', 'United Kingdom'})")
    elif kind == 'signature':
        source = source.replace(marker, 'def story_entity_names_are_bound(text, claims, extra=None):\n')
    elif kind == 'callee':
        source = source.replace(marker, marker + '    additional_reader(text)\n')
    elif kind == 'version':
        source = source.replace("VERSION = 'newsroom.native-story-source-names.v2'", "VERSION = 'changed-consumer-version'")
    elif kind == 'helper':
        source += '\ndef new_helper():\n    return True\n'
    elif kind == 'getattr':
        source = source.replace(marker, marker + "    getattr(text, '__class__')\n")
    elif kind == 'reflected-field':
        source = source.replace("getattr(claim, 'attribution', None)", "getattr(claim, '__class__', None)")
    elif kind == 'dunder-key':
        source = source.replace(marker, marker + "    text['__dict__']\n")
    elif kind == 'existing-reference-call':
        source = source.replace(marker, marker + '    text()\n')
    elif kind == 'shadow':
        source = source.replace(marker, marker + '    bounded_named_entities = text\n')
    elif kind == 'reference':
        source = source.replace(marker, marker + '    unknown_metadata\n')
    paths = [names]
    if kind == 'protected-path':
        paths.append('newsroom/control_plane/native_story_writer.py')
        _write(tmp_path, paths[-1], 'DRAFT_SYSTEM = "changed producer"\n')
    elif kind == 'other-test':
        paths.append('newsroom/tests/test_native_story_writer.py')
        _write(tmp_path, paths[-1], '# This test is selected but outside the changed-test contract.\n')
    elif kind == 'control':
        paths.append('.sdlc/gates.toml')
        _write(tmp_path, paths[-1], 'scope = "fixture"\n')
    elif kind == 'deploy':
        paths.append('deploy/fixture.json')
        _write(tmp_path, paths[-1], '{}\n')
    elif kind == 'missing-test':
        missing = 'newsroom/tests/test_native_story_writer.py'
        (tmp_path / missing).unlink()
        paths.append(missing)
    _write(tmp_path, names, source)
    head = _commit(tmp_path, 'unproved or mixed contract delta')
    route = selector.select_focus(paths, repo_root=tmp_path, base_sha=base, head_sha=head)
    assert 'explicit_source_name_date_body_contract:F2' not in route['reasons']
    assert route['selected_service_tests'] == ['newsroom/tests/test_other_neo4j_service.py']
    assert 'F3' in route['gates']
    if kind == 'control':
        assert 'sdlc_control:F2' in route['reasons']
    if kind == 'deploy':
        assert route['owner_authority_required'] is True and 'F4' in route['gates']


@pytest.mark.parametrize('declaration', [
    '    def _require(*args, **kwargs):\n        return None\n',
    '    def unused_helper():\n        return None\n',
    '    lambda: None\n',
    '    class Unused:\n        pass\n',
    '    async def unused_helper():\n        return None\n',
])
def test_nested_date_consumer_declarations_keep_normal_discovery(tmp_path, declaration):
    sources, _base = _body_contract_repo(tmp_path)
    path = _BODY_CONTRACT_SOURCES[1]
    service_test = 'newsroom/tests/test_other_neo4j_service.py'
    _write(tmp_path, service_test,
           'from newsroom.control_plane.native_story_dates import derive_and_verify\n')
    base = _commit(tmp_path, 'direct consumer of the changed date function')
    marker = '                      source_records=(), final_draft=None, date_derivation=None):\n'
    assert marker in sources[path]
    _write(tmp_path, path, sources[path].replace(marker, marker + declaration))
    head = _commit(tmp_path, 'nested declaration outside the closed top-level contract')
    route = selector.select_focus((path,), repo_root=tmp_path, base_sha=base, head_sha=head)
    assert 'explicit_source_name_date_body_contract:F2' not in route['reasons']
    assert route['selected_service_tests'] == [service_test]
    assert 'F3' in route['gates']

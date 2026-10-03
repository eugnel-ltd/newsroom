from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from scripts.sdlc.focus_gate import (
    FocusGateError,
    select_focus,
    validate_focus_contract_data,
    validate_manifest,
)


REPO_ROOT = Path(__file__).parents[2]


def _write(root: Path, relative: str, content: str = "") -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_documentation_route_stops_at_f0_without_bootstrap() -> None:
    route = select_focus(("docs/guide.md", "AGENTS.md"))

    assert route["gates"] == ["F0"]
    assert route["selected_tests"] == []
    assert route["selected_service_tests"] == []
    assert route["bootstrap_required"] is False
    assert route["full_health_required"] is False
    assert route["execution_budget"] == {
        "focus_gate_jobs": 1,
        "dependency_bootstraps": 0,
    }
    validate_manifest(route)


def test_local_source_selects_direct_importing_test(tmp_path: Path) -> None:
    _write(tmp_path, "newsroom/pricing.py", "def quoted_price() -> int:\n    return 7\n")
    _write(
        tmp_path,
        "newsroom/tests/test_pricing.py",
        "from newsroom.pricing import quoted_price\n\n"
        "def test_price() -> None:\n    assert quoted_price() == 7\n",
    )

    route = select_focus(("newsroom/pricing.py",), repo_root=tmp_path)

    assert route["gates"] == ["F0", "F1", "F2"]
    assert route["selected_tests"] == ["newsroom/tests/test_pricing.py"]
    assert route["bootstrap_required"] is True
    assert route["full_health_required"] is False


def test_package_reexport_selects_changed_symbol_without_unrelated_package_imports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.sdlc.focus_gate as focus_gate

    _write(
        tmp_path,
        "newsroom/example/models.py",
        "class ImportantReceipt:\n    pass\n",
    )
    _write(
        tmp_path,
        "newsroom/example/__init__.py",
        "from .models import ImportantReceipt\n\n"
        "class OtherThing:\n    pass\n",
    )
    _write(
        tmp_path,
        "newsroom/tests/test_00_package_smoke.py",
        "from newsroom.example import OtherThing\n\n"
        "def test_smoke() -> None:\n    assert OtherThing\n",
    )
    _write(
        tmp_path,
        "newsroom/tests/test_10_changed_receipt.py",
        "from newsroom.example import ImportantReceipt\n\n"
        "def test_receipt() -> None:\n    assert ImportantReceipt\n",
    )
    _write(
        tmp_path,
        "newsroom/tests/test_20_unrelated.py",
        "from newsroom.example import OtherThing\n\n"
        "def test_unrelated() -> None:\n    assert OtherThing\n",
    )

    class Graph:
        def dependent_paths(self, path: str) -> tuple[str, ...]:
            assert path == "newsroom/example/models.py"
            return ("newsroom/example/__init__.py",)

    monkeypatch.setattr(focus_gate, "build_dependency_graph", lambda _root: Graph())
    route = focus_gate.select_focus(
        ("newsroom/example/models.py",),
        repo_root=tmp_path,
    )

    assert route["selected_tests"] == [
        "newsroom/tests/test_10_changed_receipt.py",
    ]


def test_issue_789_shape_selects_adapter_and_consumer_without_broad_lanes(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "newsroom/graphiti_adapter/models.py",
        "class GraphitiAdapterExecution:\n    pass\n\n"
        "def adapter_outcome_for() -> str:\n    return 'COMPLETE'\n",
    )
    _write(
        tmp_path,
        "newsroom/tests/test_graphiti_adapter_execution.py",
        "from newsroom.graphiti_adapter.models import GraphitiAdapterExecution\n\n"
        "def test_execution() -> None:\n    assert GraphitiAdapterExecution\n",
    )
    _write(
        tmp_path,
        "newsroom/tests/test_graphiti_event_consumer.py",
        "from newsroom.graphiti_adapter.models import adapter_outcome_for\n\n"
        "def test_consumer() -> None:\n    assert adapter_outcome_for() == 'COMPLETE'\n",
    )

    route = select_focus(
        ("newsroom/graphiti_adapter/models.py",),
        repo_root=tmp_path,
    )

    assert route["selected_tests"] == [
        "newsroom/tests/test_graphiti_adapter_execution.py",
        "newsroom/tests/test_graphiti_event_consumer.py",
    ]
    assert route["selected_service_tests"] == []
    assert route["research_required"] is False
    assert route["full_health_required"] is False
    assert route["owner_authority_required"] is False


def test_current_graphiti_adapter_without_exact_diff_keeps_conservative_service_fallback() -> None:
    route = select_focus(
        ("newsroom/graphiti_adapter/models.py",),
        repo_root=REPO_ROOT,
    )

    assert {"F0", "F1", "F2"} <= set(route["gates"])
    assert len(route["selected_tests"]) >= 2
    assert any("graphiti" in path for path in route["selected_tests"])
    assert route["selected_service_tests"]
    assert route["research_required"] is False
    assert route["full_health_required"] is False


def test_stateful_change_adds_f2_migration_and_authority_evidence() -> None:
    route = select_focus(("newsroom/authority/example_migrations.py",))

    assert "F2" in route["gates"]
    assert any("migration" in path for path in route["selected_tests"])
    assert any("authority" in path for path in route["selected_tests"])
    assert route["bootstrap_required"] is True


def test_neo4j_change_selects_actual_service_gate() -> None:
    route = select_focus(("newsroom/projection/neo4j/adapter.py",))

    assert "F3" in route["gates"]
    assert route["selected_service_tests"]
    assert route["bootstrap_required"] is True
    assert route["research_required"] is False


def test_release_change_remains_f4_and_fail_closed() -> None:
    route = select_focus(("scripts/production_operational_admission.py",))

    assert "F4" in route["gates"]
    assert route["owner_authority_required"] is True
    assert route["full_health_required"] is True


def test_research_only_change_is_isolated_from_ordinary_tests() -> None:
    route = select_focus(
        ("newsroom/tests/test_graphiti_combined_temporal_runtime.py",)
    )

    assert route["gates"] == ["F0"]
    assert route["research_required"] is True
    assert route["selected_tests"] == []
    assert route["selected_service_tests"] == []
    assert route["bootstrap_required"] is False
    assert route["full_health_required"] is False


def test_unknown_executable_escalates_visibly_to_full_health() -> None:
    route = select_focus(("Makefile",))

    assert route["full_health_required"] is True
    assert {"F0", "F1", "F2"} <= set(route["gates"])
    assert "unknown_executable_path:full_health" in route["reasons"]
    assert route["bootstrap_required"] is True


def test_sdlc_control_change_selects_only_contract_tests() -> None:
    route = select_focus((".github/workflows/focus-gates.yml",))

    assert {"F0", "F1", "F2"} <= set(route["gates"])
    assert "newsroom/tests/test_focus_gate.py" in route["selected_tests"]
    assert "newsroom/tests/test_ci_workflow.py" in route["selected_tests"]
    assert route["full_health_required"] is False
    assert route["selected_service_tests"] == []


def test_full_health_excludes_research_fixtures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import subprocess
    from types import SimpleNamespace

    import scripts.sdlc.focus_gate as focus_gate

    captured: dict[str, object] = {}

    monkeypatch.setattr(focus_gate, "load_focus_contract", lambda _root: {})
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: (
            captured.update(command=command, kwargs=kwargs)
            or SimpleNamespace(returncode=0)
        ),
    )

    assert focus_gate.execute_full_health(
        tmp_path,
        junit="reports/full-health.xml",
    ) == 0

    command = captured["command"]
    assert isinstance(command, tuple)
    assert "newsroom/tests" in command
    assert "-n" in command
    assert "4" in command
    assert any(
        item == "--ignore-glob=newsroom/tests/test_graphiti_combined_temporal_*.py"
        for item in command
    )
    assert any(
        item == "--ignore-glob=newsroom/tests/test_graphiti_core_0293_*.py"
        for item in command
    )
    assert captured["kwargs"]["cwd"] == tmp_path.resolve()


def test_manifest_identity_is_canonical_and_tamper_evident() -> None:
    first = select_focus(("newsroom/example.py", "newsroom/tests/test_example.py"))
    second = select_focus(("newsroom/tests/test_example.py", "newsroom/example.py"))
    assert first == second
    validate_manifest(first)

    tampered = deepcopy(first)
    tampered["gates"] = ["F0"]
    with pytest.raises(FocusGateError, match="digest"):
        validate_manifest(tampered)


def test_focus_contract_rejects_reintroduced_universal_pr_suite() -> None:
    import tomllib

    data = tomllib.loads((REPO_ROOT / ".sdlc/gates.toml").read_text(encoding="utf-8"))
    validate_focus_contract_data(data)

    changed = deepcopy(data)
    changed["global"]["full_suite_is_default"] = True
    with pytest.raises(FocusGateError, match="full-suite"):
        validate_focus_contract_data(changed)


def test_import_inventory_resolves_context_without_changing_legacy_default() -> None:
    import ast
    from scripts.sdlc.focus_gate import _imported_modules

    tree = ast.parse(
        "from . import fixture as selected\n"
        "from .bridge import build as retained\n"
        "import newsroom.package.child as package\n"
        "importlib.import_module('newsroom.tests.dynamic')\n"
    )
    assert _imported_modules(tree) == {
        "bridge", "bridge.build", "newsroom.package.child",
    }
    assert _imported_modules(tree, importer="newsroom.tests.test_consumer") == {
        "newsroom.tests", "newsroom.tests.fixture", "newsroom.tests.bridge",
        "newsroom.tests.bridge.build", "newsroom.package.child", "newsroom.tests.dynamic",
    }

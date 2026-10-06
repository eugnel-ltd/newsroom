from __future__ import annotations

import ast
import subprocess
from pathlib import Path
from typing import Iterable, Sequence

from . import focus_gate as legacy
from .classify_change import matches_repository_glob
from .dependencies import DependencyError, build_dependency_graph, module_name_for_path

CONTROL_PATTERNS = (
    ".sdlc/**",
    ".github/workflows/**",
    "scripts/sdlc/focus_gate.py",
    "scripts/sdlc/focus_gate_v2.py",
    "scripts/sdlc/focus_selector.py",
    "newsroom/tests/test_focus_gate.py",
    "newsroom/tests/test_focus_gate_hardening.py",
    "newsroom/tests/test_ci_workflow.py",
    "newsroom/tests/test_sdlc_contract.py",
    "newsroom/tests/test_sdlc_evidence_workflow.py",
)
CONTROL_TESTS = (
    "newsroom/tests/test_focus_gate.py",
    "newsroom/tests/test_focus_gate_hardening.py",
    "newsroom/tests/test_ci_workflow.py",
    "newsroom/tests/test_sdlc_contract.py",
    "newsroom/tests/test_sdlc_evidence_workflow.py",
    "newsroom/tests/test_pr_lifecycle.py",
)
RESEARCH_ONLY_PATTERNS = (
    "docs/research/**/*.json",
    "docs/research/**/*.csv",
    "scripts/graphiti_combined_temporal_extraction.py",
    "scripts/graphiti_deterministic_work.py",
    "scripts/graphiti_runtime_calibration.py",
    "newsroom/tests/test_graphiti_core_0293_*.py",
    "newsroom/tests/test_graphiti_sdk_no_tool_calibration.py",
    "newsroom/tests/test_graphiti_combined_temporal_*.py",
    "newsroom/tests/test_graphiti_donor_identities.py",
    "newsroom/tests/test_graphiti_deterministic_work.py",
)
RESEARCH_TRIGGER_PATTERNS = ("pyproject.toml", "uv.lock", *RESEARCH_ONLY_PATTERNS)
SERVICE_PATTERNS = (
    "newsroom/projection/neo4j/**",
    "newsroom/*neo4j*.py",
    "newsroom/**/*neo4j*.py",
    "newsroom/tests/test_*_neo4j_service.py",
)
STATEFUL_PATTERNS = (
    "newsroom/authority/**",
    "newsroom/integrated/**",
    "newsroom/projection/models.py",
    "newsroom/projection/policy.py",
    "newsroom/schemas/**",
    "newsroom/**/*migration*.py",
    "newsroom/**/*_migrations.py",
)
STATEFUL_SENTINELS = (
    "newsroom/tests/test_authority_migration_compatibility.py",
    "newsroom/tests/test_authority_store_conformance.py",
)
F4_PATTERNS = (
    "deploy/**",
    "release/**",
    ".github/workflows/deploy*.yml",
    ".github/workflows/release*.yml",
    "newsroom/production_admission/**",
    "scripts/production_operational_admission.py",
    "newsroom/**/*credential*.py",
    "newsroom/**/*keychain*.py",
)
SHARED_BREADTH_PATTERNS = ("pyproject.toml", "uv.lock", "newsroom/tests/conftest.py")
PREPARED_CANARY_PARITY_PATTERNS = (
    "newsroom/control_plane/cycle.py",
    "newsroom/control_plane/graphiti.py",
    "newsroom/control_plane/issue_790_disposition.py",
    "newsroom/control_plane/issue_790_canary.py",
    "newsroom/control_plane/issue_790_prepared_canary.py",
    "newsroom/control_plane/issue_790_rehearsal.py",
    "scripts/issue_790_live_canary_preflight.py",
    "scripts/issue_790_prepared_canary_rehearsal.py",
)
PREPARED_CANARY_PARITY_TESTS = (
    "newsroom/tests/test_issue_790_prepared_canary.py",
    "newsroom/tests/test_issue_790_prepared_canary_artifact.py",
    "newsroom/tests/test_issue_790_retry_forbidden_safety_state.py",
)
PREPARED_CANARY_CONSUMER_PATTERNS = (
    "newsroom/control_plane/issue_790_disposition.py",
    "newsroom/control_plane/issue_790_canary.py",
    "newsroom/control_plane/issue_790_prepared_canary.py",
    "newsroom/control_plane/issue_790_rehearsal.py",
    "scripts/issue_790_live_canary_preflight.py",
    "scripts/issue_790_prepared_canary_rehearsal.py",
)
PREPARED_CANARY_CONSUMER_TESTS = (
    "newsroom/tests/test_model_usage_receipts.py",
)
EXECUTABLE_SUFFIXES = frozenset(
    {".py", ".toml", ".json", ".yml", ".yaml", ".sh", ".bash", ".sql"}
)
IGNORED_IMPORT_ROOTS = frozenset({"newsroom", "scripts"})
SOURCE_BODY_CONTRACT_PATHS = frozenset({
    "newsroom/control_plane/native_story_entities.py",
    "newsroom/control_plane/native_story_dates.py",
})
SOURCE_BODY_CHANGED_TESTS = frozenset({
    "newsroom/tests/test_native_story_entities.py",
    "newsroom/tests/test_native_story_dates.py",
})
SOURCE_BODY_CONTRACT_TESTS = tuple("newsroom/tests/" + name for name in (
    "test_native_story_entities.py", "test_native_story_writer.py",
    "test_native_story_model.py", "test_native_story_dates.py",
    "test_native_story_editorial.py", "test_native_brief_prompt_identity.py",
    "test_native_publication.py", "test_native_publication_continuation.py",
    "test_native_pipeline.py", "test_native_composition.py",
    "test_native_source_context_ranges.py", "test_native_vertical.py",
))


def _matches(path: str, patterns: Iterable[str]) -> bool:
    return any(matches_repository_glob(path, pattern) for pattern in patterns)


def _matches_any(paths: Iterable[str], patterns: Iterable[str]) -> bool:
    return any(_matches(path, patterns) for path in paths)


def _is_documentation(path: str) -> bool:
    return path.endswith(".md") or (
        path.startswith("docs/") and Path(path).suffix.lower() not in EXECUTABLE_SUFFIXES
    )


def _is_research_only(path: str) -> bool:
    return _matches(path, RESEARCH_ONLY_PATTERNS)


def _triggers_research(path: str) -> bool:
    return _matches(path, RESEARCH_TRIGGER_PATTERNS)


def _is_test(path: str) -> bool:
    return path.startswith("newsroom/tests/test_") and path.endswith(".py")


def _is_service_test(path: str) -> bool:
    return _is_test(path) and path.endswith("_neo4j_service.py")


def _assignment_names(target: ast.expr) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        return {name for item in target.elts for name in _assignment_names(item)}
    return set()


def _literal_string_set(value: ast.expr) -> set[str] | None:
    if not isinstance(value, (ast.List, ast.Tuple, ast.Set)):
        return None
    values: set[str] = set()
    for item in value.elts:
        if not isinstance(item, ast.Constant) or not isinstance(item.value, str):
            return None
        values.add(item.value)
    return values


def _defined_names(path: Path) -> set[str]:
    """Return exact top-level public symbols, including short names/constants."""

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeError):
        return set()
    names: set[str] = set()
    declared_all: set[str] | None = None
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            assigned = {name for target in node.targets for name in _assignment_names(target)}
            names.update(assigned)
            if "__all__" in assigned:
                literal = _literal_string_set(node.value)
                if literal is not None:
                    declared_all = literal
        elif isinstance(node, ast.AnnAssign):
            names.update(_assignment_names(node.target))
    public = {name for name in names if not name.startswith("_")}
    return public if declared_all is None else public & declared_all


def _revision_modules(repo_root: Path, path: str, base_sha: str, head_sha: str):
    """Read the exact declaration inputs once without importing candidate code."""
    modules = []
    for revision in (base_sha, head_sha):
        try:
            result = subprocess.run(
                ("git", "show", f"{revision}:{path}"), cwd=repo_root, capture_output=True,
            )
            if result.returncode != 0:
                return None
            source = result.stdout.decode("utf-8")
            modules.append((source, ast.parse(source, filename=path)))
        except (OSError, SyntaxError, UnicodeError):
            return None
    return modules


def _changed_public_symbols(repo_root: Path, path: str, base_sha: str, head_sha: str) -> set[str] | None:
    modules = _revision_modules(repo_root, path, base_sha, head_sha)
    if modules is None:
        return None
    (_, old_tree), (_, new_tree) = modules

    old, new = _module_declarations(old_tree), _module_declarations(new_tree)
    if old is None or new is None or old[1:] != new[1:]:
        return None
    changed = {name for name in old[0].keys() | new[0].keys()
               if old[0].get(name) != new[0].get(name)}
    if not changed:
        return None
    for tree, other in ((old_tree, new_tree), (new_tree, old_tree)):
        other_names = {node.name for node in other.body
                       if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))}
        if any(not _static_definition(tree, node) for node in tree.body
               if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
               and node.name not in other_names):
            return None
    if _local_symbol_closure(old_tree, changed) is None:
        return None
    return _local_symbol_closure(new_tree, changed)


def _closed_existing_function_bodies(modules) -> bool:
    """A baseline call/reference subset, not a general Python purity proof."""
    (before, old_tree), (after, new_tree) = modules
    old, new = _module_declarations(old_tree), _module_declarations(new_tree)
    if old is None or new is None or old[1:] != new[1:] or old[0].keys() != new[0].keys():
        return False
    previous = {node.name: node for node in old_tree.body if isinstance(node, ast.FunctionDef)}
    current = {node.name: node for node in new_tree.body if isinstance(node, ast.FunctionDef)}
    changed = {name for name in old[0] if old[0][name] != new[0][name]}
    if not changed or not changed <= previous.keys() or not changed <= current.keys():
        return False
    if _local_symbol_closure(old_tree, changed) is None or _local_symbol_closure(new_tree, changed) is None:
        return False
    # Keep other functions and module-time imports/maps/effects byte-identical.
    protected = lambda source, tree: [ast.get_source_segment(source, node) for node in tree.body
                                     if not isinstance(node, ast.FunctionDef) or node.name not in changed]
    if protected(before, old_tree) != protected(after, new_tree):
        return False
    references = (ast.Call, ast.Name, ast.Attribute, ast.Import, ast.ImportFrom,
                  ast.Global, ast.Nonlocal, ast.Yield, ast.YieldFrom, ast.Await)
    def footprint(node):
        result = set()
        for item in ast.walk(node):
            if isinstance(item, ast.Call):
                # Argument predicates may change under the same inspected data
                # callee. Reflected attribute arguments are not data-only proof.
                reflected = isinstance(item.func, ast.Name) and item.func.id in {
                    "getattr", "setattr", "delattr", "hasattr",
                }
                result.add("CALL:" + ast.dump(item if reflected else item.func))
            elif isinstance(item, references) or (
                isinstance(item, ast.Constant) and isinstance(item.value, str)
                and item.value.startswith("__") and item.value.endswith("__")
            ):
                result.add(ast.dump(item))
        return result
    for name in changed:
        left, right = previous[name], current[name]
        if footprint(right) - footprint(left):
            return False
        left_body, right_body = left.body, right.body
        left.body, right.body = [], []
        equal_header = ast.dump(left) == ast.dump(right)
        left.body, right.body = left_body, right_body
        if not equal_header:
            return False
    return True


def _source_body_contract_tests(root, changed, source_paths, base_sha, head_sha):
    allowed = SOURCE_BODY_CONTRACT_PATHS | SOURCE_BODY_CHANGED_TESTS
    if not source_paths or not set(source_paths) <= SOURCE_BODY_CONTRACT_PATHS or any(
        not _is_documentation(path) and path not in allowed for path in changed
    ):
        return None
    if any(not (root / path).is_file() or (root / path).is_symlink() for path in SOURCE_BODY_CONTRACT_TESTS):
        return None
    for path in source_paths:
        modules = _revision_modules(root, path, base_sha, head_sha)
        if modules is None or not _closed_existing_function_bodies(modules):
            return None
    return set(SOURCE_BODY_CONTRACT_TESTS)


def _module_declarations(tree: ast.Module):
    """Separate precise bindings from executable effects, without evaluating code."""
    bindings, modules, effects = {}, set(), []
    for node in tree.body:
        declared = {}
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            declared[node.name] = ast.dump(node)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom) and (node.module is None or any(a.name == '*' for a in node.names)):
                effects.append(ast.dump(node))
                continue
            for alias in node.names:
                module = ('from', node.module, node.level) if isinstance(node, ast.ImportFrom) else ('import', alias.name)
                modules.add(module)
                name = alias.asname or alias.name.split('.')[0]
                signature = (('import', name), name) if isinstance(node, ast.Import) and alias.asname is None else (module, alias.name)
                if name in bindings and not (isinstance(node, ast.Import) and alias.asname is None
                                             and bindings[name] == signature):
                    return None
                declared[name] = signature
        elif isinstance(node, ast.Assign) and all(isinstance(target, ast.Name) for target in node.targets):
            try:
                ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
                effects.append(ast.dump(node))
                continue
            declared = {target.id: ast.dump(node) for target in node.targets}
        else:
            effects.append(ast.dump(node))
        if bindings.keys() & declared.keys() and not isinstance(node, ast.Import):
            return None  # Rebinding/shadowing keeps the existing broad route.
        bindings.update(declared)
    return bindings, modules, effects


def _static_definition(tree: ast.Module, node: ast.AST) -> bool:
    """Added/removed declarations must not execute arbitrary module-time work."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if node.decorator_list:
            return False
        for value in (*node.args.defaults, *node.args.kw_defaults):
            if value is not None:
                try:
                    ast.literal_eval(value)
                except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
                    return False
        annotations = [arg.annotation for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)]
        annotations += [node.returns, node.args.vararg.annotation if node.args.vararg else None,
                        node.args.kwarg.annotation if node.args.kwarg else None]
        deferred = any(isinstance(item, ast.ImportFrom) and item.module == '__future__'
                       and any(alias.name == 'annotations' for alias in item.names) for item in tree.body)
        return deferred or all(value is None or isinstance(value, (ast.Name, ast.Constant)) for value in annotations)
    if _static_dataclass(tree, node):
        return True
    if not isinstance(node, ast.ClassDef) or node.decorator_list or node.keywords:
        return False
    declarations = _module_declarations(tree)
    for base in node.bases:
        name = base.value.id if isinstance(base, ast.Subscript) and isinstance(base.value, ast.Name) else None
        if (name is None or declarations is None
                or declarations[0].get(name) != (('from', 'collections.abc', 0), 'Mapping')
                or any(isinstance(item, (ast.Call, ast.Attribute)) for item in ast.walk(base))):
            return False
    return all(_static_definition(tree, item) if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
               else isinstance(item, ast.Pass) or isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant)
               for item in node.body)


def _static_dataclass(tree: ast.Module, node: ast.AST) -> bool:
    """Recognise only an unshadowed stdlib transform of plain literal fields."""
    if not isinstance(node, ast.ClassDef) or node.bases or node.keywords or len(node.decorator_list) != 1:
        return False
    counts, aliases = {}, set()
    for statement in tree.body:
        names = set()
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            for alias in statement.names:
                name = alias.asname or alias.name.split('.')[0]
                names.add(name)
                if isinstance(statement, ast.ImportFrom) and statement.level == 0 and statement.module == 'dataclasses' and alias.name == 'dataclass':
                    aliases.add((name,))
                elif isinstance(statement, ast.Import) and alias.name == 'dataclasses':
                    aliases.add((name, 'dataclass'))
        elif isinstance(statement, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(statement.name)
        elif isinstance(statement, ast.Assign):
            names.update(name for target in statement.targets for name in _assignment_names(target))
        elif isinstance(statement, (ast.AnnAssign, ast.AugAssign)):
            names.update(_assignment_names(statement.target))
        for name in names:
            counts[name] = counts.get(name, 0) + 1
    decorator = node.decorator_list[0]
    function = decorator.func if isinstance(decorator, ast.Call) else decorator
    chain = _attribute_chain(function)
    if chain not in aliases or counts.get(chain[0]) != 1 or _binding_mutated(tree, chain[0]):
        return False
    if isinstance(decorator, ast.Call) and (decorator.args or any(
            keyword.arg not in {'init', 'repr', 'eq', 'order', 'unsafe_hash', 'frozen', 'match_args', 'kw_only', 'slots', 'weakref_slot'}
            or not isinstance(keyword.value, ast.Constant) or type(keyword.value.value) is not bool
            for keyword in decorator.keywords)):
        return False
    builtins = {'str', 'int', 'float', 'bool', 'bytes', 'object', 'tuple', 'list', 'dict', 'set', 'frozenset', 'type'}
    fields = {item.target.id for item in node.body if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)}
    for item in node.body:
        if isinstance(item, ast.Pass) or isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant) and isinstance(item.value.value, str):
            continue
        if not isinstance(item, ast.AnnAssign) or not isinstance(item.target, ast.Name):
            return False
        for part in ast.walk(item.annotation):
            if isinstance(part, ast.Name):
                if part.id not in builtins or part.id in counts or part.id in fields or _binding_mutated(tree, part.id):
                    return False
            elif isinstance(part, ast.Constant):
                if part.value not in (None, Ellipsis):
                    return False
            elif not isinstance(part, (ast.Subscript, ast.Tuple, ast.BinOp, ast.BitOr, ast.Load)):
                return False
        if item.value is not None:
            try:
                value = ast.literal_eval(item.value)
            except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
                return False
            if type(value) not in (str, bytes, int, float, bool, type(None), tuple):
                return False
    return True


def _binding_mutated(tree: ast.AST, name: str) -> bool:
    # Reuse the existing rooted attribute/subscript escape boundary, including
    # conditional rebinding and reflected module mutation, not just top-level names.
    for item in ast.walk(tree):
        if isinstance(item, (ast.Import, ast.ImportFrom)) and item not in tree.body and any(
                (alias.asname or alias.name.split('.')[0]) == name for alias in item.names):
            return True
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and item.name == name:
            return True
        if isinstance(item, (ast.Name, ast.Attribute, ast.Subscript)) and isinstance(item.ctx, (ast.Store, ast.Del)):
            target = item
            while isinstance(target, (ast.Attribute, ast.Subscript)):
                target = target.value
            if isinstance(target, ast.Name) and target.id == name:
                return True
        if isinstance(item, ast.Call) and any(
                (chain := _attribute_chain(argument)) is not None and chain[0] == name
                for argument in (*item.args, *(keyword.value for keyword in item.keywords))):
            return True
    return False


def _local_symbol_closure(
    tree: ast.AST, symbols: set[str], module_attributes: set[tuple[str, ...]] = frozenset(),
) -> set[str] | None:
    """Close bound names over local callers; executable module effects stay broad."""
    kinds = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    definitions = {node.name: node for node in tree.body if isinstance(node, kinds)}
    module_names = set(definitions)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            module_names.update(name for target in node.targets for name in _assignment_names(target))
        elif isinstance(node, ast.AnnAssign):
            module_names.update(_assignment_names(node.target))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            module_names.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
    changed = set(symbols)
    if any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
           and (node.func.id in {"globals", "locals", "eval", "exec"}
                or node.func.id == "vars" and not node.args) for node in ast.walk(tree)):
        return None
    module_dicts = {(alias.asname or "sys", "modules") for node in ast.walk(tree)
                    if isinstance(node, ast.Import) for alias in node.names if alias.name == "sys"}
    module_dicts.update((alias.asname or alias.name,) for node in ast.walk(tree)
                       if isinstance(node, ast.ImportFrom) and node.module == "sys"
                       for alias in node.names if alias.name == "modules")
    if any(_attribute_chain(node) in module_dicts for node in ast.walk(tree)):
        return None

    def references(node):
        return any(
            isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load) and item.id in changed
            or any((chain := _attribute_chain(item)) is not None and chain[:len(prefix)] == prefix
                   for prefix in module_attributes)
            for item in ast.walk(node)
        )

    while True:
        callers = {name for name, node in definitions.items() if references(node)}
        expanded = changed | callers
        if any(node.decorator_list and not _static_dataclass(tree, node)
               or any(isinstance(item, (ast.Global, ast.Nonlocal)) for item in ast.walk(node))
               for name, node in definitions.items() if name in expanded):
            return None
        for name in expanded & definitions.keys():
            for item in ast.walk(definitions[name]):
                if isinstance(item, (ast.Attribute, ast.Subscript)) and isinstance(item.ctx, (ast.Store, ast.Del)):
                    target = item.value
                    while isinstance(target, (ast.Attribute, ast.Subscript)):
                        target = target.value
                    if isinstance(target, ast.Name) and target.id in module_names:
                        return None
        if expanded == changed:
            if any(references(node) for node in tree.body if not isinstance(node, (*kinds, ast.Import, ast.ImportFrom))):
                return None
            return changed
        changed = expanded


def _imports_module(imported: str, selected: str) -> bool:
    if imported in IGNORED_IMPORT_ROOTS or selected in IGNORED_IMPORT_ROOTS:
        return False
    return imported == selected or imported.startswith(selected + ".")


def _attribute_chain(node: ast.AST) -> tuple[str, ...] | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return tuple(reversed(parts))


def _imported_from(node: ast.ImportFrom, importer: str | None) -> str | None:
    if not node.level:
        return node.module
    if importer is None:
        return None
    parts = importer.rpartition(".")[0].split(".")
    base = parts[: len(parts) - node.level + 1]
    return ".".join((*base, *(node.module.split(".") if node.module else ())))


def _imports_any_public_symbol(
    tree: ast.AST, packages: set[str], symbols: set[str], importer: str | None = None
) -> bool:
    if not symbols or not packages:
        return False
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and _imported_from(node, importer) in packages:
            if any(alias.name == "*" or alias.name in symbols for alias in node.names):
                return True
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in packages:
                    aliases.add(alias.asname or alias.name.split(".")[0])
    package_parts = {tuple(package.split(".")) for package in packages}
    for node in ast.walk(tree):
        chain = _attribute_chain(node)
        if not chain or chain[-1] not in symbols:
            continue
        if chain[:-1] in package_parts:
            return True
        if len(chain) == 2 and chain[0] in aliases:
            return True
    return False


def _imports_changed_surface(
    tree: ast.AST, module: str, symbols: set[str], importer: str | None = None
) -> bool:
    parent, _, leaf = module.rpartition(".")
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call) and node.args
            and isinstance(node.args[0], ast.Constant) and node.args[0].value == module
        ):
            return True
        if isinstance(node, ast.Import) and any(alias.name == module for alias in node.names):
            return True
        if isinstance(node, ast.ImportFrom):
            if node.level:
                if importer is None:
                    return True
            imported_from = _imported_from(node, importer)
            if imported_from == parent and any(alias.name == leaf for alias in node.names):
                return True
            if imported_from == module and any(
                alias.name == "*" or alias.name in symbols for alias in node.names
            ):
                return True
    return False


def _importer_symbols(
    tree: ast.AST, module: str, symbols: set[str], importer: str,
) -> set[str] | None:
    """Carry imported aliases and their local callers, not every module descendant."""
    parent, _, leaf = module.rpartition(".")
    aliases: set[str] = set()
    modules: set[tuple[str, ...]] = set()
    exported_modules: set[str] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == module):
            return None
        if isinstance(node, ast.ImportFrom):
            source = _imported_from(node, importer)
            if source == module:
                if any(alias.name == "*" for alias in node.names):
                    return None
                aliases.update(alias.asname or alias.name for alias in node.names if alias.name in symbols)
            elif source == parent:
                for alias in node.names:
                    if alias.name == leaf:
                        modules.add((alias.asname or alias.name,))
                        if node in tree.body:
                            exported_modules.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == module:
                    modules.add((alias.asname,) if alias.asname else tuple(module.split(".")))
                    if node in tree.body:
                        exported_modules.add(alias.asname or alias.name.split(".")[0])
    if not aliases and not modules:
        return set()
    # Passing/reflecting on an imported module escapes the static attribute closure.
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            if _attribute_chain(child) in modules and not isinstance(node, ast.Attribute):
                return None
    attributes = {(*prefix, symbol) for prefix in modules for symbol in symbols}
    affected = _local_symbol_closure(tree, aliases, attributes)
    return None if affected is None else affected | exported_modules


def _discover_tests(
    repo_root: Path,
    source_paths: Sequence[str],
    *,
    base_sha: str = "0" * 40,
    head_sha: str = "1" * 40,
) -> tuple[set[str], bool]:
    direct: set[str] = set()
    dependents: set[str] = set()
    reexports: set[str] = set()
    direct_symbols: dict[str, set[str] | None] = {}
    surfaces: dict[str, set[str] | None] = {}
    unresolved = False
    try:
        graph = build_dependency_graph(repo_root)
    except DependencyError:
        graph = None
        unresolved = True

    def retain(relative: str) -> None:
        selected = module_name_for_path(relative)
        if selected is not None:
            (reexports if relative.endswith("/__init__.py") else dependents).add(selected)

    for relative in source_paths:
        module = module_name_for_path(relative)
        if module is None:
            continue
        direct.add(module)
        symbols = _changed_public_symbols(repo_root, relative, base_sha, head_sha)
        direct_symbols[module] = symbols
        if graph is None:
            continue
        if not hasattr(graph, "reverse_importers"):
            try:
                for nested in graph.dependent_paths(relative):
                    retain(nested)
            except DependencyError:
                unresolved = True
            continue
        if module not in graph.module_to_path:
            unresolved = True
        surfaces[module] = symbols
    if graph is not None and hasattr(graph, "reverse_importers"):
        queue = list(surfaces)
        while queue:
            module = queue.pop()
            symbols = surfaces[module]
            for importer in graph.reverse_importers.get(module, frozenset()):
                importer_path = graph.module_to_path.get(importer)
                if importer_path is None or (importer in surfaces and surfaces[importer] is None):
                    continue
                affected = None
                if symbols is not None:
                    try:
                        tree = ast.parse((repo_root / importer_path).read_text(encoding="utf-8"))
                    except (OSError, SyntaxError, UnicodeError):
                        unresolved = True
                    else:
                        affected = _importer_symbols(tree, module, symbols,
                            importer + ".__init__" if importer_path.endswith('/__init__.py') else importer)
                if affected == set():
                    continue
                previous = surfaces.get(importer, set())
                combined = None if affected is None else previous | affected
                if importer not in surfaces or combined != previous:
                    surfaces[importer] = combined
                    queue.append(importer)
    symbols: set[str] = set()
    for relative in source_paths:
        path = repo_root / relative
        if path.is_file() and not path.is_symlink():
            symbols.update(_defined_names(path))
    selected: set[str] = set()
    for path in legacy._test_files(repo_root):
        relative = path.relative_to(repo_root).as_posix()
        if _is_research_only(relative):
            continue
        importer = module_name_for_path(relative)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
            imports = legacy._imported_modules(tree, importer=importer)
        except (OSError, SyntaxError, UnicodeError, DependencyError):
            unresolved = True
            continue
        direct_hit = any(
            (
                _imports_changed_surface(tree, module, direct_symbols[module], importer)
                if direct_symbols[module] is not None
                else any(_imports_module(imported, module) for imported in imports)
            )
            for module in direct
        )
        dependent_hit = any(
            _imports_module(imported, module)
            for imported in imports
            for module in dependents
        )
        surface_hit = any(
            _imports_changed_surface(tree, module, affected, importer)
            if affected is not None else any(_imports_module(imported, module) for imported in imports)
            for module, affected in surfaces.items()
        )
        reexport_hit = _imports_any_public_symbol(tree, reexports, symbols, importer)
        if direct_hit or dependent_hit or surface_hit or reexport_hit:
            selected.add(relative)
    return selected, unresolved


def select_focus(
    paths: Iterable[str],
    *,
    repo_root: str | Path | None = None,
    base_sha: str = "0" * 40,
    head_sha: str = "1" * 40,
    base_tree_sha: str = "2" * 40,
    head_tree_sha: str = "3" * 40,
) -> dict[str, object]:
    root = None if repo_root is None else Path(repo_root).resolve()
    changed = tuple(sorted(set(paths)))
    if not changed:
        raise legacy.FocusGateError("a Focus Gate route requires at least one changed path")

    reasons: set[str] = set()
    gates = {"F0"}
    tests: set[str] = set()
    service_tests: set[str] = set()
    executable = [path for path in changed if not _is_documentation(path)]
    source_paths = [
        path
        for path in changed
        if path.endswith(".py")
        and path.startswith(("newsroom/", "scripts/"))
        and not _is_test(path)
    ]
    research_paths = [path for path in executable if _is_research_only(path)]
    normal_executable = [path for path in executable if not _is_research_only(path)]
    research_required = any(_triggers_research(path) for path in changed)
    research_only = bool(research_paths) and not normal_executable
    full_health_required = _matches_any(changed, SHARED_BREADTH_PATTERNS)
    owner_required = _matches_any(changed, F4_PATTERNS)
    service_required = _matches_any(changed, SERVICE_PATTERNS)
    stateful_required = _matches_any(changed, STATEFUL_PATTERNS)
    control_required = _matches_any(changed, CONTROL_PATTERNS)

    if all(_is_documentation(path) for path in changed):
        reasons.add("documentation_only:F0")
    if research_required:
        reasons.add("research_inputs_changed:isolated_lane")
    if normal_executable:
        gates.add("F1")
        reasons.add("executable_change:F1")
    if stateful_required or control_required:
        gates.add("F2")
        reasons.add("stateful_contract:F2" if stateful_required else "sdlc_control:F2")
    if service_required:
        gates.add("F3")
        reasons.add("actual_neo4j_semantics:F3")
    if owner_required:
        gates.add("F4")
        reasons.add("irreversible_or_public_effect:F4")

    for path in changed:
        if _is_test(path) and not _is_research_only(path):
            (service_tests if _is_service_test(path) else tests).add(path)
    if control_required:
        tests.update(legacy._existing(root, CONTROL_TESTS))

    if source_paths and not research_only and root is not None:
        contract_tests = _source_body_contract_tests(
            root, changed, source_paths, base_sha, head_sha,
        )
        if contract_tests is None:
            discovered, unresolved = _discover_tests(
                root, source_paths, base_sha=base_sha, head_sha=head_sha
            )
        else:
            discovered, unresolved = contract_tests, False
        for path in discovered:
            (service_tests if _is_service_test(path) else tests).add(path)
        if discovered:
            gates.add("F2")
            reasons.add("repository_import_or_symbol_consumers:F2" if contract_tests is None
                        else "explicit_source_name_date_body_contract:F2")
        if unresolved:
            full_health_required = True
            reasons.add("unresolved_dependency_analysis:full_health")

    if _matches_any(changed, PREPARED_CANARY_PARITY_PATTERNS):
        tests.update(legacy._existing(root, PREPARED_CANARY_PARITY_TESTS))
        gates.add("F1")
        reasons.add("prepared_canary_parity:F1")
    if _matches_any(changed, PREPARED_CANARY_CONSUMER_PATTERNS):
        tests.update(legacy._existing(root, PREPARED_CANARY_CONSUMER_TESTS))
        gates.add("F2")
        reasons.add("prepared_canary_consumers:F2")
    if stateful_required:
        if not (tests or service_tests):
            full_health_required = True
            reasons.add("stateful_without_direct_evidence:full_health")
        tests.update(legacy._existing(root, STATEFUL_SENTINELS))
        reasons.add("bounded_stateful_sentinels:F2")
    if service_tests:
        service_required = True
        gates.add("F3")
        reasons.add("actual_service_consumer:F3")
    if service_required and not service_tests:
        service_tests.update(legacy._glob_tests(root, ("test_*_neo4j_service.py",)))
        reasons.add("service_fallback_inventory:F3")

    known_roots = ("newsroom/", "scripts/", "docs/", ".github/", ".sdlc/")
    unknown = [
        path
        for path in normal_executable
        if not path.startswith(known_roots) and path not in {"pyproject.toml", "uv.lock"}
    ]
    if unknown:
        full_health_required = True
        reasons.add("unknown_executable_path:full_health")

    tests = legacy._existing(root, tests)
    service_tests = legacy._existing(root, service_tests)
    if normal_executable and not research_only and not (
        tests or service_tests or full_health_required
    ):
        full_health_required = True
        reasons.add("no_defensible_focused_test:full_health")
    if full_health_required:
        gates.update({"F1", "F2"})
        reasons.add("cross_cutting_or_unresolved:full_health")

    yaml_bootstrap = any(Path(path).suffix.lower() in {".yml", ".yaml"} for path in executable)
    if research_only:
        tests.clear()
        service_tests.clear()
        full_health_required = False
        service_required = False
        gates = {"F0"}
    bootstrap = bool(tests or service_tests or full_health_required or service_required or yaml_bootstrap)
    body = legacy._manifest_without_digest(
        changed=changed,
        base_sha=base_sha,
        head_sha=head_sha,
        base_tree_sha=base_tree_sha,
        head_tree_sha=head_tree_sha,
        gates=sorted(gates),
        tests=sorted(tests),
        service_tests=sorted(service_tests),
        research_required=research_required,
        full_health_required=full_health_required,
        owner_authority_required=owner_required,
        bootstrap_required=bootstrap,
        reasons=sorted(reasons),
    )
    return {**body, "manifest_digest": legacy._digest(body)}

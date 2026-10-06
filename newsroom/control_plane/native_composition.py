"""Actual private Hermes pipeline composition over the existing native ports."""

from __future__ import annotations

import json
import os
import secrets
import shlex
import sqlite3
import subprocess
import threading
from collections.abc import Callable, Mapping
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter_ns, process_time_ns

from newsroom.authority import AuthenticationProof, UtcTimestamp
from newsroom.authority.canonical import digest_canonical
from newsroom.authority.neo4j_projection_system import (
    open_native_retrieval_neo4j_resources,
)
from newsroom.increment5.exact_retriever import SQLiteExactRetriever
from newsroom.increment5.fulltext_journal import FullTextReceiptJournal
from newsroom.increment5.fulltext_retriever import FullTextRetriever
from newsroom.increment5.native_retrieval import NativeRetrievalHold, NativeRetrievalPort
from newsroom.increment5.receipt_journal import BranchReceiptJournal
from newsroom.increment5.retrieval_context import RetrievalContextJournal
from newsroom.increment6.work_items import RetrievalContextAuthority
from newsroom.projection.neo4j.models import Neo4jProjectorConfig
from newsroom.sources import SourceDefinitionId, SourceDefinitionVersionId

from .admission import QUALIFICATION_RELATION_POLICY_VERSION
from .diagnostic_logging import emit_diagnostic
from .evidence import FACTUAL_LOCALISATION_POLICY_VERSION, NAMED_ENTITY_POLICY_VERSION
from .govuk_evidence import GovUkEvidenceAcquisition, POLICY_DIGEST as GOVUK_TRANSPORT_POLICY
from .govuk_spreadsheet_evidence import (
    GovUkSpreadsheetEvidenceAcquisition,
    POLICY_DIGEST as GOVUK_SPREADSHEET_TRANSPORT_POLICY,
)
from .govuk_pdf_evidence import GovUkPdfEvidenceAcquisition, POLICY_DIGEST as GOVUK_PDF_TRANSPORT_POLICY

from .govuk_rights import (
    LICENCE_URL, REUSE_URL, GovUkLicenceEvidence, retain_current_govuk_licence,
    _govuk_semantic_evidence,
)
from .graphiti_operational_readiness import OPERATOR_AUTHORITY_DOMAIN, OPERATOR_PRINCIPAL_ID
from .model_usage import InvocationEfficiencyPolicy, ModelUsageService
from .native_assessor import (
    AutonomousNativeEvidenceAssessor, NativeAssessmentUsage,
    RETAINED_ASSESSMENT_POLICY_VERSION,
    QUALIFICATION_CLAUSE_CONSUMER_VERSION,
    VERSION as ASSESSOR_CONTRACT_VERSION,
)
from .native_context_materialisation import VERSION as CONTEXT_MATERIALISATION_VERSION
from .native_collision import NativeCollisionAuthority, NativeCollisionIdentity
from .native_cycle import _uuid4_for
from .native_discovery import NativeDiscovery
from .native_embeddings import NativePassageEmbedder
from .native_evidence import (
    EvidenceAssessor, EvidenceTransport, NativeEvidenceController,
    NativeEvidenceHold,
)
from .native_graphiti import NativeGraphitiProcessor
from .native_pipeline import NativePipeline
from .native_policies import VERSION, native_policy_components
from .native_progress import NativeRevisionJournal, source_header
from .native_publication import NativePublicationContinuation
from .native_retrieval import NativeRetrievalContinuation, compose_native_documents
from .native_runtime import open_native_runtime
from .native_source_intake import (
    NativeSourceIntake,
    native_evidence_sources,
    spreadsheet_asset_url, pdf_asset_url,
)
from .native_source_rights import (
    NativePortfolioRights, observe_portfolio_terms, read_rights_observation,
    retain_rights_snapshot_bundle, require_rights_assessment, fetch_licensing_observations,
)
from .native_assessor_spans import PARTITION_VERSION
from .native_source_definitions import MISSING_SOURCE_IDS, register_missing_native_source_definitions
from .native_weather_sources import poll_other_source
from .native_weather_evidence import NativeWeatherEvidenceAcquisition, POLICY_DIGEST as WEATHER_TRANSPORT_POLICY
from .store import connect
from .zh_hant import ZH_HANT_HK_SHAPE_POLICY_VERSION

ASSESSMENT_CONTRACT_VERSION = (
    f"{ASSESSOR_CONTRACT_VERSION}+{NAMED_ENTITY_POLICY_VERSION}+"
    f"{ZH_HANT_HK_SHAPE_POLICY_VERSION}+{FACTUAL_LOCALISATION_POLICY_VERSION}+"
    f"{QUALIFICATION_RELATION_POLICY_VERSION}+{RETAINED_ASSESSMENT_POLICY_VERSION}+{PARTITION_VERSION}"
    f"+{QUALIFICATION_CLAUSE_CONSUMER_VERSION}"
    f"+{CONTEXT_MATERIALISATION_VERSION}"
)

TRANSPORT_POLICY = digest_canonical({
    "version": "hermes-native-independent-evidence-v2",
    "govuk": GOVUK_TRANSPORT_POLICY,
    "govuk_spreadsheet": GOVUK_SPREADSHEET_TRANSPORT_POLICY,
    "govuk_pdf": GOVUK_PDF_TRANSPORT_POLICY,
    "weather": WEATHER_TRANSPORT_POLICY,
})


def _require_hko_current_source(package, current, *, sources, definition_version_id, locator, proof):
    """Fence one timestamped HKO warning against existing canonical source state."""
    if current.source_id != "HK-02" or current.currency_family not in {"CURRENT_VERSION", "COMPLETED_HISTORICAL_EVENT"}:
        return
    from newsroom.checks import deterministic_uuid4
    from newsroom.graphiti_adapter.identity import content_digest
    from newsroom.increment10.editorial import EditorialHold
    from newsroom.increment9.proving import SOURCE_URLS
    from newsroom.sources import SourceItemId
    from newsroom.sources.types import TimePrecision
    from .native_source_intake import VERSION as INTAKE_VERSION
    from .native_weather_evidence import _hko_warning

    def unknown():
        raise EditorialHold(reason="NATIVE_STORY_SOURCE_ORDER_UNKNOWN")

    if locator != SOURCE_URLS["HK-02"]:
        unknown()
    try:
        raw = package.passages[package.source_ids.index(current.source_id)].partition("\n\n")[0]
        warning, _label = _hko_warning(raw.encode())
        key = next(iter(json.loads(raw)))
        selected_time = UtcTimestamp.parse(current.version_reference).value
        if UtcTimestamp.parse(warning["updateTime"]).value != selected_time:
            unknown()
    except (ValueError, TypeError, KeyError, IndexError, UnicodeError):
        unknown()
    item_id = deterministic_uuid4(SourceItemId, namespace=f"{INTAKE_VERSION}:item",
        semantic_value=[str(definition_version_id), current.source_id, key])
    latest = sources.latest_revision(item_id, proof=proof)

    def timestamp(revision):
        if (revision is None or revision.request.item_id != item_id
                or str(revision.request.definition_version_id) != str(definition_version_id)
                or revision.request.source_updated_time.precision is not TimePrecision.EXACT):
            unknown()
        return UtcTimestamp.parse(revision.request.source_updated_time.value).value

    latest_time = timestamp(latest)
    if latest.request.prior_revision_id is not None:
        previous_time = timestamp(sources.revision(latest.request.prior_revision_id, proof=proof))
        # Ledger order is not a timestamp high-water. Refuse a backdated current
        # revision rather than asserting that its name establishes currentness.
        if latest_time < previous_time:
            unknown()
    if selected_time < latest_time:
        raise EditorialHold(reason="NATIVE_STORY_SOURCE_SUPERSEDED")
    if (selected_time != latest_time or latest.request.permitted_state_digest != content_digest(
            headline=warning["name"], body=raw, canonical_url=locator)):
        unknown()


def _lexical_path(path: str | Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _require_safe_deployment_path(
    path: Path, *, label: str, required: bool, directory: bool,
) -> None:
    path = _lexical_path(path)
    for candidate in (path, *path.parents):
        if candidate.is_symlink():
            raise ValueError(f"native deployment {label} path contains a symlink")
    if required and not path.exists():
        raise ValueError(f"native deployment {label} is absent")
    if path.exists() and path.is_dir() != directory:
        expected = "directory" if directory else "file"
        raise ValueError(f"native deployment {label} is not a {expected}")


def _native_deployment_preflight(
    *, supplied_ledger: str | Path, supplied_lock: str | Path,
    expected_ledger: Path, expected_lock: Path,
    required_files: Mapping[str, Path], required_directories: Mapping[str, Path],
    creatable_files: Mapping[str, Path],
) -> None:
    if _lexical_path(supplied_ledger) != _lexical_path(expected_ledger):
        raise ValueError("native service must use the canonical private ledger")
    if _lexical_path(supplied_lock) != _lexical_path(expected_lock):
        raise ValueError("native service must use its canonical singleton lock")
    for label, path in required_files.items():
        _require_safe_deployment_path(
            path, label=label, required=True, directory=False,
        )
    for label, path in required_directories.items():
        _require_safe_deployment_path(
            path, label=label, required=True, directory=True,
        )
    for label, path in creatable_files.items():
        _require_safe_deployment_path(
            path, label=label, required=False, directory=False,
        )


@contextmanager
def _native_cursor_credential():
    """Bind only the existing purpose-provisioned SDK key, never the whole .env."""
    name = "CURSOR_API_KEY"
    if os.environ.get(name):
        yield
        return
    path = Path.home() / "Coding/newsroom/.env"
    metadata = path.stat()
    if metadata.st_uid != os.getuid() or metadata.st_mode & 0o022:
        raise ValueError("native Cursor credential file ownership differs")
    values = [
        shlex.split(line.split("=", 1)[1], comments=True)
        for line in path.read_text().splitlines()
        if line.startswith(name + "=")
    ]
    if len(values) != 1 or len(values[0]) != 1 or not values[0][0].strip():
        raise ValueError("native purpose-provisioned Cursor credential is absent")
    previous = os.environ.get(name)
    os.environ[name] = values[0][0]
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def _require_semantic_current_sources(sources, binding, *, proof, rights_for):
    for source in binding.get('source_currentness', ()):
        definition = sources.current_summary(
            SourceDefinitionId.parse(source['definition_id']), proof=proof)
        if str(definition.version_id) != source['definition_version_id']:
            raise ValueError('semantic source definition changed')
        version = sources.version_details(definition.version_id, proof=proof)
        if rights_for(source['source_id'], version.request.locator) is None:
            raise ValueError('semantic current rights unavailable')


def _judgment_scope(journal, sources, acquired):
    # A fresh observation is not proof of a new material fact.
    # A source-declared first publication remains distinct from a proved change.
    if not sources or len(sources) != len(acquired):
        return {'coverage': 'PARTIAL', 'newness': 'UNKNOWN', 'prior_scope': None}
    scope = {'coverage': 'COMPLETE', 'newness': 'UNKNOWN', 'prior_scope': None,
        'current_scope': {'sources': [{'source_id': source.unit.source_id,
            'published_at': item.publication_time, 'updated_at': item.source_updated_time,
            'retrieved_at': item.retrieval_time,
            'body': item.body.decode('utf-8')} for source, item in zip(sources, acquired, strict=True)]},
        'source_currentness': [{'source_id': source.unit.source_id,
            'definition_id': str(source.unit.authority.definition_id),
            'definition_version_id': str(source.unit.authority.definition_version_id)} for source in sources]}
    # Source acquisition has already enforced this candidate's
    # declared sibling closure. An unrelated held item elsewhere
    # in the same feed is not a new blanket editorial veto.
    previous = []
    prior_found = False
    for source, item in zip(sources, acquired, strict=True):
        matches = []
        for revision_id in journal.units:
            header = source_header(journal.units, revision_id)
            if (header.source_id, header.item_key, header.canonical_url) != (
                    source.unit.source_id, source.unit.item_key, source.unit.canonical_url):
                continue
            if revision_id == str(source.unit.authority.revision_id):
                continue
            if max(header.observed_ats, default='') >= source.unit.observed_at:
                continue
            matches.append((max(header.observed_ats, default=''), revision_id))
        if not matches:
            break
        prior_found = True
        _, prior_id = max(matches)
        prior = journal.units[prior_id][0]
        if not prior.body.strip() or prior.authority is None:
            break
        previous.append({'source_id': prior.source_id, 'headline': prior.headline,
            'body': prior.body, 'published_at': prior.published_at, 'updated_at': prior.updated_at})
    if len(previous) == len(sources) and previous:
        scope['prior_scope'] = {'sources': previous}
        scope['newness'] = 'KNOWN_CHANGE' if any(
            prior['body'].strip() != source.unit.body.strip()
            or prior['headline'].strip() != source.unit.headline.strip()
            for prior, source in zip(previous, sources, strict=True)) else 'KNOWN_UNCHANGED'
    elif not prior_found:
        first_publication = []
        for source, item in zip(sources, acquired, strict=True):
            # This acquisition parser has no date fallback: publication_time
            # comes from GOV.UK first_published_at, never public_updated_at.
            # Non-GOV.UK acquisition contracts require their own provenance.
            if (source.unit.source_id not in {'UK-01', 'UK-02', 'UK-03', 'UK-05'}
                    or not item.canonical_url.startswith('https://www.gov.uk/')
                    or item.source_type != 'PRIMARY_OFFICIAL'
                    or item.currentness_basis != 'AUTHORITATIVE_CURRENT_CONTENT_ENDPOINT'
                    or not any(assignment.role.value == 'ORIGINATING_AUTHORITY'
                        for assignment in source.source_version.request.roles)):
                break
            try:
                published = UtcTimestamp.parse(item.publication_time).value
                retrieved = UtcTimestamp.parse(item.retrieval_time).value
            except (TypeError, ValueError):
                break
            if published > retrieved:
                break
            first_publication.append({'source_id': source.unit.source_id,
                'definition_id': str(source.unit.authority.definition_id),
                'definition_version_id': str(source.unit.authority.definition_version_id),
                'source_revision_digest': source.unit.revision_digest,
                'acquisition_receipt_digest': item.receipt_digest,
                'first_published_at': item.publication_time})
        if len(first_publication) == len(sources):
            scope['newness'] = 'SOURCE_DECLARED_FIRST_PUBLICATION'
            scope['first_publication'] = first_publication
    return scope


def _deployment_identity(*, revision, tree, paths, embedding_policy, assessment_policy,
                         semantic_policy_digests=()):
    from . import broker
    from .native_source_rights import POLICY_DIGEST as RIGHTS_POLICY
    identities = {}
    for name, path in sorted(paths.items()):
        path = Path(path)
        if path.is_symlink():
            raise ValueError("native deployment store is a symlink")
        stat = path.stat()
        identities[name] = {"path": str(path.resolve()), "device": stat.st_dev, "inode": stat.st_ino}
    value = {
        "version": VERSION, "revision": revision, "tree": tree,
        "stores": identities, "target": "hermes-private-serving", "public_effect": False,
        "embedding_policy": embedding_policy.canonical_digest,
        "assessment_policy": assessment_policy.canonical_digest,
        "rights_policy": RIGHTS_POLICY, "transport_policy": TRANSPORT_POLICY,
        "neo4j": {"host": broker.NEO4J_BOLT_HOST, "port": broker.NEO4J_BOLT_PORT,
                  "database": broker.NEO4J_DATABASE, "principal": broker.NEO4J_PROJECTOR_USERNAME},
    }
    if semantic_policy_digests:
        value['semantic_policies'] = list(semantic_policy_digests)
    return digest_canonical(value)


def deployed_native_service(args):
    """Compose the installed canonical private route, after the singleton lock."""
    from . import broker, native_assessor, native_embeddings
    from .cycle import assert_no_owner_emergency_stop, owner_emergency_stop_fence
    from .model_usage import WorkloadClass
    from .native_service import NativeService
    from .paths import (
        CANONICAL_PROVING_STORE, CANONICAL_UNPUBLISHED_STORE,
        CANONICAL_INCREMENT4_AUTHORITY_STORE, CANONICAL_OBJECT_CAS_ROOT,
        CANONICAL_GRAPHITI_WORKSPACE_ROOT, HOST_CONTROL_PLANE_STATE_ROOT,
    )
    from .writer import cont_writer_implementation_identity
    from newsroom.increment9.proving import SOURCE_URLS

    if _lexical_path(args.ledger) != _lexical_path(CANONICAL_UNPUBLISHED_STORE):
        raise ValueError("native service must use the canonical private ledger")
    private_root = HOST_CONTROL_PLANE_STATE_ROOT / "native"
    expected_lock = private_root / "hermes.lock"
    if _lexical_path(args.lock) != _lexical_path(expected_lock):
        raise ValueError("native service must use its canonical singleton lock")
    check = lambda: assert_no_owner_emergency_stop(str(CANONICAL_PROVING_STORE))
    fence = lambda: owner_emergency_stop_fence(str(CANONICAL_PROVING_STORE))
    service_event = threading.Event()

    def preflight():
        _native_deployment_preflight(
            supplied_ledger=args.ledger, supplied_lock=args.lock,
            expected_ledger=CANONICAL_UNPUBLISHED_STORE,
            expected_lock=expected_lock,
            required_files={
                "authority": CANONICAL_INCREMENT4_AUTHORITY_STORE,
                "private ledger": CANONICAL_UNPUBLISHED_STORE,
                "proving": CANONICAL_PROVING_STORE,
            },
            required_directories={
                "Object CAS": CANONICAL_OBJECT_CAS_ROOT,
                "Graphiti workspace": CANONICAL_GRAPHITI_WORKSPACE_ROOT,
            },
            creatable_files={
                "singleton lock": expected_lock,
                "evidence intake": private_root / "evidence-intake.sqlite3",
                "private serving": private_root / "private-serving.sqlite3",
                "retrieval": private_root / "retrieval.sqlite3",
            },
        )

    @contextmanager
    def pipeline():
        check()
        revision, clean = cont_writer_implementation_identity()
        if not clean:
            raise ValueError("native deployment implementation is not exact and clean")
        usage = ModelUsageService(str(CANONICAL_UNPUBLISHED_STORE))
        embedding = usage.qualified_policy(
            workload_class=WorkloadClass.NATIVE_RETRIEVAL_EMBEDDING,
            provider="openrouter", route=native_embeddings.ROUTE,
            model=native_embeddings.OPENROUTER_EMBEDDING_SLUG, reasoning="none",
            output_schema_digest=native_embeddings.SCHEMA_DIGEST,
        )
        assessment = usage.qualified_policy(
            workload_class=WorkloadClass.NATIVE_EVIDENCE_ASSESSOR,
            provider=native_assessor.CONT_PRIMARY_PROVIDER, route=native_assessor.ROUTE,
            model=native_assessor.MODEL, reasoning=native_assessor.REASONING,
            config_identity=native_assessor.CONFIG_IDENTITY,
            output_schema_digest=native_assessor.PROVIDER_SCHEMA_DIGEST,
        )
        if (assessment.prompt_contract_version, assessment.model, assessment.reasoning,
                assessment.command_flags, assessment.max_output_tokens) != (
            native_assessor.VERSION, native_assessor.MODEL, native_assessor.REASONING,
            native_assessor.COMMAND_FLAGS, None,
        ):
            raise ValueError("native assessor profile differs before authority OPEN")
        from .typesafe_judgment import ROUTE as JUDGMENT_ROUTE, SCHEMA_DIGEST as JUDGMENT_SCHEMA
        from .native_claim_localisation import ROUTE as LOCALISATION_ROUTE, SCHEMA_DIGEST as LOCALISATION_SCHEMA
        from . import typesafe_judgment, native_claim_localisation
        from newsroom.authority.canonical import digest_bytes
        from .model_usage import ModelUsageAdmissionError
        semantic_kwargs = {}
        key_path = Path.home() / '.config/newsroom/credentials/typesafe-jev-evaluation.key'
        if key_path.exists():
            try:
                semantic_policy = usage.qualified_policy(workload_class=WorkloadClass.TYPESAFE_JUDGMENT,
                    provider='typesafe', route=JUDGMENT_ROUTE, model='jev-latest', reasoning='none',
                    output_schema_digest=JUDGMENT_SCHEMA,
                    implementation_revision=digest_bytes(Path(typesafe_judgment.__file__).read_bytes()))
                rendering_policy = usage.qualified_policy(workload_class=WorkloadClass.NATIVE_EVIDENCE_ASSESSOR,
                    provider='grok-build-cli', route=LOCALISATION_ROUTE, model='grok-4.7', reasoning='high',
                    output_schema_digest=LOCALISATION_SCHEMA,
                    implementation_revision=digest_bytes(Path(native_claim_localisation.__file__).read_bytes()))
                if (semantic_policy.implementation_revision != typesafe_judgment.implementation_digest()
                        or rendering_policy.implementation_revision != digest_bytes(
                            Path(native_claim_localisation.__file__).read_bytes())):
                    raise ModelUsageAdmissionError('semantic implementation qualification is stale')
            except ModelUsageAdmissionError:
                from .diagnostic_logging import emit_diagnostic
                emit_diagnostic('native_semantic_configuration', {'state': 'CONFIG_HOLD'})
                # The independently qualified existing assessor remains usable;
                # missing semantic qualification is not a NOQUAL decision.
            else:
                def semantic_key():
                    metadata = key_path.lstat()
                    if key_path.is_symlink() or metadata.st_uid != os.getuid() or metadata.st_mode & 0o077:
                        raise ValueError('native semantic credential ownership differs')
                    value = key_path.read_text().strip()
                    if not value.startswith('apikey_') or len(value) < 32:
                        raise ValueError('native semantic credential shape differs')
                    return value
                semantic_kwargs = dict(judgment_api_key=semantic_key,
                    judgment_policy=semantic_policy, localisation_policy=rendering_policy)
                from . import native_source_qualification
                try:
                    exception_policy = usage.qualified_policy(
                        workload_class=WorkloadClass.NATIVE_EVIDENCE_ASSESSOR,
                        provider='grok-build-cli', route=native_source_qualification.ROUTE,
                        model=native_source_qualification.MODEL, reasoning='high',
                        config_identity=native_source_qualification.VERSION,
                        output_schema_digest=native_source_qualification.SCHEMA_DIGEST)
                    if exception_policy.implementation_revision != digest_bytes(
                            Path(native_source_qualification.__file__).read_bytes()):
                        raise ModelUsageAdmissionError('source qualification implementation is stale')
                except ModelUsageAdmissionError:
                    pass  # Typed decisions remain qualified; exceptions hold.
                else:
                    semantic_kwargs['source_qualification_policy'] = exception_policy
        tree = subprocess.check_output(
            ("/usr/bin/git", "rev-parse", f"{revision}^{{tree}}"),
            cwd=Path(__file__).resolve().parents[2], text=True, timeout=10,
        ).strip()
        identity_paths = {
            "authority": CANONICAL_INCREMENT4_AUTHORITY_STORE,
            "cas": CANONICAL_OBJECT_CAS_ROOT, "private_ledger": CANONICAL_UNPUBLISHED_STORE,
            "proving": CANONICAL_PROVING_STORE,
            "intake": private_root / "evidence-intake.sqlite3",
            "serving": private_root / "private-serving.sqlite3",
            "retrieval": private_root / "retrieval.sqlite3",
        }

        def identity(paths=identity_paths):
            return _deployment_identity(
                revision=revision, tree=tree, paths=paths,
                embedding_policy=embedding, assessment_policy=assessment,
                semantic_policy_digests=tuple(semantic_kwargs[key].canonical_digest
                    for key in ('judgment_policy', 'localisation_policy', 'source_qualification_policy') if key in semantic_kwargs),
            )

        opening_paths = {
            name: path for name, path in identity_paths.items()
            if Path(path).exists()
        }
        opening_identity = identity(opening_paths)
        # Discovery only: each selected identity is authenticated again by the
        # Source facade before an observation. No Source Definition is invented.
        connection = sqlite3.connect(CANONICAL_INCREMENT4_AUTHORITY_STORE.as_uri() + "?mode=ro", uri=True)
        try:
            connection.execute("PRAGMA query_only=ON")
            rows = connection.execute(
                "SELECT h.definition_id,v.locator FROM source_definition_version_heads h "
                "JOIN source_definition_versions v ON v.version_id=h.current_version_id"
            ).fetchall()
        finally:
            connection.close()
        bindings = {}
        for source_id, url in SOURCE_URLS.items():
            matching = [definition for definition, locator in rows if locator == url]
            if len(matching) > 1:
                raise ValueError("current native source definition identity is ambiguous")
            if matching:
                bindings[source_id] = SourceDefinitionId.parse(matching[0])
        with _native_cursor_credential(), open_native_pipeline(
            authority_path=CANONICAL_INCREMENT4_AUTHORITY_STORE,
            object_root=CANONICAL_OBJECT_CAS_ROOT, workspace_root=CANONICAL_GRAPHITI_WORKSPACE_ROOT,
            private_path=CANONICAL_UNPUBLISHED_STORE, proving_path=CANONICAL_PROVING_STORE,
            intake_path=private_root / "evidence-intake.sqlite3",
            serving_path=private_root / "private-serving.sqlite3",
            retrieval_path=private_root / "retrieval.sqlite3",
            neo4j_config=broker.neo4j_projector_config(), embedding_key=broker.openrouter_api_key(),
            embedding_policy=embedding, assessment_policy=assessment, source_definition_ids=bindings,
            licence=None, stop_check=check, stop_fence=fence, implementation_worktree_clean=clean,
            **semantic_kwargs,
            service_event=service_event,
            reassessment_quantum_seconds=args.interval,
        ) as composed:
            if identity(opening_paths) != opening_identity:
                raise ValueError("native deployment identity changed during open")
            composed.runtime_identity_digest = identity()
            yield composed

    return NativeService(
        pipeline_factory=pipeline, ledger_path=str(CANONICAL_UNPUBLISHED_STORE),
        lock_path=Path(args.lock), stop_check=check, interval_seconds=args.interval,
        failure_backoff_seconds=args.failure_backoff,
        preflight=preflight,
        service_event=service_event,
    )


@contextmanager
def open_native_pipeline(
    *, authority_path: Path, object_root: Path, workspace_root: Path,
    private_path: Path, proving_path: Path, intake_path: Path, serving_path: Path,
    retrieval_path: Path, neo4j_config: Neo4jProjectorConfig,
    embedding_key: str, embedding_policy: InvocationEfficiencyPolicy,
    assessment_policy: InvocationEfficiencyPolicy,
    source_definition_ids: Mapping[str, SourceDefinitionId],
    licence: GovUkLicenceEvidence | NativePortfolioRights | None,
    stop_check: Callable[[], None], stop_fence: Callable,
    implementation_worktree_clean: bool,
    service_event: threading.Event | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(tz=UTC),
    reassessment_quantum_seconds: float = 300,
    judgment_api_key: Callable[[], str] | None = None,
    judgment_policy: InvocationEfficiencyPolicy | None = None,
    localisation_policy: InvocationEfficiencyPolicy | None = None,
    source_qualification_policy: InvocationEfficiencyPolicy | None = None,
):
    """Open one real runtime after its invocation policies are qualified.

    Credentials remain process-local. No route fallback, legacy intake writer,
    fixture rights renewal, historical campaign or public target is composed.
    """
    stop_check()
    operator_drain_requested = (
        (lambda: False) if service_event is None else service_event.is_set
    )
    if not implementation_worktree_clean:
        raise ValueError("native provider composition requires its reviewed clean implementation")
    principal, domain = OPERATOR_PRINCIPAL_ID, OPERATOR_AUTHORITY_DOMAIN
    target_id = "hermes-private-serving"
    credential = secrets.token_urlsafe(32)
    proof = AuthenticationProof(method="STATIC_TOKEN", credential=credential)
    now = lambda: UtcTimestamp(clock().astimezone(UTC))
    policies = native_policy_components(
        principal_id=principal, authority_domain=domain,
        target_path=serving_path, target_id=target_id,
    )
    # The native passage index and admitted Increment 4 graph are different
    # projections; their identities are bound separately in the context.
    generation_basis = {
        "version": VERSION, "authority": str(authority_path.resolve()),
        "document_policy": policies.retrieval_document_definition,
        "neo4j_destination": {"uri": neo4j_config.uri, "database": neo4j_config.database,
                              "projector_principal": neo4j_config.username},
    }
    generation_digest = digest_canonical(generation_basis)
    generation_id = _uuid4_for(generation_basis)
    suffix = generation_id.replace("-", "")[:16]
    scope = f"hermes-native:{generation_digest}"
    with ExitStack() as resources:
        private = connect(str(private_path))
        resources.callback(private.close)
        proving = sqlite3.connect(proving_path.as_uri() + "?mode=ro", uri=True)
        proving.execute("PRAGMA query_only=ON")
        resources.callback(proving.close)
        journal = NativeRevisionJournal(private)
        usage = ModelUsageService(str(private_path))
        def require_current_story_sources(package, currentness):
            from newsroom.increment10.editorial import EditorialHold
            stop_check()
            if tuple(item.source_id for item in currentness) != package.source_ids:
                raise EditorialHold(reason="NATIVE_STORY_SOURCE_BINDING_HOLD")
            for item in currentness:
                definition = runtime.authority.sources.current_summary(
                    SourceDefinitionId.parse(item.source_definition_id), proof=proof)
                version = runtime.authority.sources.version_details(definition.version_id, proof=proof)
                if (version.canonical_digest != item.source_definition_revision_digest
                        or source_rights(item.source_id, version.request.locator) is None):
                    raise EditorialHold(reason="NATIVE_STORY_CURRENT_SOURCE_RIGHTS_HOLD")
                _require_hko_current_source(package, item, sources=runtime.authority.sources,
                    definition_version_id=definition.version_id, locator=version.request.locator, proof=proof)

        def source_currentness_fence(package, currentness):
            with stop_fence():
                require_current_story_sources(package, currentness)

        def story_writer(package, **identities):
            from newsroom.increment10.editorial import EditorialHold
            from .model_usage import ModelUsageAdmissionError
            from .native_story_model import NativeStoryModel, load_story_model_policies
            from .native_story_writer import NativeStoryWriterHold
            from .writer import WriterDispatchError, CliProcessError, CliTimeoutError
            currentness = identities.pop("source_currentness")
            source_records = identities.pop('source_records',())
            cached_only = identities.pop('cached_only',False)
            def require_story_sources():
                require_current_story_sources(package, currentness)
            @contextmanager
            def writer_fence():
                with stop_fence():
                    require_story_sources()
                    yield
            try:
                model = NativeStoryModel(usage, load_story_model_policies(usage),
                    fence=writer_fence, stop_check=require_story_sources, clock=clock,cached_only=cached_only)
                result = model.write(package,source_currentness=currentness,source_records=source_records, **identities)
                require_story_sources()
                return result
            except (ModelUsageAdmissionError, NativeStoryWriterHold) as exc:
                held=EditorialHold(reason=getattr(exc, "reason_code", str(exc)))
                held.stable_reason_codes=getattr(exc,'stable_reason_codes',(str(held),))
                raise held from exc
            except (WriterDispatchError, CliProcessError, CliTimeoutError, json.JSONDecodeError) as exc:
                raise EditorialHold(reason="NATIVE_STORY_PROVIDER_RESULT_HOLD") from exc
        RetrievalContextJournal(retrieval_path)
        exact = SQLiteExactRetriever(
            authority_database=authority_path, journal=BranchReceiptJournal(retrieval_path),
        )
        fulltext_journal = FullTextReceiptJournal(retrieval_path)
        neo4j_resources = open_native_retrieval_neo4j_resources(
            config=neo4j_config,
            generation_id=generation_id,
            fulltext_index=f"native_fulltext_{suffix}", vector_index=f"native_vector_{suffix}",
        )
        resources.callback(neo4j_resources.close)
        projector = neo4j_resources.projector
        reader = neo4j_resources.fulltext
        components = {}

        def dependencies(*, objects, extraction, commands, events):
            documents = compose_native_documents(
                objects=objects, extraction=extraction, commands=commands,
                events=events, projector=projector, policies=policies,
                principal_id=principal, authority_domain=domain,
            )
            collision = NativeCollisionAuthority(
                # Retrieval is ATTACHed to the authority writer. Its BEGIN
                # IMMEDIATE also locks that database, so collision observations
                # use the existing private ledger, never a second attached writer.
                authority_path=authority_path, journal_path=private_path,
                identity=NativeCollisionIdentity(scope, principal, domain),
                context_reader=documents.read_context, events=events,
            )
            components.update(documents=documents, collision=collision)
            return RetrievalContextAuthority(
                retrieval_path, {}, native_context_read_port=documents.context_read_port(proof=proof),
            ), collision.enforcer, collision.candidate_citation_read_port()

        runtime = resources.enter_context(open_native_runtime(
            authority_path=authority_path, object_root=object_root, workspace_root=workspace_root,
            intake_path=intake_path, target_path=serving_path, target_id=target_id,
            credential=credential, principal_id=principal, authority_domain=domain,
            neo4j_config=neo4j_config, native_dependency_factory=dependencies, clock=now,
            story_writer=story_writer,
            source_currentness_fence=source_currentness_fence,
        ))
        documents = components["documents"]
        # Capture the existing checked store-instance identity only after the
        # runtime owns its writer. Content, observations and releases are not epochs.
        if authority_path.is_symlink():
            raise ValueError("native rights store is a symlink")
        store_stat = authority_path.stat()
        rights_reobservation_epoch = digest_canonical({
            "path": str(authority_path.resolve()), "device": store_stat.st_dev, "inode": store_stat.st_ino,
        })
        from newsroom.increment9.proving import SOURCE_URLS
        if licence is None:
            def refresh_current_rights():
                observed = fetch_licensing_observations(stop_check=stop_check, stop_fence=stop_fence)
                def fetched(url):
                    value = observed[url]
                    if isinstance(value, Exception):
                        raise value
                    return value
                try:
                    govuk = retain_current_govuk_licence(
                        objects=runtime.authority.objects, proof=proof,
                        dispatch_fence=stop_fence, clock=clock, fetch=fetched,
                    )
                    govuk_reason = "REVIEWED_REUSE_PERMITTED"
                except NativeEvidenceHold as exc:
                    govuk = None
                    govuk_reason = exc.reason_code
                # GOV.UK failure does not suppress an independent weather or
                # portfolio observation. VetoError still propagates from both.
                evidence = observe_portfolio_terms(
                    objects=runtime.authority.objects, proof=proof,
                    stop_check=stop_check, stop_fence=stop_fence, clock=clock, fetch=fetched,
                    parallel_observation=False,
                )
                current = NativePortfolioRights(
                    govuk, evidence, govuk_reason=govuk_reason,
                )
                snapshot_inputs = {}
                for source_id, definition_url in SOURCE_URLS.items():
                    assessment = current.for_source(source_id=source_id, definition_url=definition_url)
                    source_evidence = evidence.get(source_id)
                    if source_evidence is not None:
                        observed_at = source_evidence.observed_at
                        reason = source_evidence.reason
                        observations = source_evidence.observations
                    elif govuk is not None:
                        observed_at = govuk.observed_at
                        reason = "REVIEWED_REUSE_PERMITTED"
                        observations = tuple(
                            (url, digest, str(admission), "")
                            for url, digest, admission in zip(
                                (REUSE_URL, LICENCE_URL), govuk.raw_digests,
                                govuk.admission_ids, strict=True,
                            )
                        )
                    else:
                        observed_at = clock().astimezone(UTC).isoformat()
                        reason = govuk_reason
                        observations = ()
                    snapshot_inputs[source_id] = dict(
                        definition_url=definition_url,
                        assessment=assessment,
                        observed_at=observed_at, reason=reason,
                        observations=observations,
                        govuk_semantic_evidence=(
                            _govuk_semantic_evidence(source_id=source_id, definition_url=definition_url)
                            if govuk is not None and source_evidence is None and assessment.decision == "PERMITTED"
                            else None
                        ),
                    )
                snapshots = retain_rights_snapshot_bundle(
                    objects=runtime.authority.objects, proof=proof,
                    snapshots=snapshot_inputs, stop_check=stop_check,
                    reobservation_epoch=rights_reobservation_epoch,
                )
                return govuk, govuk_reason, evidence, snapshots

            licence = NativePortfolioRights(
                None, {}, refresh_current=refresh_current_rights,
            )
            licence.refresh()
            opening_snapshot_unused = True

            def refresh_licence():
                nonlocal opening_snapshot_unused
                if opening_snapshot_unused:
                    opening_snapshot_unused = False
                    return
                licence.refresh()
        elif type(licence) is GovUkLicenceEvidence:
            licence = NativePortfolioRights(licence, {})
            refresh_licence = licence.refresh
        elif type(licence) is NativePortfolioRights:
            refresh_licence = licence.refresh
        else:
            raise ValueError("native source rights binding differs")
        licence.require_retained(objects=runtime.authority.objects, proof=proof)
        embedder = NativePassageEmbedder(
            api_key=embedding_key, objects=runtime.authority.objects, usage=usage,
            policy=embedding_policy, dispatch_fence=stop_fence,
            implementation_worktree_clean=implementation_worktree_clean, clock=clock,
        )
        assessment_usage = NativeAssessmentUsage(
            usage, assessment_policy, clock=clock
        )
        assessor = AutonomousNativeEvidenceAssessor(
            usage=assessment_usage,
            dispatch_fence=stop_fence,
        )
        typed_proposal_verifier = None
        if judgment_api_key is not None:
            from .native_assessor_judgments import NativeAssessorJudgments, VERSION as JUDGMENT_CONTRACT
            from .native_claim_localisation import NativeClaimLocaliser
            from .typesafe_judgment import TypesafeJudgment
            if judgment_policy is None or localisation_policy is None:
                raise ValueError('qualified semantic and localisation policies are required')

            @contextmanager
            def judgment_fence(binding, current_proof):
                with stop_fence():
                    stop_check()
                    if current_proof is not proof:
                        raise ValueError('semantic caller proof differs')
                    _require_semantic_current_sources(runtime.authority.sources, binding,
                        proof=proof, rights_for=source_rights)
                    yield

            def judgment_scope(candidate, base, sources, acquired):
                return _judgment_scope(journal, sources, acquired)

            judgments = TypesafeJudgment(usage=usage, objects=runtime.authority.objects,
                policy=judgment_policy, api_key=judgment_api_key, source_fence=judgment_fence,
                implementation_worktree_clean=implementation_worktree_clean, clock=clock)
            localiser = NativeClaimLocaliser(usage=usage, objects=runtime.authority.objects,
                policy=localisation_policy, source_fence=judgment_fence,
                implementation_worktree_clean=implementation_worktree_clean, clock=clock)

            def localise_claims(request):
                binding = request['source_binding']
                return localiser.localise(request, proof=proof, **{key: binding[key]
                    for key in ('candidate_id', 'hypothesis_digest', 'evidence_package_digest')})

            def read_claim_localisation(reference, request):
                binding = request['source_binding']
                return localiser.read_localisation(reference, request, proof=proof, **{key: binding[key]
                    for key in ('candidate_id', 'hypothesis_digest', 'evidence_package_digest')})

            assessor._judgments = NativeAssessorJudgments(judgments=judgments, proof=proof,
                scope_for=judgment_scope, localise=localise_claims, read_localisation=read_claim_localisation,
                require_current=stop_check)
            if source_qualification_policy is not None:
                from .native_source_qualification import NativeSourceQualifier, VERSION as QUALIFICATION_CONTRACT
                qualifier = NativeSourceQualifier(usage=usage, objects=runtime.authority.objects,
                    policy=source_qualification_policy, source_fence=judgment_fence, judgments=judgments,
                    implementation_worktree_clean=implementation_worktree_clean, clock=clock)

                def qualify_source(candidate, base, sources, acquired, fallback):
                    return qualifier.assess(candidate, base, sources, acquired, fallback,
                        scope=judgment_scope(candidate, base, sources, acquired), proof=proof)

                assessor._qualification = qualify_source
                def read_qualified_source(candidate, base, sources, acquired):
                    from .native_source_qualification import QualificationHold
                    from .native_source_qualification_replay import read_current_result
                    try:
                        return read_current_result(qualifier, candidate, base, sources, acquired,
                            scope=judgment_scope(candidate, base, sources, acquired), proof=proof)
                    except QualificationHold as exc:
                        raise NativeEvidenceHold(str(exc), sources[0].unit.source_id) from exc

                assessor._retained_qualification = read_qualified_source
                from .native_context_enrichment import NativeContextEnricher, ContextEnrichmentHold

                def enrich_source_context(original,candidate,base,sources,acquired):
                    from .native_assessor_spans import build_lossless_source_view
                    scope=judgment_scope(candidate,base,sources,acquired)
                    view=build_lossless_source_view(base.passages,base.source_ids)
                    binding=NativeAssessorJudgments._binding(candidate,base,scope,view)
                    def require_context_current():
                        stop_check()
                        _require_semantic_current_sources(runtime.authority.sources,binding,
                            proof=proof,rights_for=source_rights)
                    consumer=NativeContextEnricher(judgments=judgments,localiser=localiser,
                        objects=runtime.authority.objects,proof=proof,require_current=require_context_current)
                    try:
                        return consumer.enrich(original,candidate,base,sources,acquired,scope=scope)
                    except ContextEnrichmentHold as exc:
                        raise NativeEvidenceHold(str(exc),sources[0].unit.source_id)from exc

                assessor._context_enrichment=enrich_source_context

            from .native_graphiti_judgments import NativeGraphitiJudgments
            graph_judgments = NativeGraphitiJudgments(judgments=judgments)

            def typed_proposal_verifier(*, unit, envelope, source_revision, proposal_receipt):
                with stop_fence():
                    stop_check()
                    if rights_for_unit(unit) is None:
                        raise NativeEvidenceHold('NATIVE_CURRENT_SOURCE_RIGHTS_HOLD')
                    return graph_judgments.evaluate(proposal_receipt, source_revision,
                        envelope=envelope, unit=unit, cycle_id=envelope.cycle_id,
                        caller_identity='GRAPHITI_VERIFIER', proof=proof)
        definitions = dict(source_definition_ids)
        intake = None

        def register_current_definitions(*, fenced: bool = True) -> None:
            missing_rights = {
                source_id: rights
                for source_id in MISSING_SOURCE_IDS if source_id not in definitions
                if (rights := licence.for_source(
                    source_id=source_id, definition_url=SOURCE_URLS[source_id],
                )).decision == "PERMITTED"
            }
            if not missing_rights:
                return
            if fenced:
                stop_check()
                with stop_fence():
                    retained = register_missing_native_source_definitions(
                        sources=runtime.authority.sources, proof=proof,
                        rights_by_source=missing_rights,
                    )
            else:
                retained = register_missing_native_source_definitions(
                    sources=runtime.authority.sources, proof=proof,
                    rights_by_source=missing_rights,
                )
            if intake is not None:
                intake.bind_definitions(retained)
            definitions.update(retained)

        with stop_fence():
            register_current_definitions(fenced=False)
            projector.bootstrap()

        def source_rights(source_id, url, _at=None):
            stop_check()
            rights = licence.for_source(source_id=source_id, definition_url=url)
            if rights.decision != "PERMITTED":
                return None
            snapshot = licence.snapshot_for(source_id)
            if snapshot is None:
                return None
            observed = read_rights_observation(
                objects=runtime.authority.objects, proof=proof,
                reference=snapshot, source_id=source_id, definition_url=url,
            )
            current_evidence = licence.evidence.get(source_id)
            if current_evidence is not None:
                facts = {"observed_at": current_evidence.observed_at,
                         "reason": current_evidence.reason,
                         "observations": current_evidence.observations}
            else:
                current_licence = licence.govuk
                if current_licence is None:
                    return None
                facts = {"observed_at": current_licence.observed_at,
                         "reason": "REVIEWED_REUSE_PERMITTED", "observations": tuple(
                    (endpoint, digest, str(admission), "")
                    for endpoint, digest, admission in zip(
                        (REUSE_URL, LICENCE_URL), current_licence.raw_digests,
                        current_licence.admission_ids, strict=True,
                    )
                )}
            if digest_canonical({field: observed[field] for field in facts}) != digest_canonical(facts):
                raise ValueError("rights observation is not the current fetched snapshot")
            require_rights_assessment(
                objects=runtime.authority.objects, proof=proof,
                reference=snapshot, assessment=rights, observation=observed,
            )
            packet = {"source_id": source_id, "source_url": url,
                    "packet_digest": rights.evidence_digest,
                    "rights_decision_id": rights.record_id,
                    "assessment_admission_id": snapshot.assessment_admission_id,
                    "assessment_blob_digest": snapshot.assessment_blob_digest,
                    "observation_admission_id": snapshot.observation_admission_id,
                    "observation_blob_digest": snapshot.observation_blob_digest,
                    "policy_digest": rights.policy_digest,
                    "scope": "NATIVE_RETAINED_SOURCE_TEXT"}
            if snapshot.observation_source_id is not None:
                packet.update(observation_source_id=snapshot.observation_source_id,
                              observation_member_digest=snapshot.observation_member_digest)
            return packet

        def rights_for_unit(unit):
            current = runtime.authority.sources.current_summary(
                SourceDefinitionId.parse(unit.authority.definition_id), proof=proof,
            )
            if str(current.version_id) != unit.authority.definition_version_id:
                return None
            version = runtime.authority.sources.version_details(current.version_id, proof=proof)
            if version.request.locator != unit.source_definition_url:
                return None
            return source_rights(unit.source_id, version.request.locator)

        def require_rights(unit):
            rights = rights_for_unit(unit)
            if rights is None:
                raise NativeRetrievalHold("NATIVE_CURRENT_SOURCE_RIGHTS_HOLD")
            # Current permission is checked above on every call. Its policy
            # binds reviewed substantive terms and permitted use; raw page or
            # observation identities are provenance, not retrieval inputs.
            # Acquisition/publication retain their exact evidence bindings.
            return digest_canonical({
                "source_id": unit.source_id,
                "source_url": unit.source_definition_url,
                "definition_version_id": unit.authority.definition_version_id,
                "decision": "PERMITTED",
                "policy_digest": rights["policy_digest"],
                "scope": rights["scope"],
            })

        @contextmanager
        def retrieval_rights_cohort():
            # The native daemon owns the authority writer for its lifetime;
            # Source/terms refresh and retrieval execute synchronously in tick.
            # This snapshot ends before any retrieval port or provider effect.
            checked = {}
            active = True

            def rights(unit):
                if not active:
                    raise NativeRetrievalHold("NATIVE_RIGHTS_COHORT_EXPIRED")
                stop_check()
                key = (unit.source_id, unit.source_definition_url,
                       unit.authority.definition_id, unit.authority.definition_version_id)
                if key not in checked:
                    checked[key] = (unit, require_rights(unit))
                return checked[key][1]

            try:
                yield rights
                for unit, digest in checked.values():
                    stop_check()
                    if require_rights(unit) != digest:
                        raise NativeRetrievalHold("NATIVE_CURRENT_SOURCE_RIGHTS_HOLD")
            finally:
                # Keep only representatives during this operation; no source
                # body or permission remains cached in the continuation.
                active = False
                checked.clear()

        @contextmanager
        def source_fence(source_id, url):
            with stop_fence():
                stop_check()
                yield

        def port_for(subjects, document_inventory, rights_inventory_digest):
            receipts = tuple(item.document_receipt for item in subjects)
            retained_by_event = documents.require_authenticated_inventory(
                document_inventory, receipts,
            )
            retained = tuple(retained_by_event[item.event_id] for item in receipts)
            for missing in projector.reconcile_membership(receipts):
                documents.reproject(missing, proof=proof)
            watermark = documents.authenticated_inventory_watermark(
                document_inventory, receipts, proof=proof,
            )
            snapshot = projector.snapshot(
                generation_identity_digest=generation_digest,
                rights_manifest_digest=digest_canonical(tuple(sorted(
                    (item.passage_id, item.rights_digest) for item in retained
                ))), contiguous_ledger_seq=watermark,
                expected_document_count=len(receipts), clock=now,
            )
            view = documents.fulltext_authority_view_from_inventory(
                document_inventory, receipts, snapshot,
            )
            return NativeRetrievalPort(
                documents=documents, exact=exact,
                fulltext=FullTextRetriever(graph_reader=reader, journal=fulltext_journal,
                                          authority_view_provider=lambda _: view),
                increment4=runtime.authority.increment4, fulltext_view=view,
                subjects=subjects, document_inventory=document_inventory,
                authority_scope_id=scope,
                rights_inventory_digest=rights_inventory_digest,
                minimum_authority_watermark=watermark,
            )

        retrieval = NativeRetrievalContinuation(
            system=runtime.authority, documents=documents, journal=journal,
            connection=private, embedder=embedder, generation_id=generation_id,
            port_for=port_for, rights_check=require_rights,
            rights_cohort=retrieval_rights_cohort,
            unit_headers_for=journal.units.retrieval_headers,
        )
        govuk_acquisition = GovUkEvidenceAcquisition(
            sources=runtime.authority.sources, proof=proof,
            dispatch_fence=lambda request: source_fence(request.source_id, request.canonical_url),
            clock=clock, licence_evidence=licence,
            transport_policy_digest=TRANSPORT_POLICY,
        )
        weather_acquisition = NativeWeatherEvidenceAcquisition(
            sources=runtime.authority.sources, objects=runtime.authority.objects,
            proof=proof, rights=licence, transport_policy_digest=TRANSPORT_POLICY,
            dispatch_fence=lambda request: source_fence(request.source_id, request.canonical_url),
            retained_units=journal.units, observations=journal.observations,
            clock=clock,
        )
        spreadsheet_acquisition = GovUkSpreadsheetEvidenceAcquisition(
            sources=runtime.authority.sources,
            objects=runtime.authority.objects,
            proof=proof,
            licence=licence,
            transport_policy_digest=TRANSPORT_POLICY,
            dispatch_fence=lambda request: source_fence(
                request.source_id, request.canonical_url
            ),
            retained_units=journal.units,
            observations=journal.observations,
            clock=clock,
        )

        pdf_acquisition = GovUkPdfEvidenceAcquisition(
            sources=runtime.authority.sources, objects=runtime.authority.objects,
            proof=proof, licence=licence, transport_policy_digest=TRANSPORT_POLICY,
            dispatch_fence=lambda request: source_fence(request.source_id, request.canonical_url),
            retained_units=journal.units, observations=journal.observations, clock=clock,
        )

        def acquire(request):
            retained = journal.units.get(request.source_revision_id, ())
            spreadsheet = bool(retained) and all(
                spreadsheet_asset_url(unit) is not None
                and unit.canonical_url == request.canonical_url
                for unit in retained
            )
            pdf = bool(retained) and all(pdf_asset_url(unit) is not None
                and unit.canonical_url == request.canonical_url for unit in retained)
            transport = (
                weather_acquisition
                if request.source_id in {"HK-02", "UK-10"}
                else spreadsheet_acquisition
                if spreadsheet
                else pdf_acquisition
                if pdf
                else govuk_acquisition
            )
            return transport(request)

        evidence = NativeEvidenceController(
            objects=runtime.authority.objects, candidate_port=runtime.authority.candidate_read_port,
            evidence_packages=runtime.evidence,
            transport=EvidenceTransport(acquire), assessor=EvidenceAssessor(assessor),
            policy_bundle_digest=policies.publication.editorial_policy_bundle_digest,
            transport_policy_digest=TRANSPORT_POLICY, clock=now,
        )

        class Publication:
            copy_correction_due = staticmethod(lambda facts, decide=NativePublicationContinuation.copy_correction_due:
                decide(facts, "newsroom.native-story-writer.v1"))
            source_binding_recovery_due = staticmethod(lambda facts, decide=NativePublicationContinuation.source_binding_recovery_due:
                decide(facts, ASSESSMENT_CONTRACT_VERSION))

            def semantic_intent_revalidation_due(self, facts):
                return judgment_api_key is not None and NativePublicationContinuation.semantic_intent_revalidation_due(
                    facts, JUDGMENT_CONTRACT + ('+' + QUALIFICATION_CONTRACT if source_qualification_policy is not None else ''))

            def restore_current_output(self):
                return runtime.publication.restore_current_publisher_output(journal,proof=proof)

            def sources_for(self, revision_id):
                started = None
                try:
                    started = (perf_counter_ns(), process_time_ns())
                except Exception:
                    pass
                result, status, failure_class = None, 'COMPLETE', None
                try:
                    result = native_evidence_sources(
                        units=journal.units[revision_id], sources=runtime.authority.sources,
                        objects=runtime.authority.objects, licence=licence, proof=proof,
                        observations=journal.observations,
                    )
                    return result
                except BaseException as exc:
                    status = 'HOLD' if isinstance(exc, NativeEvidenceHold) else 'FAILED'
                    failure_class = type(exc).__name__
                    raise
                finally:
                    if started is not None:
                        try:
                            emit_diagnostic('native_source_binding_cost', {
                                'revision_id': revision_id,
                                'wall_ms': (perf_counter_ns() - started[0]) / 1_000_000,
                                'cpu_ms': (process_time_ns() - started[1]) / 1_000_000,
                                'status': status, 'failure_class': failure_class,
                                'source_count': len(result) if failure_class is None else None,
                            })
                        except Exception:
                            # Optional diagnostics never alter Source authority or failures.
                            pass

            def continuation(self, sources):
                return NativePublicationContinuation(
                    journal=journal, runtime=runtime, evidence_controller=evidence,
                    sources=sources,
                    assessment_contract_failure=(
                        assessment_usage.retained_output_contract_failure
                    ),
                    assessment_pre_dispatch_failure=(
                        assessment_usage.retained_pre_dispatch_failure
                    ),
                    assessment_old_provider_failure=assessment_usage.retained_old_provider_failure,
                    semantic_origin_failure=(assessment_usage.retained_semantic_origin_failure
                        if judgment_api_key is not None else None),
                    semantic_intent_contract=(JUDGMENT_CONTRACT + ('+' + QUALIFICATION_CONTRACT if source_qualification_policy is not None else '')
                        if judgment_api_key is not None else None),
                    context_enrichment_contract=('newsroom.native-context-package.v2'
                        if judgment_api_key is not None and source_qualification_policy is not None else None),
                    evidence_sources_for=self.sources_for,
                    assessment_contract_version=ASSESSMENT_CONTRACT_VERSION,
                    clock=now,
                )

            def recover_pre_dispatch(self, revision_ids, *, before_revision):
                return self.continuation({}).recover_pre_dispatch(
                    revision_ids,
                    failure_many=assessment_usage.retained_pre_dispatch_failure_many,
                    denial_many=lambda ids, **binding: assessment_usage.retained_pre_dispatch_allocation_denials(
                        ids, authority_path=runtime.authority.authority_store_path, **binding,
                    ),
                    before_revision=before_revision,
                )

            def advance(self, *, revision_id, candidate_version_id):
                progress = journal.summary(revision_id)
                sources = ()
                if (progress.get("stage") != "ASSESSMENT_INTERRUPTED"
                        and not self.source_binding_recovery_due(progress.get("facts", {}))):
                    try:
                        sources = self.sources_for(revision_id)
                    except NativeEvidenceHold:
                        if progress.get("stage") not in {"ACKNOWLEDGED", "COPY_CORRECTION_PREPARED"}:
                            raise
                        # The correction continuation retains the old ACK and
                        # records a current-source HOLD, without blocking peers.
                return self.continuation(
                    {revision_id: sources} if sources else {},
                ).advance(revision_id=revision_id, candidate_version_id=candidate_version_id)

            @staticmethod
            def writer_revalidation_due(facts):
                return NativePublicationContinuation.writer_revalidation_due(facts)

            def context_enrichment_due(self,facts):
                return (judgment_api_key is not None and source_qualification_policy is not None
                    and NativePublicationContinuation.context_enrichment_due(facts))

        intake = NativeSourceIntake(
            sources=runtime.authority.sources, objects=runtime.authority.objects,
            proof=proof, definition_ids=definitions, licence=licence,
            dispatch_fence=source_fence, clock=clock,
            retained_units=journal.units, observations=journal.observations,
            reobservation_epoch=rights_reobservation_epoch,
            other_source_poll=lambda **request: poll_other_source(intake, **request),
        )

        def refresh_rights() -> None:
            refresh_licence()
            register_current_definitions()

        yield NativePipeline(
            runtime=runtime, journal=journal, source_intake=intake,
            graphiti=NativeGraphitiProcessor(
                system=runtime.authority, connection=private, usage=usage, proof=proof,
                rights_for=rights_for_unit, stop_check=stop_check, dispatch_fence=stop_fence,
                operator_drain_requested=operator_drain_requested,
                typed_proposal_verifier=typed_proposal_verifier,
                clock=clock,
            ), discovery=NativeDiscovery(
                sources=runtime.authority.sources, checks=runtime.authority.checks,
                discovery=runtime.authority.discovery, proving=proving,
                rights_for=source_rights,
            ), retrieval_for=lambda _: retrieval, collision=components["collision"],
            publish=Publication(), actor_identity_digest=runtime.actor_identity_digest,
            stop_check=stop_check, stop_fence=stop_fence, clock=now,
            operator_drain_requested=operator_drain_requested,
            refresh_rights=refresh_rights,
            assessment_contract_version=ASSESSMENT_CONTRACT_VERSION,
            reassessment_quantum_seconds=reassessment_quantum_seconds,
        )

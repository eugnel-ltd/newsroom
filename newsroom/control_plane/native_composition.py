"""Actual private Hermes pipeline composition over the existing native ports."""

from __future__ import annotations

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
from .evidence import FACTUAL_LOCALISATION_POLICY_VERSION, NAMED_ENTITY_POLICY_VERSION
from .govuk_evidence import GovUkEvidenceAcquisition, POLICY_DIGEST as GOVUK_TRANSPORT_POLICY
from .govuk_spreadsheet_evidence import (
    GovUkSpreadsheetEvidenceAcquisition,
    POLICY_DIGEST as GOVUK_SPREADSHEET_TRANSPORT_POLICY,
)
from .govuk_rights import (
    LICENCE_URL, REUSE_URL, GovUkLicenceEvidence, retain_current_govuk_licence,
)
from .graphiti_operational_readiness import OPERATOR_AUTHORITY_DOMAIN, OPERATOR_PRINCIPAL_ID
from .model_usage import InvocationEfficiencyPolicy, ModelUsageService
from .native_assessor import (
    AutonomousNativeEvidenceAssessor, NativeAssessmentUsage,
    RETAINED_ASSESSMENT_POLICY_VERSION,
    VERSION as ASSESSOR_CONTRACT_VERSION,
)
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
from .native_progress import NativeRevisionJournal
from .native_publication import NativePublicationContinuation
from .native_retrieval import NativeRetrievalContinuation, compose_native_documents
from .native_runtime import open_native_runtime
from .native_source_intake import (
    NativeSourceIntake,
    native_evidence_sources,
    spreadsheet_asset_url,
)
from .native_source_rights import (
    NativePortfolioRights, observe_portfolio_terms, retain_rights_snapshot,
)
from .native_source_definitions import MISSING_SOURCE_IDS, register_missing_native_source_definitions
from .native_weather_sources import poll_other_source
from .native_weather_evidence import NativeWeatherEvidenceAcquisition, POLICY_DIGEST as WEATHER_TRANSPORT_POLICY
from .store import connect
from .zh_hant import ZH_HANT_HK_SHAPE_POLICY_VERSION

ASSESSMENT_CONTRACT_VERSION = (
    f"{ASSESSOR_CONTRACT_VERSION}+{NAMED_ENTITY_POLICY_VERSION}+"
    f"{ZH_HANT_HK_SHAPE_POLICY_VERSION}+{FACTUAL_LOCALISATION_POLICY_VERSION}+"
    f"{QUALIFICATION_RELATION_POLICY_VERSION}+{RETAINED_ASSESSMENT_POLICY_VERSION}"
)

TRANSPORT_POLICY = digest_canonical({
    "version": "hermes-native-independent-evidence-v2",
    "govuk": GOVUK_TRANSPORT_POLICY,
    "govuk_spreadsheet": GOVUK_SPREADSHEET_TRANSPORT_POLICY,
    "weather": WEATHER_TRANSPORT_POLICY,
})


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


def _deployment_identity(*, revision, tree, paths, embedding_policy, assessment_policy):
    from . import broker
    from .native_source_rights import POLICY_DIGEST as RIGHTS_POLICY
    identities = {}
    for name, path in sorted(paths.items()):
        path = Path(path)
        if path.is_symlink():
            raise ValueError("native deployment store is a symlink")
        stat = path.stat()
        identities[name] = {"path": str(path.resolve()), "device": stat.st_dev, "inode": stat.st_ino}
    return digest_canonical({
        "version": VERSION, "revision": revision, "tree": tree,
        "stores": identities, "target": "hermes-private-serving", "public_effect": False,
        "embedding_policy": embedding_policy.canonical_digest,
        "assessment_policy": assessment_policy.canonical_digest,
        "rights_policy": RIGHTS_POLICY, "transport_policy": TRANSPORT_POLICY,
        "neo4j": {"host": broker.NEO4J_BOLT_HOST, "port": broker.NEO4J_BOLT_PORT,
                  "database": broker.NEO4J_DATABASE, "principal": broker.NEO4J_PROJECTOR_USERNAME},
    })


def deployed_native_service(args):
    """Compose the installed canonical private route, after the singleton lock."""
    from . import broker, native_assessor, native_embeddings
    from .cycle import assert_no_owner_emergency_stop, owner_emergency_stop_fence
    from .model_usage import WorkloadClass
    from .native_service import NativeService
    from .native_qualification import record_qualification
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
            model=native_assessor.CONT_PRIMARY_MODEL, reasoning=native_assessor.CONT_PRIMARY_REASONING,
            config_identity=native_assessor.CONFIG_IDENTITY,
            output_schema_digest=native_assessor.SCHEMA_DIGEST,
        )
        if assessment.prompt_contract_version != native_assessor.VERSION:
            raise ValueError("native assessor prompt contract differs before authority OPEN")
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
        qualify_once=record_qualification,
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
        ))
        documents = components["documents"]
        from newsroom.increment9.proving import SOURCE_URLS
        if licence is None:
            def refresh_current_rights():
                try:
                    govuk = retain_current_govuk_licence(
                        objects=runtime.authority.objects, proof=proof,
                        dispatch_fence=stop_fence, clock=clock,
                    )
                    govuk_reason = "REVIEWED_REUSE_PERMITTED"
                except NativeEvidenceHold as exc:
                    govuk = None
                    govuk_reason = exc.reason_code
                # GOV.UK failure does not suppress an independent weather or
                # portfolio observation. VetoError still propagates from both.
                evidence = observe_portfolio_terms(
                    objects=runtime.authority.objects, proof=proof,
                    stop_check=stop_check, stop_fence=stop_fence, clock=clock,
                )
                current = NativePortfolioRights(
                    govuk, evidence, govuk_reason=govuk_reason,
                )
                snapshots = {}
                for source_id, definition_url in SOURCE_URLS.items():
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
                    snapshots[source_id] = retain_rights_snapshot(
                        objects=runtime.authority.objects, proof=proof,
                        source_id=source_id, definition_url=definition_url,
                        assessment=current.for_source(
                            source_id=source_id, definition_url=definition_url,
                        ),
                        observed_at=observed_at, reason=reason,
                        observations=observations,
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
            return {"source_id": source_id, "source_url": url,
                    "packet_digest": rights.evidence_digest,
                    "rights_decision_id": rights.record_id,
                    "assessment_admission_id": snapshot.assessment_admission_id,
                    "assessment_blob_digest": snapshot.assessment_blob_digest,
                    "observation_admission_id": snapshot.observation_admission_id,
                    "observation_blob_digest": snapshot.observation_blob_digest,
                    "policy_digest": rights.policy_digest,
                    "scope": "NATIVE_RETAINED_SOURCE_TEXT"}

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
            watermark = max(runtime.authority.events.provenance(item.event_id, proof=proof).event.ledger_seq for item in receipts)
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

        def acquire(request):
            retained = journal.units.get(request.source_revision_id, ())
            spreadsheet = bool(retained) and all(
                spreadsheet_asset_url(unit) is not None
                and unit.canonical_url == request.canonical_url
                for unit in retained
            )
            transport = (
                weather_acquisition
                if request.source_id in {"HK-02", "UK-10"}
                else spreadsheet_acquisition
                if spreadsheet
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
            copy_correction_due = staticmethod(NativePublicationContinuation.copy_correction_due)

            def advance(self, *, revision_id, candidate_version_id):
                progress = journal.progress.get(revision_id, {})
                sources = ()
                if progress.get("stage") != "ASSESSMENT_INTERRUPTED":
                    try:
                        sources = native_evidence_sources(
                            units=journal.units[revision_id],
                            sources=runtime.authority.sources,
                            objects=runtime.authority.objects,
                            licence=licence,
                            proof=proof,
                            observations=journal.observations,
                        )
                    except NativeEvidenceHold:
                        if progress.get("stage") not in {"ACKNOWLEDGED", "COPY_CORRECTION_PREPARED"}:
                            raise
                        # The correction continuation retains the old ACK and
                        # records a current-source HOLD, without blocking peers.
                return NativePublicationContinuation(
                    journal=journal, runtime=runtime, evidence_controller=evidence,
                    sources={revision_id: sources} if sources else {},
                    assessment_contract_failure=(
                        assessment_usage.retained_output_contract_failure
                    ),
                    assessment_pre_dispatch_failure=(
                        assessment_usage.retained_pre_dispatch_failure
                    ),
                    assessment_contract_version=ASSESSMENT_CONTRACT_VERSION,
                    clock=now,
                ).advance(revision_id=revision_id, candidate_version_id=candidate_version_id)

        intake = NativeSourceIntake(
            sources=runtime.authority.sources, objects=runtime.authority.objects,
            proof=proof, definition_ids=definitions, licence=licence,
            dispatch_fence=source_fence, clock=clock,
            retained_units=journal.units,
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

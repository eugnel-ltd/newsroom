"""Explicit DEV selected-row copy into a NEW native store, never a live upgrade.

Only exact CURRENT identities enter the parent closure. Superseded Discovery
predecessors are opaque in the destination contract. Other selected business
records keep their original full authority proof, including causation. External
journal/serving/usage/stop files and the immutable CAS are not rewritten here.
"""
from __future__ import annotations

from collections import defaultdict, deque
import sqlite3

from .native_current_checkpoint import discovery_current_root_ids
from .native_current_checkpoint_migrations import require_checkpoint_schema
from .persistence import AuthorityPersistenceError


def _quote(name):
    return '"'+name.replace('"','""')+'"'


def source_roots_from_current(journal):
    """Exact Source/CAS roots from retained CURRENT units and observations.

    This is the Source inventory, not a claim that publication/pending/usage
    cross-store inventories have also been qualified.
    """
    from newsroom.control_plane.native_progress import NativeRevisionJournal
    roots=defaultdict(set)
    binding_fields={'source_definitions':'definition_id','source_definition_versions':'definition_version_id',
        'source_items':'item_id','source_revisions':'revision_id','discovery_representations':'representation_id',
        'object_admissions':'admission_id','object_access_decisions':'access_decision_id'}
    for revision,units in journal.units.items():
        journal.current(revision)
        NativeRevisionJournal._validate_units(units)
        for unit in units:
            for table,field in binding_fields.items():roots[table].add((getattr(unit.authority,field),))
    for _,_,admission,access in journal.observations.values():
        roots['object_admissions'].add((admission,));roots['object_access_decisions'].add((access,))
    return {table:tuple(sorted(ids)) for table,ids in roots.items()}


def publication_event_roots(connection: sqlite3.Connection, facts):
    """Retain exact ACK and correction-intent/result event receipts, not logs."""
    names=('story_event_id','publication_event_id','delivery_attempt_event_id','delivery_evidence_event_id')
    roots=set()
    packages=set()
    def refs(value):
        if type(value) is not dict:raise AuthorityPersistenceError('CURRENT publication refs differ')
        for field in names:
            if field in value:
                if type(value[field]) is not str or not value[field]:raise AuthorityPersistenceError('CURRENT publication identity differs')
                roots.add((value[field],))
        if 'package_admission_id' in value:
            identity=value['package_admission_id']
            if type(identity) is not str or not identity:raise AuthorityPersistenceError('CURRENT package identity differs')
            packages.add((identity,))
    refs(facts)
    for field in ('copy_correction_origin','copy_correction_of','publication_predecessor','factual_correction_result'):
        if field in facts:refs(facts[field])
    if 'factual_correction_intent' in facts:
        intent=facts['factual_correction_intent']
        if type(intent) is not dict:
            raise AuthorityPersistenceError('CURRENT correction intent refs differ')
        refs(intent)
        for field in ('superseded','reviewed'):
            if field not in intent:raise AuthorityPersistenceError('CURRENT correction intent refs differ')
            refs(intent[field])
    sequences=[]
    for (event_id,) in sorted(roots):
        row=connection.execute('SELECT ledger_seq FROM ledger_events WHERE event_id=?',(event_id,)).fetchone()
        if row is None:raise AuthorityPersistenceError('CURRENT publication event is absent')
        sequences.append((row[0],))
    return {'ledger_events':tuple(sorted(sequences)),'object_admissions':tuple(sorted(packages))}


def projection_roots_from_current(connection: sqlite3.Connection):
    """Existing native head contract plus required/pending delivery reservations."""
    roots=defaultdict(set)
    for generation,lifecycle in connection.execute(
        "SELECT generation_id,lifecycle_version FROM projection_generations WHERE state!='RETIRED'"
    ):
        roots['projection_generations'].add((generation,))
        roots['projection_generation_versions'].add((generation,lifecycle))
        checkpoint=connection.execute('SELECT checkpoint_version FROM projection_checkpoint_versions WHERE generation_id=? '
            'ORDER BY checkpoint_version DESC LIMIT 1',(generation,)).fetchone()
        if checkpoint is None:raise AuthorityPersistenceError('CURRENT projection checkpoint is absent')
        roots['projection_checkpoint_versions'].add((generation,checkpoint[0]))
        for (digest,) in connection.execute('SELECT promotion_digest FROM projection_generation_promotions WHERE generation_id=?',(generation,)):
            roots['projection_generation_promotions'].add((digest,))
        for (gap,) in connection.execute("SELECT gap_id FROM projection_gaps WHERE generation_id=? AND state='OPEN'",(generation,)):
            roots['projection_gaps'].add((gap,))
        for (letter,) in connection.execute('SELECT dead_letter_id FROM projection_dead_letters WHERE generation_id=?',(generation,)):
            roots['projection_dead_letters'].add((letter,))
        for (sequence,) in connection.execute('SELECT ledger_seq FROM projection_delivery_states WHERE generation_id=? '
            'AND (required=1 OR finalized=0)',(generation,)):
            roots['projection_delivery_states'].add((generation,sequence))
    return {table:tuple(sorted(ids)) for table,ids in roots.items()}


def copy_selected_native_store(store, destination: sqlite3.Connection, *, roots, dev_rebuild=False):
    """Copy supplied verified CURRENT roots and exact parents under writer lock.

    ``roots`` maps table names to tuples of primary-key tuples. The operational
    planner must derive cross-store roots from authenticated CURRENT records;
    a hand-picked fixture is not a complete live rebuild inventory.
    No obsolete wide row is copied merely to delete it afterwards.
    """
    if not dev_rebuild or not store._current_state_only or store._lock_fd is None:
        raise AuthorityPersistenceError('selected copy requires explicit DEV writer ownership')
    require_checkpoint_schema(destination)
    if destination.in_transaction or destination.execute('SELECT 1 FROM ledger_events LIMIT 1').fetchone():
        raise AuthorityPersistenceError('selected copy requires a NEW empty destination')
    source=store._connection
    tables=tuple(row[0] for row in destination.execute(
        "SELECT name FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%'"))
    keys={table:tuple(row[1] for row in sorted(destination.execute('PRAGMA table_info('+_quote(table)+')'),key=lambda r:r[5]) if row[5]) for table in tables}
    columns={table:tuple(row[1] for row in destination.execute('PRAGMA table_info('+_quote(table)+')')) for table in tables}
    parents={}
    for table in tables:
        groups=defaultdict(list)
        for row in destination.execute('PRAGMA foreign_key_list('+_quote(table)+')'):groups[row[0]].append(row)
        parents[table]=tuple(tuple(sorted(group,key=lambda r:r[1])) for group in groups.values())
    queued=deque();seen=set();counts=defaultdict(int)

    def retain(table, identity):
        if table not in keys or not keys[table] or len(identity)!=len(keys[table]):
            raise AuthorityPersistenceError('CURRENT root primary key differs')
        token=(table,tuple(identity))
        if token not in seen:seen.add(token);queued.append(token)

    def matching(table, names, values, *, required=False):
        if any(value is None for value in values):return
        found=False
        for row in source.execute('SELECT '+','.join(_quote(k) for k in keys[table])+' FROM '+_quote(table)+
            ' WHERE '+' AND '.join(_quote(name)+'=?' for name in names),tuple(values)):
            found=True;retain(table,tuple(row))
        if required and not found:raise AuthorityPersistenceError('selected CURRENT parent is absent')

    guards=tuple(destination.execute("SELECT name,sql FROM sqlite_schema WHERE type='trigger'"))
    try:
      with store._lock,store._transaction(),store._exact_event_read_scope(source):
        destination.execute('BEGIN IMMEDIATE')
        destination.execute('PRAGMA defer_foreign_keys=ON')
        for name,_ in guards:destination.execute('DROP TRIGGER '+_quote(name))
        for table,identities in roots.items():
            for identity in identities:retain(table,identity)
        # Existing Discovery recovery roots, never every superseded decision.
        for table,identities in discovery_current_root_ids(source).items():
            for identity in identities:retain(table,(identity,))
        for table in ('discovery_gate_decision_heads','lead_disposition_heads','news_leads','discovery_signals',
                      'source_definition_version_heads','baseline_decision_heads','object_admission_idempotency','command_definitions','payload_schema_contracts',
                      'rights_policy_contracts','hydration_policy_contracts','object_admission_definitions'):
            for row in source.execute('SELECT '+','.join(_quote(k) for k in keys[table])+' FROM '+_quote(table)):
                retain(table,tuple(row))
        while queued:
            table,identity=queued.popleft()
            row=source.execute('SELECT * FROM '+_quote(table)+' WHERE '+
                ' AND '.join(_quote(k)+'=?' for k in keys[table]),identity).fetchone()
            if row is None:raise AuthorityPersistenceError('selected CURRENT parent is absent')
            values=dict(row)
            if table=='ledger_events':store._validate_retained_event(values['event_id'])
            names=tuple(name for name in columns[table] if name in values)
            retained=destination.execute('SELECT '+','.join(_quote(n) for n in names)+' FROM '+_quote(table)+' WHERE '+
                ' AND '.join(_quote(k)+'=?' for k in keys[table]),identity).fetchone()
            if retained is not None:
                if tuple(retained)!=tuple(values[name] for name in names):
                    raise AuthorityPersistenceError('NEW-store seeded contract differs from selected original')
            else:
                destination.execute('INSERT INTO '+_quote(table)+'('+','.join(_quote(n) for n in names)+') VALUES('+','.join('?' for _ in names)+')',
                    tuple(values[name] for name in names))
            counts[table]+=1
            for group in parents[table]:
                parent=group[0][2]
                matching(parent,tuple(r[4] or keys[parent][r[1]] for r in group),tuple(values.get(r[3]) for r in group),required=True)
            if table=='authority_commands':
                for child in ('ledger_events','authority_audit_events','authority_aggregate_versions','object_lifecycle_operations'):
                    matching(child,('command_id',),(values['command_id'],))
            elif table=='authority_aggregate_versions':
                matching('authority_aggregates',('aggregate_type','aggregate_id'),(values['aggregate_type'],values['aggregate_id']))
            elif table=='authority_aggregates':
                matching('authority_aggregate_versions',('aggregate_type','aggregate_id','aggregate_version'),
                    (values['aggregate_type'],values['aggregate_id'],values['current_version']))
            elif table=='authority_payloads' and values['mode']=='OBJECT_ADMISSION':
                # Admission/initial activation are typed payload proof, not SQL FKs.
                matching('object_admissions',('admission_id',),(values['object_admission_id'],),required=True)
                matching('object_admission_versions',('admission_id','lifecycle_version'),(values['object_admission_id'],1),required=True)
            elif table=='object_admissions':
                # Pending/staged admissions need not have an initial activation.
                matching('object_admission_versions',('admission_id','lifecycle_version'),(values['admission_id'],1))
                matching('object_admission_heads',('admission_id',),(values['admission_id'],))
                matching('blob_lifecycle_heads',('blob_digest',),(values['blob_digest'],))
            elif table=='source_definition_versions':
                for child in ('source_version_roles','source_version_portfolio_functions','source_version_gaps','source_version_coverage_mappings','source_version_dependencies'):
                    matching(child,('version_id',),(values['version_id'],))
            elif table=='graphiti_input_manifests':
                matching('graphiti_input_manifest_passages',('manifest_id',),(values['manifest_id'],))
            elif table=='extraction_runs':
                for child in ('extraction_run_passages','extraction_run_versions','extraction_run_heads'):
                    matching(child,('run_id',),(values['run_id'],))
            elif table=='extraction_outputs':
                matching('extraction_proposal_sets',('output_id',),(values['output_id'],))
            elif table=='extraction_proposal_sets':
                matching('extraction_proposals',('proposal_set_id',),(values['proposal_set_id'],))
            elif table=='extraction_proposals':
                matching('extraction_proposal_evidence',('proposal_id',),(values['proposal_id'],))
            elif table=='graphiti_adapter_attempts':
                matching('graphiti_adapter_attempt_replays',('attempt_id',),(values['attempt_id'],))
            elif table=='projection_generations':
                matching('projection_generation_versions',('generation_id','lifecycle_version'),
                    (values['generation_id'],values['lifecycle_version']),required=True)
                checkpoint=source.execute('SELECT checkpoint_version FROM projection_checkpoint_versions WHERE generation_id=? ORDER BY checkpoint_version DESC LIMIT 1',(values['generation_id'],)).fetchone()
                if checkpoint is None:raise AuthorityPersistenceError('selected generation checkpoint is absent')
                matching('projection_checkpoint_versions',('generation_id','checkpoint_version'),(values['generation_id'],checkpoint[0]),required=True)
            elif table=='projection_generation_validations':
                matching('projection_generation_versions',('generation_id','lifecycle_version'),
                    (values['generation_id'],values['lifecycle_version']),required=True)
                matching('projection_generation_versions',('generation_id','lifecycle_version'),
                    (values['generation_id'],values['lifecycle_version']-1),required=True)
            elif table=='projection_delivery_states':
                matching('projection_delivery_attempts',('generation_id','ledger_seq'),(values['generation_id'],values['ledger_seq']))
            elif table=='projection_gaps':
                matching('projection_gap_versions',('gap_id','lifecycle_version'),(values['gap_id'],values['lifecycle_version']))
            elif table=='canonical_entities':
                matching('entity_aliases',('entity_id',),(values['entity_id'],))
                matching('entity_preferred_identities',('entity_id',),(values['entity_id'],))
                matching('entity_mention_resolutions',('entity_id',),(values['entity_id'],))
            elif table=='editorial_relation_proposal_versions':
                for child in ('editorial_relation_evidence_items','editorial_relation_extraction_evidence','editorial_relation_workflow_evidence','editorial_relation_resolution_dependencies'):
                    matching(child,('proposal_version_id',),(values['proposal_version_id'],))
            elif table=='ledger_events':
                if values.get('native_checkpoint_id') is not None:
                    matching('native_current_checkpoint_members',('event_id',),(values['event_id'],))
                elif values['retired_header_digest'] is None:
                    # Preserve existing selected proof, not arbitrary historical
                    # tokens embedded in JSON. Causation is a typed header field.
                    if values['causation_kind']=='COMMAND':matching('authority_commands',('command_id',),(values['causation_identifier'],),required=True)
                    elif values['causation_kind']=='EVENT':matching('ledger_events',('event_id',),(values['causation_identifier'],),required=True)
        # Join a small TEMP selection to the source index, then batch only
        # reservation headers. No per-old-row destination query or wide RPC copy.
        source.execute('CREATE TEMP TABLE _native_rebuild_selected_events(event_id TEXT PRIMARY KEY) WITHOUT ROWID')
        source.executemany('INSERT INTO _native_rebuild_selected_events VALUES(?)',destination.execute('SELECT event_id FROM ledger_events'))
        reservations=source.execute('''SELECT coalesce(e.retired_namespace,c.idempotency_namespace),
            coalesce(e.retired_key,c.idempotency_key),e.command_id,e.event_id,e.ledger_seq,e.security_scope,e.trust_scope,e.aggregate_type,e.aggregate_id
            FROM ledger_events e LEFT JOIN authority_commands c ON c.command_id=e.command_id
            WHERE NOT EXISTS(SELECT 1 FROM temp._native_rebuild_selected_events s WHERE s.event_id=e.event_id)''')
        while batch:=reservations.fetchmany(256):
            if any(row[0] is None or row[1] is None for row in batch):raise AuthorityPersistenceError('old key reservation is absent')
            destination.executemany('INSERT INTO native_expired_command_keys VALUES(?,?,?,?,?,?,?,?,?)',batch)
            counts['native_expired_command_keys']+=len(batch)
        source.execute('DROP TABLE temp._native_rebuild_selected_events')
        if store._native_checkpoint_schema:
            for row in source.execute('SELECT * FROM native_expired_command_keys'):
                destination.execute('INSERT INTO native_expired_command_keys VALUES(?,?,?,?,?,?,?,?,?)',tuple(row))
                counts['native_expired_command_keys']+=1
        seq=source.execute("SELECT seq FROM sqlite_sequence WHERE name='ledger_events'").fetchone()
        if seq is not None:
            destination.execute("DELETE FROM sqlite_sequence WHERE name='ledger_events'")
            destination.execute("INSERT INTO sqlite_sequence VALUES('ledger_events',?)",(seq[0],))
        if destination.execute('PRAGMA foreign_key_check').fetchone():raise AuthorityPersistenceError('selected CURRENT FK closure differs')
        for _,sql in guards:destination.execute(sql)
        require_checkpoint_schema(destination)
        destination.commit()
    except BaseException:
        if destination.in_transaction:destination.rollback()
        if source.in_transaction:source.rollback()
        raise
    finally:
        source.execute('DROP TABLE IF EXISTS temp._native_rebuild_selected_events')
    return dict(counts)

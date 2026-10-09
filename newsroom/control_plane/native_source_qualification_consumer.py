"""Application-only post-consumer of an authenticated frozen SourceQA proposal.

The qualified producer module/bytes, original paid receipts and purposes remain
unchanged. Witness verification and localisation use separate existing phases.
"""
from __future__ import annotations
import json
from newsroom.authority import HydrationRequest, ObjectAdmissionId, ObjectAdmissionRequest
from newsroom.authority.canonical import canonical_json_bytes, digest_canonical
from .native_assessor import NativeAssessmentExecution
from .native_assessor_spans import build_lossless_source_view
from .native_source_qualification import QualificationHold, QualificationReference, _materialisation

CONSUMER_VERSION = 'newsroom.source-qualification-consumer.v4'
REPLAY_CONSUMER_VERSION = 'newsroom.source-qualification-replay.v1'
# Pure consumer repairs never reopen an identical paid witness/render purpose.
PAID_BINDING_VERSION = 'newsroom.source-qualification-consumer.v1'
TYPED_CONSUMER_VERSION = 'newsroom.source-qualification-typed-rendering-consumer.v1'
RESOLUTION_CONSUMER_VERSION = 'newsroom.source-qualification-resolution-consumer.v1'
CORRECTED_RESOLUTION_CONSUMER_VERSION = 'newsroom.source-qualification-resolution-consumer.v2'
RENDERING_REPAIR_CONSUMER_VERSION = 'newsroom.source-qualification-rendering-repair.v1'


def _resolution_state(qualifier, candidate, base, parent, references, *, proof,
                      contract='newsroom.qualification-semantic-resolution.v1'):
    """Reconstruct one complete, settled disagreement from original authorities."""
    from .evidence import (QualificationEvidence, Evid012QualificationTest,
        semantic_witness_reference, SEMANTIC_WITNESS_CONTRACT, SEMANTIC_RESOLUTION_CONTRACT)
    from .native_assessor_judgments import _semantic_witness_inputs
    from .native_source_qualification_replay import original_qualification_state
    from .typesafe_judgment import JudgmentReference
    from .admission import _qualification_relation_is_proven
    from types import SimpleNamespace
    raw = qualifier.objects.rehydrate(HydrationRequest(
        ObjectAdmissionId.parse(parent['receipt_admission_id']), 'evidence.record'), proof=proof).data
    receipt = json.loads(raw)
    if canonical_json_bytes(receipt) != raw or receipt.get('invocation_id') != parent['invocation_id']:
        raise QualificationHold('QUALIFICATION_RESOLUTION_PARENT_HOLD')
    binding = receipt['source_binding']
    original = original_qualification_state(qualifier, candidate, base, binding, proof=proof)
    checked = qualifier.read_qualification(QualificationReference(parent['invocation_id'],
        ObjectAdmissionId.parse(parent['raw_admission_id']), ObjectAdmissionId.parse(parent['receipt_admission_id'])),
        original, proof=proof, candidate_id=candidate.candidate_id,
        hypothesis_digest=candidate.governing_manifest.canonical_digest, evidence_package_digest=base.digest)
    package = json.loads(checked['materialisation']['materialised_text'])['package']
    if not package['substantive_new_information']:
        raise QualificationHold('QUALIFICATION_RESOLUTION_PARENT_NOT_AFFIRMATIVE')
    claims = {row['claim_id']: SimpleNamespace(**{**row, 'source_ids': tuple(row['source_ids'])})
              for row in package['governed_claims']}
    required = {}
    for row in package['qualification_evidence']:
        claim = claims[row['governed_claim_id']]
        q = QualificationEvidence(Evid012QualificationTest(row['test']), claim.claim_id,
            'resolution-original-qualification', tuple(row['test_evidence'].items()), row['policy_version'])
        if not _qualification_relation_is_proven(q, claim, source_context=base.passages[claim.passage_index]):
            required[(claim.claim_id, q.test.value)] = q
    keys = tuple((row['claim_id'], row['test']) for row in references)
    if (not 0 < len(keys) <= 32 or keys != tuple(sorted(required))):
        raise QualificationHold('QUALIFICATION_RESOLUTION_WITNESS_SET_HOLD')
    answers = []
    dissent = False
    for row in references:
        if set(row) != {'claim_id', 'test', 'reference'}:
            raise QualificationHold('QUALIFICATION_RESOLUTION_REFERENCE_HOLD')
        ref = semantic_witness_reference(tuple(sorted(row['reference'].items())))
        if ref['contract'] != SEMANTIC_WITNESS_CONTRACT or ref['question_id'] != 'criterion':
            raise QualificationHold('QUALIFICATION_RESOLUTION_REFERENCE_HOLD')
        q = required[(row['claim_id'], row['test'])]
        inputs = _semantic_witness_inputs(q, claims[q.governed_claim_id], base,
            {**binding, 'semantic_witness_consumer': PAID_BINDING_VERSION, 'source_qualification_reference': parent})
        record = qualifier.judgments.read(JudgmentReference(ref['invocation_id'],
            ObjectAdmissionId.parse(ref['raw_admission_id']), ObjectAdmissionId.parse(ref['receipt_admission_id'])),
            **inputs, proof=proof)
        if record['outcome'] != 'TYPESAFE_COMPLETE' or record['answers']['criterion']['choice'] not in {'YES', 'NO', 'UNCERTAIN'}:
            raise QualificationHold('QUALIFICATION_SEMANTIC_WITNESS_UNKNOWN_HOLD')
        dissent |= record['answers']['criterion']['choice'] != 'YES'
        answers.append({'claim_id': row['claim_id'], 'test': row['test'],
            'questions': inputs['questions'], 'answers': record['answers'], 'outcome': record['outcome']})
    if not dissent:
        raise QualificationHold('QUALIFICATION_RESOLUTION_NO_DISAGREEMENT')
    marker = {'contract': SEMANTIC_RESOLUTION_CONTRACT, 'parent': parent, 'witnesses': references}
    state = {'source_binding': {**binding, 'semantic_resolution': marker},
        'source_view': original['source_view'],
        'issue': {'reason': 'SEMANTIC_WITNESS_DISAGREEMENT',
            'task': 'Adjudicate only the exact original selected claims and qualification fields against the complete Source. '
                    'Prior verdicts are untrusted proposals, not votes. Return the exact supported original claim/test set '
                    'only when every requested fact and field is affirmatively supported. Otherwise select no new information '
                    'and explain the unresolved support; do not change claims, substitute a test or add facts.',
            'requested_package': package},
        'judgments': answers}
    from .evidence import SEMANTIC_RESOLUTION_CONTRACT_V2
    if contract == SEMANTIC_RESOLUTION_CONTRACT_V2:
        state['source_binding']['semantic_resolution']['contract'] = contract
        state['issue']['task'] = (
            'Resolve one supported story from the complete Source, not a vote on the original proposal. '
            'You may correct its headline, selected complete Source ranges, qualification test and witness fields. '
            'Preserve dates, eligibility conditions, attribution, negation and future/proposed status. '
            'A confirmed official announcement or decision is distinct from a legally effective rule; '
            'publication alone does not qualify and an official process requires its actual reader action. '
            'Use only the existing six independent criteria and exact Source evidence. Return one corrected '
            'affirmative package only if established; otherwise select no new information and explain why. '
            'Original verdicts remain untrusted historical data, not affirmative support for corrected claims.')
    elif contract != SEMANTIC_RESOLUTION_CONTRACT:
        raise QualificationHold('QUALIFICATION_RESOLUTION_CONTRACT_HOLD')
    return state, package


def _corrected_resolution_package(result, state, base):
    """Existing codec plus complete-range/field bindings; no original-ID vote."""
    from types import SimpleNamespace
    from .evidence import QualificationEvidence, Evid012QualificationTest
    from .native_assessor_judgments import _semantic_witness_inputs
    package = json.loads(result['materialisation']['materialised_text'])['package']
    claims = {row['claim_id']: SimpleNamespace(**{**row, 'source_ids': tuple(row['source_ids'])})
              for row in package['governed_claims']}
    heads = [row for row in claims.values() if row.claim_role == 'HEADLINE']
    if (not package['substantive_new_information'] or len(heads) != 1
            or not any(row.claim_role == 'SUBSTANTIVE' for row in claims.values())
            or any(row.status != 'CONFIRMED_FACT' for row in claims.values())
            or not any(row['governed_claim_id'] == heads[0].claim_id for row in package['qualification_evidence'])):
        raise QualificationHold('QUALIFICATION_RESOLUTION_NOT_AFFIRMATIVE_HOLD')
    for row in package['qualification_evidence']:
        q = QualificationEvidence(Evid012QualificationTest(row['test']), row['governed_claim_id'],
            'corrected-resolution-qualification', tuple(row['test_evidence'].items()), row['policy_version'])
        _semantic_witness_inputs(q, claims[q.governed_claim_id], base, state['source_binding'])
    return package


def _require_resolution_result(result, original):
    package = json.loads(result['materialisation']['materialised_text'])['package']
    fields = ('claim', 'supporting_excerpt', 'claim_role', 'status', 'source_ids', 'passage_index', 'quotations')
    def facts(value):
        return {canonical_json_bytes({key: row[key] for key in fields}): row['claim_id'] for row in value['governed_claims']}
    expected, actual = facts(original), facts(package)
    # The frozen codec derives IDs from its input binding. Compare exact Source
    # facts, not new purpose IDs; the original package is never rewritten.
    mapping = {actual[key]: expected[key] for key in actual.keys() & expected.keys()}
    def qualifications(value, ids):
        return sorted(({**{key: row[key] for key in ('test', 'test_evidence', 'policy_version')},
            'governed_claim_id': ids.get(row['governed_claim_id'])} for row in value['qualification_evidence']),
            key=lambda row: (str(row['governed_claim_id']), row['test']))
    if (not package['substantive_new_information'] or expected.keys() != actual.keys()
            or len(actual) != len(package['governed_claims']) or len(expected) != len(original['governed_claims'])
            or qualifications(package, mapping) != qualifications(original, {cid: cid for cid in expected.values()})):
        raise QualificationHold('QUALIFICATION_RESOLUTION_NOT_AFFIRMATIVE_HOLD')


def current_source_passage(source):
    """Reproduce the paid text recipe only from an already verified CURRENT Source."""
    from .govuk_evidence import _api_url
    unit=source.unit
    if unit.source_id in {'HK-02','UK-10'}:
        raise ValueError('CURRENT weather paid projection is unsupported')
    # GOV.UK Content API, PDF and spreadsheet acquisitions share this recipe.
    _api_url(unit.canonical_url)
    return unit.headline+'\n\n'+unit.body


class NativeQualifiedSourceConsumer:
    def __init__(self, qualifier, *, semantic_witnesses, localise, read_localisation, resolve_disagreements=False,
                 resolution_contract='newsroom.qualification-semantic-resolution.v1'):
        if type(resolve_disagreements) is not bool:
            raise ValueError('semantic resolution opt-in differs')
        self.qualifier, self.objects = qualifier, qualifier.objects
        self.semantic_witnesses = semantic_witnesses
        self.localise, self.read_localisation = localise, read_localisation
        self.resolve_disagreements = resolve_disagreements
        from .evidence import SEMANTIC_RESOLUTION_CONTRACT, SEMANTIC_RESOLUTION_CONTRACT_V2
        if resolution_contract not in {SEMANTIC_RESOLUTION_CONTRACT, SEMANTIC_RESOLUTION_CONTRACT_V2}:
            raise ValueError('semantic resolution contract differs')
        self.resolution_contract = resolution_contract
        if resolve_disagreements:
            semantic_witnesses.resolution_reader = self.read_resolution

    def read_resolution(self, qualification, claim, package):
        """Admission authenticates a resolved original dissent; never allocates."""
        from .evidence import semantic_witness_reference
        from newsroom.increment10.evidence import _base_package
        value = semantic_witness_reference(qualification.semantic_witness_ref)
        from .evidence import SEMANTIC_RESOLUTION_CONTRACT_V2
        corrected = value['contract'] == SEMANTIC_RESOLUTION_CONTRACT_V2
        raw = self.objects.rehydrate(HydrationRequest(ObjectAdmissionId.parse(
            value['receipt_admission_id' if corrected else 'resolution_receipt_admission_id']), 'evidence.record'), proof=self.semantic_witnesses.proof).data
        receipt = json.loads(raw)
        if not self.resolve_disagreements or canonical_json_bytes(receipt) != raw:
            raise QualificationHold('QUALIFICATION_RESOLUTION_RECEIPT_HOLD')
        binding = receipt['source_binding']
        candidate = self.semantic_witnesses.candidate_for(binding['candidate_version_id'])
        base = _base_package(package)
        marker = binding['semantic_resolution']
        state, original = _resolution_state(self.qualifier, candidate, base, marker['parent'], marker['witnesses'],
            proof=self.semantic_witnesses.proof, contract=marker['contract'])
        prefix = '' if corrected else 'resolution_'
        ref = QualificationReference(value[prefix+'invocation_id'], ObjectAdmissionId.parse(value[prefix+'raw_admission_id']),
            ObjectAdmissionId.parse(value[prefix+'receipt_admission_id']))
        checked = self.qualifier.read_qualification(ref, state, proof=self.semantic_witnesses.proof,
            candidate_id=candidate.candidate_id, hypothesis_digest=candidate.governing_manifest.canonical_digest,
            evidence_package_digest=base.digest)
        if corrected:
            if marker['contract'] != value['contract'] or value['question_id'] != 'criterion':
                raise QualificationHold('QUALIFICATION_RESOLUTION_CONTRACT_HOLD')
            output = _corrected_resolution_package(checked, state, base)
            current_claims = {row.claim_id: row for row in package.governed_claims}
            for source_row in output['governed_claims']:
                current = current_claims.get(source_row['claim_id'])
                if (current is None or any(source_row[key] != getattr(current, key) for key in
                        ('claim', 'supporting_excerpt', 'claim_role', 'passage_index'))
                        or source_row['source_ids'] != list(current.source_ids) or source_row['status'] != str(current.status)):
                    raise QualificationHold('QUALIFICATION_RESOLUTION_PACKAGE_HOLD')
                if (source_row['rendered_assertion_zh_hant_hk'] != current.rendered_assertion_zh_hant_hk
                        or source_row['quotations'] != list(current.quotations)
                        or tuple(map(tuple, source_row['localised_factual_expressions'])) != current.localised_factual_expressions):
                    from .evidence import source_rendering_reference, SOURCE_RENDERING_CONTRACT_V2
                    presentation = source_rendering_reference(current.source_rendering_ref)
                    if (presentation['contract'] != SOURCE_RENDERING_CONTRACT_V2
                            or self.semantic_witnesses._read_typed_source_rendering(presentation, current, package) is not True):
                        raise QualificationHold('QUALIFICATION_RESOLUTION_PACKAGE_HOLD')
            expected = next((row for row in output['qualification_evidence'] if row['governed_claim_id'] == claim.claim_id
                             and row['test'] == qualification.test.value), None)
            source_claim = next((row for row in output['governed_claims'] if row['claim_id'] == claim.claim_id), None)
            if (expected is None or source_claim is None or expected['test_evidence'] != dict(qualification.test_evidence)
                    or any(source_claim[key] != getattr(claim, key) for key in ('claim', 'supporting_excerpt', 'claim_role', 'passage_index'))
                    or source_claim['source_ids'] != list(claim.source_ids) or source_claim['status'] != str(claim.status)):
                raise QualificationHold('QUALIFICATION_RESOLUTION_CLAIM_HOLD')
            return True
        _require_resolution_result(checked, original)
        from .evidence import SEMANTIC_WITNESS_CONTRACT
        original_ref = {key: value[key] for key in ('invocation_id', 'raw_admission_id', 'receipt_admission_id', 'question_id')}
        original_ref['contract'] = SEMANTIC_WITNESS_CONTRACT
        selected = next((row for row in marker['witnesses'] if row['claim_id'] == claim.claim_id
                         and row['test'] == qualification.test.value), None)
        expected = next((row for row in original['qualification_evidence'] if row['governed_claim_id'] == claim.claim_id
                         and row['test'] == qualification.test.value), None)
        source_claim = next((row for row in original['governed_claims'] if row['claim_id'] == claim.claim_id), None)
        if (selected is None or selected['reference'] != original_ref or expected is None or source_claim is None
                or expected['test_evidence'] != dict(qualification.test_evidence)
                or any(source_claim[key] != getattr(claim, key) for key in ('claim', 'supporting_excerpt', 'claim_role', 'passage_index'))
                or source_claim['source_ids'] != list(claim.source_ids) or source_claim['status'] != str(claim.status)):
            raise QualificationHold('QUALIFICATION_RESOLUTION_CLAIM_HOLD')
        return True

    def read_semantic_parent(self, binding, candidate, base, *, proof):
        """The original SourceQA is re-read, never evaluated during admission."""
        from .native_source_qualification_replay import original_qualification_state
        parent = binding['source_qualification_reference']
        original_binding = {key:value for key,value in binding.items()
            if key not in {'semantic_witness_consumer', 'source_qualification_reference'}}
        if 'semantic_resolution' in original_binding:
            marker = original_binding['semantic_resolution']
            state, _ = _resolution_state(self.qualifier, candidate, base, marker['parent'], marker['witnesses'],
                proof=proof, contract=marker['contract'])
        else:
            state = original_qualification_state(self.qualifier, candidate, base, original_binding, proof=proof)
        reference = QualificationReference(parent['invocation_id'], ObjectAdmissionId.parse(parent['raw_admission_id']),
            ObjectAdmissionId.parse(parent['receipt_admission_id']))
        return self.qualifier.read_qualification(reference, state, proof=proof, candidate_id=candidate.candidate_id,
            hypothesis_digest=candidate.governing_manifest.canonical_digest, evidence_package_digest=base.digest)['materialisation']

    def read_current_disposition(self, candidate, base, sources, *, proof, source_passages=None):
        """Proof-only retained CURRENT projection, not external-now acquisition."""
        from types import SimpleNamespace
        from .native_source_qualification_replay import original_qualification_state, original_qualification_reference
        from .evidence import QualificationEvidence,Evid012QualificationTest
        from .admission import _qualification_relation_is_proven
        located = original_qualification_reference(self.qualifier, candidate, base, proof=proof, optional=True)
        if located is None:
            return None
        reference, receipt = located
        binding=receipt['source_binding']
        passages=tuple(source.unit.body for source in sources)if source_passages is None else tuple(current_source_passage(source)for source in sources)
        if source_passages is not None and source_passages!=passages:return None
        if type(passages)is not tuple or len(passages)!=len(sources) or passages!=base.passages:return None
        current=binding.get('current_scope',{}).get('sources',[])
        if (len(current)!=len(sources) or any(row['source_id']!=source.unit.source_id or row['body']!=passage
                or row['published_at']!=source.unit.published_at or row['updated_at']!=source.unit.updated_at
                for row,source,passage in zip(current,sources,passages,strict=True))):return None
        pins=binding.get('source_currentness',[])
        if len(pins)!=len(sources) or any(row['definition_id']!=str(source.unit.authority.definition_id)
                or row['definition_version_id']!=str(source.unit.authority.definition_version_id)
                for row,source in zip(pins,sources,strict=True)):return None
        for row in binding.get('first_publication',[]):
            source=next((item for item in sources if item.unit.source_id==row['source_id']),None)
            if source is None or row['source_revision_digest']!=source.unit.revision_digest:return None
        state=original_qualification_state(self.qualifier,candidate,base,binding,proof=proof)
        checked=self.qualifier.read_qualification(reference,state,proof=proof,candidate_id=candidate.candidate_id,
            hypothesis_digest=candidate.governing_manifest.canonical_digest,evidence_package_digest=base.digest)
        package=json.loads(checked['materialisation']['materialised_text'])['package']
        if not package['substantive_new_information']:return None
        claims={row['claim_id']:SimpleNamespace(**{**row,'source_ids':tuple(row['source_ids'])})for row in package['governed_claims']}
        for item in package['qualification_evidence']:
            claim=claims[item['governed_claim_id']]
            q=QualificationEvidence(Evid012QualificationTest(item['test']),claim.claim_id,'retained-qualification',tuple(item['test_evidence'].items()),item['policy_version'])
            if not _qualification_relation_is_proven(q,claim,source_context=base.passages[claim.passage_index]):
                from .native_evidence import NativeEvidenceHold
                try:
                    self.semantic_witnesses.read_existing(q,claim,base,{**binding,'semantic_witness_consumer':PAID_BINDING_VERSION,
                        'source_qualification_reference':{'invocation_id':reference.invocation_id,'raw_admission_id':str(reference.raw_admission_id),
                            'receipt_admission_id':str(reference.receipt_admission_id)}})
                except NativeEvidenceHold as error:
                    if not self.resolve_disagreements or error.reason_code not in {
                            'QUALIFICATION_SEMANTIC_WITNESS_NO', 'QUALIFICATION_SEMANTIC_WITNESS_UNCERTAIN'}:
                        raise
        return None

    def compose_selected(self, original, candidate, base, sources, acquired, *, proof):
        """A new witness/render purpose, never qualification-model redispatch."""
        from types import SimpleNamespace
        from .native_assessor import _qualification_record_id, _reference_binding, _document
        from .native_assessor_judgments import (JudgedAssessment, SemanticWitnessMetadata, SourceRenderingMetadata,
            source_rendering_details, source_rendering_names, source_rendering_projection, original_rendering_slots)
        from .evidence import (QualificationEvidence, Evid012QualificationTest, bounded_named_entities,
            _canonical_localised_fact, SOURCE_RENDERING_CONTRACT, SOURCE_RENDERING_CONTRACT_V2,
            _localised_fact_is_bound)
        from .admission import _qualification_relation_is_proven
        import re
        decision = json.loads(original.decision_record)
        document = _document(original.execution.text)
        package = document['package']
        if not package.get('substantive_new_information'):
            return original
        if self.semantic_witnesses is None:
            return original
        binding = decision['source_binding']
        view = build_lossless_source_view(base.passages, base.source_ids)
        if _reference_binding(view) != binding['source_reference_binding'] or binding['content_digest'] != base.digest:
            raise QualificationHold('QUALIFICATION_CURRENT_SNAPSHOT_HOLD')
        from .evidence import SEMANTIC_RESOLUTION_CONTRACT_V2
        active_contract = self.resolution_contract
        if self.resolve_disagreements and active_contract == SEMANTIC_RESOLUTION_CONTRACT_V2:
            from .native_source_qualification_replay import original_qualification_reference
            parent_ref, _, prior_resolution = original_qualification_reference(
                self.qualifier, candidate, base, proof=proof, include_resolution=True)
            if decision['qualification_reference'] != {'invocation_id': parent_ref.invocation_id,
                    'raw_admission_id': str(parent_ref.raw_admission_id), 'receipt_admission_id': str(parent_ref.receipt_admission_id)}:
                raise QualificationHold('QUALIFICATION_RESOLUTION_PARENT_HOLD')
            if prior_resolution is not None:
                active_contract = prior_resolution[1]['source_binding']['semantic_resolution']['contract']
        claims = {row['claim_id']: SimpleNamespace(**{**row,'source_ids':tuple(row['source_ids'])}) for row in package['governed_claims']}
        witnesses = []
        dissent = False
        if not package['qualification_evidence'] or not any(row['governed_claim_id'] == next(
                c['claim_id']for c in package['governed_claims']if c['claim_role']=='HEADLINE')for row in package['qualification_evidence']):
            raise QualificationHold('QUALIFICATION_HEADLINE_UNPROVEN')
        for item in package['qualification_evidence']:
            claim = claims[item['governed_claim_id']]
            fields = tuple(item['test_evidence'].items())
            q = QualificationEvidence(test=Evid012QualificationTest(item['test']), governed_claim_id=claim.claim_id,
                qualification_record_id=_qualification_record_id(claim.claim_id, item['test'], [list(p)for p in fields]),
                test_evidence=fields, policy_version=item['policy_version'])
            if not _qualification_relation_is_proven(q, claim, source_context=base.passages[claim.passage_index]):
                verified_binding = {**binding, 'semantic_witness_consumer': PAID_BINDING_VERSION,
                    'source_qualification_reference': decision['qualification_reference']}
                if self.resolve_disagreements:
                    ref, choice = self.semantic_witnesses.collect(q, claim, base, verified_binding)
                    dissent |= choice != 'YES'
                else:
                    ref = self.semantic_witnesses.evaluate(q, claim, base, verified_binding)
                witnesses.append(((claim.claim_id, q.test.value), ref))
        if dissent:
            references = [{'claim_id': key[0], 'test': key[1], 'reference': dict(ref)}
                          for key, ref in sorted(witnesses)]
            state, expected = _resolution_state(self.qualifier, candidate, base, decision['qualification_reference'],
                references, proof=proof, contract=active_contract)
            identities = dict(candidate_id=candidate.candidate_id, hypothesis_digest=candidate.governing_manifest.canonical_digest,
                evidence_package_digest=base.digest)
            ref = self.qualifier.qualify(state, proof=proof, **identities)
            checked = self.qualifier.read_qualification(ref, state, proof=proof, **identities)
            from .evidence import SEMANTIC_RESOLUTION_CONTRACT
            if active_contract == SEMANTIC_RESOLUTION_CONTRACT_V2:
                package = _corrected_resolution_package(checked, state, base)
                decision = {**decision, 'original_qualification_reference': decision['qualification_reference'],
                    'qualification_reference': {'invocation_id': ref.invocation_id, 'raw_admission_id': str(ref.raw_admission_id),
                        'receipt_admission_id': str(ref.receipt_admission_id)}, 'source_binding': state['source_binding'],
                    'materialisation_receipt': checked['materialisation']}
                binding = decision['source_binding']
                claims = {row['claim_id']: SimpleNamespace(**{**row, 'source_ids': tuple(row['source_ids'])})
                          for row in package['governed_claims']}
                reference = tuple(sorted({'contract': SEMANTIC_RESOLUTION_CONTRACT_V2, 'question_id': 'criterion',
                    **decision['qualification_reference']}.items()))
                witnesses = [((row['governed_claim_id'], row['test']), reference) for row in package['qualification_evidence']]
            else:
                _require_resolution_result(checked, expected)
                witnesses = [(key, tuple(sorted({**dict(original), 'contract': SEMANTIC_RESOLUTION_CONTRACT,
                    'resolution_invocation_id': ref.invocation_id, 'resolution_raw_admission_id': str(ref.raw_admission_id),
                    'resolution_receipt_admission_id': str(ref.receipt_admission_id)}.items()))) for key, original in witnesses]
        # A verifier does not certify actor identity, unsupported factual types,
        # or rendering. Deny impossible current capabilities before any render fee.
        derived = {}
        rendering_sources = []
        typed_capable = getattr(self.localise, 'rendering_contract', None) == SOURCE_RENDERING_CONTRACT_V2
        needs_typed = False
        for claim in claims.values():
            body = base.passages[claim.passage_index]
            try:
                terms, year = source_rendering_details(claim,body,binding['current_scope']['sources'][claim.passage_index])
            except ValueError as exc:
                raise QualificationHold('QUALIFICATION_SOURCE_RENDERING_UNPROVEN') from exc
            names = source_rendering_names(claim,body)
            typed_names = source_rendering_names(claim, body, contract=SOURCE_RENDERING_CONTRACT_V2) if typed_capable else names
            recognised = {name for name,_kind in typed_names}
            if typed_capable:
                from .writer import _CHINESE_NUMERAL_FACT, _remove_exact_expressions
                pairs = tuple(map(tuple, claim.localised_factual_expressions))
                source_residue = _remove_exact_expressions(claim.claim, tuple(name for name, _kind in names) + tuple(p[0] for p in pairs))
                target_residue = _remove_exact_expressions(claim.rendered_assertion_zh_hant_hk,
                    tuple(name for name, _kind in names) + tuple(p[1] for p in pairs))
                needs_typed |= (typed_names != names
                    or _CHINESE_NUMERAL_FACT.findall(source_residue) != _CHINESE_NUMERAL_FACT.findall(target_residue)
                    or any(not _localised_fact_is_bound(source, target, claim.claim, claim.supporting_excerpt,
                        claim.rendered_assertion_zh_hant_hk) for source, target in pairs))
            if any(token not in recognised for token in re.findall(r'\b[A-Z]{2,}\b', claim.claim)):
                raise QualificationHold('QUALIFICATION_TYPED_ACTOR_UNSUPPORTED')
            for money in re.findall(r'£[0-9][0-9,.]*(?:\s+(?:million|billion|thousand))?',claim.claim,re.I):
                if _canonical_localised_fact(money) is None:
                    raise QualificationHold('QUALIFICATION_TYPED_VALUE_UNSUPPORTED')
            derived[claim.claim_id] = terms, year, names
            if names != bounded_named_entities(claim.claim,source_context=body) or year is not None:
                ref = tuple(sorted({'contract':SOURCE_RENDERING_CONTRACT,'operation':'SOURCE_RENDERING',
                    **decision['qualification_reference']}.items()))
                rendering_sources.append((claim.claim_id,ref))
        # A new immutable rendering view leaves the paid SourceQA view untouched.
        from .admission import _valid_zh_hant_hk_rendering
        from .evidence import rendered_named_entities
        valid_rendering = all(_valid_zh_hant_hk_rendering(SimpleNamespace(**vars(claim),
            named_entities=tuple(name for name,_kind in derived[claim.claim_id][2])))
            and rendered_named_entities(claim.rendered_assertion_zh_hant_hk,
                frozenset(derived[claim.claim_id][2])) == set(derived[claim.claim_id][2])
            and (derived[claim.claim_id][1] is None or derived[claim.claim_id][1][:2] in
                 tuple(map(tuple,claim.localised_factual_expressions)))for claim in claims.values())
        valid_rendering = valid_rendering and not needs_typed
        if dissent and active_contract == SEMANTIC_RESOLUTION_CONTRACT_V2 and not valid_rendering and not typed_capable:
            raise QualificationHold('QUALIFICATION_RESOLUTION_RENDERING_HOLD')
        if not witnesses and not rendering_sources and valid_rendering:
            return original
        rendering_ref = None
        consumer_version = RESOLUTION_CONSUMER_VERSION if self.resolve_disagreements else CONSUMER_VERSION
        if self.resolve_disagreements and self.resolution_contract == SEMANTIC_RESOLUTION_CONTRACT_V2:
            consumer_version = CORRECTED_RESOLUTION_CONSUMER_VERSION
        if not valid_rendering:
            if self.localise is None or self.read_localisation is None:
                raise QualificationHold('QUALIFICATION_RENDERING_UNAVAILABLE')
            raw_wire = json.loads(self.objects.rehydrate(HydrationRequest(
                ObjectAdmissionId.parse(decision['qualification_reference']['raw_admission_id']), 'evidence.record'), proof=proof).data)
            # SourceQA original ranges, roles and classifier fields stay fixed.
            selected = {}
            for index, (wire_claim, claim) in enumerate(zip(raw_wire['package']['governed_claims'], claims.values(), strict=True)):
                terms, year, source_names = derived[claim.claim_id]
                kinds = dict(source_names)
                names = [(name,kinds[name])for name,_kind,_start,_end in terms]
                selected[str(index)] = {'source_id':claim.source_ids[0], 'text':claim.claim,
                    'source_range':wire_claim['source_range'], 'entities':[list(p)for p in names], 'rendering_fragment_count':len(names)+1,
                    **({'source_derived_facts':[list(year[:2])]}if year is not None else {})}
            request = {'source_binding': {**binding, 'source_qualification_rendering': PAID_BINDING_VERSION,
                'qualification_reference': decision['qualification_reference'], 'semantic_witnesses': [[list(key), dict(ref)]for key,ref in witnesses]}, 'claims':selected}
            rendering_ref = self.localise(request)
            rendered = self.read_localisation(rendering_ref, request)
            if rendered.get('source_binding') != request['source_binding'] or set(rendered.get('renderings', {})) != set(selected):
                raise QualificationHold('QUALIFICATION_RENDERING_BINDING_HOLD')
            from .native_claim_localisation import TYPED_VERSION
            typed_result = rendered.get('version') == TYPED_VERSION
            if dissent and active_contract == SEMANTIC_RESOLUTION_CONTRACT_V2 and not typed_result:
                raise QualificationHold('QUALIFICATION_RENDERING_BINDING_HOLD')
            if typed_result:
                if rendered.get('original_state') != request or rendered.get('projected_state') != source_rendering_projection(request):
                    raise QualificationHold('QUALIFICATION_RENDERING_BINDING_HOLD')
                consumer_version = (TYPED_CONSUMER_VERSION + '+' + consumer_version
                                    if self.resolve_disagreements else TYPED_CONSUMER_VERSION)
                consumer_version += '+' + RENDERING_REPAIR_CONSUMER_VERSION
                reference = tuple(sorted({'contract': SOURCE_RENDERING_CONTRACT_V2, 'operation': 'SOURCE_RENDERING',
                    'invocation_id': rendering_ref.invocation_id, 'raw_admission_id': str(rendering_ref.raw_admission_id),
                    'receipt_admission_id': str(rendering_ref.receipt_admission_id)}.items()))
                rendering_sources = [(claim.claim_id, reference) for claim in claims.values()]
            for index, wire_claim in enumerate(raw_wire['package']['governed_claims']):
                result = dict(rendered['renderings'][str(index)])
                if typed_result:
                    result = original_rendering_slots({'entities': decision['materialisation_receipt']['claim_entity_order'][index]},
                        rendered['projected_state']['claims'][str(index)], result)
                    wire_claim.update(result)
                    continue
                fragments = result['rendered_assertion_zh_hant_hk_fragments']
                terms = selected[str(index)]['entities']
                text = fragments[0] + ''.join(name+fragment for (name,_kind),fragment in zip(terms,fragments[1:],strict=True))
                # Re-express the derived text in the original codec's N+1 slots.
                # No original range, identity or entity inventory is changed.
                old_entities = decision['materialisation_receipt']['claim_entity_order'][index]
                old_fragments = []
                for name,_kind in old_entities:
                    before, found, text = text.partition(name)
                    if not found:raise QualificationHold('QUALIFICATION_RENDERING_IDENTITY_HOLD')
                    old_fragments.append(before)
                result['rendered_assertion_zh_hant_hk_fragments'] = [*old_fragments,text]
                wire_claim.update(result)
            materialisation = _materialisation(canonical_json_bytes(raw_wire), {'source_binding':binding,
                'source_view':{'passages':list(base.passages),'source_ids':list(base.source_ids)}})
        else:
            materialisation = decision['materialisation_receipt']
        composed = {**decision, 'consumer_contract':consumer_version, 'materialisation_receipt':materialisation,
            'semantic_witnesses':[[list(key),dict(ref)]for key,ref in witnesses],
            'source_renderings':[[key,dict(ref)]for key,ref in rendering_sources],
            **({'rendering_reference':{'invocation_id':rendering_ref.invocation_id,
                'raw_admission_id':str(rendering_ref.raw_admission_id),'receipt_admission_id':str(rendering_ref.receipt_admission_id)}}if rendering_ref else {})}
        raw = canonical_json_bytes(composed)
        admitted = self.objects.admit(ObjectAdmissionRequest('evidence.record',
            'source-qualification-consumer:'+digest_canonical([consumer_version, decision['qualification_reference']])), raw, proof=proof).admission
        return JudgedAssessment(NativeAssessmentExecution(materialisation['materialised_text'], {}), raw, admitted.admission_id,
            semantic_witnesses=SemanticWitnessMetadata(tuple(witnesses)),
            source_renderings=SourceRenderingMetadata(tuple(rendering_sources)))


def validate_source_literal_copy(copy, package, *, semantic_witness_reader=None):
    """Only correct quote delimiters within currently authenticated literal names."""
    from .writer import validate_writer_copy
    from .admission import source_rendering_is_admitted
    from .evidence import _entity_pattern
    import re
    original = validate_writer_copy(copy, package)
    names = set()
    for claim in package.governed_claims:
        if claim.source_rendering_ref and source_rendering_is_admitted(
                claim, package, semantic_witness_reader=semantic_witness_reader):
            names.update(name for name,kind,_ref in claim.named_entity_evidence if kind == 'SOURCE_LITERAL')
    def mask(text):
        positions = set()
        for name in names:
            for match in re.finditer(_entity_pattern(name),text):
                positions.update(match.start()+index for index,char in enumerate(name)
                    if char in {"'",'’'} and 0 < index < len(name)-1
                    and name[index-1].isalpha() and name[index+1].isalpha())
        return ''.join('x'if index in positions else char for index,char in enumerate(text))
    text = f"{copy.title}\n{copy.body}"
    scanned = mask(text)
    if scanned == text:
        return original
    from .writer import _has_unicode_quote_delimiter, WriterValidatorResult
    # Scan positions are unchanged; comparator values remain authoritative text.
    patterns = (r'"([^"\n]+)"',r"“([^”\n]+)”",r"「([^」\n]+)」",r"『([^』\n]+)』",
        r"‘([^’\n]+)’",r"〝([^〞\n]+)〞",r"﹁([^﹂\n]+)﹂",r"❝([^❞\n]+)❞",
        r"﹃([^﹄\n]+)﹄",r"«([^»\n]+)»",r"‹([^›\n]+)›",
        r"(?<![A-Za-z])'([^'\n]+)'(?![A-Za-z])")
    quoted = {text[match.start(1):match.end(1)]for pattern in patterns for match in re.finditer(pattern,scanned)}
    positions = tuple(index for index,char in enumerate(scanned) if _has_unicode_quote_delimiter(char))
    quoted.update(text[start+1:end]for start,end in zip(positions[::2],positions[1::2])if start+1<end)
    segments = tuple(segment.strip()for segment in re.split(r'\n+',text)if segment.strip())
    passed = len(positions)%2 == 0 and all(any(value in claim.quotations
        and claim.attribution in claim.rendered_assertion_zh_hant_hk
        and any(value in segment and claim.attribution in segment for segment in segments)
        for claim in package.governed_claims)for value in quoted)
    corrected = WriterValidatorResult('QUOTE_FIDELITY','PASS'if passed else'FAIL','UNSUPPORTED_OR_UNATTRIBUTED_QUOTATION')
    return tuple(corrected if row.validator == 'QUOTE_FIDELITY'else row for row in original)

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

CONSUMER_VERSION = 'newsroom.source-qualification-consumer.v2'
# Pure consumer repairs never reopen an identical paid witness/render purpose.
PAID_BINDING_VERSION = 'newsroom.source-qualification-consumer.v1'


def current_source_passage(source):
    """Reproduce the paid text recipe only from an already verified CURRENT Source."""
    from .native_source_intake import pdf_asset_url, spreadsheet_asset_url
    from .govuk_evidence import _api_url
    unit=source.unit
    if unit.source_id in {'HK-02','UK-10'} or pdf_asset_url(unit) or spreadsheet_asset_url(unit):
        return unit.body
    # This is the existing acquisition router's Content API branch.
    _api_url(unit.canonical_url)
    return unit.headline+'\n\n'+unit.body


class NativeQualifiedSourceConsumer:
    def __init__(self, qualifier, *, semantic_witnesses, localise, read_localisation):
        self.qualifier, self.objects = qualifier, qualifier.objects
        self.semantic_witnesses = semantic_witnesses
        self.localise, self.read_localisation = localise, read_localisation

    def read_semantic_parent(self, binding, candidate, base, *, proof):
        """The original SourceQA is re-read, never evaluated during admission."""
        from .native_source_qualification_replay import original_qualification_state
        parent = binding['source_qualification_reference']
        original_binding = {key:value for key,value in binding.items()
            if key not in {'semantic_witness_consumer', 'source_qualification_reference'}}
        state = original_qualification_state(self.qualifier, candidate, base, original_binding, proof=proof)
        reference = QualificationReference(parent['invocation_id'], ObjectAdmissionId.parse(parent['raw_admission_id']),
            ObjectAdmissionId.parse(parent['receipt_admission_id']))
        return self.qualifier.read_qualification(reference, state, proof=proof, candidate_id=candidate.candidate_id,
            hypothesis_digest=candidate.governing_manifest.canonical_digest, evidence_package_digest=base.digest)['materialisation']

    def read_current_disposition(self, candidate, base, sources, *, proof, source_passages=None):
        """Proof-only retained CURRENT projection, not external-now acquisition."""
        import sqlite3,time
        from pathlib import Path
        from types import SimpleNamespace
        from .native_source_qualification import ROUTE,VERSION,QualificationReference
        from .native_source_qualification_replay import original_qualification_state
        from .evidence import QualificationEvidence,Evid012QualificationTest
        from .admission import _qualification_relation_is_proven
        with sqlite3.connect(Path(self.qualifier.usage.path).resolve().as_uri()+'?mode=ro',uri=True)as c:
            c.execute('PRAGMA query_only=ON');deadline=time.monotonic()+5
            c.set_progress_handler(lambda:int(time.monotonic()>deadline),1000)
            rows=c.execute('SELECT a.invocation_id FROM model_invocation_allocations a JOIN model_work_envelopes e USING(envelope_id) '
                "WHERE a.route=? AND json_extract(e.record_json,'$.candidate_id')=? "
                "AND json_extract(e.record_json,'$.hypothesis_digest')=? AND json_extract(e.record_json,'$.evidence_package_digest')=? LIMIT 2",
                (ROUTE,candidate.candidate_id,candidate.governing_manifest.canonical_digest,base.digest)).fetchall()
        if len(rows)!=1:return None
        admitted=self.objects.committed_admission(ObjectAdmissionRequest('evidence.record','source-qualification-receipt:'+rows[0][0]),proof=proof)
        if admitted is None:return None
        raw=self.objects.rehydrate(HydrationRequest(admitted.admission.admission_id,'evidence.record'),proof=proof).data
        receipt=json.loads(raw);binding=receipt['source_binding']
        if canonical_json_bytes(receipt)!=raw or receipt['version']!=VERSION:return None
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
        reference=QualificationReference(rows[0][0],ObjectAdmissionId.parse(receipt['raw_admission_id']),admitted.admission.admission_id)
        checked=self.qualifier.read_qualification(reference,state,proof=proof,candidate_id=candidate.candidate_id,
            hypothesis_digest=candidate.governing_manifest.canonical_digest,evidence_package_digest=base.digest)
        package=json.loads(checked['materialisation']['materialised_text'])['package']
        if not package['substantive_new_information']:return None
        claims={row['claim_id']:SimpleNamespace(**{**row,'source_ids':tuple(row['source_ids'])})for row in package['governed_claims']}
        for item in package['qualification_evidence']:
            claim=claims[item['governed_claim_id']]
            q=QualificationEvidence(Evid012QualificationTest(item['test']),claim.claim_id,'retained-qualification',tuple(item['test_evidence'].items()),item['policy_version'])
            if not _qualification_relation_is_proven(q,claim,source_context=base.passages[claim.passage_index]):
                self.semantic_witnesses.read_existing(q,claim,base,{**binding,'semantic_witness_consumer':PAID_BINDING_VERSION,
                    'source_qualification_reference':{'invocation_id':reference.invocation_id,'raw_admission_id':str(reference.raw_admission_id),
                        'receipt_admission_id':str(reference.receipt_admission_id)}})
        return None

    def compose_selected(self, original, candidate, base, sources, acquired, *, proof):
        """A new witness/render purpose, never qualification-model redispatch."""
        from types import SimpleNamespace
        from .native_assessor import _qualification_record_id, _reference_binding, _document
        from .native_assessor_judgments import (JudgedAssessment, SemanticWitnessMetadata, SourceRenderingMetadata,
            source_rendering_details, source_rendering_names)
        from .evidence import (QualificationEvidence, Evid012QualificationTest, bounded_named_entities,
            _canonical_localised_fact, SOURCE_RENDERING_CONTRACT)
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
        claims = {row['claim_id']: SimpleNamespace(**{**row,'source_ids':tuple(row['source_ids'])}) for row in package['governed_claims']}
        witnesses = []
        for item in package['qualification_evidence']:
            claim = claims[item['governed_claim_id']]
            fields = tuple(item['test_evidence'].items())
            q = QualificationEvidence(test=Evid012QualificationTest(item['test']), governed_claim_id=claim.claim_id,
                qualification_record_id=_qualification_record_id(claim.claim_id, item['test'], [list(p)for p in fields]),
                test_evidence=fields, policy_version=item['policy_version'])
            if not _qualification_relation_is_proven(q, claim, source_context=base.passages[claim.passage_index]):
                verified_binding = {**binding, 'semantic_witness_consumer': PAID_BINDING_VERSION,
                    'source_qualification_reference': decision['qualification_reference']}
                ref = self.semantic_witnesses.evaluate(q, claim, base, verified_binding)
                witnesses.append(((claim.claim_id, q.test.value), ref))
        if not package['qualification_evidence'] or not any(row['governed_claim_id'] == next(
                c['claim_id']for c in package['governed_claims']if c['claim_role']=='HEADLINE')for row in package['qualification_evidence']):
            raise QualificationHold('QUALIFICATION_HEADLINE_UNPROVEN')
        # A verifier does not certify actor identity, unsupported factual types,
        # or rendering. Deny impossible current capabilities before any render fee.
        derived = {}
        rendering_sources = []
        for claim in claims.values():
            body = base.passages[claim.passage_index]
            try:
                terms, year = source_rendering_details(claim,body,binding['current_scope']['sources'][claim.passage_index])
            except ValueError as exc:
                raise QualificationHold('QUALIFICATION_SOURCE_RENDERING_UNPROVEN') from exc
            names = source_rendering_names(claim,body)
            recognised = {name for name,_kind in names}
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
        valid_rendering = all(_valid_zh_hant_hk_rendering(SimpleNamespace(**vars(claim),
            named_entities=tuple(name for name,_kind in derived[claim.claim_id][2])))
            and (derived[claim.claim_id][1] is None or derived[claim.claim_id][1][:2] in
                 tuple(map(tuple,claim.localised_factual_expressions)))for claim in claims.values())
        if not witnesses and not rendering_sources and valid_rendering:
            return original
        rendering_ref = None
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
            for index, wire_claim in enumerate(raw_wire['package']['governed_claims']):
                result = dict(rendered['renderings'][str(index)])
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
        composed = {**decision, 'consumer_contract':CONSUMER_VERSION, 'materialisation_receipt':materialisation,
            'semantic_witnesses':[[list(key),dict(ref)]for key,ref in witnesses],
            'source_renderings':[[key,dict(ref)]for key,ref in rendering_sources],
            **({'rendering_reference':{'invocation_id':rendering_ref.invocation_id,
                'raw_admission_id':str(rendering_ref.raw_admission_id),'receipt_admission_id':str(rendering_ref.receipt_admission_id)}}if rendering_ref else {})}
        raw = canonical_json_bytes(composed)
        admitted = self.objects.admit(ObjectAdmissionRequest('evidence.record',
            'source-qualification-consumer:'+digest_canonical([CONSUMER_VERSION, decision['qualification_reference']])), raw, proof=proof).admission
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

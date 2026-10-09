"""Call-local watermark uses event evidence already read by actual inventory."""
from copy import copy
from dataclasses import replace

import pytest

from newsroom.authority import AggregateId, GovernedObjects, AuthenticationProof, ObjectAdmissionId
from newsroom.authority.persistence import AuthorityEvents
from newsroom.increment5.native_retrieval import NativeDocumentReceipt, NativeRetrievalDocuments, NativeRetrievalError
from newsroom.tests import test_native_retrieval_authority as authority_case


def test_complete_corpus_scope_over_4096_never_hydrates_unselected_documents(monkeypatch):
    documents = object.__new__(NativeRetrievalDocuments)
    proof = AuthenticationProof(method="STATIC_TOKEN", credential="fixture")
    receipts = tuple(NativeDocumentReceipt(
        f"event-{index}", f"command-{index}", AggregateId.new(), 1,
        ObjectAdmissionId.new(), "sha256:" + "1" * 64,
        ObjectAdmissionId.new(), ObjectAdmissionId.new(),
    ) for index in range(4197))
    verified = []
    def metadata(self, receipt, supplied):
        assert self is documents and supplied is proof
        verified.append(receipt.event_id)
        return int(receipt.event_id.removeprefix("event-")) + 1
    monkeypatch.setattr(NativeRetrievalDocuments, "_verify_event", metadata)
    monkeypatch.setattr(NativeRetrievalDocuments, "_read_with_sequence",
                        lambda *_args, **_kwargs: pytest.fail("unselected document/vector hydration"))
    scope = documents.authenticated_corpus_scope(receipts, proof=proof)
    assert documents.corpus_scope_receipts(scope, proof=proof) == receipts
    assert documents.corpus_scope_watermark(scope, proof=proof) == 4197
    assert len(verified) == len(receipts) + 1
    for owner, supplied in ((copy(documents), proof), (documents, replace(proof, credential="wrong"))):
        with pytest.raises(NativeRetrievalError):
            owner.corpus_scope_receipts(scope, proof=supplied)


def test_valid_but_current_excluded_vector_event_is_rejected_before_body_read(monkeypatch):
    documents = object.__new__(NativeRetrievalDocuments)
    receipt = NativeDocumentReceipt("registered", "command", AggregateId.new(), 1,
        ObjectAdmissionId.new(), "sha256:" + "1" * 64, ObjectAdmissionId.new(), ObjectAdmissionId.new())
    excluded = replace(receipt, event_id="current-excluded")
    monkeypatch.setattr(documents, "_read", lambda *_args: pytest.fail("excluded body/vector read"))
    with pytest.raises(NativeRetrievalError, match="outside current corpus"):
        documents._documents(({**excluded.projection_value(), "score": 1.0},), "generation", None,
                             allowed_receipts=(receipt,))


def test_real_inventory_watermark_binds_proof_and_avoids_second_full_event_pass(tmp_path, monkeypatch):
    reads = []
    denial = {"events": False, "vector": None}
    original_provenance = AuthorityEvents.provenance
    def provenance(self, event_id, *, proof):
        reads.append(event_id)
        if denial["events"]:
            raise PermissionError("fixture current read revoked")
        return original_provenance(self, event_id, proof=proof)
    monkeypatch.setattr(AuthorityEvents, "provenance", provenance)
    original_rehydrate = GovernedObjects.rehydrate
    def rehydrate(self, request, *, proof):
        if request.admission_id == denial["vector"]:
            raise PermissionError("fixture current vector rights revoked")
        return original_rehydrate(self, request, proof=proof)
    monkeypatch.setattr(GovernedObjects, "rehydrate", rehydrate)
    admitted_request = []
    original_admit = NativeRetrievalDocuments.admit
    def admit(self, request, *, proof):
        admitted_request.append(request)
        return original_admit(self, request, proof=proof)
    monkeypatch.setattr(NativeRetrievalDocuments, "admit", admit)
    factory = NativeRetrievalDocuments.authenticated_document_inventory
    exercised = []
    def inventory(self, receipts, *, proof):
        if not exercised:
            exercised.append(True)
            projection_rows = list(self._projector.rows)
            extra = tuple(original_admit(self, replace(
                admitted_request[0], aggregate_id=AggregateId.new(),
                idempotency_key=f"watermark-extra-document-{number}",
            ), proof=proof)[0] for number in range(2))
            # Distinct real document events authenticate through the actual
            # facade. Restore the original projection fixture before its normal
            # single-passage four-branch/context test continues.
            self._projector.rows[:] = projection_rows
            exact = receipts + extra
            expected = max(original_provenance(self._events, item.event_id, proof=proof).event.ledger_seq for item in exact)
            start = len(reads)
            verified = factory(self, exact, proof=proof)
            assert len(reads) - start == len(exact)
            assert self.authenticated_inventory_watermark(verified, exact, proof=proof) == expected
            assert len(reads) - start == len(exact) + 1  # One current proof check, not another N pass.
            for owner, selected, supplied in (
                (copy(self), exact, proof),
                (self, tuple(reversed(exact)), proof),
                (self, exact[:-1], proof),
                (self, exact, replace(proof, credential="wrong-proof")),
            ):
                before = len(reads)
                with pytest.raises(NativeRetrievalError):
                    owner.authenticated_inventory_watermark(verified, selected, proof=supplied)
                assert len(reads) == before
            denial["events"] = True
            with pytest.raises(PermissionError, match="current read revoked"):
                self.authenticated_inventory_watermark(verified, exact, proof=proof)
            denial["events"] = False
            denial["vector"] = exact[-1].vector_admission_id
            with pytest.raises(PermissionError, match="current vector rights revoked"):
                factory(self, exact, proof=proof)
            denial["vector"] = None
            self.consume_authenticated_inventory(verified, exact, proof=proof)
            with pytest.raises(NativeRetrievalError):
                self.authenticated_inventory_watermark(verified, exact, proof=proof)
        return factory(self, receipts, proof=proof)
    monkeypatch.setattr(NativeRetrievalDocuments, "authenticated_document_inventory", inventory)
    authority_case.test_native_documents_admit_retain_context_and_reopen(tmp_path, monkeypatch)
    assert exercised == [True]

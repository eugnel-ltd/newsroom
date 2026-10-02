from __future__ import annotations

import sqlite3
from contextlib import nullcontext

from ._object_capability import _HydrationGrant
from .canonical import canonical_json_bytes, digest_bytes, digest_canonical
from .objects import (
    BlobIdentity,
    ObjectAccessDecisionId,
    ObjectAdmissionId,
    ObjectAccessDecisionView,
    ObjectAdmissionDenied,
    ObjectHydrationDenied,
)
from .persistence import AuthorityPersistenceError
from .types import (
    AuthenticationContextId,
    AuthorizationDecisionId,
    UtcTimestamp,
)


class _ObjectHydrationStoreMixin:
    """Authenticated, purpose-bound hydration with exact current-state cutoff."""

    def hydrate(
        self, grant: _HydrationGrant, *, in_transaction: bool = False,
        reuse_retained: bool = False
    ) -> tuple[bytes, ObjectAccessDecisionView]:
        if type(reuse_retained) is not bool:
            raise TypeError("reuse_retained must be boolean")
        if type(in_transaction) is not bool:
            raise TypeError("in_transaction must be boolean")
        if in_transaction and not self._connection.in_transaction:
            raise AuthorityPersistenceError(
                "transaction-bound hydration requires an active transaction"
            )
        now = self._clock()
        self._object_issuer.verify_hydration(grant, now=now)
        policy = self._hydration_policies.resolve_exact(
            grant.policy.policy_id,
            grant.policy.contract_version,
            grant.policy.contract_digest,
        )
        transaction = (
            nullcontext(self._connection)
            if in_transaction
            else self._transaction()
        )
        with self._lock:
            with transaction as conn:
                decided_at = self._clock()
                self._object_issuer.verify_hydration(
                    grant, now=decided_at
                )
                row = self._current_admission_row(
                    conn,
                    str(grant.request.admission_id),
                    now=decided_at,
                    require_active=True,
                    require_bytes=True,
                )
                definition = self._admission_registry.resolve_exact(
                    str(row["admission_type"]),
                    str(row["definition_version"]),
                    str(row["definition_digest"]),
                )
                if policy.contract_digest not in (
                    definition.hydration_policy_contract_digests
                ):
                    raise ObjectHydrationDenied(
                        "hydration policy is not admitted for this object use"
                    )
                authentication = grant.authentication
                if (
                    authentication.principal_id
                    not in policy.allowed_principal_ids
                    or authentication.authority_domain
                    not in policy.allowed_authority_domains
                ):
                    raise ObjectHydrationDenied(
                        "principal or authority domain is outside hydration policy"
                    )
                if str(row["object_class"]) not in policy.allowed_object_classes:
                    raise ObjectHydrationDenied(
                        "object class is outside hydration policy"
                    )
                if str(row["allowed_use"]) not in policy.allowed_uses:
                    raise ObjectHydrationDenied(
                        "object use is outside hydration policy"
                    )
                if (
                    str(row["security_scope"])
                    not in policy.allowed_security_scopes
                ):
                    raise ObjectHydrationDenied(
                        "security scope is outside hydration policy"
                    )
                if (
                    str(row["retention_scope"])
                    not in policy.allowed_retention_scopes
                ):
                    raise ObjectHydrationDenied(
                        "retention scope is outside hydration policy"
                    )
                blob = BlobIdentity(
                    str(row["blob_digest"]), int(row["size_bytes"])
                )
                offset = grant.request.offset
                length = (
                    blob.size_bytes - offset
                    if grant.request.length is None
                    else grant.request.length
                )
                if not policy.allow_ranges and (
                    offset != 0 or length != blob.size_bytes
                ):
                    raise ObjectHydrationDenied(
                        "hydration policy permits only a complete object read"
                    )
                if length > policy.max_bytes:
                    raise ObjectHydrationDenied(
                        "requested bytes exceed hydration policy"
                    )
                self._cas.limits.require_range(
                    total_size=blob.size_bytes,
                    offset=offset,
                    length=length,
                )
                pinned = self._cas.pin(blob)
                try:
                    self._cas.verify_pinned(pinned)
                    blob_lifecycle = self._blob_lifecycle_row(
                        blob.blob_digest, conn=conn
                    )
                    deletion = self._active_deletion_for_blob(
                        conn, blob.blob_digest
                    )
                    state_cutoff_value = {
                        "admission_id": str(grant.request.admission_id),
                        "admission_lifecycle_version": int(
                            row["admission_lifecycle_version"]
                        ),
                        "admission_state": str(row["state"]),
                        "rights_decision_id": str(row["rights_decision_id"]),
                        "rights_decision_digest": str(
                            row["rights_decision_digest"]
                        ),
                        "rights_valid_from": str(row["rights_valid_from"]),
                        "rights_valid_until": (
                            None
                            if row["rights_valid_until"] is None
                            else str(row["rights_valid_until"])
                        ),
                        "blob_digest": blob.blob_digest,
                        "blob_lifecycle_version": int(
                            blob_lifecycle["current_version"]
                        ),
                        "blob_state": str(blob_lifecycle["state"]),
                        "blob_integrity_state": str(
                            blob_lifecycle["integrity_state"]
                        ),
                        "deletion_id": (
                            None
                            if deletion is None
                            else str(deletion["deletion_id"])
                        ),
                        "deletion_lifecycle_version": (
                            None
                            if deletion is None
                            else int(deletion["current_version"])
                        ),
                        "deletion_state": (
                            None if deletion is None else str(deletion["state"])
                        ),
                        "offset": offset,
                        "length": length,
                    }
                    state_cutoff_bytes = canonical_json_bytes(
                        state_cutoff_value
                    )
                    state_cutoff = digest_bytes(state_cutoff_bytes)
                    retained = (
                        self._retained_hydration(conn, grant, state_cutoff, offset, length, row)
                        if reuse_retained else None
                    )
                    if retained is not None:
                        access_decision_id = retained.access_decision_id
                    else:
                        access_decision_id = ObjectAccessDecisionId.new()
                        canonical_value = {
                            "access_decision_id": str(access_decision_id),
                            "policy_contract_digest": policy.contract_digest,
                            "authentication_context_id": str(
                                authentication.authentication_context_id
                            ),
                            "authorization_request_digest": (
                                grant.authorization_request.request_digest
                            ),
                            "authorization_decision_id": str(
                                grant.authorization.authorization_decision_id
                            ),
                            "principal_id": authentication.principal_id,
                            "authority_domain": authentication.authority_domain,
                            "purpose": policy.purpose,
                            "admission_id": str(grant.request.admission_id),
                            "object_class": str(row["object_class"]),
                            "allowed_use": str(row["allowed_use"]),
                            "security_scope": str(row["security_scope"]),
                            "retention_scope": str(row["retention_scope"]),
                            "offset": offset,
                            "allowed_bytes": length,
                            "state_cutoff": state_cutoff_value,
                            "state_cutoff_digest": state_cutoff,
                            "decided_at": decided_at.to_text(),
                        }
                        canonical = canonical_json_bytes(canonical_value)
                        canonical_digest = digest_bytes(canonical)
                        self._persist_security_records(
                            conn,
                            authentication=grant.authentication,
                            request=grant.authorization_request,
                            decision=grant.authorization,
                            recorded_at=decided_at.to_text(),
                        )
                        conn.execute(
                            "INSERT INTO object_access_decisions("
                            "access_decision_id,hydration_policy_contract_digest,"
                            "authentication_context_id,authorization_request_digest,"
                            "authorization_decision_id,principal_id,authority_domain,"
                            "purpose,admission_id,object_class,allowed_use,"
                            "security_scope,retention_scope,byte_offset,allowed_bytes,"
                            "state_cutoff_bytes,state_cutoff_digest,decided_at,"
                            "canonical_bytes,canonical_digest) "
                            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                            (
                                str(access_decision_id),
                                policy.contract_digest,
                                str(authentication.authentication_context_id),
                                grant.authorization_request.request_digest,
                                str(grant.authorization.authorization_decision_id),
                                authentication.principal_id,
                                authentication.authority_domain,
                                policy.purpose,
                                str(grant.request.admission_id),
                                str(row["object_class"]),
                                str(row["allowed_use"]),
                                str(row["security_scope"]),
                                str(row["retention_scope"]),
                                offset,
                                length,
                                state_cutoff_bytes,
                                state_cutoff,
                                decided_at.to_text(),
                                canonical,
                                canonical_digest,
                            ),
                        )
                    # The state and pinned bytes are rechecked immediately before
                    # leaving the authority transaction.
                    self._current_admission_row(
                        conn,
                        str(grant.request.admission_id),
                        now=self._clock(),
                        require_active=True,
                        require_bytes=True,
                    )
                    self._cas.verify_pinned(pinned)
                    data = self._cas.read_range(
                        pinned, offset=offset, length=length
                    )
                    # The read can cross a rights-expiry boundary even though
                    # SQLite prevents concurrent lifecycle mutation.  Recheck
                    # current rights and admission time after the exact bytes
                    # have been read, before the access decision commits or any
                    # bytes can leave the authority boundary.
                    final_now = self._clock()
                    self._current_admission_row(
                        conn,
                        str(grant.request.admission_id),
                        now=final_now,
                        require_active=True,
                        require_bytes=True,
                    )
                    self._object_issuer.verify_hydration(
                        grant, now=final_now
                    )
                finally:
                    pinned.close()
            return data, self.access_decision_view(access_decision_id)

    def _retained_hydration(self, conn, grant, cutoff, offset, length, admission):
        # Use the existing admission/time index; never scan all access history.
        row = conn.execute(
            "SELECT * FROM object_access_decisions WHERE admission_id=? "
            "AND hydration_policy_contract_digest=? AND principal_id=? "
            "AND authority_domain=? AND purpose=? AND byte_offset=? "
            "AND allowed_bytes=? AND state_cutoff_digest=? "
            "ORDER BY decided_at DESC,rowid DESC LIMIT 1",
            (str(grant.request.admission_id), grant.policy.contract_digest,
             grant.authentication.principal_id, grant.authentication.authority_domain,
             grant.policy.purpose, offset, length, cutoff),
        ).fetchone()
        if row is None:
            return None
        value = self._require_canonical_record(row)
        for field in ("object_class", "allowed_use", "security_scope", "retention_scope"):
            if row[field] != admission[field]:
                raise AuthorityPersistenceError("retained access admission semantics differ")
        if UtcTimestamp.parse(row["decided_at"]).value > self._clock().value:
            return None
        for field, column in (("policy_contract_digest", "hydration_policy_contract_digest"),
                              ("offset", "byte_offset")):
            if value.get(field) != row[column]:
                raise AuthorityPersistenceError("retained access indexed fields differ")
        for field in ("authentication_context_id", "authorization_request_digest",
                      "authorization_decision_id", "principal_id", "authority_domain",
                      "purpose", "admission_id", "object_class", "allowed_use",
                      "security_scope", "retention_scope", "allowed_bytes", "decided_at"):
            if value.get(field) != row[field]:
                raise AuthorityPersistenceError("retained access indexed fields differ")
        decision_row = conn.execute(
            "SELECT * FROM authorization_decisions WHERE authorization_decision_id=?",
            (row["authorization_decision_id"],),
        ).fetchone()
        if decision_row is None:
            raise AuthorityPersistenceError("retained access authorisation is missing")
        decision = self._decision_record_from_row(decision_row)
        context_row = conn.execute(
            "SELECT * FROM authentication_contexts WHERE authentication_context_id=?",
            (row["authentication_context_id"],),
        ).fetchone()
        request_row = conn.execute(
            "SELECT * FROM authorization_requests WHERE request_digest=?",
            (row["authorization_request_digest"],),
        ).fetchone()
        if context_row is None or request_row is None:
            raise AuthorityPersistenceError("retained access security records are missing")
        context = self._authentication_record_from_row(context_row)
        request = self._request_record_from_row(request_row)
        if (decision.authentication_context_id != context.authentication_context_id
                or decision.authorization_request_digest != request.request_digest
                or request.authentication_context_id != context.authentication_context_id):
            raise AuthorityPersistenceError("retained access security binding differs")
        recorded_request = self._decode_canonical_object(request.canonical_bytes)
        current_request = grant.authorization_request.canonical_value()
        recorded_semantic = recorded_request.pop("stable_semantic_request_digest")
        current_semantic = current_request.pop("stable_semantic_request_digest")
        for ephemeral in ("authentication_context_id", "request_digest"):
            recorded_request.pop(ephemeral)
            current_request.pop(ephemeral)
        if recorded_request != current_request:
            raise AuthorityPersistenceError("retained access request semantics differ")
        # An explicit complete range and length=None can read the same bytes
        # with different request identities. Record the new exact request once.
        if recorded_semantic != current_semantic:
            return None
        for field in ("principal_id", "authority_domain", "authentication_method",
                      "assurance_class", "credential_binding_digest"):
            if getattr(context, field) != getattr(grant.authentication, field):
                return None
        if (decision.authorization_policy_version != grant.authorization.authorization_policy_version
                or decision.effective_scopes != grant.authorization.effective_scopes
                or not decision.allowed):
            return None
        return self.access_decision_view(
            ObjectAccessDecisionId.parse(str(row["access_decision_id"]))
        )

    def access_decision_view(
        self, access_decision_id: ObjectAccessDecisionId
    ) -> ObjectAccessDecisionView:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM object_access_decisions WHERE access_decision_id=?",
                (str(access_decision_id),),
            ).fetchone()
            if row is None:
                raise KeyError(str(access_decision_id))
            value = self._require_canonical_record(row)
            if value.get("access_decision_id") != str(access_decision_id):
                raise AuthorityPersistenceError(
                    "access decision canonical identity mismatch"
                )
            indexed_fields = (
                ("policy_contract_digest", "hydration_policy_contract_digest"),
                ("authentication_context_id", "authentication_context_id"),
                ("authorization_request_digest", "authorization_request_digest"),
                ("authorization_decision_id", "authorization_decision_id"),
                ("principal_id", "principal_id"),
                ("authority_domain", "authority_domain"),
                ("purpose", "purpose"),
                ("admission_id", "admission_id"),
                ("object_class", "object_class"),
                ("allowed_use", "allowed_use"),
                ("security_scope", "security_scope"),
                ("retention_scope", "retention_scope"),
                ("offset", "byte_offset"),
                ("allowed_bytes", "allowed_bytes"),
                ("decided_at", "decided_at"),
            )
            if any(
                value.get(field) != row[column]
                for field, column in indexed_fields
            ):
                raise AuthorityPersistenceError(
                    "access decision indexed fields differ"
                )
            cutoff_bytes = bytes(row["state_cutoff_bytes"])
            cutoff_digest = str(row["state_cutoff_digest"])
            cutoff_value = self._decode_canonical_object(cutoff_bytes)
            if digest_bytes(cutoff_bytes) != cutoff_digest:
                raise AuthorityPersistenceError(
                    "access decision state cutoff digest mismatch"
                )
            if (
                not isinstance(cutoff_value, dict)
                or value.get("state_cutoff") != cutoff_value
                or value.get("state_cutoff_digest") != cutoff_digest
                or cutoff_value.get("admission_id") != row["admission_id"]
                or cutoff_value.get("offset") != row["byte_offset"]
                or cutoff_value.get("length") != row["allowed_bytes"]
            ):
                raise AuthorityPersistenceError(
                    "access decision canonical cutoff differs from indexed record"
                )
            decision_row = self._connection.execute(
                "SELECT * FROM authorization_decisions "
                "WHERE authorization_decision_id=?",
                (row["authorization_decision_id"],),
            ).fetchone()
            context_row = self._connection.execute(
                "SELECT * FROM authentication_contexts "
                "WHERE authentication_context_id=?",
                (row["authentication_context_id"],),
            ).fetchone()
            request_row = self._connection.execute(
                "SELECT * FROM authorization_requests WHERE request_digest=?",
                (row["authorization_request_digest"],),
            ).fetchone()
            if decision_row is None or context_row is None or request_row is None:
                raise AuthorityPersistenceError(
                    "access decision security records are missing"
                )
            decision = self._decision_record_from_row(decision_row)
            context = self._authentication_record_from_row(context_row)
            request = self._request_record_from_row(request_row)
            request_value = self._decode_canonical_object(request.canonical_bytes)
            policy = self._hydration_policies.resolve_digest(
                str(row["hydration_policy_contract_digest"])
            )
            admission = self._connection.execute(
                "SELECT * FROM object_admissions WHERE admission_id=?",
                (row["admission_id"],),
            ).fetchone()
            if admission is None:
                raise AuthorityPersistenceError(
                    "access decision admission is missing"
                )
            blob = self._connection.execute(
                "SELECT * FROM blob_identities WHERE blob_digest=?",
                (admission["blob_digest"],),
            ).fetchone()
            rights = self._connection.execute(
                "SELECT * FROM object_rights_decisions WHERE rights_decision_id=?",
                (admission["rights_decision_id"],),
            ).fetchone()
            if blob is None or rights is None:
                raise AuthorityPersistenceError(
                    "access decision admission authority is missing"
                )
            rights_value = self._verified_rights_value(rights)
            definition = self._admission_registry.resolve_exact(
                str(admission["admission_type"]),
                str(admission["definition_version"]),
                str(admission["definition_digest"]),
            )
            size = int(blob["size_bytes"])
            offset = int(row["byte_offset"])
            allowed = int(row["allowed_bytes"])
            explicit_semantic = digest_canonical({
                "policy_contract_digest": row["hydration_policy_contract_digest"],
                "admission_id": row["admission_id"],
                "purpose": row["purpose"],
                "offset": offset,
                "length": allowed,
            })
            read_to_end_semantic = digest_canonical({
                "policy_contract_digest": row["hydration_policy_contract_digest"],
                "admission_id": row["admission_id"],
                "purpose": row["purpose"],
                "offset": offset,
                "length": None,
            })
            retained_semantic = request_value.get(
                "stable_semantic_request_digest"
            )
            accessed_at = UtcTimestamp.parse(str(row["decided_at"]))
            rights_decided_at = UtcTimestamp.parse(str(rights["decided_at"]))
            rights_valid_from = UtcTimestamp.parse(str(rights["valid_from"]))
            rights_valid_until = (
                None
                if rights["valid_until"] is None
                else UtcTimestamp.parse(str(rights["valid_until"]))
            )
            admission_binding_differs = (
                row["object_class"] != admission["object_class"]
                or row["allowed_use"] != admission["allowed_use"]
                or row["security_scope"] != admission["security_scope"]
                or row["retention_scope"] != admission["retention_scope"]
                or cutoff_value["admission_id"] != admission["admission_id"]
                or cutoff_value.get("blob_digest") != admission["blob_digest"]
                or cutoff_value.get("rights_decision_id")
                != admission["rights_decision_id"]
                or cutoff_value.get("rights_decision_digest")
                != rights["canonical_digest"]
                or cutoff_value.get("rights_valid_from") != rights["valid_from"]
                or cutoff_value.get("rights_valid_until") != rights["valid_until"]
                or rights_value.get("rights_decision_id")
                != rights["rights_decision_id"]
                or rights_value.get("valid_from") != rights["valid_from"]
                or rights_value.get("valid_until") != rights["valid_until"]
                or rights["blob_digest"] != admission["blob_digest"]
                or rights["admission_definition_digest"]
                != admission["definition_digest"]
                or rights["object_class"] != admission["object_class"]
                or rights["allowed_use"] != admission["allowed_use"]
                or rights["security_scope"] != admission["security_scope"]
                or rights["retention_scope"] != admission["retention_scope"]
                or not bool(rights["allowed"])
                or int(rights["size_bytes"]) != size
                or admission["valid_from"] != rights["valid_from"]
                or admission["valid_until"] != rights["valid_until"]
                or rights_decided_at.value > accessed_at.value
                or rights_valid_from.value > accessed_at.value
                or (
                    rights_valid_until is not None
                    and accessed_at.value >= rights_valid_until.value
                )
                or definition.object_class != admission["object_class"]
                or definition.allowed_use != admission["allowed_use"]
                or definition.security_scope != admission["security_scope"]
                or definition.retention_scope != admission["retention_scope"]
                or definition.rights_policy_contract_digest
                != rights["policy_contract_digest"]
                or policy.contract_digest
                not in definition.hydration_policy_contract_digests
                or offset + allowed > size
                or (
                    not policy.allow_ranges
                    and (offset != 0 or allowed != size)
                )
                or retained_semantic not in {
                    explicit_semantic, read_to_end_semantic,
                }
                or (
                    retained_semantic == read_to_end_semantic
                    and allowed != size - offset
                )
            )
            if admission_binding_differs:
                raise AuthorityPersistenceError(
                    "access decision admission binding differs"
                )
            expected_scope_digest = digest_canonical({
                "authentication_context_digest": context.canonical_digest,
                "effective_scopes": list(decision.effective_scopes),
            })
            authenticated_at = UtcTimestamp.parse(context.authenticated_at)
            expires_at = UtcTimestamp.parse(context.expires_at)
            authorised_at = UtcTimestamp.parse(decision.decided_at)
            if (
                decision.authentication_context_id
                != context.authentication_context_id
                or decision.authorization_request_digest != request.request_digest
                or request.authentication_context_id
                != context.authentication_context_id
                or request.principal_id != context.principal_id
                or request.authority_domain != context.authority_domain
                or request.principal_id != row["principal_id"]
                or request.authority_domain != row["authority_domain"]
                or request.operation_type != f"object:hydrate:{row['purpose']}"
                or request_value.get("command_definition_digest")
                != policy.contract_digest
                or policy.purpose != row["purpose"]
                or request.required_scope != policy.required_scope
                or request.required_scope not in decision.effective_scopes
                or row["principal_id"] not in policy.allowed_principal_ids
                or row["authority_domain"] not in policy.allowed_authority_domains
                or row["object_class"] not in policy.allowed_object_classes
                or row["allowed_use"] not in policy.allowed_uses
                or row["security_scope"] not in policy.allowed_security_scopes
                or row["retention_scope"] not in policy.allowed_retention_scopes
                or row["allowed_bytes"] > policy.max_bytes
                or (not policy.allow_ranges and row["byte_offset"] != 0)
                or decision.effective_scope_digest != expected_scope_digest
                or not decision.allowed
                or not (
                    authenticated_at.value
                    <= authorised_at.value
                    < expires_at.value
                )
                or authorised_at.value > accessed_at.value
                or accessed_at.value >= expires_at.value
            ):
                raise AuthorityPersistenceError(
                    "access decision security binding differs"
                )
            return ObjectAccessDecisionView(
                access_decision_id=access_decision_id,
                policy_contract_digest=str(
                    row["hydration_policy_contract_digest"]
                ),
                authentication_context_id=AuthenticationContextId.parse(
                    str(row["authentication_context_id"])
                ),
                authorization_request_digest=str(
                    row["authorization_request_digest"]
                ),
                authorization_decision_id=AuthorizationDecisionId.parse(
                    str(row["authorization_decision_id"])
                ),
                principal_id=str(row["principal_id"]),
                authority_domain=str(row["authority_domain"]),
                purpose=str(row["purpose"]),
                admission_id=ObjectAdmissionId.parse(str(row["admission_id"])),
                object_class=str(row["object_class"]),
                allowed_use=str(row["allowed_use"]),
                security_scope=str(row["security_scope"]),
                retention_scope=str(row["retention_scope"]),
                offset=int(row["byte_offset"]),
                allowed_bytes=int(row["allowed_bytes"]),
                state_cutoff_bytes=cutoff_bytes,
                state_cutoff_digest=cutoff_digest,
                decided_at=UtcTimestamp.parse(str(row["decided_at"])),
                canonical_digest=str(row["canonical_digest"]),
            )

    def latest_access_decision(
        self,
        *,
        admission_id: ObjectAdmissionId,
        policy_contract_digest: str,
        principal_id: str,
        authority_domain: str,
        purpose: str,
    ) -> ObjectAccessDecisionView:
        """Return the latest exact retained complete-object access decision."""

        with self._lock:
            admission = self._current_admission_row(
                self._connection,
                str(admission_id),
                now=self._clock(),
                require_active=True,
                require_bytes=True,
            )
            row = self._connection.execute(
                "SELECT access_decision_id FROM object_access_decisions "
                "WHERE admission_id=? "
                "AND hydration_policy_contract_digest=? "
                "AND principal_id=? AND authority_domain=? AND purpose=? "
                "AND byte_offset=0 AND allowed_bytes=? "
                "ORDER BY decided_at DESC,access_decision_id DESC LIMIT 1",
                (
                    str(admission_id),
                    policy_contract_digest,
                    principal_id,
                    authority_domain,
                    purpose,
                    int(admission["size_bytes"]),
                ),
            ).fetchone()
            if row is None:
                raise KeyError(str(admission_id))
            return self.access_decision_view(
                ObjectAccessDecisionId.parse(str(row["access_decision_id"]))
            )


__all__ = ["_ObjectHydrationStoreMixin"]

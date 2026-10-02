# ADR 0010: current operational state and disposable diagnostics

Status: Accepted, 2 October 2026. Owner direction: [#981](https://github.com/fol2/newsroom/issues/981#issuecomment-5947985064).

## Decision

The private development engine starts from compact durable **current state**. Debug and maintenance history is optional, bounded and independent of boot and ordinary work. Permanent availability and eager validation of every historical execution record are no longer runtime requirements.

Durable state includes canonical sources/revisions and required evidence, current graph and Candidate state, pending effects, article/ACK records, idempotency identities, current route blockers and provider accounting. Creation and transitions remain transactional. Selected business reads and effects verify their actual dependencies. A corrupt current record must not grant an effect; unrelated obsolete history must not prevent startup.

Native OPEN checks schema, connection/writer ownership, current heads and pending recovery. Whole-history physical, foreign-key, semantic and CAS sweeps remain explicit maintenance. CAS content is verified when rehydrated for use; pending installation/deletion recovery remains durable.

Revision continuation uses current rows and content-addressed retrieval pairs. An explicit initial import validates the legacy representation once. Normal boot never replays diagnostic history or silently reconstructs missing current state from old logs. Usage admission reads current invocation blockers; settlement validation occurs when those facts are created or changed, rather than re-proving unchanged historical source/CLI records for every call.

Ordinary cycle/progress diagnostics use a bounded asynchronous file sink. Full queues, unavailable files and expired diagnostics do not block work. Business proof records required for provider recovery and ACK remain durable. Obsolete diagnostic payloads may be cleared while retained ledger identities and business pins are preserved.

## Consequences

Acceptance tests now distinguish current-state corruption from unavailable diagnostic history. Historical-only replay guarantees are not ordinary runtime gates. The development operator may expire old diagnostics within this contract without per-batch approval.

The cutover is an explicit maintenance operation with current-state, source, pending, accounting and ACK read-back. Once current-only transitions exist, recovery uses compatible current-state readers; an old history-replay binary is not a valid rollback. No public publication or platform migration is enabled by this decision.

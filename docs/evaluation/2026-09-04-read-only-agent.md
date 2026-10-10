Checked published #895/#916 claims against current origin/main 044bc29e2159c40fc5ff083bca356cf9dac17136 (tree f0feb080f8460258117ca0002fc938117510a263) and local artefacts under /Users/jamesto/Coding/newsroom-evidence/issue-895/. No live F4 process was observed. Evidence was not rerun.

1. Published “latest runtime report” is stale versus local exact-main packet

Reported in the #895 body and comments 5531444458 / 5546351957: the latest fetched runtime report is 3 September on 4b7bf781…, last event 8fab380a… / ledger 1600 held, PRE151_GRAPHITI_STEADY_STATE_READY=false.

Observed locally, unpublished on GitHub:

┌──────────┬────────────────────────────────────────────────────────────────┐
│ Field    │ Value                                                          │
├──────────┼────────────────────────────────────────────────────────────────┤
│ Artefact │ /Users/jamesto/Coding/newsroom-evidence/issue-895/exact-main-  │
│          │ evaluator-044bc29e2159c40fc5ff083bca356cf9dac17136/graphiti    │
│          │ -steady-state-                                                 │
│          │ 0e587d27cc738da8cbf94e882644822718381f3eac0ba48dc09335a92a5294 │
│          │ f0                                                             │
│          │ .json                                                          │
├──────────┼────────────────────────────────────────────────────────────────┤
│ observed │ 2026-09-04T15:35:18.637456Z                                    │
│ _at      │                                                                │
├──────────┼────────────────────────────────────────────────────────────────┤
│ code_    │ 044bc29e… / f0feb080…                                          │
│ identity │                                                                │
├──────────┼────────────────────────────────────────────────────────────────┤
│ Canonica │ sha256:0e587d27cc738da8cbf94e882644822718381f3eac0ba48dc09335a │
│ l        │ 92a5294f0                                                      │
│ packet_  │                                                                │
│ digest   │                                                                │
├──────────┼────────────────────────────────────────────────────────────────┤
│ File     │ 9607827e2560e3a40eeb38ea8829d47cd09cdb493e7b8cfd7fb6c176f0c4e8 │
│ SHA-256  │ f3                                                             │
├──────────┼────────────────────────────────────────────────────────────────┤
│ verdict  │ READY_FOR_OWNER_DECISION                                       │
├──────────┼────────────────────────────────────────────────────────────────┤
│ readines │ F4_CAMPAIGN_READY_FOR_OWNER_DECISION                           │
│ s        │                                                                │
├──────────┼────────────────────────────────────────────────────────────────┤
│ blockers │ []                                                             │
├──────────┼────────────────────────────────────────────────────────────────┤
│ campaign │ false                                                          │
│ _        │                                                                │
│ authoris │                                                                │
│ ed       │                                                                │
├──────────┼────────────────────────────────────────────────────────────────┤
│ caps.    │ 212                                                            │
│ total.   │                                                                │
│ events / │                                                                │
│ selected │                                                                │
│ cohort   │                                                                │
├──────────┼────────────────────────────────────────────────────────────────┤
│ Ramp     │ [1, 10, 212]                                                   │
│ limits   │                                                                │
└──────────┴────────────────────────────────────────────────────────────────┘

Exit marker exit-status.json records exit_code=0, provider_calls=0, finished 2026-09-04T15:36:01Z. .execution-claimed is an empty leftover directory, not a live claim.

This packet is not PRE151_GRAPHITI_STEADY_STATE_READY (objectives_are_prospective=true). It does not retire #916.

The 3 September 72f2f504 stop matches its local terminal evidence (not a conflict with this finding):

• f4-campaign-72f2f504…/terminal-evidence-manifest.json SHA-256 969e3ef1ebc9a593119354e15dd65d82ac0e886505dfa4578ebd59dee874b34b
• packet digest sha256:72f2f504…, return code 2, event 8fab380a… ledger 1600 RETRY_HELD, receipt sha256:c64e9476…, reserved 500000 / actual 0
• that event is absent from the 212-member 044bc29e cohort

UNOBSERVED: GitHub comments do not contain packet 0e587d27…. Cumulative programme reservation 2000000 for five starts was not fully reconstructed here (only the fifth campaign’s 500000 reservation was in that terminal manifest).

2. #916 packet-path coverage is missing; the helper still heals ramps

Reported in #916 (helper-level reproduction, explicitly not the repository suite): input checkpoints [1, 2, 20] become [1, 10, 15]; later-phase extra conditions dropped; empty/invalid ramps replaced.

Observed in current main, not re-executed:

def _campaign_ramp_for_event_count(...):
    # copies only template_phases[0] conditions
    # rebuilds limits via campaign_event_limits → (1, min(10, N), N)
...
        "ramp": _campaign_ramp_for_event_count(
            selected_event_count, template_phases=template_phases
        ),

Caller always narrows when a cohort exists:

    if derived_event_ids:
        campaign = _narrow_campaign_input_to_selected_cohort(
            campaign, selected_event_count=len(derived_event_ids)
        )

Fixture-only / already gated (Focus run 33850961092, local junit 177 tests, 0 failures; observed remote conclusion success):

┌──────────────────────────────┬────────────────────┬───────────────────────┐
│ Test                         │ What it covers     │ What it does not      │
├──────────────────────────────┼────────────────────┼───────────────────────┤
│ test_bootstrap_overcount     │ 3→2 default ramp   │ 213→212; checkpoint 2 │
│ _narrows_to_selected_cohort  │ becomes [1, 2]     │ vs 10; later-phase    │
│ _after_retry_held_exclusion  │                    │ conditions            │
├──────────────────────────────┼────────────────────┼───────────────────────┤
│ test_undercounted_campaign   │ undercount stays   │ n/a                   │
│ _cap_does_not_widen_to_      │ NO_GO              │                       │
│ selected_cohort              │                    │                       │
├──────────────────────────────┼────────────────────┼───────────────────────┤
│ test_fallback_stop_and_ramp_ │ duplicate phase-1  │ narrowing does not    │
│ contracts_fail_closed        │ entry conditions   │ run (supplied count   │
│                              │ fail closed        │ equals selected)      │
├──────────────────────────────┼────────────────────┼───────────────────────┤
│ test_campaign_event_limits_  │ limit collapse     │ extra/malformed phase │
│ collapse_to_unique_          │ (1,), (1,2),       │ conditions            │
│ increasing_bounds            │ (1,10), (1,10,25)  │                       │
└──────────────────────────────┴────────────────────┴───────────────────────┘

Missing coverage relative to #916 acceptance (no matching test names in newsroom/tests/test_graphiti_steady_state.py or test_graphiti_operational_readiness.py):

• malformed/empty ramp remains rejected on the packet-builder path when caps.total.events > selected count
• valid narrower intermediate checkpoint 2 is not rewritten to 10
• a later-phase extra condition is preserved or the packet is explicit NO_GO, never dropped
• inexpensive 213→212 default-ramp shape (live packet ramp is [1,10,212], but no test pins that shape)

The live 044bc29e packet has bootstrap candidate_event_count=212 already, so that READY result is not live proof of the overcount-healing path.
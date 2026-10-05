"""Source-bound display-name equivalence; never infer actors or geography.

These are linguistic identities, not new Source facts. Government bodies and
speakers remain literal: publisher metadata does not establish a policy actor.
"""
from .evidence import bounded_named_entities

VERSION = 'newsroom.native-story-place-names.v1'
# Unicode CLDR zh-Hant territory names. Only an already admitted typed Source
# name can select an equivalence class. No network, dependency or global alias
# admission is involved: https://www.unicode.org/cldr/charts/45/summary/zh_Hant.html
_PLACE_NAMES = (frozenset({'UK', '英國'}), frozenset({'Hong Kong', '香港'}))


def story_entity_names_are_bound(text, claims):
    approved = {name for claim in claims for name in claim.named_entities}
    typed = {name for claim in claims for name, kind, _ in claim.named_entity_evidence
             if kind == 'PLACE'}
    for names in _PLACE_NAMES:
        if typed & names:
            approved.update(names)
    return {name for name, _ in bounded_named_entities(text)} <= approved


def declared_source_speakers_are_retained(claims, links):
    """A publisher must not replace an explicitly named quoted Source speaker."""
    from .native_source_context_ranges import declared_speakers

    for claim in claims:
        linked = '\n'.join(link.rendered_assertion for link in links
                           if link.governed_claim_id == claim.claim_id)
        speakers = declared_speakers(claim.claim, tuple((name, kind)
            for name, kind, _ in claim.named_entity_evidence))
        if any(name not in linked for name in speakers):
            return False
    return True

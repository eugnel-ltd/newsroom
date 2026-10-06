"""Source-bound display-name equivalence; never infer actors or geography.

These are linguistic identities, not new Source facts. Policy actors and quoted
Source speakers remain literal. Qualified publisher reporting labels may be
linguistically equivalent; publisher metadata never establishes a policy actor.
"""
import re

from .evidence import bounded_named_entities

VERSION = 'newsroom.native-story-source-names.v2'
# Unicode CLDR zh-Hant territory names. Only an already admitted typed Source
# name can select an equivalence class. No network, dependency or global alias
# admission is involved: https://www.unicode.org/cldr/charts/45/summary/zh_Hant.html
_PLACE_NAMES = (frozenset({'UK', '英國'}), frozenset({'Hong Kong', '香港'}))


# Official GOV.UK bilingual naming; selected only by exact governed publisher
# attribution, never a feed anchor or permission to perform a policy action.
# https://www.gov.uk/government/news/311079.zh-tw
_PUBLISHER_REPORTING_NAMES = ('Home Office', '英國內政部')
_REPORTING_VERB = re.compile(r"\s*(?:表示|指出|稱)\s*[,，:：]")
_REPORTING_START = re.compile(r"(?:^|[\n。！？!?])\s*$")


def _outside_quotation(text, position):
    closes = {'「': '」', '『': '』', '“': '”', '‘': '’'}
    stack = []
    for index, char in enumerate(text[:position]):
        if char == "'":
            before = text[index - 1] if index else ''
            after = text[index + 1] if index + 1 < len(text) else ''
            # Contractions/possessives are not quotation boundaries. A closing
            # quote is still honoured when its matching quote is already open.
            if before.isascii() and before.isalpha() and after.isascii() and after.isalpha():
                continue
            if stack and stack[-1] == "'":
                stack.pop()
            elif before.isascii() and before.isalpha():
                continue
            else:
                stack.append("'")
        elif char == '"':
            if stack and stack[-1] == '"':
                stack.pop()
            else:
                stack.append('"')
        elif char in closes:
            stack.append(closes[char])
        elif stack and char == stack[-1]:
            stack.pop()
    return not stack


def source_publisher_reporting_names(text, claims):
    """Return only observed, unquoted publisher-attribution display names.

    Every occurrence must be reporting attribution: one allowed mention cannot
    turn later occurrences into policy actors or quoted Source speakers.
    """
    if not any(getattr(claim, 'attribution', None) == 'Home Office' for claim in claims):
        return set()
    allowed = set()
    for name in _PUBLISHER_REPORTING_NAMES:
        occurrences = tuple(re.finditer(re.escape(name), text))
        if occurrences and all(
                _REPORTING_START.search(text[:match.start()]) is not None
                and _REPORTING_VERB.match(text, match.end()) is not None
                and _outside_quotation(text, match.start())
                for match in occurrences):
            allowed.add(name)
    return allowed


def story_entity_names_are_bound(text, claims):
    claims = tuple(claims)
    approved = {name for claim in claims for name in claim.named_entities}
    typed = {name for claim in claims for name, kind, _ in claim.named_entity_evidence
             if kind == 'PLACE'}
    for names in _PLACE_NAMES:
        if typed & names:
            approved.update(names)
    approved.update(source_publisher_reporting_names(text, claims))
    # The general scanner can see only the country inside a Chinese actor name;
    # do not let admitted UK turn an unqualified organisation into a policy actor.
    if any(name in text and name not in approved for name in _PUBLISHER_REPORTING_NAMES):
        return False
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

"""Complete canonical inputs from the two approved weather feed/API endpoints.

RSS remains an RSS observation, never a claimed full maintained warning page.
The HKO response is the complete current summary, not an inferred activation
history. An empty, valid inventory is recorded explicitly rather than as a
parser failure or a fabricated warning. Complete HKO state revisions also
preserve disappearances, separately from source-timestamped warning records.
"""
from __future__ import annotations

import json
from urllib.parse import urlsplit

from lxml import etree

from newsroom.authority.canonical import canonical_json_bytes, digest_bytes
from newsroom.increment9.proving import MAX_BODY_BYTES, SOURCE_URLS
from .govuk_evidence import _instant, _unique_object, _utc
from .items import SourceItem, parse_source_time
from .veto import VetoError


def weather_items(source_id: str, raw: bytes, *, observed_at) -> tuple[SourceItem, ...]:
    if source_id == "HK-02":
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
        if type(value) is not dict:
            raise ValueError("HKO warning summary must be a complete object")
        items = []
        for key, warning in sorted(value.items()):
            if type(warning) is not dict or not key or not all(
                type(warning.get(name)) is str and warning[name]
                for name in ("name", "code", "actionCode", "issueTime", "updateTime")
            ):
                raise ValueError("HKO warning record is incomplete")
            issued, updated = _instant(warning["issueTime"]), _instant(warning["updateTime"])
            if issued > updated or updated > observed_at:
                raise ValueError("HKO warning time differs")
            if "expireTime" in warning and warning["expireTime"] and _instant(warning["expireTime"]) < issued:
                raise ValueError("HKO warning expiry differs")
            body = canonical_json_bytes({key: warning}).decode()
            items.append(SourceItem(source_id, key, warning["name"], body,
                                    SOURCE_URLS[source_id], _utc(issued), _utc(updated), body))
        # A per-warning inventory alone drops disappearance/cancellation. Keep
        # the complete state as one stable item, including {}, without inventing
        # a source activation/update time from our observation clock.
        body = canonical_json_bytes(value).decode()
        items.append(SourceItem(
            source_id, "current-warning-summary", "Hong Kong Observatory complete current warning summary",
            body, SOURCE_URLS[source_id], None, None, body,
        ))
        return tuple(items)
    if source_id != "UK-10":
        raise ValueError("source is outside the fixed weather portfolio")
    root = etree.fromstring(raw, etree.XMLParser(resolve_entities=False, no_network=True, recover=False))
    if root.tag != "rss" or root.getroottree().docinfo.doctype or len(root.findall("channel")) != 1:
        raise ValueError("Met Office RSS inventory differs")
    channel = root.find("channel")
    if channel.findtext("title") != "Met Office warnings for UK":
        raise ValueError("Met Office RSS identity differs")
    items, keys = [], set()
    for entry in channel.findall("item"):
        title, link = entry.findtext("title"), entry.findtext("link")
        key = entry.findtext("guid") or link
        published = parse_source_time(entry.findtext("pubDate") or "")
        if (not title or not link or not key or key in keys or not published
                or urlsplit(link).scheme != "https"
                or urlsplit(link).netloc not in {"www.metoffice.gov.uk", "weather.metoffice.gov.uk"}
                or _instant(published) > observed_at):
            raise ValueError("Met Office warning record is incomplete")
        keys.add(key)
        # Retain every entry field, not the clipped drafting summary parser.
        body = etree.tostring(entry, encoding="unicode", with_tail=False)
        items.append(SourceItem(source_id, key, title, body, link, published, None, body))
    return tuple(items)


def poll_other_source(intake, source_id, definition_id, version_id, version):
    """NativeSourceIntake callback; reuse its canonical retain/hydration path."""
    from .native_source_intake import NativeSourceDisposition
    url = SOURCE_URLS[source_id]
    rights = intake._licence.for_source(source_id=source_id, definition_url=url)
    observations = list(intake._licence.observations_for(source_id))
    if rights.decision != "PERMITTED":
        return NativeSourceDisposition(source_id, "HOLD", intake._licence.reason_for(source_id),
                                       observations=tuple(observations))
    if source_id not in {"HK-02", "UK-10"}:
        raise ValueError("observed source rights exceed the implemented transport")
    with intake._fence(source_id, url):
        status, raw = intake._fetch(url)
    if status != 200 or not raw or len(raw) > MAX_BODY_BYTES:
        return NativeSourceDisposition(source_id, "HOLD", "SOURCE_FETCH_INCOMPLETE", observations=tuple(observations))
    observed = intake._clock()
    admission, access = intake._admit_observation(source_id, raw, url=url)
    observations.append((url, digest_bytes(raw), str(admission), str(access.access_decision_id)))
    try:
        items = weather_items(source_id, raw, observed_at=observed)
    except (ValueError, TypeError, KeyError, UnicodeError, etree.XMLSyntaxError):
        return NativeSourceDisposition(source_id, "HOLD", "WEATHER_INVENTORY_INCOMPLETE",
                                       observations=tuple(observations))
    units, holds = [], []
    for item in items:
        try:
            units.extend(intake._retain_item(source_id, definition_id, version_id, version,
                                            item, digest_bytes(raw), _utc(observed), rights.record_id))
        except VetoError:
            raise
        except Exception as exc:
            holds.append((item.canonical_url, getattr(exc, "reason_code", "SOURCE_ITEM_RETAIN_FAILED")))
    return NativeSourceDisposition(
        source_id, "HOLD" if holds else "READY",
        "SOURCE_ITEMS_HELD" if holds else "GOVERNED_REVISIONS_RETAINED" if items else "NO_ACTIVE_WARNINGS_OBSERVED",
        tuple(units), str(admission), str(access.access_decision_id),
        tuple(observations), tuple(holds),
    )

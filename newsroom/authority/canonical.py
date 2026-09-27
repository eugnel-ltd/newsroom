from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

import orjson

MIN_SAFE_INTEGER = -9_007_199_254_740_991
MAX_SAFE_INTEGER = 9_007_199_254_740_991
DIGEST_ALGORITHM = "sha256"
_SURROGATE = re.compile(r"[\ud800-\udfff]")


class CanonicalizationError(ValueError):
    """Raised when a value cannot enter the canonical authority domain."""


def _validate_string(value: str, path: str) -> None:
    # ASCII cannot contain a surrogate. Use the built-in implementation even
    # for str subclasses, then retain the existing Unicode/error-path checks.
    if str.isascii(value):
        return
    for character in value:
        if 0xD800 <= ord(character) <= 0xDFFF:
            raise CanonicalizationError(f"lone surrogate is unsupported at {path}")


def _is_restricted_builtin(value: Any) -> bool:
    """Check ordinary JSON values without building unused diagnostic paths.

    Subclasses/custom collections use the existing diagnostic validator. No
    accepted type or value rule is added here; False means validate normally.
    """

    kind = type(value)
    if kind is str:
        return value.isascii() or _SURROGATE.search(value) is None
    if value is None or kind is bool:
        return True
    if kind is int:
        return MIN_SAFE_INTEGER <= value <= MAX_SAFE_INTEGER
    if kind is dict:
        for key, item in value.items():
            if type(key) is not str or not (
                key.isascii() or _SURROGATE.search(key) is None
            ):
                return False
            if not _is_restricted_builtin(item):
                return False
        return True
    if kind is list or kind is tuple:
        for item in value:
            if not _is_restricted_builtin(item):
                return False
        return True
    return False


def _validate_restricted_value(value: Any, path: str = "$") -> bool:
    """Validate fully and identify trees composed only of exact built-ins."""
    builtin = _is_restricted_builtin(value)
    if not builtin:
        _validate_restricted_value_with_path(value, path)
    return builtin


def _validate_restricted_value_with_path(value: Any, path: str = "$") -> None:
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, int):
        if not MIN_SAFE_INTEGER <= value <= MAX_SAFE_INTEGER:
            raise CanonicalizationError(
                f"integer outside the interoperable safe range at {path}"
            )
        return
    if isinstance(value, float):
        raise CanonicalizationError(f"floating-point values are unsupported at {path}")
    if isinstance(value, str):
        _validate_string(value, path)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalizationError(f"object names must be strings at {path}")
            _validate_string(key, f"{path}.<key>")
            _validate_restricted_value_with_path(item, f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        for index, item in enumerate(value):
            _validate_restricted_value_with_path(item, f"{path}[{index}]")
        return
    raise CanonicalizationError(
        f"unsupported value type at {path}: {type(value).__name__}"
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON for the restricted authority domain."""

    if _validate_restricted_value(value):
        try:
            return orjson.dumps(
                value, option=orjson.OPT_SORT_KEYS | orjson.OPT_STRICT_INTEGER,
            )
        except orjson.JSONEncodeError:
            # A valid tree may exceed the native encoder's depth limit. Keep
            # the established stdlib result/error, as for custom collections.
            pass
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError, UnicodeError) as exc:
        raise CanonicalizationError(f"canonical JSON encoding failed: {exc}") from exc
    return text.encode("utf-8", errors="strict")


def digest_bytes(data: bytes) -> str:
    if not isinstance(data, bytes):
        raise CanonicalizationError("digest input must be immutable bytes")
    return f"{DIGEST_ALGORITHM}:{hashlib.sha256(data).hexdigest()}"


def digest_canonical(value: Any) -> str:
    return digest_bytes(canonical_json_bytes(value))


def validate_sha256_digest(value: str, *, field: str = "digest") -> str:
    if not isinstance(value, str) or value != value.strip().lower():
        raise CanonicalizationError(f"{field} must be lowercase canonical text")
    if not value.startswith("sha256:"):
        raise CanonicalizationError(f"{field} must use sha256:<hex>")
    hexadecimal = value.removeprefix("sha256:")
    if len(hexadecimal) != 64:
        raise CanonicalizationError(f"{field} must contain 64 hexadecimal characters")
    try:
        int(hexadecimal, 16)
    except ValueError as exc:
        raise CanonicalizationError(f"{field} contains non-hexadecimal characters") from exc
    return value

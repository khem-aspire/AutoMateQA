"""
path_template — URL normalization + match predicate for API assertions.

At record time, `normalize_url_to_template` converts a concrete URL's path
into a template by replacing numeric ids, UUIDs, Mongo ObjectIds, and ISO
dates with placeholders.

At playback time, `path_matches_template` checks whether a live URL's path
matches the recorded template (placeholder segments match any non-empty
segment; literal segments must match exactly).
"""

from __future__ import annotations

import re
from urllib.parse import urlparse, parse_qs

_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^\d{4}-\d{2}-\d{2}$"), "{date}"),
    (re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"), "{uuid}"),
    (re.compile(r"^[0-9a-fA-F]{24}$"), "{oid}"),
    (re.compile(r"^\d+$"), "{id}"),
]

_PLACEHOLDER_RE = re.compile(r"^\{[a-zA-Z_][a-zA-Z0-9_]*\}$")


def normalize_url_to_template(url: str) -> str:
    """Convert a URL's path into a path template with dynamic-segment placeholders."""
    path = urlparse(url).path or "/"
    parts = path.split("/")
    normalized: list[str] = []
    for seg in parts:
        if not seg:
            normalized.append(seg)
            continue
        replaced = seg
        for pattern, placeholder in _RULES:
            if pattern.fullmatch(seg):
                replaced = placeholder
                break
        normalized.append(replaced)
    return "/".join(normalized)


def query_keys_of(url: str) -> list[str]:
    """Return sorted unique query-parameter names for a URL."""
    q = urlparse(url).query
    if not q:
        return []
    return sorted(parse_qs(q, keep_blank_values=True).keys())


def path_matches_template(path: str, template: str) -> bool:
    """True iff `path` matches `template` segment-by-segment."""
    ps, ts = path.split("/"), template.split("/")
    if len(ps) != len(ts):
        return False
    for p, t in zip(ps, ts):
        if _PLACEHOLDER_RE.fullmatch(t):
            if p == "":
                return False
            continue
        if p != t:
            return False
    return True

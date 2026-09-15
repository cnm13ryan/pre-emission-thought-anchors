"""Released sentence segmentation and strict sequential character alignment."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Span:
    text: str
    start: int
    end: int
    method: str


def _ws_pattern(text: str) -> str:
    return "".join(r"\s+" if p.isspace() else re.escape(p)
                   for p in re.split(r"(\s+)", text) if p)


def align_sequential(trace: str, chunks: list[str], start: int = 0) -> list[Span]:
    """Align each released chunk after the previous match; never global-replace."""
    spans: list[Span] = []
    cursor = start
    for chunk in chunks:
        exact = trace.find(chunk, cursor)
        if exact >= 0:
            spans.append(Span(chunk, exact, exact + len(chunk), "exact"))
            cursor = exact + len(chunk)
            continue
        match = re.search(_ws_pattern(chunk), trace[cursor:])
        if match is None:
            raise ValueError(f"Unresolved chunk at cursor {cursor}: {chunk[:80]!r}")
        begin = cursor + match.start()
        end = cursor + match.end()
        spans.append(Span(trace[begin:end], begin, end, "whitespace_tolerant"))
        cursor = end
    if any(a.end > b.start for a, b in zip(spans, spans[1:], strict=False)):
        raise ValueError("Non-monotonic overlapping spans")
    return spans


def span_v2(trace: str) -> list[Span]:
    """Conservative sentence splitter retaining terminal fragments."""
    spans = []
    cursor = 0
    for match in re.finditer(r"(?<=[.!?])(?=\s+)|\n\s*\n+", trace):
        end = match.start()
        if trace[cursor:end].strip():
            left = cursor + len(trace[cursor:end]) - len(trace[cursor:end].lstrip())
            right = cursor + len(trace[cursor:end].rstrip())
            spans.append(Span(trace[left:right], left, right, "span_v2"))
        cursor = match.end()
    if trace[cursor:].strip():
        left = cursor + len(trace[cursor:]) - len(trace[cursor:].lstrip())
        spans.append(Span(trace[left:len(trace.rstrip())], left, len(trace.rstrip()), "span_v2"))
    return spans

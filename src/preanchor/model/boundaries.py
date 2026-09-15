"""Once-tokenised trace boundaries (SPEC §3.8, §10.2): CPU-only, tokenizer-agnostic."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

from preanchor.config import config_hash


@dataclass(frozen=True)
class Boundary:
    sentence_id: str
    sentence_idx: int
    char_start: int
    char_end: int
    target_token_start: int
    boundary_token_idx: int
    target_token_end: int  # exclusive: first token starting at or after char_end
    overlapping_token: bool  # the target's first token starts before char_start
    prefix_token_count: int  # tokens strictly before the target == boundary_token_idx + 1


def tokenize_trace(tokenizer, full_cot: str) -> tuple[list[int], list[tuple[int, int]]]:
    """Tokenise the exact stored trace once with the upstream special-token behaviour."""
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Pinned fast tokenizer with offset mappings required")
    encoded = tokenizer(full_cot, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(encoded["input_ids"])
    offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"]]
    if tokenizer.decode(ids) != full_cot:
        raise ValueError("Tokenisation does not round-trip the stored trace")
    for (a, b), (c, _) in zip(offsets, offsets[1:], strict=False):
        if a > b or b > c:
            raise ValueError("Offset mapping is not monotone")
    return ids, offsets


def boundary_for(offsets: list[tuple[int, int]], char_start: int,
                 char_end: int) -> tuple[int, int, int, bool]:
    if not 0 <= char_start < char_end <= offsets[-1][1]:
        raise ValueError("Character span outside the trace")
    target_token_start = next(k for k, (_, end) in enumerate(offsets) if end > char_start)
    boundary_token_idx = target_token_start - 1
    if boundary_token_idx < 0:
        raise ValueError("boundary_token_idx < 0: sentence starts at the first token")
    target_token_end = next((k for k, (start, _) in enumerate(offsets) if start >= char_end),
                            len(offsets))
    return (target_token_start, boundary_token_idx, target_token_end,
            offsets[target_token_start][0] < char_start)


def boundary_table(offsets: list[tuple[int, int]], sentences: list[dict]) -> list[Boundary]:
    """`sentences` are Stage 3 rows of one trajectory; output is ordered by sentence_idx."""
    rows = sorted(sentences, key=lambda r: int(r["sentence_idx"]))
    if [int(r["sentence_idx"]) for r in rows] != list(range(len(rows))):
        raise ValueError("Sentence indices must be contiguous from zero")
    table = []
    for row in rows:
        start, boundary, end, overlap = boundary_for(offsets, int(row["char_start"]),
                                                     int(row["char_end"]))
        table.append(Boundary(row["sentence_id"], int(row["sentence_idx"]),
                              int(row["char_start"]), int(row["char_end"]), start, boundary,
                              end, overlap, boundary + 1))
    return table


def audit_boundaries(ids: list[int], offsets: list[tuple[int, int]],
                     table: list[Boundary]) -> dict:
    """SPEC §10.2 structural checks; raises on the first violation."""
    for b in table:
        if any(offsets[k][1] > b.char_start for k in range(b.target_token_start)):
            raise ValueError(f"Prefix token ends after the target boundary: {b.sentence_id}")
        if offsets[b.target_token_start][1] <= b.char_start:
            raise ValueError(f"Target character before target_token_start: {b.sentence_id}")
        if ids[: b.boundary_token_idx + 1] != ids[: b.prefix_token_count]:
            raise ValueError("Prefix ids are not the full-trace prefix")
    return {"schema_version": 1, "n_tokens": len(ids), "n_sentences": len(table),
            "overlapping_tokens": sum(b.overlapping_token for b in table),
            "min_boundary_token_idx": min((b.boundary_token_idx for b in table), default=None),
            "max_boundary_token_idx": max((b.boundary_token_idx for b in table), default=None),
            "token_ids_sha256": token_ids_sha256(ids),
            "boundary_mapping_sha256": boundary_mapping_sha256(table)}


def token_ids_sha256(ids: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()


def boundary_mapping_sha256(table: list[Boundary]) -> str:
    return config_hash([[b.sentence_idx, b.target_token_start, b.boundary_token_idx]
                        for b in table])


def boundary_records(table: list[Boundary]) -> list[dict]:
    return [asdict(b) for b in table]

"""Final-answer parsing: the upstream v1 rule (verbatim copy) and the repaired v2 rule.

v1 provenance: ``extract_boxed_answers`` and ``normalize_latex`` are copied without
behaviour changes from thought-anchors ``utils.py`` at
b53ed8c75d3f6112f68adfaec9a13d4d708c442e (via the MVP port
``src/preanchor/labels/upstream_answers.py`` at commit f11823f).

MIT License

Copyright (c) 2025 interp-reasoning

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

v2 (this study): known v1 defects and their repairs:
1. v1 takes the FIRST box anywhere, so an intermediate boxed value can beat the final
   answer. v2 takes the LAST balanced box in the answer region, which is the text after
   the last ``</think>`` when present, otherwise the whole completion.
2. v1 accepts an unterminated box (``\\boxed{12`` at a truncation point) as an answer.
   v2 gives it status ``unterminated_box``.
3. v1 maps every failure to one ``<NO_ANSWER>``. v2 keeps the reason: ``length_truncated``,
   ``unterminated_box``, ``unboxed_stated``, ``no_answer``, ``empty``.
4. ``normalize_latex`` is not idempotent (nested braces). v2 iterates it to a fixed point
   and canonicalises plain numbers (``6.40`` → ``6.4``, ``1,000`` → ``1000``; v1 turns
   ``1,000`` into the decimal ``1.000``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ----------------------------------------------------------------------------- v1


def extract_boxed_answers(text: str) -> list[str]:
    """Upstream v1, verbatim behaviour."""
    boxed_starts = [m.start() for m in re.finditer(r"\\boxed\{", text)]
    if not boxed_starts:
        return [""]
    answers = []
    for start_idx in boxed_starts:
        idx = start_idx + 7
        brace_count = 1
        answer = ""
        while idx < len(text) and brace_count > 0:
            char = text[idx]
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    break
            if brace_count > 0:
                answer += char
            idx += 1
        if answer:
            answers.append(answer)
    return answers if answers else [""]


def normalize_latex(latex_str: str) -> str:
    """Upstream v1, verbatim behaviour."""
    normalized = latex_str.strip().lower()
    normalized = normalized.replace("dfrac", "frac")
    normalized = normalized.replace("tfrac", "frac")
    normalized = re.sub(r"\s+", "", normalized)
    normalized = normalized.replace("\\%", "")
    normalized = normalized.replace("{,}", "")
    normalized = normalized.replace("\\times", "*")
    normalized = normalized.replace("\\cdot", "*")
    normalized = re.sub(r"(\d+)[\.,](\d+)", r"\1.\2", normalized)
    normalized = re.sub(r"{([^{}]+)}", r"\1", normalized)
    normalized = normalized.replace("\\pi", "pi")
    normalized = re.sub(r"\\text\{([^{}]+)\}", r"\1", normalized)
    normalized = re.sub(r"\\mathrm\{([^{}]+)\}", r"\1", normalized)
    normalized = re.sub(r"([a-z]+)\\+\s*(\d+)", r"\1\2", normalized)
    normalized = normalized.replace("\\text", "")
    return normalized


V1_MISSING = "<NO_ANSWER>"


def category_v1(stored_answer: object) -> str:
    """MVP canonical category of a stored ``answer`` field (missing → <NO_ANSWER>)."""
    if not isinstance(stored_answer, str) or stored_answer in ("", "None"):
        return V1_MISSING
    return normalize_latex(stored_answer) or V1_MISSING


def answer_v1_from_text(continuation: str) -> str:
    """MVP rule applied to raw text: first box, normalised."""
    return normalize_latex(extract_boxed_answers(continuation)[0]) or V1_MISSING


# ----------------------------------------------------------------------------- v2

TRUNCATED = "<TRUNCATED>"
NON_ANSWER = "<NON_ANSWER>"
_STATED = re.compile(
    r"(?:\*\*\s*(?:final\s+)?answer\s*:?\s*\*\*\s*:?|(?:the\s+)?final\s+answer\s+is\s*:?|"
    r"\banswer\s*:)\s*(.+?)\s*$", re.IGNORECASE | re.DOTALL)
_NUM = re.compile(r"^[+-]?(\d{1,3}(,\d{3})+|\d+)(\.\d+)?$")


@dataclass(frozen=True)
class Parsed:
    status: str          # answered | unboxed_stated | unterminated_box | length_truncated
    #                      | no_answer | empty
    answer: str | None   # normalised answer when status in {answered, unboxed_stated}


def balanced_boxes(text: str) -> tuple[list[str], bool]:
    """All balanced ``\\boxed{...}`` contents in order, and whether an unterminated one exists."""
    out, unterminated = [], False
    for m in re.finditer(r"\\boxed\s*\{", text):
        idx, depth, buf = m.end(), 1, []
        while idx < len(text) and depth:
            ch = text[idx]
            depth += (ch == "{") - (ch == "}")
            if depth:
                buf.append(ch)
            idx += 1
        if depth:
            unterminated = True
        else:
            out.append("".join(buf))
    return out, unterminated


_THOUSANDS = re.compile(r"^\s*[+-]?\d{1,3}(,\d{3})+(\.\d+)?\s*$")


def normalize_v2(raw: str) -> str:
    # v1 rewrites "1,000" as the decimal "1.000"; drop thousands separators first.
    value = raw.replace(",", "") if _THOUSANDS.match(raw) else raw
    for _ in range(10):   # fixed point of the v1 normaliser
        nxt = normalize_latex(value)
        if nxt == value:
            break
        value = nxt
    value = value.replace("\\left", "").replace("\\right", "").replace("\\!", "")
    value = value.strip().strip("$").rstrip(".").strip()
    if _NUM.match(value):
        number = value.replace(",", "")
        if "." in number:
            number = number.rstrip("0").rstrip(".")
        value = number.lstrip("+") or "0"
    return value


def parse_v2(continuation: str | None, *, finish_reason: str | None = None,
             token_count: int | None = None, cap: int = 16384, margin: int = 64) -> Parsed:
    """Repaired parser. ``finish_reason='length'`` or ``token_count >= cap - margin``
    identifies truncation when there is no complete answer."""
    if not isinstance(continuation, str) or not continuation.strip():
        return Parsed("empty", None)
    region = continuation.rsplit("</think>", 1)[1] if "</think>" in continuation \
        else continuation
    boxes, unterminated = balanced_boxes(region)
    boxes = [b for b in boxes if b.strip()]
    if boxes:
        value = normalize_v2(boxes[-1])
        if value:
            return Parsed("answered", value)
    truncated = finish_reason == "length" or (token_count is not None
                                              and token_count >= cap - margin)
    if truncated:
        return Parsed("length_truncated", None)
    if unterminated:
        return Parsed("unterminated_box", None)
    tail = region[-400:]
    m = _STATED.search(tail)
    if m:
        stated = m.group(1).strip().splitlines()[0] if m.group(1).strip() else ""
        stated = re.sub(r"^\\\(|\\\)$", "", stated).strip()
        stated = re.sub(r"\s*(?:units?|cm|feet|ft|inches|degrees)\.?$", "", stated,
                        flags=re.IGNORECASE)
        value = normalize_v2(stated) if stated else ""
        if value and len(value) <= 40:
            return Parsed("unboxed_stated", value)
    return Parsed("no_answer", None)


def category_v2(parsed: Parsed, *, include_unboxed: bool = True) -> str:
    """Outcome category for distribution comparisons (primary handling, PROVISIONAL).

    Answers are their normalised value; truncation is its own outcome; every other
    non-answer is ``<NON_ANSWER>``. ``include_unboxed=False`` is the sensitivity that
    treats prose answers as non-answers.
    """
    if parsed.status == "answered":
        return parsed.answer  # type: ignore[return-value]
    if parsed.status == "unboxed_stated":
        return parsed.answer if include_unboxed else NON_ANSWER  # type: ignore[return-value]
    if parsed.status == "length_truncated":
        return TRUNCATED
    return NON_ANSWER

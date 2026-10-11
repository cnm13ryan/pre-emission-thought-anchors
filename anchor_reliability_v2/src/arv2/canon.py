"""Shared answer canonicaliser for the v3 follow-up (EXPLORATORY, post hoc).

Why: ``answers.normalize_v2`` groups by a cleaned *string*, and the v1 step it wraps strips
braces, so ``\\frac{5}{12}`` becomes ``\\frac512`` and equal values written differently
(``0.5``, ``\\frac12``, ``1/2``, ``50\\%``-free forms, ``5 cm``) can fall into different
categories. This module maps a RAW answer surface (the boxed string, before any v1/v2
normalisation) to a canonical key:

1. strip presentation: ``$``, ``\\left``/``\\right``, spacing commands, ``\\displaystyle``,
   ``\\text{..}``/``\\mathrm{..}`` wrappers, ``%``/``\\%``, degree marks, trailing unit
   words, a leading single-variable assignment (``x = 5``), a trailing period;
2. numbers: thousands separators, decimals, ``\\frac``/``\\dfrac``/``\\tfrac``, ``a/b``,
   signs -> an exact rational, keyed ``"q:p/q"`` (or ``"q:p"``);
3. other closed-form constants (``\\sqrt``, ``\\pi``) -> sympy; exact rationals as in 2,
   otherwise ``"r:"`` + 12 significant digits;
4. symbolic expressions -> ``"e:"`` + ``str(sympy.expand(...))``;
5. anything else -> ``"s:"`` + ``normalize_v2`` of the stripped string.

Non-answer outcomes (``<TRUNCATED>``, ``<NON_ANSWER>``) pass through unchanged. Rounding
variants (6.17 vs 6.18) are different values and are NOT merged.
"""

from __future__ import annotations

import re
from fractions import Fraction
from functools import lru_cache

from . import answers as A

SPECIAL = {A.TRUNCATED, A.NON_ANSWER}

_UNITS = (r"units?|square\s+units|sq\.?\s*units|cm|mm|km|m|meters?|metres?|feet|foot|ft|"
          r"inches|inch|in|miles?|mph|degrees?|deg|seconds?|sec|s|minutes?|min|hours?|hrs?|"
          r"days?|years?|dollars?|cents?|percent|people|students|ways|cm\^2|m\^2|cm\^3")
_UNIT_TAIL = re.compile(rf"(?<=[\d}})\s])\s*(?:\\,|\\;|\\ |~)*\s*(?:{_UNITS})\.?\s*$", re.IGNORECASE)
_WRAP = re.compile(r"\\(?:text|mathrm|textbf|mbox|operatorname|mathbf|rm)\s*\{([^{}]*)\}")
_ASSIGN = re.compile(r"^\s*[a-zA-Z](?:_\{?\w+\}?)?\s*=\s*(?=\S)")
_NUM = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")
_SLASH = re.compile(r"^([+-]?)\(?([+-]?(?:\d+(?:\.\d*)?|\.\d+))\)?/\(?(\d+(?:\.\d*)?)\)?$")


def _strip(raw: str) -> str:
    s = raw.strip()
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    for tok in ("\\displaystyle", "\\left", "\\right", "\\!", "\\,", "\\;", "\\:", "\\quad",
                "\\qquad", "\\ ", "~", "\\$", "$"):
        s = s.replace(tok, " " if tok in ("\\,", "\\;", "\\:", "\\ ", "~") else "")
    s = s.replace("{,}", ",")
    for _ in range(3):
        s = _WRAP.sub(lambda m: " " + m.group(1) + " ", s)
    s = re.sub(r"\^\s*\{?\s*\\circ\s*\}?", "", s).replace("\\circ", "").replace("°", "")
    s = s.replace("\\%", "").replace("%", "")
    s = s.strip().rstrip(".").strip()
    for _ in range(2):
        s = _UNIT_TAIL.sub("", s).strip()
    m = _ASSIGN.match(s)
    if m and "=" not in s[m.end():]:
        s = s[m.end():]
    s = s.strip().rstrip(".").strip()
    # thousands separators, incl. thin-space forms "1 000" left by spacing commands
    if re.fullmatch(r"[+-]?\d{1,3}(?:[, ]\d{3})+(?:\.\d+)?", s):
        s = s.replace(",", "").replace(" ", "")
    return re.sub(r"\s+", " ", s).strip()


def _frac_arg(s: str, i: int) -> tuple[str, int] | None:
    """Argument of \\frac starting at s[i]: a braced group or a single character."""
    while i < len(s) and s[i] == " ":
        i += 1
    if i >= len(s):
        return None
    if s[i] != "{":
        return s[i], i + 1
    depth, j = 1, i + 1
    while j < len(s) and depth:
        depth += (s[j] == "{") - (s[j] == "}")
        j += 1
    return (s[i + 1:j - 1], j) if depth == 0 else None


def _rational(s: str) -> Fraction | None:
    """Exact value of a plain number, a/b, or (signed) \\frac{A}{B} with numeric A, B."""
    s = s.replace(" ", "")
    if _NUM.match(s):
        return Fraction(s)
    m = _SLASH.match(s)
    if m:
        d = Fraction(m.group(3))
        return None if d == 0 else (-1 if m.group(1) == "-" else 1) * Fraction(m.group(2)) / d
    sign = 1
    if s[:1] in "+-" and s[1:].startswith("\\frac"):
        sign, s = (-1 if s[0] == "-" else 1), s[1:]
    if s.startswith("\\frac"):
        a = _frac_arg(s, 5)
        b = _frac_arg(s, a[1]) if a else None
        if a and b and b[1] == len(s):
            num, den = _rational(a[0]), _rational(b[0])
            if num is not None and den not in (None, 0):
                return sign * num / den
    return None


def _key_rational(q: Fraction) -> str:
    if abs(q.numerator).bit_length() + q.denominator.bit_length() > 200:   # (9/8)^2007: hash
        import hashlib   # hex formatting has no int-digit limit, unlike str()

        return "q#" + hashlib.sha1(f"{q.numerator:x}/{q.denominator:x}".encode()).hexdigest()[:16]
    return f"q:{q.numerator}" if q.denominator == 1 else f"q:{q.numerator}/{q.denominator}"


def _latex_to_sympy(s: str) -> str:
    out = s
    for _ in range(8):   # innermost-first rewrite of powers, \frac and \sqrt
        nxt = re.sub(r"\^\s*\{([^{}]*)\}", r"**(\1)", out)
        nxt = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"((\1)/(\2))", nxt)
        nxt = re.sub(r"\\frac\s*(\w)\s*(\w)", r"((\1)/(\2))", nxt)
        nxt = re.sub(r"\\sqrt\s*\[([^\]]*)\]\s*\{([^{}]*)\}", r"((\2)**(1/(\1)))", nxt)
        nxt = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"sqrt(\1)", nxt)
        nxt = re.sub(r"\\sqrt\s*(\w)", r"sqrt(\1)", nxt)
        if nxt == out:
            break
        out = nxt
    out = (out.replace("\\pi", " pi ").replace("\\cdot", "*").replace("\\times", "*")
           .replace("^", "**").replace("{", "(").replace("}", ")"))
    return out


@lru_cache(maxsize=100_000)
def _sympy_key(s: str) -> str | None:
    if "\\" in s.replace("\\frac", "").replace("\\sqrt", "").replace("\\pi", "") \
            .replace("\\cdot", "").replace("\\times", ""):
        return None                                  # other LaTeX: not attempted
    try:
        import sympy
        from sympy.parsing.sympy_parser import (factorial_notation,
                                                implicit_multiplication_application,
                                                parse_expr, standard_transformations)

        expr = parse_expr(_latex_to_sympy(s), evaluate=True,
                          transformations=standard_transformations
                          + (factorial_notation, implicit_multiplication_application))
        if not isinstance(expr, sympy.Basic):
            return None
        if not expr.free_symbols:
            expr = sympy.nsimplify(expr) if expr.is_Float else expr
            if expr.is_Rational:
                return _key_rational(Fraction(int(expr.p), int(expr.q)))
            val = complex(sympy.N(expr, 30))
            if abs(val.imag) > 1e-12:
                return None
            return f"r:{val.real:.12g}"
        return "e:" + str(sympy.expand(expr))
    except Exception:   # noqa: BLE001 - any parse failure falls back to the string rule
        return None


@lru_cache(maxsize=200_000)
def canonical_answer(raw: str) -> str:
    """Canonical key of one raw answer surface (see module docstring)."""
    if raw in SPECIAL:
        return raw
    s = _strip(raw)
    if not s:
        return A.NON_ANSWER
    q = _rational(s)
    if q is not None:
        return _key_rational(q)
    k = _sympy_key(s)
    if k is not None:
        return k
    return "s:" + (A.normalize_v2(s) or s)


def raw_surface(continuation: str | None, *, finish_reason: str | None = None,
                token_count: int | None = None) -> tuple[str, str | None]:
    """(v2 status, raw answer surface) with exactly the ``answers.parse_v2`` rules, but
    returning the un-normalised string (last balanced box in the answer region, or the
    stated text for ``unboxed_stated``)."""
    p = A.parse_v2(continuation, finish_reason=finish_reason, token_count=token_count)
    if p.status not in ("answered", "unboxed_stated"):
        return p.status, None
    region = continuation.rsplit("</think>", 1)[1] if "</think>" in continuation \
        else continuation
    if p.status == "answered":
        boxes = [b for b in A.balanced_boxes(region)[0] if b.strip()]
        return p.status, boxes[-1]
    m = A._STATED.search(region[-400:])
    stated = m.group(1).strip().splitlines()[0] if m and m.group(1).strip() else p.answer
    stated = re.sub(r"^\\\(|\\\)$", "", stated).strip()
    return p.status, stated


def category_canonical(status: str, raw: str | None) -> str:
    """Canonical counterpart of ``answers.category_v2`` (primary handling)."""
    if status in ("answered", "unboxed_stated") and raw is not None:
        return canonical_answer(raw)
    if status == "length_truncated":
        return A.TRUNCATED
    return A.NON_ANSWER

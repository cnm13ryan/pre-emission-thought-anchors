"""Pinned answer extraction/normalization, ported without behavior changes.

Source: thought-anchors/utils.py at b53ed8c75d3f6112f68adfaec9a13d4d708c442e

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
"""

import re


def extract_boxed_answers(text: str) -> list[str]:
    """
    Extract answers enclosed in \boxed{} from the text with improved handling
    of nested braces and complex LaTeX expressions.

    Args:
        text: The text to extract boxed answers from

    Returns:
        List of extracted boxed answers
    """
    # Find all occurrences of \boxed{
    boxed_starts = [m.start() for m in re.finditer(r"\\boxed\{", text)]

    if not boxed_starts:
        return [""]

    answers = []

    for start_idx in boxed_starts:
        # Start after \boxed{
        idx = start_idx + 7
        brace_count = 1  # We've already opened one brace
        answer = ""

        # Parse until we find the matching closing brace
        while idx < len(text) and brace_count > 0:
            char = text[idx]

            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1

                # Skip the closing brace of \boxed{}
                if brace_count == 0:
                    break

            if brace_count > 0:  # Only add if we're still inside the boxed content
                answer += char

            idx += 1

        if answer:
            answers.append(answer)

    return answers if answers else [""]


def normalize_latex(latex_str: str) -> str:
    """
    Normalize LaTeX string by applying various transformations.

    Args:
        latex_str: The LaTeX string to normalize

    Returns:
        Normalized LaTeX string
    """
    normalized = latex_str.strip().lower()

    # Replace different fraction notations
    normalized = normalized.replace("dfrac", "frac")
    normalized = normalized.replace("tfrac", "frac")

    # Normalize spaces
    normalized = re.sub(r"\s+", "", normalized)

    # Normalize percentages
    normalized = normalized.replace("\\%", "")

    # Normalize funny commas
    normalized = normalized.replace("{,}", "")

    # Normalize common mathematical notations
    normalized = normalized.replace("\\times", "*")
    normalized = normalized.replace("\\cdot", "*")

    # Normalize decimal representation
    normalized = re.sub(r"(\d+)[\.,](\d+)", r"\1.\2", normalized)

    # Remove unnecessary braces in simple expressions
    normalized = re.sub(r"{([^{}]+)}", r"\1", normalized)

    # Normalize common constants
    normalized = normalized.replace("\\pi", "pi")

    # Remove LaTeX text commands
    normalized = re.sub(r"\\text\{([^{}]+)\}", r"\1", normalized)
    normalized = re.sub(r"\\mathrm\{([^{}]+)\}", r"\1", normalized)

    # Normalize date formats (e.g., "October 30" vs "October\\ 30")
    normalized = re.sub(r"([a-z]+)\\+\s*(\d+)", r"\1\2", normalized)
    normalized = normalized.replace("\\text", "")

    return normalized

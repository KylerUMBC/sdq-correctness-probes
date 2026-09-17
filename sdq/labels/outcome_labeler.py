"""Derive correct/incorrect labels from model generations vs gold answers.

Each task family in the SDQ benchmark has a known answer_id format:
  - arithmetic, multi_step_arithmetic, variable_chain: numeric strings
  - syllogistic, relational, multi_hop: word/name strings
  - contradiction, set_inclusion: yes/no strings

The labeler normalizes both the model output and the gold answer, then
compares them to produce a binary correctness label.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class OutcomeLabel:
    """Correctness label for a single generation."""

    correct: bool
    parsed_answer: str
    gold_answer: str
    task_family: str
    confidence: float = 1.0  # 1.0 = deterministic match, <1.0 = fuzzy


def _normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


# Preambles the model often emits before the real answer. These are stripped
# iteratively from the left so the answer extractor sees the meaningful content.
# Ordered most-specific first.
_PREAMBLE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^\s*\[\s*answer\s*\d*\s*\]\s*[:.]?\s*",   # [Answer 1]
        r"^\s*\[\s*user\s*\d*\s*\]\s*[:.]?\s*",      # [User 0001]
        r"^\s*<[^>]+>\s*",                            # <blockquote>, <strong>, ...
        r"^\s*the\s+answer\s+is\s*[:.]?\s*",
        r"^\s*answer\s*[:.]?\s+",
        r"^\s*so\s*[,.]?\s+",
        r"^\s*therefore\s*[,.]?\s+",
        r"^\s*thus\s*[,.]?\s+",
        r"^\s*first\s*[,.]?\s+",
        r"^\s*well\s*[,.]?\s+",
        r"^\s*sure\s*[,.]?\s+",
        r"^\s*let\s+me\s+think\s*[:.]?\s*",
        r"^\s*okay\s*[,.]?\s+",
    )
)


def _strip_preambles(text: str) -> str:
    """Iteratively strip leading whitespace and common reasoning preambles."""
    prev = None
    while prev != text:
        prev = text
        text = text.lstrip()
        for pat in _PREAMBLE_PATTERNS:
            new_text, n = pat.subn("", text, count=1)
            if n:
                text = new_text
                break
    return text


def _first_sentence_window(text: str, max_chars: int = 120) -> str:
    """Return the first sentence / clause of text (up to max_chars)."""
    m = re.search(r"[.!?\n]", text)
    if m is not None:
        return text[: m.start()]
    return text[:max_chars]


def _extract_numeric(text: str, gold: str | None = None) -> str | None:
    """Pull a numeric answer from text.

    Strategy, in order:
    1. If the first number in the text equals gold, return it. Gemma often
       answers directly (' 16\\n\\n16 + 1 + 6 = 23...') and then spirals into
       unrelated arithmetic — the real answer is the first number.
    2. If show-your-work of the form 'X = Y' appears, take the number after
       the LAST '='. This handles variable_chain reasoning where the answer
       is computed at the end ('a = 7. b = 14.').
    3. Otherwise return the first number.
    """
    text = _strip_preambles(text)
    if not text:
        return None

    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)

    # Priority 1: direct answer at the start matches gold.
    if gold is not None and numbers:
        try:
            if float(numbers[0]) == float(gold):
                return numbers[0]
        except (ValueError, TypeError):
            pass

    # Priority 2: show-your-work — number after the last '='.
    eq_matches = list(re.finditer(r"=\s*(-?\d+(?:\.\d+)?)", text))
    if eq_matches:
        return eq_matches[-1].group(1)

    # Priority 3: first number as a fallback.
    return numbers[0] if numbers else None


def _extract_single_step_numeric(text: str) -> str | None:
    """Extract the first answer-bearing value from single-step arithmetic.

    Single-step generations often answer with an equation and then continue
    into unrelated arithmetic. The first equation's right-hand side is the
    committed answer; choosing the final equation mislabels that continuation.
    """
    text = _strip_preambles(text)
    if not text:
        return None

    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    direct = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*[.!]?", first_line)
    if direct:
        return direct.group(1)

    first_equation = re.search(r"=\s*(-?\d+(?:\.\d+)?)", text)
    if first_equation:
        return first_equation.group(1)

    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    return numbers[0] if numbers else None


def _extract_yes_no(text: str) -> str | None:
    """Pull a yes/no answer from text.

    Returns whichever of 'yes'/'no' appears first as a whole word after
    stripping preambles. Handles phrasings like 'No, there is no ...' by
    returning the first whole-word match.
    """
    text = _strip_preambles(text).lower()
    if not text:
        return None

    yes_match = re.search(r"\byes\b", text)
    no_match = re.search(r"\bno\b", text)
    if yes_match and no_match:
        return "yes" if yes_match.start() < no_match.start() else "no"
    if yes_match:
        return "yes"
    if no_match:
        return "no"
    return None


def _extract_word(text: str, gold: str) -> str | None:
    """Extract an answer word/name from generated text.

    Strategy:
    1. Strip preambles ([Answer 1], <blockquote>, ...).
    2. Restrict to the first sentence/clause window.
    3. If gold (or a singular/plural variant) appears as a whole word,
       return what the model actually wrote for it.
    4. Fall back to the first alphabetic content word.
    """
    text = _strip_preambles(text)
    if not text:
        return None

    window = _first_sentence_window(text)
    window_lower = window.lower()
    gold_lower = gold.lower().strip()

    candidates: set[str] = {gold_lower}
    if gold_lower.endswith("s") and len(gold_lower) > 1:
        candidates.add(gold_lower[:-1])
    else:
        candidates.add(gold_lower + "s")

    best: tuple[int, str] | None = None
    for cand in candidates:
        if not cand:
            continue
        m = re.search(rf"\b{re.escape(cand)}\b", window_lower)
        if m is not None and (best is None or m.start() < best[0]):
            best = (m.start(), m.group(0))
    if best is not None:
        return best[1]

    words = re.findall(r"[A-Za-z]+", window)
    return words[0].lower() if words else None


_NUMERIC_FAMILIES = {"arithmetic", "multi_step_arithmetic", "variable_chain"}
_YES_NO_FAMILIES = {"contradiction", "set_inclusion"}
_WORD_FAMILIES = {"syllogistic", "relational", "multi_hop"}


def _parse_answer(
    generated_text: str,
    gold_answer: str,
    task_family: str,
) -> tuple[str, float]:
    """Parse the model's answer and return (parsed, confidence)."""
    text = generated_text.strip()
    if not text:
        return "", 1.0

    if task_family == "arithmetic":
        parsed = _extract_single_step_numeric(text)
        if parsed is not None:
            return parsed, 1.0
        return text.strip()[:50], 0.5

    if task_family in _NUMERIC_FAMILIES:
        parsed = _extract_numeric(text, gold_answer)
        if parsed is not None:
            return parsed, 1.0
        return text.strip()[:50], 0.5

    if task_family in _YES_NO_FAMILIES:
        parsed = _extract_yes_no(text)
        if parsed is not None:
            return parsed, 1.0
        return text.strip()[:50], 0.5

    if task_family in _WORD_FAMILIES:
        parsed = _extract_word(text, gold_answer)
        if parsed is not None:
            return parsed, 1.0
        return text.strip()[:50], 0.5

    return text.strip()[:50], 0.5


def _answers_match(parsed: str, gold: str, task_family: str) -> bool:
    """Check if parsed answer matches gold.

    Word families allow singular/plural variants (animal vs animals) because
    Gemma's natural phrasing and the benchmark's gold form do not always agree.
    """
    if task_family in _NUMERIC_FAMILIES:
        try:
            return float(parsed) == float(gold)
        except (ValueError, TypeError):
            return parsed.strip().lower() == gold.strip().lower()

    p = parsed.strip().lower()
    g = gold.strip().lower()
    if p == g:
        return True
    if task_family in _WORD_FAMILIES:
        if p and g and (p + "s" == g or p == g + "s"):
            return True
    return False


def label_run(
    new_tokens: str,
    answer_id: str,
    task_family: str,
) -> OutcomeLabel:
    """Label a single generation as correct or incorrect.

    Args:
        new_tokens: The newly generated text (not including the prompt).
        answer_id: The gold answer string from the benchmark.
        task_family: The task family for parser selection.

    Returns:
        OutcomeLabel with correctness verdict.
    """
    parsed, confidence = _parse_answer(new_tokens, answer_id, task_family)
    correct = _answers_match(parsed, answer_id, task_family)

    return OutcomeLabel(
        correct=correct,
        parsed_answer=parsed,
        gold_answer=answer_id,
        task_family=task_family,
        confidence=confidence,
    )


def label_batch(
    runs: list[dict[str, Any]],
) -> list[OutcomeLabel]:
    """Label a batch of runs.

    Each dict must have keys: new_tokens (str), answer_id (str), task_family (str).
    """
    return [
        label_run(
            new_tokens=r["new_tokens"],
            answer_id=r["answer_id"],
            task_family=r["task_family"],
        )
        for r in runs
    ]


def make_future_window_labels(
    correct: bool,
    seq_len: int,
    horizon: int = 5,
) -> list[int]:
    """Convert a sequence-level correct/incorrect label into per-timestep labels.

    For an incorrect sequence, all timesteps get y_t=1 (bad outcome ahead).
    For a correct sequence, all timesteps get y_t=0.

    A more refined version could use partial labels (e.g., based on when the
    model commits to the wrong answer), but this simple version is sufficient
    for the initial signal-validation experiment.

    Args:
        correct: Whether the final answer was correct.
        seq_len: Length of the generation sequence.
        horizon: Future window size (currently unused; reserved for
                 fine-grained per-step labeling in Phase 2).

    Returns:
        List of length seq_len with binary labels.
    """
    label = 0 if correct else 1
    return [label] * seq_len

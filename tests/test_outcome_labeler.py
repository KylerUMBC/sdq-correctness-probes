"""Regression tests for sdq.labels.outcome_labeler.

These fixtures are drawn from real Gemma-2-2B generations observed in
data/runs/. The v1 labeler mislabeled most of them because it greedily
grabbed the first token out of show-your-work or reasoning preambles.
"""

from __future__ import annotations

import pytest

from sdq.labels.outcome_labeler import (
    _extract_numeric,
    _extract_single_step_numeric,
    _extract_word,
    _extract_yes_no,
    _strip_preambles,
    label_run,
    make_future_window_labels,
)


# ── Preamble stripping ──────────────────────────────────────────────────────

class TestStripPreambles:
    def test_answer_bracket(self):
        assert _strip_preambles("\n\n[Answer 1]\n\nAlice is older") == "Alice is older"

    def test_user_bracket(self):
        assert _strip_preambles("\n\n[User 0001]\n\nI'm not sure") == "I'm not sure"

    def test_html_tag(self):
        assert _strip_preambles("<blockquote>Alice is older").startswith("Alice")

    def test_the_answer_is(self):
        assert _strip_preambles("The answer is 14") == "14"

    def test_chained_preambles(self):
        # Multiple preambles in a row should all strip
        assert _strip_preambles("\n\n[Answer 1]\n\nSo, Alice") == "Alice"

    def test_preserves_bare_text(self):
        assert _strip_preambles("Alice is the oldest") == "Alice is the oldest"

    def test_empty(self):
        assert _strip_preambles("") == ""
        assert _strip_preambles("   \n\n") == ""


# ── Numeric extraction ──────────────────────────────────────────────────────

class TestExtractNumeric:
    def test_simple_arithmetic(self):
        # " 5\n\n2 + 3 = 5\n\n..."
        assert _extract_numeric(" 5\n\n2 + 3 = 5\n\n2 + 3 =", gold="5") == "5"

    def test_show_your_work_last_equals(self):
        # Multi-step: first number is "5" but the final answer after last = is "15"
        assert _extract_numeric(" 15\n\n5 + 7 + 3 = 15", gold="15") == "15"

    def test_variable_chain_show_work(self):
        # Canonical variable_chain failure mode:
        # "First, a = 7. Then b = a * 2 = 14. So b = 14."
        text = "\n\nFirst, a = 3 + 4 = 7. Then b = a * 2 = 7 * 2 = 14. So b = 14."
        assert _extract_numeric(text, gold="14") == "14"

    def test_variable_chain_intermediate_number(self):
        # Make sure we don't grab the intermediate "7" from the reasoning.
        text = "a = 7, b = 14"
        assert _extract_numeric(text, gold="14") == "14"

    def test_no_equals_just_answer(self):
        assert _extract_numeric(" 14", gold="14") == "14"

    def test_off_topic_model(self):
        # Model goes off-topic with a different question. First number is "10"
        # which doesn't match gold "5" → "10" returned → labeled incorrect.
        text = "\n\n[User 0001]\n\nWhat is 10 minus 5?"
        assert _extract_numeric(text, gold="5") == "10"

    def test_answer_first_then_self_continuation(self):
        # Multi-step failure mode: model answers correctly, then spirals into
        # unrelated arithmetic. v2 "last =" heuristic was picking up "30" by
        # mistake. Correct behavior: recognize the first number as the answer.
        text = " 16\n\n16 + 1 + 6 = 23\n\n23 + 1 + 6 = 30"
        assert _extract_numeric(text, gold="16") == "16"

    def test_single_step_uses_first_answer_bearing_equation(self):
        text = "11 - 8 = 3\n\nNow add 3 to 11.\n\n11 + 3 = 14"
        assert _extract_single_step_numeric(text) == "3"

    def test_single_step_direct_answer(self):
        assert _extract_single_step_numeric("36\n\n6 x 6 = 36") == "36"

    def test_negative_number(self):
        assert _extract_numeric(" -3\n", gold="-3") == "-3"

    def test_float(self):
        assert _extract_numeric(" 2.5", gold="2.5") == "2.5"

    def test_empty_text(self):
        assert _extract_numeric("", gold="5") is None


# ── Yes/no extraction ───────────────────────────────────────────────────────

class TestExtractYesNo:
    def test_yes_with_preamble(self):
        # "\n\n[Answer 1]\n\nYes, cats are mammals."
        assert _extract_yes_no("\n\n[Answer 1]\n\nYes, cats are mammals.") == "yes"

    def test_no_with_preamble(self):
        # "\n\n[Answer 1]\n\nNo, there is no contradiction."
        assert _extract_yes_no("\n\n[Answer 1]\n\nNo, there is no contradiction.") == "no"

    def test_yes_after_so(self):
        assert _extract_yes_no("So, yes, definitely.") == "yes"

    def test_first_occurrence_wins(self):
        # "Yes, this is not the case" → yes first
        assert _extract_yes_no("Yes, this is not the case") == "yes"
        # "No, yes is wrong" → no first
        assert _extract_yes_no("No, yes is wrong") == "no"

    def test_no_within_sentence(self):
        # "There is no contradiction" → no
        assert _extract_yes_no("There is no contradiction here.") == "no"

    def test_no_word_boundary(self):
        # "note" and "notice" should NOT match "no"
        assert _extract_yes_no("Notice that this is a note.") is None

    def test_neither(self):
        assert _extract_yes_no("Maybe it could be.") is None

    def test_empty(self):
        assert _extract_yes_no("") is None


# ── Word extraction ─────────────────────────────────────────────────────────

class TestExtractWord:
    def test_syllogistic_singular_plural(self):
        # Gold is "animals" (plural); model says "an animal" (singular).
        # The v1 parser would take "an" and miss. v2 should find "animal".
        assert _extract_word(" an animal.\n\n", gold="animals") == "animal"

    def test_syllogistic_direct_plural(self):
        # Direct plural form matches.
        assert _extract_word(" animals.\n", gold="animals") == "animals"

    def test_relational_answer_bracket(self):
        # "\n\n[Answer 1]\n\nAlice is the oldest."
        assert _extract_word("\n\n[Answer 1]\n\nAlice is the oldest.", gold="Alice") == "alice"

    def test_relational_bare(self):
        # " Alice.\n\n"
        assert _extract_word(" Alice.\n\n", gold="Alice") == "alice"

    def test_relational_misidentified(self):
        # Model says wrong name — should still extract something, not gold.
        # " Bob.\n\n" with gold Alice → fall back to first word "bob"
        assert _extract_word(" Bob.\n\n", gold="Alice") == "bob"

    def test_multi_hop_name(self):
        # Synthetic multi_hop: " Ashwick.\n"
        assert _extract_word(" Ashwick.\n", gold="Ashwick") == "ashwick"

    def test_blockquote_wrapper(self):
        assert _extract_word("<blockquote>Alice is older", gold="Alice") == "alice"

    def test_empty(self):
        assert _extract_word("", gold="Alice") is None


# ── End-to-end label_run ────────────────────────────────────────────────────

class TestLabelRun:
    """These are the canonical regression fixtures. Each one was being
    mislabeled by the v1 parser."""

    def test_arithmetic_correct(self):
        label = label_run(" 5\n\n2 + 3 = 5", answer_id="5", task_family="arithmetic")
        assert label.correct is True
        assert label.parsed_answer == "5"

    def test_multi_step_arithmetic_show_work(self):
        # v1 bug: took "5", said wrong. Gold is "15".
        label = label_run(" 15\n\n5 + 7 + 3 = 15", answer_id="15",
                          task_family="multi_step_arithmetic")
        assert label.correct is True

    def test_variable_chain_show_work(self):
        # v1 bug: took "3" (first number in "3 + 4"), said wrong. Gold is "14".
        label = label_run(
            "\n\nFirst, a = 3 + 4 = 7. Then b = a * 2 = 14. So b = 14.",
            answer_id="14",
            task_family="variable_chain",
        )
        assert label.correct is True

    def test_syllogistic_singular_plural(self):
        # v1 bug: took "an", said wrong. Gold "animals", model "an animal".
        label = label_run(
            " an animal.\n\nI'm not sure",
            answer_id="animals",
            task_family="syllogistic",
        )
        assert label.correct is True

    def test_relational_answer_bracket(self):
        # v1 bug: took "[Answer", said wrong. Model did say "Alice".
        label = label_run(
            "\n\n[Answer 1]\n\nAlice is the oldest.",
            answer_id="Alice",
            task_family="relational",
        )
        assert label.correct is True

    def test_set_inclusion_yes(self):
        # v1 technically got this via the "yes," tuple match — verify we kept it.
        label = label_run(
            "\n\n[Answer 1]\n\nYes, cats are mammals.",
            answer_id="yes",
            task_family="set_inclusion",
        )
        assert label.correct is True

    def test_contradiction_correct_no_when_gold_yes(self):
        # Genuine model failure: gold yes, model says no.
        label = label_run(
            "\n\n[Answer 1]\n\nNo, there is no contradiction.",
            answer_id="yes",
            task_family="contradiction",
        )
        assert label.correct is False
        assert label.parsed_answer == "no"

    def test_multi_hop_preamble(self):
        # Canonical multi_hop failure fixture.
        label = label_run(
            "\n\n[Answer 1]\n\nAshwick is where Zara lives.",
            answer_id="Ashwick",
            task_family="multi_hop",
        )
        assert label.correct is True

    def test_off_topic_arithmetic_still_wrong(self):
        # Genuine failure: model went off-topic. First number is "10", not
        # the gold "5", so it's correctly marked incorrect.
        label = label_run(
            "\n\nWhat is 10 minus 5?",
            answer_id="5",
            task_family="arithmetic",
        )
        assert label.correct is False
        assert label.parsed_answer == "10"

    def test_multi_step_answer_first_then_continuation(self):
        # Regression: " 16\n\n16 + 1 + 6 = 23 ..." should be marked correct
        # because the model's real answer is at the start, not after the
        # last =.
        label = label_run(
            " 16\n\n16 + 1 + 6 = 23\n\n23 + 1 + 6 = 30",
            answer_id="16",
            task_family="multi_step_arithmetic",
        )
        assert label.correct is True
        assert label.parsed_answer == "16"

    def test_empty_generation(self):
        label = label_run("", answer_id="5", task_family="arithmetic")
        assert label.correct is False


# ── Future window labels ────────────────────────────────────────────────────

class TestFutureWindowLabels:
    def test_correct_all_zero(self):
        assert make_future_window_labels(correct=True, seq_len=5) == [0, 0, 0, 0, 0]

    def test_incorrect_all_one(self):
        assert make_future_window_labels(correct=False, seq_len=3) == [1, 1, 1]

    def test_zero_length(self):
        assert make_future_window_labels(correct=True, seq_len=0) == []

"""Tests for the benchmark data generation system."""

from __future__ import annotations

import json
from collections import Counter

import pytest

from sdq.data.generators import (
    BenchmarkDataset,
    BenchmarkExample,
    BenchmarkSplit,
    EventSpan,
    assemble_benchmark,
    make_all_splits,
    make_benchmark_splits,
    print_benchmark_stats,
)
from sdq.data.generators.arithmetic import generate_multi_step, generate_single_step
from sdq.data.generators.contradiction import generate_contradiction
from sdq.data.generators.relational import generate_relational
from sdq.data.generators.set_inclusion import generate_set_inclusion
from sdq.data.generators.syllogistic import generate_syllogistic


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestSchema:
    def test_event_span_to_dict(self):
        span = EventSpan("setup", 0, 10)
        d = span.to_dict()
        assert d == {"event": "setup", "start_char": 0, "end_char": 10}

    def test_benchmark_example_to_dict(self):
        ex = BenchmarkExample(
            example_id="test_001",
            task_family="arithmetic",
            semantic_task_id="add_2_3",
            reasoning_graph_id="single_+",
            surface_template_id="equation",
            reasoning_variant_id="canonical",
            answer_id="5",
            prompt_text="2 + 3 =",
            event_spans=(EventSpan("operand_load", 0, 1),),
        )
        d = ex.to_dict()
        assert d["example_id"] == "test_001"
        assert d["task_family"] == "arithmetic"
        assert len(d["event_spans"]) == 1
        assert "same_reasoning_group" not in d  # empty strings are omitted

    def test_benchmark_example_prompt_id(self):
        ex = BenchmarkExample(
            example_id="bench_00042",
            task_family="t",
            semantic_task_id="s",
            reasoning_graph_id="r",
            surface_template_id="f",
            reasoning_variant_id="v",
            answer_id="a",
            prompt_text="text",
        )
        assert ex.prompt_id == "bench_00042"

    def test_benchmark_dataset_properties(self):
        examples = [
            BenchmarkExample("e1", "f1", "s1", "r1", "sf1", "v1", "a1", "t1"),
            BenchmarkExample("e2", "f2", "s2", "r2", "sf2", "v2", "a2", "t2"),
        ]
        ds = BenchmarkDataset(examples)
        assert ds.task_families == {"f1", "f2"}
        assert ds.semantic_task_ids == {"s1", "s2"}
        assert ds.surface_template_ids == {"sf1", "sf2"}
        assert ds.answer_ids == {"a1", "a2"}

    def test_benchmark_dataset_filter(self):
        examples = [
            BenchmarkExample("e1", "arith", "s1", "r1", "sf1", "v1", "5", "t1"),
            BenchmarkExample("e2", "arith", "s2", "r2", "sf2", "v2", "10", "t2"),
            BenchmarkExample("e3", "logic", "s3", "r3", "sf3", "v3", "yes", "t3"),
        ]
        ds = BenchmarkDataset(examples)
        assert len(ds.filter(task_family="arith")) == 2
        assert len(ds.filter(answer_id="5")) == 1


# ---------------------------------------------------------------------------
# Generator tests
# ---------------------------------------------------------------------------

class TestArithmeticGenerator:
    def test_single_step_generates_examples(self):
        examples = list(generate_single_step())
        assert len(examples) > 100  # 25 problems × 6 templates = 150

    def test_single_step_has_valid_fields(self):
        ex = next(iter(generate_single_step()))
        assert ex.task_family == "arithmetic"
        assert ex.prompt_text
        assert ex.answer_id
        assert ex.event_spans
        assert all(isinstance(s, EventSpan) for s in ex.event_spans)

    def test_single_step_event_spans_within_bounds(self):
        for ex in generate_single_step():
            for span in ex.event_spans:
                assert 0 <= span.start_char <= span.end_char <= len(ex.prompt_text), \
                    f"Span {span} out of bounds for '{ex.prompt_text}' (len={len(ex.prompt_text)})"

    def test_multi_step_generates_examples(self):
        examples = list(generate_multi_step())
        assert len(examples) > 80  # 20 problems × 6 templates = 120

    def test_multi_step_has_reasoning_variants(self):
        examples = list(generate_multi_step())
        variants = {e.reasoning_variant_id for e in examples}
        assert "left_assoc" in variants
        assert "right_assoc" in variants

    def test_multi_step_event_spans_within_bounds(self):
        for ex in generate_multi_step():
            for span in ex.event_spans:
                assert 0 <= span.start_char <= span.end_char <= len(ex.prompt_text), \
                    f"Span {span} out of bounds for '{ex.prompt_text}' (len={len(ex.prompt_text)})"


class TestSyllogisticGenerator:
    def test_generates_examples(self):
        examples = list(generate_syllogistic())
        assert len(examples) >= 120  # 20 entities × 6 templates = 120 + transitive

    def test_has_both_graph_types(self):
        examples = list(generate_syllogistic())
        graphs = {e.reasoning_graph_id for e in examples}
        assert "simple_universal" in graphs
        assert "transitive_universal" in graphs

    def test_event_spans_within_bounds(self):
        for ex in generate_syllogistic():
            for span in ex.event_spans:
                assert 0 <= span.start_char <= span.end_char <= len(ex.prompt_text), \
                    f"Span {span} out of bounds for '{ex.prompt_text}' (len={len(ex.prompt_text)})"

    def test_surface_template_diversity(self):
        examples = list(generate_syllogistic())
        templates = {e.surface_template_id for e in examples}
        assert len(templates) >= 4  # therefore, since, given, because, reorder, if_then

    def test_because_surface_requests_a_completion(self):
        because = [
            e for e in generate_syllogistic()
            if e.surface_template_id == "because"
        ]
        assert because
        assert all(e.prompt_text.startswith("Because ") for e in because)
        assert all(e.prompt_text.rstrip().endswith(" is") for e in because)


class TestRelationalGenerator:
    def test_generates_examples(self):
        examples = list(generate_relational())
        assert len(examples) >= 200  # 5 relations × 8 name sets × 6 templates + 4-chains

    def test_has_both_chain_lengths(self):
        examples = list(generate_relational())
        graphs = {e.reasoning_graph_id for e in examples}
        assert any("transitive_3" in g for g in graphs)
        assert any("transitive_4" in g for g in graphs)

    def test_event_spans_within_bounds(self):
        for ex in generate_relational():
            for span in ex.event_spans:
                assert 0 <= span.start_char <= span.end_char <= len(ex.prompt_text), \
                    f"Span {span} out of bounds for '{ex.prompt_text}' (len={len(ex.prompt_text)})"


class TestSetInclusionGenerator:
    def test_generates_examples(self):
        examples = list(generate_set_inclusion())
        assert len(examples) >= 100  # 20 valid × 6 + 10 invalid × 4

    def test_has_valid_and_invalid(self):
        examples = list(generate_set_inclusion())
        answers = {e.answer_id for e in examples}
        assert "yes" in answers
        assert "no" in answers

    def test_event_spans_within_bounds(self):
        for ex in generate_set_inclusion():
            for span in ex.event_spans:
                assert 0 <= span.start_char <= span.end_char <= len(ex.prompt_text), \
                    f"Span {span} out of bounds for '{ex.prompt_text}' (len={len(ex.prompt_text)})"

    def test_every_surface_uses_the_yes_no_contract(self):
        for ex in generate_set_inclusion():
            assert ex.answer_id in {"yes", "no"}
            assert "yes or no" in ex.prompt_text.lower()
            assert not ex.prompt_text.rstrip().endswith(" are")

    def test_invalid_converse_prompts_state_the_valid_direction(self):
        invalid = [
            ex for ex in generate_set_inclusion()
            if ex.reasoning_graph_id == "invalid_converse"
        ]
        assert invalid
        assert all("follow" in ex.prompt_text.lower() or "entailed" in ex.prompt_text.lower()
                   or "must" in ex.prompt_text.lower() for ex in invalid)


class TestContradictionGenerator:
    def test_generates_examples(self):
        examples = list(generate_contradiction())
        assert len(examples) >= 80  # 15 contradictions × 5 + 10 consistent × 5

    def test_has_yes_and_no(self):
        examples = list(generate_contradiction())
        answers = {e.answer_id for e in examples}
        assert "yes" in answers
        assert "no" in answers

    def test_event_spans_within_bounds(self):
        for ex in generate_contradiction():
            for span in ex.event_spans:
                assert 0 <= span.start_char <= span.end_char <= len(ex.prompt_text), \
                    f"Span {span} out of bounds for '{ex.prompt_text}' (len={len(ex.prompt_text)})"

    def test_surface_forms_keep_yes_equal_to_contradiction(self):
        examples = list(generate_contradiction())
        true_false = [e for e in examples if e.surface_template_id == "true_false"]
        can_both = [e for e in examples if e.surface_template_id == "can_both"]
        assert true_false and can_both
        assert all("contradict each other" in e.prompt_text for e in true_false)
        assert all("impossible for both" in e.prompt_text for e in can_both)


# ---------------------------------------------------------------------------
# Assembly tests
# ---------------------------------------------------------------------------

class TestAssembly:
    @pytest.fixture(scope="class")
    def dataset(self):
        return assemble_benchmark()

    def test_total_count_near_1000(self, dataset):
        assert 800 <= len(dataset.examples) <= 1500

    def test_globally_unique_ids(self, dataset):
        ids = [e.example_id for e in dataset.examples]
        assert len(ids) == len(set(ids))

    def test_all_benchmark_ids_prefixed(self, dataset):
        for e in dataset.examples:
            assert e.example_id.startswith("bench_")

    def test_all_task_families_present(self, dataset):
        families = dataset.task_families
        assert "arithmetic" in families
        assert "multi_step_arithmetic" in families
        assert "syllogistic" in families
        assert "relational" in families
        assert "set_inclusion" in families
        assert "contradiction" in families

    def test_all_have_event_spans(self, dataset):
        for e in dataset.examples:
            assert e.event_spans, f"{e.example_id} has no event spans"

    def test_all_have_same_reasoning_group(self, dataset):
        for e in dataset.examples:
            assert e.same_reasoning_group, f"{e.example_id} missing same_reasoning_group"

    def test_stats_string(self, dataset):
        s = print_benchmark_stats(dataset)
        assert "Total examples:" in s
        assert "arithmetic" in s

    def test_to_dict_roundtrip(self, dataset):
        d = dataset.to_dict()
        assert d["num_examples"] == len(dataset.examples)
        assert isinstance(d["examples"], list)
        assert d["examples"][0]["example_id"] == dataset.examples[0].example_id

    def test_no_duplicate_prompt_texts(self, dataset):
        texts = [e.prompt_text for e in dataset.examples]
        # Some duplicates are acceptable for reorder variants, but should be rare
        duplicates = len(texts) - len(set(texts))
        assert duplicates < len(texts) * 0.05, f"Too many duplicate texts: {duplicates}"

    def test_subtraction_surface_forms_preserve_the_answer(self, dataset):
        subtraction = [
            e for e in dataset.examples
            if e.task_family == "arithmetic" and e.reasoning_graph_id == "single_-"
        ]
        assert subtraction
        for example in subtraction:
            assert not (
                example.surface_template_id == "reorder"
            ), f"subtraction operands were swapped in {example.example_id}"


# ---------------------------------------------------------------------------
# Split tests
# ---------------------------------------------------------------------------

class TestBenchmarkSplits:
    @pytest.fixture(scope="class")
    def dataset(self):
        return assemble_benchmark()

    def test_semantic_family_split(self, dataset):
        split = make_benchmark_splits(dataset, "semantic_family")
        assert split.train_size > 0
        assert split.test_size > 0
        assert split.train_size + split.test_size == len(dataset.examples)

    def test_surface_form_split(self, dataset):
        split = make_benchmark_splits(dataset, "surface_form")
        assert split.train_size > 0
        assert split.test_size > 0
        assert split.train_size + split.test_size == len(dataset.examples)

    def test_reasoning_variant_split(self, dataset):
        split = make_benchmark_splits(dataset, "reasoning_variant")
        assert split.train_size > 0
        assert split.test_size > 0
        # Should not hold out the dominant variant
        assert split.train_size > split.test_size

    def test_combination_split(self, dataset):
        split = make_benchmark_splits(dataset, "combination")
        assert split.train_size > 0
        assert split.test_size > 0
        assert split.train_size + split.test_size == len(dataset.examples)

    def test_all_splits(self, dataset):
        splits = make_all_splits(dataset)
        assert len(splits) == 4
        for name, split in splits.items():
            assert split.split_type == name

    def test_invalid_split_type_raises(self, dataset):
        with pytest.raises(ValueError):
            make_benchmark_splits(dataset, "nonexistent")

    def test_semantic_family_no_leakage(self, dataset):
        split = make_benchmark_splits(dataset, "semantic_family")
        train_sems = {(e.task_family, e.semantic_task_id) for e in split.train}
        test_sems = {(e.task_family, e.semantic_task_id) for e in split.test}
        assert not train_sems & test_sems

    def test_surface_form_no_leakage(self, dataset):
        split = make_benchmark_splits(dataset, "surface_form")
        train_surfs = {e.surface_template_id for e in split.train}
        test_surfs = {e.surface_template_id for e in split.test}
        assert not train_surfs & test_surfs

    def test_reproducible(self, dataset):
        s1 = make_benchmark_splits(dataset, "semantic_family", seed="test")
        s2 = make_benchmark_splits(dataset, "semantic_family", seed="test")
        assert [e.example_id for e in s1.train] == [e.example_id for e in s2.train]

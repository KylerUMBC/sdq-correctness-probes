"""Tests for sdq.eval.intervention_data — shared loader for Phases 4/4b/5.

Builds synthetic run directories on disk and verifies:
  - h_0 source semantics (prompt_final vs first_gen are different states,
    loaded from the right files, never silently mixed)
  - run-directory resolution (exact name and <eid>_<timestamp> fallback)
  - multi-layer prompt-final collection for per-layer calibration
  - family-stratified example selection
"""

from __future__ import annotations

import json

import pytest
import torch

from sdq.eval.intervention_data import (
    collect_h0_and_labels,
    collect_h0_multi_layer,
    load_h0,
    resolve_run_dirs,
    select_examples,
)

NUM_LAYERS = 3
SEQ_LEN = 4
DIM = 8


def _write_run(
    run_dir,
    eid: str,
    new_tokens: str = " 7",
    with_gen: bool = True,
    with_prompt: bool = True,
    seed: int = 0,
):
    """Create a synthetic run directory with distinguishable tensors."""
    run_dir.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    if with_prompt:
        act = torch.randn(NUM_LAYERS, SEQ_LEN, DIM, generator=g)
        torch.save(act, run_dir / "activations.pt")
    if with_gen:
        gen = torch.randn(2, NUM_LAYERS, DIM, generator=g)
        torch.save(gen, run_dir / "gen_activations.pt")
    meta = {
        "prompt": {"id": eid, "text": f"prompt for {eid}"},
        "output": {"new_tokens": new_tokens},
    }
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)


def _example(eid: str, family: str = "arithmetic", answer: str = "7") -> dict:
    return {"example_id": eid, "task_family": family, "answer_id": answer,
            "prompt_text": f"prompt for {eid}"}


class TestLoadH0:
    def test_sources_are_different_states(self, tmp_path):
        d = tmp_path / "run1"
        _write_run(d, "run1")
        pf = load_h0(d, layer=-1, source="prompt_final")
        fg = load_h0(d, layer=-1, source="first_gen")
        act = torch.load(d / "activations.pt", weights_only=True)
        gen = torch.load(d / "gen_activations.pt", weights_only=True)
        assert torch.allclose(pf, act[-1, -1, :].float())
        assert torch.allclose(fg, gen[0, -1, :].float())
        assert not torch.allclose(pf, fg)

    def test_layer_indexing(self, tmp_path):
        d = tmp_path / "run1"
        _write_run(d, "run1")
        act = torch.load(d / "activations.pt", weights_only=True)
        h1 = load_h0(d, layer=1, source="prompt_final")
        assert torch.allclose(h1, act[1, -1, :].float())

    def test_missing_file_returns_none(self, tmp_path):
        d = tmp_path / "run1"
        _write_run(d, "run1", with_prompt=False)
        assert load_h0(d, source="prompt_final") is None
        assert load_h0(d, source="first_gen") is not None

    def test_unknown_source_raises(self, tmp_path):
        d = tmp_path / "run1"
        _write_run(d, "run1")
        with pytest.raises(ValueError):
            load_h0(d, source="bogus")


class TestResolveRunDirs:
    def test_exact_match(self, tmp_path):
        _write_run(tmp_path / "ex1", "ex1")
        out = resolve_run_dirs(tmp_path, ["ex1"])
        assert out["ex1"] == tmp_path / "ex1"

    def test_timestamp_fallback_latest_wins(self, tmp_path):
        _write_run(tmp_path / "ex1_20260101_000000", "ex1")
        _write_run(tmp_path / "ex1_20260201_000000", "ex1")
        out = resolve_run_dirs(tmp_path, ["ex1"])
        assert out["ex1"].name == "ex1_20260201_000000"

    def test_missing_not_resolved(self, tmp_path):
        _write_run(tmp_path / "ex1", "ex1")
        out = resolve_run_dirs(tmp_path, ["ex1", "ex2"])
        assert "ex2" not in out


class TestCollectH0AndLabels:
    def test_basic_collection_and_labels(self, tmp_path):
        _write_run(tmp_path / "ex1", "ex1", new_tokens=" 7")   # correct
        _write_run(tmp_path / "ex2", "ex2", new_tokens=" 9")   # incorrect
        examples = [_example("ex1"), _example("ex2")]
        out = collect_h0_and_labels(examples, tmp_path, source="prompt_final")
        assert len(out) == 2
        by_id = {e["example_id"]: e for e in out}
        assert by_id["ex1"]["_correct"] is True
        assert by_id["ex2"]["_correct"] is False
        assert by_id["ex1"]["_h_0"].shape == (DIM,)
        assert "_run_dir" in by_id["ex1"]

    def test_never_mixes_sources(self, tmp_path):
        _write_run(tmp_path / "ex1", "ex1")                       # both files
        _write_run(tmp_path / "ex2", "ex2", with_prompt=False)   # gen only
        examples = [_example("ex1"), _example("ex2")]
        out = collect_h0_and_labels(examples, tmp_path, source="prompt_final")
        # ex2 has no activations.pt -> must be skipped, not silently
        # substituted with the first_gen state
        assert [e["example_id"] for e in out] == ["ex1"]

    def test_first_gen_source(self, tmp_path):
        _write_run(tmp_path / "ex1", "ex1")
        out = collect_h0_and_labels([_example("ex1")], tmp_path,
                                    source="first_gen")
        gen = torch.load(tmp_path / "ex1" / "gen_activations.pt",
                         weights_only=True)
        assert torch.allclose(out[0]["_h_0"], gen[0, -1, :].float())


class TestCollectH0MultiLayer:
    def test_layers_align_with_activations(self, tmp_path):
        _write_run(tmp_path / "ex1", "ex1", seed=1)
        _write_run(tmp_path / "ex2", "ex2", seed=2)
        examples = collect_h0_and_labels(
            [_example("ex1"), _example("ex2")], tmp_path)
        per_layer = collect_h0_multi_layer(examples, [0, 2])
        assert set(per_layer.keys()) == {0, 2}
        assert per_layer[0].shape == (2, DIM)
        act1 = torch.load(tmp_path / "ex1" / "activations.pt", weights_only=True)
        assert torch.allclose(per_layer[2][0], act1[2, -1, :].float())

    def test_missing_activations_raises(self, tmp_path):
        _write_run(tmp_path / "ex1", "ex1", with_prompt=False)
        examples = collect_h0_and_labels([_example("ex1")], tmp_path,
                                         source="first_gen")
        with pytest.raises(FileNotFoundError):
            collect_h0_multi_layer(examples, [0])


class TestSelectExamples:
    def _pool(self):
        # 3 families x 4 correct examples, probe scores spread around 0.5
        examples, scores = [], []
        s = 0.1
        for fam in ("a", "b", "c"):
            for i in range(4):
                examples.append({**_example(f"{fam}{i}", family=fam),
                                 "_correct": True})
                scores.append(s)
                s += 0.07
        return examples, torch.tensor(scores)

    def test_family_stratified(self):
        examples, scores = self._pool()
        sel = select_examples(examples, scores, 6, 0.2, 0.8)
        fams = [e["task_family"] for e in sel]
        assert fams.count("a") == 2
        assert fams.count("b") == 2
        assert fams.count("c") == 2

    def test_only_correct_selected(self):
        examples, scores = self._pool()
        examples[0]["_correct"] = False
        sel = select_examples(examples, scores, 12, 0.2, 0.8)
        assert all(e["_correct"] for e in sel)
        assert len(sel) == 11

    def test_boundary_preferred_within_family(self):
        examples = [
            {**_example("x0"), "_correct": True},
            {**_example("x1"), "_correct": True},
        ]
        scores = torch.tensor([0.95, 0.5])  # x1 in boundary, x0 not
        sel = select_examples(examples, scores, 1, 0.2, 0.8)
        assert sel[0]["example_id"] == "x1"

    def test_respects_n_prompts(self):
        examples, scores = self._pool()
        sel = select_examples(examples, scores, 5, 0.2, 0.8)
        assert len(sel) == 5

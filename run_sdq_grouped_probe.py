#!/usr/bin/env python3
"""Leakage-resistant correctness decodability audit on cached SDQ runs.

This is intentionally narrower than the legacy Phase 3 script.  It asks only:
does a linear probe predict final correctness when all paraphrases of an
underlying semantic task are held out together?  It compares prompt-final
states with crude input-surface features, a family prior, and (when cached)
the model's next-token confidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

import torch

from sdq.eval.commitment_probe import (
    LinearCommitmentProbe,
    Standardizer,
    compute_auroc,
    extract_logit_features,
    fit_probe,
    mean_within_family_auroc,
    per_family_auroc,
    predict_probe_scores,
)
from sdq.eval.grouped_splits import cluster_bootstrap_auroc, stratified_group_kfold
from sdq.eval.benchmark_audit import load_excluded_example_ids
from sdq.eval.intervention_data import load_benchmark, resolve_run_dirs
from sdq.labels.outcome_labeler import label_run


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Group-disjoint audit of prompt-final correctness probes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--benchmark", default="data/prompts/benchmark_v2.json")
    p.add_argument("--runs-dir", default="data/runs")
    p.add_argument(
        "--exclude-file", default="data/prompts/benchmark_v2_exclusions.json",
        help="JSON manifest of invalid cached examples to exclude; use an empty string for none",
    )
    p.add_argument("--layers", type=int, nargs="*", default=[0, 13, 25])
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--skip-logits", action="store_true")
    p.add_argument("--text-hash-dim", type=int, default=2048)
    p.add_argument("--skip-text-hash", action="store_true")
    p.add_argument("--output", default="outputs/reproduction/grouped_probe_results.json")
    return p.parse_args()


_NUM_RE = re.compile(r"\d+")
_WORD_RE = re.compile(r"[A-Za-z]+")


def surface_features(text: str) -> list[float]:
    """Small, interpretable prompt-only baseline (no transformer forward pass)."""
    words = text.split()
    numbers = _NUM_RE.findall(text)
    alpha_words = _WORD_RE.findall(text)
    n_words = max(len(words), 1)
    max_number = max((int(v) for v in numbers), default=0)
    return [
        float(len(text)),
        float(len(words)),
        len(set(words)) / n_words,
        sum(len(word) for word in words) / n_words,
        float(len(numbers)),
        float(torch.log1p(torch.tensor(float(max_number)))),
        float(sum(char.isdigit() for char in text)),
        float(sum(text.count(char) for char in "+-*/=<>")),
        float(text.count(",")),
        float(text.count(".")),
        float(text.count("?")),
        float(text.count(":")),
        float(sum(1 for word in alpha_words if word[:1].isupper())),
    ]


def hashed_char_ngrams(text: str, dim: int, min_n: int = 3, max_n: int = 5) -> torch.Tensor:
    """Leakage-free signed hashing baseline for prompt text.

    Unlike a learned TF-IDF vocabulary, the hashing map needs no fitting, so
    it can be computed once without leaking document frequencies across folds.
    It is much stronger than length/punctuation statistics and can represent
    template wording, names, operators, and digits.
    """
    normalized = " " + " ".join(text.lower().split()) + " "
    vector = torch.zeros(dim, dtype=torch.float32)
    for n in range(min_n, max_n + 1):
        for start in range(max(0, len(normalized) - n + 1)):
            gram = normalized[start:start + n].encode("utf-8")
            digest = hashlib.blake2b(gram, digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            index = value % dim
            sign = 1.0 if (value >> 63) == 0 else -1.0
            vector[index] += sign
    return vector / vector.norm().clamp_min(1.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_cached_features(
    benchmark_path: Path,
    runs_dir: Path,
    layers: list[int],
    include_logits: bool,
    text_hash_dim: int | None,
    exclude_ids: set[str] | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, list[str], list[str], list[str]]:
    examples = load_benchmark(benchmark_path)
    dir_map = resolve_run_dirs(runs_dir, [ex["example_id"] for ex in examples])
    features: dict[str, list[torch.Tensor]] = {f"hidden_L{layer}": [] for layer in layers}
    surface: list[torch.Tensor] = []
    text_hash: list[torch.Tensor] = []
    logit_features: list[torch.Tensor] = []
    labels: list[int] = []
    families: list[str] = []
    groups: list[str] = []
    example_ids: list[str] = []
    logits_complete = include_logits
    exclude_ids = exclude_ids or set()

    for ex in examples:
        if ex["example_id"] in exclude_ids:
            continue
        run_dir = dir_map.get(ex["example_id"])
        if run_dir is None or not (run_dir / "activations.pt").exists():
            continue
        with (run_dir / "metadata.json").open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        output = metadata.get("output", {}).get("new_tokens", "")
        outcome = label_run(output, ex["answer_id"], ex["task_family"])
        activations = torch.load(
            run_dir / "activations.pt", map_location="cpu", weights_only=True
        )
        for layer in layers:
            resolved = layer if layer >= 0 else activations.shape[0] + layer
            if not 0 <= resolved < activations.shape[0]:
                raise IndexError(f"layer {layer} is unavailable for {ex['example_id']}")
            features[f"hidden_L{layer}"].append(activations[resolved, -1].float())

        surface.append(torch.tensor(surface_features(ex["prompt_text"])))
        if text_hash_dim is not None:
            text_hash.append(hashed_char_ngrams(ex["prompt_text"], text_hash_dim))
        if include_logits:
            logit_path = run_dir / "logits.pt"
            if logit_path.exists():
                logits = torch.load(logit_path, map_location="cpu", weights_only=True)
                logit_features.append(extract_logit_features(logits[0, -1].float()))
            else:
                logits_complete = False
        labels.append(0 if outcome.correct else 1)
        families.append(ex["task_family"])
        groups.append(ex.get("semantic_task_id", ex.get("same_reasoning_group", ex["example_id"])))
        example_ids.append(ex["example_id"])

    out = {name: torch.stack(values) for name, values in features.items()}
    out["input_surface"] = torch.stack(surface)
    if text_hash_dim is not None:
        out["input_charhash_3to5"] = torch.stack(text_hash)
    if include_logits and logits_complete and len(logit_features) == len(labels):
        out["next_token_confidence"] = torch.stack(logit_features)
    return out, torch.tensor(labels), families, groups, example_ids


def evaluate_feature(
    X: torch.Tensor,
    y: torch.Tensor,
    families: list[str],
    groups: list[str],
    folds: list[tuple[torch.Tensor, torch.Tensor]],
    args: argparse.Namespace,
) -> dict:
    oof = torch.full((len(y),), float("nan"))
    fold_results = []
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        torch.manual_seed(args.seed + fold_idx)
        scaler = Standardizer.fit(X[train_idx])
        probe = fit_probe(
            LinearCommitmentProbe(X.shape[1]),
            scaler.transform(X[train_idx]),
            y[train_idx],
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=args.device,
        )
        scores = predict_probe_scores(
            probe, scaler.transform(X[test_idx]), device=args.device
        )
        oof[test_idx] = scores
        test_families = [families[i] for i in test_idx.tolist()]
        fold_results.append({
            "fold": fold_idx,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "n_train_groups": len({groups[i] for i in train_idx.tolist()}),
            "n_test_groups": len({groups[i] for i in test_idx.tolist()}),
            "pooled_auroc": compute_auroc(scores, y[test_idx]),
            "mean_within_family_auroc": mean_within_family_auroc(
                scores, y[test_idx], test_families
            ),
        })
    if torch.isnan(oof).any():
        raise RuntimeError("some examples did not receive an out-of-fold score")

    result = {
        "dim": X.shape[1],
        "pooled_auroc": compute_auroc(oof, y),
        "mean_within_family_auroc": mean_within_family_auroc(oof, y, families),
        "family_aurocs": per_family_auroc(oof, y, families),
        "family_counts": {
            family: {
                "n": sum(f == family for f in families),
                "incorrect": sum(f == family and int(label) == 1 for f, label in zip(families, y)),
            }
            for family in sorted(set(families))
        },
        "folds": fold_results,
        "oof_scores": oof.tolist(),
    }
    result.update(cluster_bootstrap_auroc(
        oof, y, families, groups, n_bootstrap=args.bootstrap, seed=args.seed
    ))
    return result


def evaluate_family_prior(
    y: torch.Tensor,
    families: list[str],
    groups: list[str],
    folds: list[tuple[torch.Tensor, torch.Tensor]],
    bootstrap: int,
    seed: int,
) -> dict:
    oof = torch.empty(len(y))
    for train_idx, test_idx in folds:
        rates = {}
        for family in set(families):
            idx = [i for i in train_idx.tolist() if families[i] == family]
            rates[family] = float(y[idx].float().mean()) if idx else 0.5
        oof[test_idx] = torch.tensor([rates[families[i]] for i in test_idx.tolist()])
    # Cross-fitting makes the family rate vary slightly by fold.  Those
    # arbitrary fold-to-fold differences can create a below/above-chance
    # within-family AUROC even though the baseline contains no within-family
    # information.  Report the mathematical within-family value (0.5) and
    # retain the cross-fitted value only as a diagnostic.
    cross_fitted_within = mean_within_family_auroc(oof, y, families)
    bootstrap_result = cluster_bootstrap_auroc(
        oof, y, families, groups, n_bootstrap=bootstrap, seed=seed
    )
    result = {
        "dim": 1,
        "pooled_auroc": compute_auroc(oof, y),
        "mean_within_family_auroc": 0.5,
        "within_family_95ci": [0.5, 0.5],
        "family_aurocs": {family: 0.5 for family in sorted(set(families))},
        "cross_fitted_within_family_diagnostic": cross_fitted_within,
        "oof_scores": oof.tolist(),
    }
    result["pooled_95ci"] = bootstrap_result["pooled_95ci"]
    return result


def main() -> None:
    args = parse_args()
    started = time.time()
    benchmark_path = Path(args.benchmark)
    runs_dir = Path(args.runs_dir)
    exclude_path = Path(args.exclude_file) if args.exclude_file else None
    exclude_ids: set[str] = set()
    if exclude_path is not None and exclude_path.exists():
        benchmark_examples = load_benchmark(benchmark_path)
        exclude_ids = load_excluded_example_ids(exclude_path, benchmark_examples)
        print(f"Excluding {len(exclude_ids)} benchmark-invalid cached examples")
    print("Loading cached prompt states and labels...")
    features, y, families, groups, example_ids = load_cached_features(
        benchmark_path, runs_dir, args.layers, not args.skip_logits,
        None if args.skip_text_hash else args.text_hash_dim,
        exclude_ids,
    )
    print(
        f"Loaded {len(y)} examples in {len(set(groups))} semantic groups: "
        f"{int((y == 0).sum())} correct, {int((y == 1).sum())} incorrect"
    )
    folds = stratified_group_kfold(y, families, groups, args.folds, args.seed)
    for train_idx, test_idx in folds:
        train_groups = {groups[i] for i in train_idx.tolist()}
        test_groups = {groups[i] for i in test_idx.tolist()}
        if train_groups.intersection(test_groups):
            raise RuntimeError("semantic group leakage detected")

    results = {
        "status": "audit_rerun",
        "question": "correctness decodability under semantic-task-disjoint evaluation",
        "provenance": {
            "git_head": git_head(),
            "benchmark": str(benchmark_path),
            "benchmark_sha256": sha256(benchmark_path),
            "exclusion_manifest": str(exclude_path) if exclude_path else None,
            "exclusion_manifest_sha256": (
                sha256(exclude_path) if exclude_path is not None and exclude_path.exists() else None
            ),
            "excluded_example_ids": sorted(exclude_ids),
            "runs_dir": str(runs_dir),
            "example_ids_sha256": hashlib.sha256(
                "\n".join(example_ids).encode("utf-8")
            ).hexdigest(),
            "torch_version": torch.__version__,
        },
        "args": vars(args),
        "dataset": {
            "n": len(y),
            "n_correct": int((y == 0).sum()),
            "n_incorrect": int((y == 1).sum()),
            "n_semantic_groups": len(set(groups)),
            "n_families": len(set(families)),
            "example_ids": example_ids,
            "semantic_groups": groups,
            "labels": y.tolist(),
            "families": families,
        },
        "results": {},
    }

    results["results"]["family_prior"] = evaluate_family_prior(
        y, families, groups, folds, args.bootstrap, args.seed
    )
    for name, X in features.items():
        print(f"Evaluating {name} ({X.shape[1]} dimensions)...")
        result = evaluate_feature(X, y, families, groups, folds, args)
        results["results"][name] = result
        lo, hi = result["within_family_95ci"]
        print(
            f"  within-family AUROC {result['mean_within_family_auroc']:.3f} "
            f"(group-bootstrap 95% CI {lo:.3f}-{hi:.3f}); "
            f"pooled {result['pooled_auroc']:.3f}"
        )

    results["elapsed_seconds"] = time.time() - started
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"Saved {args.output} in {results['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()

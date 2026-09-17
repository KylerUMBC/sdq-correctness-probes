"""SDQ Phase 6a: layer x position commitment map.

Trains the standard linear commitment probe at every (layer, prompt position)
cell of the cached prefill activations and maps test AUROC over the grid.
Pure analysis of existing activations.pt files — no model, no generation.

Question this answers: WHERE (depth) and WHEN (sequence position) does the
correctness-predictive signal arise? This bounds the interpretation of the
Phase 4/4b/5 intervention nulls: if the signal is already saturated well
before the intervened layers (13/19), the interventions were downstream of
commitment and the nulls say nothing about the signal's causal role at its
formation site.

Position spec:
    "f0.25"  fractional position round(0.25 * (seq_len - 1))
    "-4"     offset from prompt end (clamped to 0)
The same examples appear in every cell, so cells are directly comparable.

Run (pod):
    python run_sdq_commitment_map.py --runs-dir data/runs/gemma-2-2b/gemma-2-2b
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import Tensor

from sdq.eval.commitment_probe import LinearCommitmentProbe, train_probe
from sdq.eval.intervention_data import load_benchmark, resolve_run_dirs
from sdq.labels.outcome_labeler import label_run

DEFAULT_POSITIONS = ["f0.0", "f0.25", "f0.5", "f0.75",
                     "-16", "-8", "-4", "-2", "-1"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SDQ Phase 6a: layer x position commitment map",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--benchmark", default="data/prompts/benchmark_v2.json")
    p.add_argument("--runs-dir", required=True)
    p.add_argument("--layers", type=int, nargs="*", default=None,
                   help="Layer indices to map (default: all captured layers)")
    p.add_argument("--positions", nargs="*", default=DEFAULT_POSITIONS,
                   help="Position specs: fN.N (fraction) or -N (offset from end)")
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--test-frac", type=float, default=0.25)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default="commitment_map_results.json")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def position_index(spec: str, seq_len: int) -> int:
    if spec.startswith("f"):
        frac = float(spec[1:])
        return round(frac * (seq_len - 1))
    return max(0, seq_len + int(spec))


def load_grid(
    examples: list[dict],
    runs_dir: Path,
    layers: list[int] | None,
    position_specs: list[str],
) -> tuple[Tensor, Tensor, list[str], list[int]]:
    """Load activations[layer, pos_spec] for every labeled example.

    Returns (X [N, L, P, D] float16, y [N], families [N], layers).
    Each activations.pt is read once and immediately reduced to the needed
    slices; the full grid for ~1250 examples x 26 layers x 9 positions is
    ~0.5 GB in fp16.
    """
    eids = [ex["example_id"] for ex in examples]
    dir_map = resolve_run_dirs(runs_dir, eids)

    feats: list[Tensor] = []
    labels: list[int] = []
    families: list[str] = []
    resolved_layers = layers
    n_missing = 0
    t0 = time.time()

    for i, ex in enumerate(examples):
        run_dir = dir_map.get(ex["example_id"])
        if run_dir is None or not (run_dir / "activations.pt").exists():
            n_missing += 1
            continue
        with open(run_dir / "metadata.json", encoding="utf-8") as f:
            meta = json.load(f)
        new_tokens = meta.get("output", {}).get("new_tokens", "")
        outcome = label_run(new_tokens, ex["answer_id"], ex["task_family"])

        act = torch.load(run_dir / "activations.pt", map_location="cpu",
                         weights_only=True)  # [L, S, D]
        if resolved_layers is None:
            resolved_layers = list(range(act.shape[0]))
        seq_len = act.shape[1]
        pos_idx = [position_index(s, seq_len) for s in position_specs]
        # [L_sel, P, D]
        sl = act[resolved_layers][:, pos_idx, :].to(torch.float16)
        feats.append(sl)
        labels.append(0 if outcome.correct else 1)
        families.append(ex["task_family"])

        if (i + 1) % 200 == 0:
            print(f"  [load] {i + 1}/{len(examples)}  {time.time() - t0:.0f}s")

    if n_missing:
        print(f"  [load] skipped {n_missing} examples without activations.pt")
    X = torch.stack(feats)  # [N, L, P, D]
    y = torch.tensor(labels, dtype=torch.long)
    return X, y, families, resolved_layers


def balanced_split(
    y: Tensor,
    families: list[str],
    test_frac: float,
    seed: int,
) -> tuple[Tensor, Tensor]:
    """Class-balanced subsample + stratified train/test split.

    Mirrors Phase 3: subsample the majority class to match the minority,
    then split within each (family, label) stratum. Returns (train_idx,
    test_idx) — the SAME split is reused for every (layer, position) cell so
    cells differ only in features.
    """
    g = torch.Generator().manual_seed(seed)
    idx_by_class = {c: (y == c).nonzero(as_tuple=True)[0] for c in (0, 1)}
    n_min = min(len(v) for v in idx_by_class.values())
    kept = []
    for c, idx in idx_by_class.items():
        perm = idx[torch.randperm(len(idx), generator=g)]
        kept.append(perm[:n_min])
    kept = torch.cat(kept)

    strata: dict[tuple[str, int], list[int]] = {}
    for i in kept.tolist():
        strata.setdefault((families[i], int(y[i])), []).append(i)

    train_idx, test_idx = [], []
    for key in sorted(strata):
        members = torch.tensor(strata[key])
        perm = members[torch.randperm(len(members), generator=g)]
        n_test = max(1, round(len(perm) * test_frac))
        test_idx.extend(perm[:n_test].tolist())
        train_idx.extend(perm[n_test:].tolist())
    return torch.tensor(train_idx), torch.tensor(test_idx)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    t_start = time.time()

    print("=" * 70)
    print("SDQ PHASE 6a: LAYER x POSITION COMMITMENT MAP")
    print("=" * 70)

    examples = load_benchmark(args.benchmark)
    runs_dir = Path(args.runs_dir)
    print(f"Loading activation grid from {runs_dir} "
          f"(positions: {args.positions}) ...")
    X, y, families, layers = load_grid(examples, runs_dir, args.layers,
                                       args.positions)
    n, L, P, D = X.shape
    print(f"Loaded grid: {n} examples x {L} layers x {P} positions x {D} dims "
          f"({X.element_size() * X.nelement() / 1e9:.2f} GB)")
    print(f"Labels: {int((y == 0).sum())} correct, {int((y == 1).sum())} incorrect")

    train_idx, test_idx = balanced_split(y, families, args.test_frac, args.seed)
    test_families = [families[i] for i in test_idx.tolist()]
    print(f"Split: {len(train_idx)} train / {len(test_idx)} test "
          f"(balanced, family-stratified, shared across all cells)")

    grid: list[dict] = []
    n_cells = L * P
    done = 0
    for li, layer in enumerate(layers):
        for pi, pos in enumerate(args.positions):
            Xc = X[:, li, pi, :].float()
            res = train_probe(
                LinearCommitmentProbe(D),
                Xc[train_idx], y[train_idx],
                Xc[test_idx], y[test_idx],
                test_families,
                name=f"L{layer}/{pos}",
                epochs=args.epochs,
                device=args.device,
            )
            grid.append({
                "layer": layer,
                "position": pos,
                "pooled_auroc": res.pooled_auroc,
                "mean_within_family_auroc": res.mean_within_family_auroc,
                "family_aurocs": res.family_aurocs,
            })
            done += 1
        elapsed = time.time() - t_start
        print(f"  [{done:4d}/{n_cells}] layer {layer} done  {elapsed:.0f}s")

    # ASCII heat tables
    by_cell = {(c["layer"], c["position"]): c for c in grid}
    for metric, label in (("mean_within_family_auroc", "WITHIN-FAMILY"),
                          ("pooled_auroc", "POOLED")):
        print(f"\n  === {label} AUROC (rows: layer, cols: position) ===")
        print("  layer  " + "  ".join(f"{p:>6s}" for p in args.positions))
        for layer in layers:
            row = "  ".join(f"{by_cell[(layer, p)][metric]:6.3f}"
                            for p in args.positions)
            print(f"  {layer:5d}  {row}")

    best = max(grid, key=lambda c: c["mean_within_family_auroc"])
    print(f"\nBest cell (within-family): layer {best['layer']} pos "
          f"{best['position']} AUROC {best['mean_within_family_auroc']:.3f}")

    results = {
        "summary": {
            "n_examples": n,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "layers": layers,
            "positions": args.positions,
            "epochs": args.epochs,
            "seed": args.seed,
            "metric_note": ("mean_within_family_auroc is the honest headline; "
                            "pooled includes family-identity signal "
                            "(family prior alone ~0.78 pooled)"),
        },
        "grid": grid,
        "elapsed_seconds": time.time() - t_start,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")
    print(f"Total time: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()

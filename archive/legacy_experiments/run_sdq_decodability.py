"""SDQ Phase 6d: input-decodability + linear-vs-nonlinear readout.

Two cheap (CPU-only) questions on cached data, answered with one shared,
balanced, family-stratified split (the same recipe as the 6a map):

(1) INPUT-DETERMINED?  Is correctness already decodable from the *input*
    alone — hand-crafted surface features and/or the mean token embedding
    (bag-of-embeddings, no transformer computation) — at a within-family
    AUROC close to the hidden-state probes? If yes, the predictive signal is
    a shallow property of the prompt, which is the mechanistic content of
    "thermometer, not lever".

(2) NONLINEAR READOUT?  Does a 1-hidden-layer MLP read the hidden state
    meaningfully better than a linear probe? Compared on the SAME PCA(k)
    features so the test isolates the function class (linear vs nonlinear),
    not raw capacity. A robust MLP>linear gap across seeds is the first
    positive evidence that the signal has nonlinear structure a linear
    intervention could never actuate; a tie means "linearly" is not doing
    hidden work in the headline claim.

No GPU: surface features need nothing; mean embeddings need only the
embedding matrix (a CPU lookup, never a forward pass); probes are tiny.

Run (pod, CPU fine):
    python run_sdq_decodability.py --runs-dir data/runs/gemma-2-2b/gemma-2-2b
Skip the model entirely (surface + hidden only):
    python run_sdq_decodability.py --runs-dir ... --no-embeddings
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from sklearn.decomposition import PCA

from sdq.eval.commitment_probe import (
    LinearCommitmentProbe,
    LowRankCommitmentProbe,
    Standardizer,
    compute_auroc,
    fit_probe,
    mean_within_family_auroc,
    predict_probe_scores,
)
from sdq.eval.intervention_data import load_benchmark, resolve_run_dirs
from sdq.labels.outcome_labeler import label_run
from run_sdq_commitment_map import balanced_split


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SDQ Phase 6d: input decodability + nonlinear readout",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--benchmark", default="data/prompts/benchmark_v2.json")
    p.add_argument("--runs-dir", required=True)
    p.add_argument("--layers", type=int, nargs="*", default=[0, 13],
                   help="Hidden layers to probe (last prompt token)")
    p.add_argument("--embeddings", dest="embeddings", action="store_true",
                   default=True, help="Include mean-token-embedding features")
    p.add_argument("--no-embeddings", dest="embeddings", action="store_false")
    p.add_argument("--config", default="configs/model.yaml")
    p.add_argument("--model-path", default=None)
    p.add_argument("--pca-dim", type=int, default=64,
                   help="PCA dim for the linear-vs-MLP comparison")
    p.add_argument("--mlp-hidden", type=int, default=32)
    p.add_argument("--mlp-seeds", type=int, default=5)
    p.add_argument("--n-splits", type=int, default=3,
                   help="Train/test splits averaged in the linear-vs-MLP "
                        "comparison (guards the gap against partition noise)")
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--test-frac", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--map-results", default="commitment_map_results.json",
                   help="Optional 6a map JSON for side-by-side reference")
    p.add_argument("--output", default="decodability_results.json")
    return p.parse_args()


# ── Input-only feature sets ─────────────────────────────────────────────────

_NUM_RE = re.compile(r"\d+")
_WORD_RE = re.compile(r"[A-Za-z]+")

SURFACE_NAMES = [
    "n_chars", "n_words", "n_unique_word_ratio", "avg_word_len",
    "n_numbers", "log_max_number", "n_digits",
    "n_math_ops", "n_commas", "n_periods", "n_question", "n_colons",
    "n_capitalized_words",
]


def surface_features(text: str) -> list[float]:
    """Interpretable, model-free surface descriptors of a prompt.

    Deliberately crude: if even these predict correctness within-family,
    the signal is unambiguously shallow. They exclude family identity (the
    within-family metric already controls for it)."""
    words = text.split()
    nums = _NUM_RE.findall(text)
    max_num = max((int(n) for n in nums), default=0)
    alpha_words = _WORD_RE.findall(text)
    cap = sum(1 for w in alpha_words if w[:1].isupper())
    nw = max(len(words), 1)
    return [
        float(len(text)),
        float(len(words)),
        len(set(words)) / nw,
        sum(len(w) for w in words) / nw,
        float(len(nums)),
        float(torch.log1p(torch.tensor(float(max_num)))),
        float(sum(c.isdigit() for c in text)),
        float(sum(text.count(c) for c in "+-*/=<>")),
        float(text.count(",")),
        float(text.count(".")),
        float(text.count("?")),
        float(text.count(":")),
        float(cap),
    ]


# ── Loading ─────────────────────────────────────────────────────────────────

def load_data(benchmark, runs_dir, layers):
    """Returns examples in benchmark order that have activations.pt, with
    prompt text, family, label (1=incorrect), and last-token hidden states.

    Requiring activations.pt keeps the population identical to the 6a map,
    so AUROCs are directly comparable to it."""
    examples = load_benchmark(benchmark)
    eids = [ex["example_id"] for ex in examples]
    dir_map = resolve_run_dirs(Path(runs_dir), eids)

    prompts, families, labels = [], [], []
    hidden = {l: [] for l in layers}
    n_missing = 0
    for ex in examples:
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
        for l in layers:
            hidden[l].append(act[l, -1, :].float())
        prompts.append(ex.get("prompt_text", ex.get("prompt", "")))
        families.append(ex["task_family"])
        labels.append(0 if outcome.correct else 1)

    if n_missing:
        print(f"  [load] skipped {n_missing} examples without activations.pt")
    y = torch.tensor(labels, dtype=torch.long)
    hidden = {l: torch.stack(v) for l, v in hidden.items()}
    return prompts, families, y, hidden


def mean_embeddings(prompts, config, model_path):
    """Mean token embedding per prompt — the input with zero transformer
    computation. CPU embedding-matrix lookup only; returns None on failure."""
    try:
        from sdq.instrumentation.model_loader import load_model
        bundle = load_model(config, model_path_override=model_path)
        tok = bundle.tokenizer
        emb = bundle.model.get_input_embeddings().weight.detach().cpu().float()
    except Exception as e:  # noqa: BLE001 — embeddings are optional
        print(f"  [embed] could not load embedding matrix ({e}); skipping.")
        return None
    feats = []
    for text in prompts:
        ids = tok(text, return_tensors="pt")["input_ids"][0]
        feats.append(emb[ids].mean(0))
    print(f"  [embed] built mean embeddings for {len(prompts)} prompts "
          f"(dim {emb.shape[1]})")
    return torch.stack(feats)


# ── Probe fit/eval (train AND test, for overfit visibility) ─────────────────

def fit_eval(make_probe, Xtr, ytr, Xte, yte, te_fams, epochs, seed):
    torch.manual_seed(seed)
    sc = Standardizer.fit(Xtr)
    probe = fit_probe(make_probe(), sc.transform(Xtr), ytr, epochs=epochs)
    s_te = predict_probe_scores(probe, sc.transform(Xte))
    s_tr = predict_probe_scores(probe, sc.transform(Xtr))
    return {
        "test_within": mean_within_family_auroc(s_te, yte, te_fams),
        "test_pooled": compute_auroc(s_te, yte),
        "train_pooled": compute_auroc(s_tr, ytr),
    }


def _mean_std(xs):
    n = len(xs)
    m = sum(xs) / n
    sd = (sum((x - m) ** 2 for x in xs) / n) ** 0.5 if n > 1 else 0.0
    return m, sd


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    t0 = time.time()
    print("=" * 70)
    print("SDQ PHASE 6d: INPUT DECODABILITY + NONLINEAR READOUT")
    print("=" * 70)

    prompts, families, y, hidden = load_data(args.benchmark, args.runs_dir,
                                             args.layers)
    n = len(y)
    print(f"Loaded {n} examples: {int((y == 0).sum())} correct, "
          f"{int((y == 1).sum())} incorrect")

    # Feature sets
    feature_sets: dict[str, torch.Tensor] = {}
    feature_sets["input_surface"] = torch.tensor(
        [surface_features(t) for t in prompts], dtype=torch.float32)
    if args.embeddings:
        me = mean_embeddings(prompts, args.config, args.model_path)
        if me is not None:
            feature_sets["input_meanembed"] = me
    for l in args.layers:
        feature_sets[f"hidden_L{l}"] = hidden[l]

    # Shared split (identical recipe to the 6a map)
    train_idx, test_idx = balanced_split(y, families, args.test_frac, args.seed)
    tr, te = train_idx.tolist(), test_idx.tolist()
    te_fams = [families[i] for i in te]
    ytr, yte = y[train_idx], y[test_idx]
    print(f"Split: {len(tr)} train / {len(te)} test (balanced, stratified)\n")

    results: dict = {"summary": {
        "n": n, "n_train": len(tr), "n_test": len(te),
        "layers": args.layers, "pca_dim": args.pca_dim,
        "mlp_hidden": args.mlp_hidden, "mlp_seeds": args.mlp_seeds,
        "seed": args.seed, "feature_sets": list(feature_sets.keys()),
    }, "full_dim_linear": {}, "pca_linear_vs_mlp": {}}

    # ── (1) Full-dim LINEAR probe on every feature set ──
    print("=== FULL-DIM LINEAR probe (within-family / pooled AUROC) ===")
    print(f"  {'feature_set':>18s}  {'dim':>5s}  {'within':>7s}  {'pooled':>7s}")
    for name, X in feature_sets.items():
        d = X.shape[1]
        r = fit_eval(lambda d=d: LinearCommitmentProbe(d), X[train_idx], ytr,
                     X[test_idx], yte, te_fams, args.epochs, args.seed)
        results["full_dim_linear"][name] = {"dim": X.shape[1], **r}
        print(f"  {name:>18s}  {X.shape[1]:5d}  {r['test_within']:7.3f}  "
              f"{r['test_pooled']:7.3f}")

    # ── (2) LINEAR vs MLP on the SAME PCA(k) features ──
    # Averaged over n_splits partitions (partition noise) x mlp_seeds (init
    # noise). PCA is refit on each split's TRAIN only. train_pooled exposes
    # MLP overfit: a gap is only believable if MLP test rises without
    # train_pooled being pathologically higher than linear's.
    print(f"\n=== PCA({args.pca_dim}) LINEAR vs MLP (nonlinear-readout test, "
          f"{args.n_splits} splits) ===")
    print(f"  {'feature_set':>18s}  {'probe':>7s}  {'within(test)':>14s}  "
          f"{'pooled(test)':>13s}  {'pooled(train)':>13s}")
    nonlin_sets = [k for k in feature_sets
                   if k.startswith("hidden_") or k == "input_meanembed"]
    for name in nonlin_sets:
        X = feature_sets[name]
        lin_w, lin_p, lin_tr = [], [], []
        mlp_w, mlp_p, mlp_tr = [], [], []
        kk = None
        for sp in range(args.n_splits):
            tri, tei = balanced_split(y, families, args.test_frac,
                                      args.seed + sp)
            yt, ye = y[tri], y[tei]
            ef = [families[i] for i in tei.tolist()]
            kk = min(args.pca_dim, X.shape[1], len(tri) - 1)
            pca = PCA(n_components=kk, random_state=args.seed)
            Xtr_p = torch.tensor(pca.fit_transform(X[tri].numpy()),
                                 dtype=torch.float32)
            Xte_p = torch.tensor(pca.transform(X[tei].numpy()),
                                 dtype=torch.float32)
            r = fit_eval(lambda kk=kk: LinearCommitmentProbe(kk), Xtr_p, yt,
                         Xte_p, ye, ef, args.epochs, args.seed)
            lin_w.append(r["test_within"]); lin_p.append(r["test_pooled"])
            lin_tr.append(r["train_pooled"])
            for s in range(args.mlp_seeds):
                m = fit_eval(lambda kk=kk: LowRankCommitmentProbe(kk, args.mlp_hidden),
                             Xtr_p, yt, Xte_p, ye, ef, args.epochs, s)
                mlp_w.append(m["test_within"]); mlp_p.append(m["test_pooled"])
                mlp_tr.append(m["train_pooled"])

        lw_m, lw_s = _mean_std(lin_w)
        mw_m, mw_s = _mean_std(mlp_w)
        gap = mw_m - lw_m
        results["pca_linear_vs_mlp"][name] = {
            "pca_dim": kk, "n_splits": args.n_splits,
            "linear": {"test_within_mean": lw_m, "test_within_std": lw_s,
                       "test_pooled_mean": _mean_std(lin_p)[0],
                       "train_pooled_mean": _mean_std(lin_tr)[0]},
            "mlp": {"test_within_mean": mw_m, "test_within_std": mw_s,
                    "test_pooled_mean": _mean_std(mlp_p)[0],
                    "train_pooled_mean": _mean_std(mlp_tr)[0]},
            "mlp_minus_linear_within": gap,
        }
        print(f"  {name:>18s}  {'linear':>7s}  {lw_m:6.3f}±{lw_s:.3f}   "
              f"{_mean_std(lin_p)[0]:13.3f}  {_mean_std(lin_tr)[0]:13.3f}")
        print(f"  {'':>18s}  {'mlp':>7s}  {mw_m:6.3f}±{mw_s:.3f}   "
              f"{_mean_std(mlp_p)[0]:13.3f}  {_mean_std(mlp_tr)[0]:13.3f}")
        print(f"  {'':>18s}  -> MLP - linear (within-family) = {gap:+.3f}")

    # ── Reference: 6a map numbers ──
    ref = {}
    map_path = Path(args.map_results)
    if map_path.exists():
        mp = json.load(open(map_path))
        by = {(c["layer"], c["position"]): c for c in mp["grid"]}
        best = max(mp["grid"], key=lambda c: c["mean_within_family_auroc"])
        for l in args.layers:
            if (l, "-1") in by:
                ref[f"map_L{l}_pos-1_within"] = by[(l, "-1")]["mean_within_family_auroc"]
        ref["map_best_within"] = {"layer": best["layer"], "position": best["position"],
                                  "auroc": best["mean_within_family_auroc"]}
        results["map_reference"] = ref
        print("\n=== 6a MAP REFERENCE (hidden-state linear, last token) ===")
        for l in args.layers:
            key = f"map_L{l}_pos-1_within"
            if key in ref:
                print(f"  layer {l} pos -1 within-family: {ref[key]:.3f}")
        print(f"  best map cell: L{best['layer']} {best['position']} "
              f"= {best['mean_within_family_auroc']:.3f}")

    results["elapsed_seconds"] = time.time() - t0
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")
    print(f"Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

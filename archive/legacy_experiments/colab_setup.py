#!/usr/bin/env python3
"""Colab setup helper for SDQ v5.

Run this LOCALLY (not on Colab) to create a minimal data tarball,
then use the generated Colab notebook cell to set everything up.

Usage (local machine):
    python colab_setup.py pack          # creates sdq_data.tar.gz (~700MB)
    python colab_setup.py pack --slim   # activations only, no logits (~700MB)

Then upload sdq_data.tar.gz to Google Drive and run the Colab cells below.
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
from pathlib import Path


def pack_data(slim: bool = True):
    """Create a tarball with only the files needed for training.

    Includes:
      - data/prompts/benchmark_v1.json
      - data/prompts/splits/split_semantic_family.json
      - data/runs/bench_*/activations.pt   (hidden states)
      - data/runs/bench_*/metadata.json    (prompt info)
      - data/runs/bench_*/benchmark_meta.json (if exists)

    Excludes:
      - logits.pt (~3MB each, not needed for training)
      - non-bench_ runs (legacy test runs)
    """
    root = Path(__file__).parent
    out_path = root / "sdq_data.tar.gz"

    files_to_pack: list[tuple[str, Path]] = []

    # Prompt data
    prompts_dir = root / "data" / "prompts"
    for f in [
        prompts_dir / "benchmark_v1.json",
        prompts_dir / "splits" / "split_semantic_family.json",
    ]:
        if f.exists():
            files_to_pack.append((str(f.relative_to(root)), f))
        else:
            print(f"WARNING: {f} not found!")

    # Bench runs (activations + metadata only)
    runs_dir = root / "data" / "runs"
    bench_count = 0
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith("bench_"):
            continue

        for fname in ["activations.pt", "metadata.json", "benchmark_meta.json"]:
            fpath = run_dir / fname
            if fpath.exists():
                rel = str(fpath.relative_to(root))
                files_to_pack.append((rel, fpath))

        bench_count += 1

    print(f"Packing {bench_count} bench runs, {len(files_to_pack)} files total")

    # Create tarball
    with tarfile.open(out_path, "w:gz") as tar:
        for arcname, fpath in files_to_pack:
            tar.add(str(fpath), arcname=arcname)
            if bench_count <= 5 or files_to_pack.index((arcname, fpath)) < 10:
                print(f"  + {arcname}")

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\nCreated {out_path} ({size_mb:.1f} MB)")
    print(f"Upload this to Google Drive, then use the Colab cells below.")


def print_colab_cells():
    """Print the Colab notebook cells to copy-paste."""
    print("""
# ============================================================
# COLAB CELL 1: Setup environment
# ============================================================
# Paste this into the first cell of a new Colab notebook:

!pip install -q torch transformers pyyaml python-dotenv numpy

# Clone the repo
!git clone https://github.com/KylerUMBC/SDQ-v1.git /content/SDQ-v1
%cd /content/SDQ-v1

# Install the sdq package in editable mode
!pip install -q -e .

# ============================================================
# COLAB CELL 2: Mount Drive & extract data
# ============================================================
# Upload sdq_data.tar.gz to your Google Drive root, then:

from google.colab import drive
drive.mount('/content/drive')

import shutil, os
os.makedirs('data/runs', exist_ok=True)
os.makedirs('data/prompts/splits', exist_ok=True)

# Extract the data tarball
!tar xzf /content/drive/MyDrive/sdq_data.tar.gz -C /content/SDQ-v1/

# Verify
!ls data/prompts/benchmark_v1.json
!ls data/runs/bench_* | head -5
!echo "Total bench runs:" && ls -d data/runs/bench_* | wc -l

# ============================================================
# COLAB CELL 3: Run training
# ============================================================

!python run_sdq_v5.py \\
    --device cuda \\
    --epochs-phase1 100 \\
    --epochs-phase2 50 \\
    --output sdq_v5_results.json

# ============================================================
# COLAB CELL 4: Save results back to Drive
# ============================================================

!cp sdq_v5_results.json /content/drive/MyDrive/sdq_v5_results.json
print("Results saved to Drive!")

# ============================================================
# COLAB CELL 5 (optional): Quick results summary
# ============================================================

import json
with open('sdq_v5_results.json') as f:
    r = json.load(f)
s = r['summary']
print(f"Retrieval accuracy:  {s['retrieval_accuracy']:.1%}")
print(f"Scorecard:           {s['scorecard']:.1f}/100")
print(f"Task family probe:   {s['task_family_probe_acc']:.1%}")
print(f"Semantic group probe:{s['semantic_group_probe_acc']:.1%}")
print(f"Attractors:          {s['num_attractors']}")
print(f"Attractor purity:    {s['answer_attractor_purity']:.1%}")

# Check Phase 1 convergence (should see supcon decreasing, within_cos rising)
hist = r['training_history']
p1 = [h for h in hist if h['phase'] == 1]
if p1:
    print(f"\\nPhase 1: {len(p1)} epochs")
    print(f"  Start supcon: {p1[0]['supcon']:.4f}, End: {p1[-1]['supcon']:.4f}")
    print(f"  Start w_cos:  {p1[0]['within_cos']:.3f}, End: {p1[-1]['within_cos']:.3f}")
    print(f"  Start x_cos:  {p1[0]['cross_cos']:.3f}, End: {p1[-1]['cross_cos']:.3f}")
""")


def main():
    parser = argparse.ArgumentParser(description="SDQ Colab setup helper")
    parser.add_argument("action", choices=["pack", "cells"],
                        help="'pack' = create data tarball, 'cells' = print Colab cells")
    parser.add_argument("--slim", action="store_true",
                        help="Exclude logits (default: already excluded)")
    args = parser.parse_args()

    if args.action == "pack":
        pack_data(slim=args.slim)
        print("\n" + "=" * 60)
        print_colab_cells()
    elif args.action == "cells":
        print_colab_cells()


if __name__ == "__main__":
    main()

# Historical material

The current writeup and analysis live in [`paper/`](../paper/),
[`results/`](../results/), and the two top-level analysis runners. This archive
preserves the exploratory history; its claims, statistics, and environment
instructions are not the current paper's conclusions.

| Location | Contents | Status |
| --- | --- | --- |
| Existing top-level files here | Original v1–v6 trajectory experiments and early results | Historical; retained at their previous archive paths |
| `legacy_experiments/` | EWS, random-split probes, layer maps, intervention runners, result JSONs, and the later commitment direction | Exploratory/superseded; known limitations in the research audit |
| `legacy_docs/` | Original formalism, assumptions, protocol, and certification notes | Historical hypotheses, not established findings |
| `project_notes/` | Old writeups, session/planning notes, and superseded preprint plan | Kept locally; excluded from the public snapshot |
| `visualizations/` | Earlier figures and their builder/source tables | Historical populations and statistical methods; not paper figures |
| `local_checkpoints/` | Large checkpoints present on this workstation | Ignored by Git; not part of a public checkout |

[`relocation_manifest.json`](relocation_manifest.json) maps old paths to new
paths and records SHA-256 hashes at relocation on 2026-09-17. Hashes of the
frozen experimental JSON files are unchanged. Entries marked
`resumed_after_move` were inventoried after resuming an interrupted relocation.
The manifest also lists local-only files; it is not a list of files distributed
in the public snapshot. The archived visualization builder and README subsequently received path/status
updates; the manifest records their pre-update bytes. Nothing was discarded.

Historical scripts may require old environments or unavailable Colab caches.
They have not been revalidated as current reproduction entry points. For an
archived script with sibling imports, invoke its file path from the repo root
after installing the `sdq` package. Old default output/checkpoint paths may
need explicit overrides. The one-time relocation script must not be rerun as
an analysis step.

Completed analyses retain original source-path strings inside their JSON
metadata for provenance. Use the relocation manifest to resolve those paths.
The package's historical modules remain in `sdq/` to preserve imports and tests.

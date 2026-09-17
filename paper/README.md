# Writeup and figures

[`preprint.md`](preprint.md) is the technical writeup. The directory and filename
are retained from an earlier paper draft; no publication or submission is
implied. Figures and tables in
[`generated/`](generated/) are built from the frozen JSON files under
[`results/`](../results/). They do not combine the audited probe population with
the older intervention populations.

## Rebuild the artifacts

```powershell
pip install -e ".[paper]"
python paper/build_artifacts.py
```

Run from the repository root. The builder writes Markdown tables, editable SVG
figures, and PNG previews. Matplotlib is pinned in the `paper` optional
dependency. `--tables-only` requires only the Python standard library.

Numeric labels occupy a separate column so that they cannot overlap interval
bars. Values use decimal half-up rounding to three places in both tables and
figures; the underlying stored estimates are unchanged. Intervals are pointwise,
not simultaneous intervals adjusted for the search.

To inspect a reproduction without replacing the paper artifacts:

```powershell
python paper/build_artifacts.py --results-dir outputs/reproduction --output-dir outputs/reproduction/figures
```

Both analysis JSON files must exist in the supplied results directory. Methods
and commands are in [`docs/reproducibility.md`](../docs/reproducibility.md).
Personal planning notes are kept locally and are not included in the public snapshot.

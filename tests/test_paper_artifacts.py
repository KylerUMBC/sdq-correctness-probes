"""Keep displayed values, frozen inputs, and figure layout consistent."""
import hashlib
import json
from pathlib import Path

import pytest

from paper.build_artifacts import (dataset_table, fmt, forest_plot,
                                   intervention_table, probe_table)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("value,signed,expected", [
    (0.0875, True, "+0.088"), (0.852637891, False, "0.853"),
    (-0.01190476, True, "-0.012"),
])
def test_decimal_display_rounding(value, signed, expected):
    assert fmt(value, signed) == expected


@pytest.mark.parametrize("name,digest", [
    ("grouped_probe_results.json", "6BE50835D9FBB3555BD598E9D0AFAB2A1D6BB6A04FF2F6960638F2A8EDAAA416"),
    ("intervention_paired_reanalysis.json", "BCD3492FD75A0A5E2A4AFF4DFE54272D0C754616AC0F6848A223A6DC62F74792"),
    ("interventions/mesoscale_results.json", "1E1AC6668144AB34882F4CA998802C94B92258F9A3CA5946F04346C93345CCF8"),
    ("interventions/confirmation_results.json", "D58EEE00D8394F2874959C9ED5F457FC424E21EEA97376297078F1F99AF92F1F"),
    ("interventions/prefill_all_results.json", "B03DED4188E1293EC3B2BBB2BDC67BA343965F716299B512CE356590E5C24CF7"),
])
def test_frozen_results_unchanged(name, digest):
    assert hashlib.sha256((ROOT / "results" / name).read_bytes()).hexdigest().upper() == digest


def test_generated_tables_match_frozen_results():
    grouped = json.loads((ROOT / "results/grouped_probe_results.json").read_text())
    intervention = json.loads((ROOT / "results/intervention_paired_reanalysis.json").read_text())
    for filename, expected in [
        ("probe_results.md", probe_table(grouped)),
        ("dataset_counts.md", dataset_table(grouped)),
        ("intervention_results.md", intervention_table(intervention)),
    ]:
        assert (ROOT / "paper/generated" / filename).read_text(encoding="utf-8") == expected


def test_numeric_labels_are_outside_plot_and_inside_figure():
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    rows = [("Exploratory, magnitude 3", .0875, (.0208, .1583), "#D97706"),
            ("Confirmation, magnitude 2", -.0119, (-.034, .0102), "#2563EB")]
    fig = forest_plot(rows, title="Layout regression", subtitle="Interval labels",
                      xlabel="Difference", limits=(-.06, .18), ticks=[0, .1], reference=0, signed=True)
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        plot = fig.axes[0].get_window_extent(renderer)
        for label in fig.axes[1].texts:
            bounds = label.get_window_extent(renderer)
            assert bounds.x0 > plot.x1
            assert bounds.x1 < fig.bbox.x1
            assert bounds.y0 >= 0 and bounds.y1 <= fig.bbox.y1
    finally:
        plt.close(fig)

# Archived SDQ visualizations

Historical figures, not the current paper figures. These retain legacy
populations, exploratory choices, and superseded statistical interpretations.
Current figures are in `paper/generated/`; the methodological audit explains
why the older claims are restricted. Source results now live under `archive/`
and `archive/legacy_experiments/`.

## Rebuild

```bash
python archive/visualizations/make_visualizations.py
```

The builder also needs pandas and seaborn, which are not required by the active
paper workflow. It has not been rerun as part of the current release check.
It writes only within this archived folder.

## Figures

- `family_auroc_breakdown`: within-family AUROC by task family for `h_0`, `z_0`, family prior, and the full EWS model.
- `probe_summary_aurocs`: headline AUROC comparison across the main probes and sequence models.
- `family_balance_vs_h0_auroc`: task-family accuracy/base-rate context against `h_0` probe strength.
- `subspace_rank_sweep`: pooled and within-family AUROC as PCA rank increases, with the 64/96 dimensionality markers.
- `cross_family_transfer`: held-out-family generalization of the commitment probe.
- `ews_timestep_auroc`: AUROC trajectory across generated-token timesteps.
- `flip_rates_by_magnitude`: presentation-focused layer-13 persistent-hook flip rates plus the directed/random ratio; the caption notes that the raw layer-25 prefill run was 0% across magnitudes.
- `layer13_flip_ratio`: directed/random flip ratio at layer 13, highlighting the narrow 1x std anomaly.
- `layer19_subspace_specificity`: layer-19 persistent-hook subspace intervention results, showing all perturbation conditions clustered near 6% and key ratios near 1x.
- `layer13_layer19_subspace_specificity`: combined layer-13/layer-19 subspace-specificity view. It shows the saved mean in-subspace/random-subspace ratios for both layers and the recovered layer-19 condition-level details.
- `layer_sweep_summary`: compact layer-depth view of the Phase 4 causal sweep story, including the layer-6 7.5% directed-flip result without inventing an unavailable ratio.
- `pca_spectrum_top20`: top-20 PCA scree plot. This says raw variance is front-loaded, but it is not the main predictive result; the rank sweep is the evidence that useful commitment prediction needs about 64-96 dimensions.

## Sources

Most tables are read directly from `commitment_results.json`, `commitment_intervention_results.json`, `ews_results2.json`, and `ews_probe_results2.json`. The post-fix layer-13, layer-19, and layer-depth intervention summaries are preserved from raw console/writeup tables, because the root `commitment_intervention_results.json` artifact contains the earlier layer-25 prefill run.

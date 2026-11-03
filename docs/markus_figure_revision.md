# Markus figure revision — 3 November 2026

This pass changes presentation only. It preserves the input data, estimators,
censoring conventions, fleet-level resampling and confidence intervals.
The existing filenames are preserved for Overleaf.

| Manuscript figure | Implementation |
| --- | --- |
| Main Figure 2 | Larger, dark annotation stating the capped-mean interpretation. Distinct line styles and markers. |
| Supplement S2(a) | Legend above the axes, away from the curves near evaluation 3. |
| Supplement S2(b) | Legend above the axes, clear of the bar labels. Subtle hatch patterns distinguish methods in grayscale. |
| Supplement S3 | Display offsets of −0.10, 0 and +0.10 separate confidence intervals. Tick labels retain actual source counts. Wider canvas reduces final text size. |
| Supplement S4 | Wider canvas and restrained fonts match the manuscript scale. |

A shared style module supplies serif typography, embedded TrueType fonts in vector
PDFs, consistent colors, light grids and 600-dpi PNG exports. The maturity plots
use the same orange for global warm start as the other plots.

## Run locally

From the repository root in the existing project environment:

```bash
python code/scripts/make_paper_figures.py --strict
python code/scripts/make_maturity_figures.py
```

The first command requires the original processed campaign files, including
`reference_robustness`, `reference_robustness_rgpe`,
`phase2_coverage_dev_c1cds02`, `calibration_landscape_similarity`, and
`phase2_fbo_comparison_campaign_v3_c1cds05` under `code/results/`.
Parquet inputs require the project's parquet dependency (e.g. pyarrow).
Without `--strict`, missing files are reported and their existing PDFs preserved.
Do not mistake preserved old PDFs for regenerated figures.

After regeneration, compile the current Overleaf manuscript and inspect figures
at actual display size. Its blue and purple text revisions are newer than the
Git LaTeX sources. Upload the figure PDFs without replacing the LaTeX sources.
Add this sentence to the S3 caption when integrating:

> Markers are horizontally offset for readability.

## Validation performed here

- Python compilation and whitespace checks passed.
- All three maturity plots regenerated using committed experimental data.
- The lane-change reference regenerated from the existing deterministic function.
- Convergence, S2(a), S2(b), and S3 rendered and visually inspected using synthetic
  layout fixtures outside the repository. These are not experimental results and
  are not included in the publication outputs.
- The remaining historical source packages were absent, so their existing PDFs
  were preserved. Final data-specific overlap and page-layout checks remain for
  the local regeneration and subsequent Overleaf integration.

## Final layout polish

The landscape count annotation and two-column family legend now occupy separate
bands above the axes. The main-paper filename remains unchanged.

`make_paper_figures.py` additionally writes `fig4_threshold_summary.pdf` and PNG.
This combines S2(a/b) at full text width with one two-row legend, showing both
line styles and bar hatches. Bar labels are now 8 pt. Existing standalone S2
files are still generated for compatibility. The combined figure calls the same
renderers, preserving all calculations and censoring conventions.

When integrating in Overleaf, replace the two S2 minipages with
`\figurepdf{\textwidth}{figures/fig4_threshold_summary.pdf}` inside the existing
figure environment. Retain the caption and label, and remove the old external
(a)/(b) labels because the combined graphic includes them. No LaTeX source has
been changed in this commit.

Validation: compilation and whitespace checks passed. Both modified layouts
were rendered and visually inspected with synthetic fixtures outside the repo.
The final experimental outputs must be regenerated on the experiment machine.

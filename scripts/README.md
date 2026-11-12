# Data processing and report tools

These scripts operate on saved measurements. They do not run PDE experiments.

## Validate saved results

```sh
python3 scripts/check_saved_results.py
```

This standard-library check works without the numerical installation. It checks local evidence hashes, recomputes the reported convergence rates, iteration sums, candidate choices and runtime ratios, and verifies report tables/figure/PDF hashes. Its output explicitly states that native binaries and fresh numerical fields were not checked.

`check_report_evidence.py` is the original, more detailed audit. It also checks the native installation and the historical source/evidence chain, so it requires that installation and Git history. Individual `audit_*.py` files support these records and remain unchanged.

## Build the report

```sh
python3 scripts/prepare_report.py --check
python3 scripts/build_report.py
```

The build requires `pdflatex` and `pdfinfo`. If generated data/copies need refreshing, run `prepare_report.py` without `--check` first. `report/technical_report/evidence/claims.json` maps generated tables and ratios to their source records. Report source changes require a rebuild so the source and PDF hashes agree.

## Recreate scientific figures

Use the separate plotting environment documented in [the reproduction guide](../docs/reproducibility.md). The five report figures come from:

| Processor | Figure |
|---|---|
| `process_spatial.py` | Mesh refinement and spatial error |
| `process_temporal.py` | Same-mesh temporal convergence |
| `process_accuracy_cost.py` | Four temporal accuracy-cost candidates |
| `process_selected_trajectories.py` | Full-trajectory iterations and Rana factor entries |
| `process_selected_cost.py` | Final matched preconditioner runtimes |

The other processors retain initial interpolation, preliminary timing, earlier pairwise timing and first-step sensitivity plots. Each processor checks source data and its plotting inventory, then saves a table, PNG/SVG figure and processing hashes.

Outputs remain under `results/`; no raw record is modified. Different experiment drivers have different timing boundaries, so do not pool their absolute runtimes.

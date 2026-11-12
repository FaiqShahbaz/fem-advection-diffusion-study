# Scientific figures

The report uses these five figures, each available as PNG and SVG:

| Figure | Source stem | Meaning |
|---|---|---|
| 1 | `spatial-20261008T133314Z-38837` | Solution/gradient errors as the mesh is refined |
| 2 | `temporal-20261018T113908Z-75912` | Same-mesh time errors for BE and Radau |
| 3 | `accuracy-cost-20261019T102618Z-70377` | Four temporal accuracy-cost candidates |
| 4 | `selected-trajectories-20261021T193925Z-25133` | Full-run outer iterations and Rana factor entries |
| 5 | `selected-cost-20261022T095152Z-28440` | Repeated common-path solver runtimes at three points |

Additional figures show initial interpolation, the battery-power timing pilot, earlier matched ASM/Rana comparisons and first-step solver sensitivity. They document intermediate experiments. Their absolute timings are not pooled across drivers.

Error plots distinguish solution values (`L2`) from gradients (`H1` seminorm). Factor entries indicate storage structure, not measured memory bytes. Timing plots retain all three repetitions; their ranges are not confidence intervals. A missing finer-Jacobi result reflects its failed first-step pilot.

Processors and plotting requirements are documented under [scripts](../../scripts/README.md). Raw data, output hashes and processing records remain under `results/`. The report uses byte-identical copies of the five source PNG files.

# Results

The [technical report](../report/technical_report/technical_report.pdf) explains the five main figures and conclusions. [Processed tables](processed/README.md) contain the numerical values, and [the figure guide](figures/README.md) identifies main and supporting plots.

- `raw/` retains original measurements, terminal transcripts and manifests.
- `processed/` retains validation summaries, tables and processing hashes.
- `figures/` contains scientific PNG/SVG plots.

The final selected-cost study contains 43 trajectories and 430 steps. Its baseline outer iteration totals are Jacobi 7258, ASM 600 and RanaLD 70 over ten Radau steps. Median ASM/Rana runtime ratios at the three selected points are 2.005, 4.638 and 1.337.

These are serial measurements within the reported timing boundary. Failed attempts, the finer-Jacobi exclusion and all timing samples remain available. Follow [the reproduction guide](../docs/reproducibility.md) for saved-data checking, plotting or numerical reruns.

# Tables and validation summaries

The five main result tables are:

- [Spatial convergence](spatial-20261008T133314Z-38837-table.md)
- [Temporal convergence](temporal-20261018T113908Z-75912-table.md)
- [Temporal accuracy versus cost](accuracy-cost-20261019T102618Z-70377-table.md)
- [Full-trajectory solver work](selected-trajectories-20261021T193925Z-25133-table.md)
- [Matched solver runtime](selected-cost-20261022T095152Z-28440-table.md)

`*_review.json` files retain validation outcomes and source hashes. `*-table.json`/`.md` files are generated measurements in readable form. `*-processing.json` files identify the source data, plotting environment and generated output hashes. Historical status sentences describe the individual experiment at the time it was processed.

Earlier interpolation, pilot, pairwise and failure records remain part of the supporting evidence. The final common-path solver timings should be compared within each selected point; earlier different-driver absolute timings are separate observations.

Use [the scripts](../../scripts/README.md) to regenerate derived outputs and [the reproduction guide](../../docs/reproducibility.md) to distinguish saved-data checking from new numerical execution.

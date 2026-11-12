# Finite-element advection-diffusion

How do the mesh, time step and linear solver affect the accuracy and cost of a simulation?

This project solves a smooth, two-dimensional transient advection-diffusion problem using **Firedrake, Irksome and PETSc**. Advection carries a quantity with a flow; diffusion spreads it out. A manufactured exact solution makes it possible to measure the numerical error.

The serial study is complete. Results were collected in October 2026; the report and documentation were revised on 12 November 2026.

## Read the study

Start with the **[technical report](report/technical_report/technical_report.pdf)**. It explains the equation, finite elements, time integration and preconditioning before discussing each figure. The [mathematical notes](docs/mathematics.md) give a compact derivation, and the [reproduction guide](docs/reproducibility.md) explains how to inspect or rerun the work.

## Main findings

- Mesh refinement produced the expected P1/P2 convergence. The finest solution-error rates were approximately 1.993 and 2.999.
- Backward Euler and two-stage Radau IIA approached first- and third-order temporal convergence: measured finest rates were 1.007 and 2.944.
- Among four tested time-step choices, selected backward Euler cost less at a loose temporal-error target. Selected Radau had about **9.79 times lower median timed runtime** at a tighter target. These are time-error comparisons on one fixed mesh, not total PDE accuracy targets.
- Over ten Radau steps in the baseline case, Jacobi, ASM/ILU(0) and RanaLD used **7,258, 600 and 70 outer GMRES iterations**. The fields agreed with independent direct-LU references within the verification tolerances.
- RanaLD with full auxiliary sparse LU had **1.34-4.64 times lower median timed runtime than ASM/ILU(0)** across three matched serial cases. Finer-mesh Jacobi reached its first-step iteration limit and has no accepted full-run timing.

The [spatial](results/processed/spatial-20261008T133314Z-38837-table.md), [temporal](results/processed/temporal-20261018T113908Z-75912-table.md), [accuracy-cost](results/processed/accuracy-cost-20261019T102618Z-70377-table.md), [solver-work](results/processed/selected-trajectories-20261021T193925Z-25133-table.md) and [matched-runtime](results/processed/selected-cost-20261022T095152Z-28440-table.md) tables retain the measurements behind these statements.

## Check the saved results

From the repository root, with Python 3.10 or newer:

```sh
python3 scripts/check_saved_results.py
```

This needs no Firedrake or PETSc installation. It checks saved-file hashes, recomputes convergence rates and timing ratios, checks recorded iteration/residual information, and compares the report tables and figures with their sources. It reads existing measurements; it does not run a simulation or verify a native installation.

To rebuild the PDF, install a LaTeX distribution providing `pdflatex` and Poppler providing `pdfinfo`, then run:

```sh
python3 scripts/build_report.py
```

For the original native-installation audit, use `python3 scripts/check_report_evidence.py`. For actual numerical reruns and figure generation, follow [the reproduction guide](docs/reproducibility.md).

## What was tested?

The domain is the unit square, with zero boundary values, velocity `(1, 1)` and exact solution `exp(-t) sin(pi x) sin(pi y)`. Spatial verification uses continuous P1/P2 elements on triangular meshes. Later comparisons use P2, backward Euler and two-stage Radau IIA, with GMRES and existing preconditioners.

All performance results use **one MPI process and one computational thread** on an Apple M4 Pro. There is no parallel scaling benchmark. The Rana configuration factors the full auxiliary matrix by sparse LU; it is not an evaluation of scalable multigrid stage subsolves. Three timing repetitions show observed variability. These results do not establish a universal method ranking or a new numerical method.

## Repository layout

| Folder | Contents |
|---|---|
| [src](src/README.md) | Numerical implementations and shared helpers |
| [configs](configs/README.md) | Mesh, PDE, time-step, solver and tolerance inputs |
| [environment](environment/README.md) | Native setup scripts, pinned versions and build records |
| [results](results/processed/README.md) | Raw measurements, validation summaries, tables and figures |
| [scripts](scripts/README.md) | Saved-data validation, plotting and report generation |
| [docs](docs/study_scope.md) | Mathematics, scope, reproduction, references and a short research log |
| [report](report/README.md) | PDF, editable LaTeX, tables and figure copies |

Numerical source paths retain their original `phase*` names so saved records still identify the exact code used. Earlier validation modules are also imported by the final experiments. Their code, inputs, raw data and hash-linked validation records are retained. Repeated planning and lesson documents have been removed from the current tree; The original development Git history preserves earlier versions.

## License

Original project code and documentation are available under the [MIT license](LICENSE). The vendored Firedrake helper keeps its upstream license; see [third-party notices](NOTICE.md).

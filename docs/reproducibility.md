# Reproducing the study

There are three different tasks: inspect saved results, regenerate the report/plots, or run fresh simulations. Only the last requires the numerical software stack.

## 1. Inspect saved results on another machine

From the repository root, with Python 3.10 or newer:

```sh
python3 scripts/check_saved_results.py
```

This uses the Python standard library. It verifies the report's six source evidence sets and their immediate input/source hashes, recomputes spatial and temporal rates, checks the four accuracy-cost candidates, and checks the final selected-cost iteration totals, recorded residuals, repeats and runtime ratios. It also compares generated tables, figure copies and PDF hashes. Numerical imports are blocked.

A successful check means the saved records and reported arithmetic agree. It does not regenerate the solution fields, repeat timing measurements, check installed binaries, or replace numerical verification. The original detailed audit is available as:

```sh
python3 scripts/check_report_evidence.py
```

That command checks the historical validation chain and installation hashes. It requires the original native installation files and Git history; a downloaded source archive alone is insufficient for it.

## 2. Rebuild the report

Install a LaTeX distribution with `pdflatex` and Poppler with `pdfinfo`. Then:

```sh
python3 scripts/prepare_report.py --check
python3 scripts/build_report.py
python3 scripts/check_saved_results.py
```

If the report tables or figure copies need to be regenerated from the unchanged source records, run `python3 scripts/prepare_report.py` before building. Tables and headline values are generated rather than manually edited. The build checks its input data, compiles twice, rejects unresolved references/overflows and records source/PDF hashes. Identical PDF bytes across operating systems and TeX versions are not promised.

## 3. Regenerate figures from saved measurements

Plotting is separate from the numerical environment. The original plotting environment is `environment/build/postprocessing-venv/`; its requirements are in `environment/requirements-postprocessing-resolved.txt`.

```sh
export MPLBACKEND=Agg
export MPLCONFIGDIR="$PWD/environment/build/plot-cache"
environment/build/postprocessing-venv/bin/python scripts/process_spatial.py
environment/build/postprocessing-venv/bin/python scripts/process_temporal.py
environment/build/postprocessing-venv/bin/python scripts/process_accuracy_cost.py
environment/build/postprocessing-venv/bin/python scripts/process_selected_trajectories.py
environment/build/postprocessing-venv/bin/python scripts/process_selected_cost.py
```

These processors read saved JSON and create tables, PNG/SVG figures and processing records. Their environment inventory checks are deliberate. A different plotting environment may change output bytes and must be recorded separately. After regeneration, refresh the report copies and rebuild it. Do not overwrite original raw measurements to change a plot.

## 4. Run fresh numerical experiments

The [environment guide](../environment/README.md) describes the recorded native ARM macOS setup. Firedrake 2026.10.0, Irksome 2026.0.0 and PETSc 3.26.0 were used. Exact Python packages, PETSc source commit, configuration options, compiler/MPI/BLAS details and original build logs are retained. A package list alone does not reproduce those compiled libraries.

The original experiment wrappers are tied to that verified installation and earlier saved records. A fresh rebuild can produce different binary hashes even with the same version numbers. Validate and document a new installation before treating its measurements as a comparable reproduction; do not replace old hashes merely to make a guard pass. This repository documents a tested native route, not an automated installer tested on every platform.

On the existing tested Mac installation, keep the parent shell outside Conda and Python virtual environments. The wrappers select the numerical `.venv` themselves:

```sh
while [ "${CONDA_SHLVL:-0}" -gt 0 ]; do
    conda deactivate || break
done
export PATH="/opt/homebrew/opt/python@3.13/libexec/bin:/opt/homebrew/opt/open-mpi/bin:/opt/homebrew/bin:/opt/homebrew/sbin:$PATH"
hash -r
```

Run the wrappers below in order. Each has `--show-commands` for a read-only preview. Timing runs require AC power before and after measurement. The later wrappers verify their saved prerequisites before launching.

| Stage | Wrapper in `environment/` |
|---|---|
| Runtime/binding and vendor checks | `check_python_runtime.py` |
| Manual BE, Irksome BE, Radau construction | `check_be_smoke.py`, `check_irksome_be.py`, `check_radau_smoke.py` |
| Initial interpolation and transient spatial accuracy | `check_interpolation.py`, `check_spatial.py` |
| Same-mesh time accuracy | `check_temporal.py` |
| Timing instrumentation and four-candidate accuracy-cost study | `check_cost_pilot.py`, `check_accuracy_cost.py` |
| ASM construction, trajectory and matched cost | `check_asm_construction.py`, `check_asm_trajectory.py`, `check_asm_cost.py` |
| Rana construction, trajectory and matched cost | `check_rana_construction.py`, `check_rana_trajectory.py`, `check_rana_cost.py` |
| Selected points, full trajectories and common-path cost | `check_solver_sensitivity.py`, `check_selected_trajectories.py`, `check_selected_cost.py` |

For example, `python3 environment/check_temporal.py --show-commands` previews the command; omitting the flag runs the experiment. Numerical reruns create new timestamped records. Retain the original results and compare new results with them.

## Measurement conventions

All formal performance runs use one process and one thread, warm compilation/code caches, one warmup and three measured repetitions per case. The final preconditioner comparison times fresh construction, assembly/setup and ten advances. Shared mesh construction, imports, verification and file output are outside the timer. All three samples are retained and the middle one is the median. Earlier drivers have different boundaries, so their absolute runtimes must not be pooled.

The mesh uses left-diagonal triangles and `h = sqrt(2)/n`. Initial data are interpolated, with zero boundary coefficients. Error integration is checked at quadrature degrees 10 and 14. Nominal GMRES uses right preconditioning, an unpreconditioned norm, restart 50, relative tolerance `1e-11`, absolute tolerance `1e-13` and a 1000-iteration limit per solve. Independent physical-residual checks and reference-field comparisons supplement solver stopping messages.

## Preserved records

- `results/raw/`: measurements, transcripts and run manifests, including unsuccessful attempts.
- `results/processed/`: validation summaries and generated tables/processing hashes.
- `environment/records/`: original toolchain, installation and small runtime checks.
- `report/technical_report/evidence/`: report data selectors and build hashes.

The earlier spatial iteration-limit attempt, initial Rana interface failure, battery timing pilot and finer-mesh Jacobi failure remain identifiable. A failed or excluded case is not silently converted into a successful result. Paths and hostnames in historical records describe the original machine. Build trees, caches and virtual environments are ignored by Git.

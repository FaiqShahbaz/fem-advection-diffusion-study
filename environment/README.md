# Numerical environment

The original experiments used native ARM macOS on an Apple M4 Pro. [The recorded snapshot](verified_environment.json) identifies the exact installed packages, PETSc build, MPI/BLAS/compiler choices and supporting records. The [resolved package list](requirements-verified.txt) contains 52 packages.

| Component | Recorded version |
|---|---|
| Python | Homebrew 3.13.16 |
| Firedrake / Irksome | 2026.10.0 / 2026.0.0 |
| PETSc / petsc4py | 3.26.0 / 3.26.0 |
| mpi4py / Open MPI | 4.1.2 / 5.0.11 |
| h5py | 3.16.0 |
| PETSc arithmetic | Optimized real double, 32-bit indices |

## Setup sequence

The [native setup guide](setup.md) consolidates toolchain selection, PETSc configuration/build, Python installation and initial runtime verification. The subsequent experiment sequence is in the [reproduction guide](../docs/reproducibility.md).

The setup scripts support display-only previews before installation/build commands. Dependency builds live under ignored `build/`, and the numerical virtual environment lives at the project root in ignored `.venv/`. Plotting uses a separate environment and package list.

## What can be reproduced where?

The archived wrappers check the original installation and saved prerequisite hashes before launching new measurements. These guards are specific to that verified installation. A new compiler, platform or binary rebuild needs its own validation record; matching package versions alone is insufficient.

To inspect saved results or rebuild the PDF on another machine, use the portable commands in [the reproduction guide](../docs/reproducibility.md). They do not need Firedrake or PETSc.

## Records and dependencies

`records/` keeps original setup transcripts, manifests, small runtime/MPI checks and compressed build logs. `*_review.json` files summarize the corresponding checks. `vendor/` contains the exact upstream configuration helper and its license. These records form part of the experiment provenance and are retained unchanged.

The original environment is already installed and verified. Routine report/plot work does not require reinstalling it. Numerical performance results are serial; the early MPI checks are correctness tests, not scaling measurements.

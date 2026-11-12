# Recorded native setup

This is the ARM macOS route used for the October 2026 experiments. It documents the original installation; Homebrew and upstream packages may change. Read the recorded plans and preview commands before building on a new machine. A fresh installation needs its own validation record rather than replacing the old binary hashes.

## Select the toolchain

Use a Terminal with Conda and any Python virtual environment inactive. Homebrew should be installed under `/opt/homebrew`.

```sh
/opt/homebrew/bin/brew install autoconf automake boost gcc libtool make ninja pkg-config python@3.13 cmake fftw hwloc metis openblas open-mpi scalapack suite-sparse zlib
export PATH="/opt/homebrew/opt/python@3.13/libexec/bin:/opt/homebrew/opt/open-mpi/bin:/opt/homebrew/bin:/opt/homebrew/sbin:$PATH"
hash -r
bash environment/check_native_tools.sh
bash environment/check_build_prerequisites.sh
```

The package installation changes the local Homebrew installation. The diagnostics record command selection, compiler and library versions, and a small compiled MPI test. Build outputs go under ignored `environment/build/` and records under `environment/records/`.

## Configure and build PETSc

The saved upstream helper selects PETSc `v3.26.0`, source commit `1136c0bd63ce5f2898c194cf902c865c0c60698c`. The configuration uses `arch-firedrake-default`, optimized shared libraries, real double precision and 32-bit indices. Native MPI, OpenBLAS and zlib paths are set explicitly; source/configuration plans and dependency records are retained.

```sh
python3 environment/configure_petsc.py --show-command
python3 environment/configure_petsc.py
python3 environment/build_check_petsc.py --show-commands
python3 environment/build_check_petsc.py
```

The first script downloads/configures PETSc and its selected dependencies. The build/check script compiles PETSc with four build jobs and runs its supplied checks. Both save logs and manifests. They do not install the Python finite-element stack.

## Install Python bindings and finite-element packages

```sh
python3 environment/install_python_stack.py --show-commands
python3 environment/install_python_stack.py
```

The installer creates the root `.venv` with Homebrew Python and the project package pins. It builds the required MPI/HDF5/PETSc Python bindings against the selected native libraries. It uses `requirements-phase0.in` and `constraints-phase0-build.txt`; the retained failed first attempt explains why the corrected build constraints are necessary. It checks dependency metadata, not the numerical correctness of a PDE solve.

## Verify runtime and the first equations

```sh
python3 environment/check_python_runtime.py --show-commands
python3 environment/check_python_runtime.py
python3 environment/check_be_smoke.py
python3 environment/check_irksome_be.py
python3 environment/check_radau_smoke.py
```

The wrappers select the child `.venv`, record thread/cache settings and run binding/vendor, small serial/MPI and BE/Radau construction checks. Firedrake initializes PETSc before the binding probes. The original outcomes and source/package inventory are in the [verified installation record](verified_environment.json).

The archived wrapper chain checks the original installation and earlier evidence. A different machine or rebuild can require a separately recorded setup and adapted wrapper configuration. There is no cross-platform installer or binary-reproducibility guarantee. The [reproduction guide](../docs/reproducibility.md) lists the subsequent experiment sequence and the installation-independent saved-data route.

## Separate plotting environment

The original plotting environment is `environment/build/postprocessing-venv/`, with its package list in `requirements-postprocessing-resolved.txt`. It is separate from the numerical `.venv`. A new plotting environment can be created with a suitable Python and that requirements file; processing records should identify it. The report itself needs system TeX and Poppler, not Firedrake.

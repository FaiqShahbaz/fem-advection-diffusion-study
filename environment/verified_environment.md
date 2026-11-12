# Recorded numerical installation

The [snapshot](verified_environment.json) was recorded on 8 October 2026 after installation, runtime and backward-Euler checks. Later Radau, convergence and timing records use the same numerical installation.

It records Homebrew Python 3.13.16, Firedrake 2026.10.0, Irksome 2026.0.0, PETSc/petsc4py 3.26.0, mpi4py 4.1.2, h5py 3.16.0 and Open MPI 5.0.11. PETSc uses optimized real double arithmetic with 32-bit indices. Source commits, configure options and supporting native-library hashes are in the JSON record and linked manifests.

The [resolved inventory](requirements-verified.txt) lists all 52 Python packages. Binary hashes identify the original installation; they are not expected to match every rebuild. [The reproduction guide](../docs/reproducibility.md) distinguishes portable saved-data checks from installation-specific numerical reruns.

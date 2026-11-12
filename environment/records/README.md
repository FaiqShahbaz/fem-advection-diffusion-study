# Original environment records

This directory retains the transcripts and manifests used to establish the native toolchain and numerical installation. They include initial tool-selection issues, the h5py build-dependency conflict and its successful retry, PETSc configuration/build logs, vendor tests, and the first BE/Radau checks.

Files with a shared timestamp belong to the same run. Text files contain terminal output, JSON manifests identify commands and source/configuration hashes, `*-serial.json`/`*-two-rank.json` contain small numerical records, and `.log.gz` files preserve full build diagnostics compactly.

The [environment snapshot](../verified_environment.json) and corresponding `*_review.json` records link these files to the tested installation. These are historical records, so early output can describe work that was incomplete at that time. Later transient and timing measurements are under `results/raw/`.

Original bytes are retained for hash checking. Local build trees, libraries and virtual environments are not tracked.

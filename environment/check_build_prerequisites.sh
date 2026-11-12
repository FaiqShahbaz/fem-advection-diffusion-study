#!/usr/bin/env bash
# Capture build prerequisites and test native MPI before building PETSc.
# This script installs nothing; it compiles the small environment test below.
set -u
set -o pipefail

task_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
task_stamp="$(date -u '+%Y%m%dT%H%M%SZ')-$$"
task_record_path="$task_repo_root/environment/records/build-prerequisites-$task_stamp.txt"
task_build_dir="$task_repo_root/environment/build/mpi-smoke-$task_stamp"
mkdir -p "$task_repo_root/environment/records" "$task_build_dir" || exit 1

collect_report() {
    printf 'Phase 0 build prerequisite check\n'
    printf 'UTC: %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    printf '\nRecheck selected native tools after dependency installation\n'
    bash "$task_repo_root/environment/check_native_tools.sh" || return 1

    printf '\nHelper integrity and build plan\n'
    python3 - "$task_repo_root" <<'PYTHON'
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1])
metadata = json.loads((root / "environment/vendor/firedrake-configure-source.json").read_text())
plan = json.loads((root / "environment/petsc_plan.json").read_text())
actual = hashlib.sha256((root / "environment/vendor/firedrake-configure").read_bytes()).hexdigest()
if actual != metadata["sha256"] or actual != plan["helper_sha256"]:
    raise SystemExit("CHECK FAILED: helper differs from the recorded snapshot.")
print("Helper commit:", metadata["source_commit"])
print("Helper SHA-256:", actual)
print("Planned PETSc tag:", plan["petsc_tag"])
print("Planned PETSc arch:", plan["petsc_arch"])
print("Planning status:", plan["status"])

# Query the installed inventory without asking Homebrew to load missing formulae.
env = os.environ.copy()
env.update(HOMEBREW_NO_AUTO_UPDATE="1", HOMEBREW_NO_INSTALL_FROM_API="1")
listing = subprocess.check_output(
    ["/opt/homebrew/bin/brew", "list", "--formula", "--versions"],
    text=True, env=env,
)
installed = {line.split()[0]: line for line in listing.splitlines() if line.strip()}
print("\nInstalled required formulae:")
missing = []
for formula in plan["project_homebrew_formulae"]:
    candidates = ("pkgconf", "pkg-config") if formula == "pkg-config" else (formula,)
    matches = [installed[name] for name in candidates if name in installed]
    if matches:
        print(matches[0])
    else:
        missing.append(formula)
if missing:
    raise SystemExit("CHECK FAILED: missing required formulae: " + ", ".join(missing))
PYTHON
    if [ "$?" -ne 0 ]; then return 1; fi

    printf '\nC and Fortran compiler identities\n'
    clang --version || return 1
    gfortran --version || return 1
    printf '\nCompile the MPI collective-sum test\n'
    /opt/homebrew/opt/open-mpi/bin/mpicc -std=c99 -Wall -Wextra \
        "$task_repo_root/environment/mpi_smoke.c" -o "$task_build_dir/mpi_smoke" || return 1
    printf '\nCompiled binary architecture and linked libraries\n'
    file "$task_build_dir/mpi_smoke" || return 1
    otool -L "$task_build_dir/mpi_smoke" || return 1
    printf '\nOne-process sum (expected 1)\n'
    /opt/homebrew/opt/open-mpi/bin/mpiexec -n 1 "$task_build_dir/mpi_smoke" || return 1
    printf '\nTwo-process sum (expected 3 on both processes)\n'
    /opt/homebrew/opt/open-mpi/bin/mpiexec -n 2 "$task_build_dir/mpi_smoke" || return 1
    printf '\nPlanned upstream PETSc configure options; not applied yet\n'
    python3 "$task_repo_root/environment/vendor/firedrake-configure" \
        --os macos-arm64 --show-petsc-configure-options || return 1
    printf '\nBUILD PREREQUISITE CHECK PASSED. PETSc and Firedrake remain unverified.\n'
}

collect_report 2>&1 | tee "$task_record_path"
task_status=$?
printf '\nSaved report: %s\n' "$task_record_path"
exit "$task_status"

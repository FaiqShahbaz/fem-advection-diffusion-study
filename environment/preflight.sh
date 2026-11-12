#!/usr/bin/env bash
# Prerequisite diagnostic only: no installation, package imports, or MPI launch.
# Compatible with the Bash 3.2 supplied by macOS.
set -u
set -o pipefail

task_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
task_records_dir="$task_repo_root/environment/records"
mkdir -p "$task_records_dir" || exit 1
task_stamp="$(date -u '+%Y%m%dT%H%M%SZ')"
task_record_path="$task_records_dir/preflight-${task_stamp}-$$.txt"

show_path() {
    local task_program="$1"
    local task_selected_path
    if task_selected_path="$(command -v "$task_program")"; then
        printf '%s: %s\n' "$task_program" "$task_selected_path"
    else
        printf '%s: NOT FOUND\n' "$task_program"
    fi
}

collect_report() {
    printf 'Phase 0 prerequisite diagnostic\n'
    printf 'UTC: %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    printf 'Status: tool selection only; numerical stack NOT functionally verified\n\n'

    printf 'Architecture and OS\n'
    uname -m
    if command -v sw_vers >/dev/null 2>&1; then sw_vers; fi
    if command -v xcode-select >/dev/null 2>&1; then xcode-select -p; fi

    printf '\nSelected programs\n'
    for task_program in python3 brew mpicc mpiexec git gh; do
        show_path "$task_program"
    done

    printf '\nSelected Python metadata and package discovery\n'
    if command -v python3 >/dev/null 2>&1; then
        python3 - <<'PY'
import importlib.util
import os
import platform
import sys

print("Executable:", sys.executable)
print("Version:", sys.version.split()[0])
print("Process architecture:", platform.machine())
print("Inside Python venv (PEP 405):", sys.prefix != sys.base_prefix)
print("CONDA_SHLVL:", os.environ.get("CONDA_SHLVL", "unset"))
print("CONDA_DEFAULT_ENV:", os.environ.get("CONDA_DEFAULT_ENV", "unset"))
print("CONDA_PREFIX:", os.environ.get("CONDA_PREFIX", "unset"))
for name in ("firedrake", "petsc4py", "mpi4py", "irksome"):
    try:
        spec = importlib.util.find_spec(name)
        print(name + ":", "discoverable, functionality untested" if spec else "not discoverable")
    except Exception as exc:
        print(name + ": discovery error:", str(exc))
PY
    else
        printf 'Python unavailable; package discovery skipped.\n'
    fi

    printf '\nRelevant installed Homebrew formulae (local listing only)\n'
    if command -v brew >/dev/null 2>&1; then
        HOMEBREW_NO_AUTO_UPDATE=1 HOMEBREW_NO_INSTALL_FROM_API=1 brew list --formula --versions \
            | awk '$1 ~ /^(python(@[0-9.]+)?|open-mpi|mpich|gcc(@[0-9]+)?|cmake|ninja|openblas|fftw|hwloc|metis|scalapack|suite-sparse|suitesparse)$/ {print; found=1} END {if (!found) print "None of the listed numerical prerequisites found."}'
    else
        printf 'Homebrew unavailable; listing skipped.\n'
    fi

    printf '\nGit version\n'
    if command -v git >/dev/null 2>&1; then git --version; fi
    printf '\nNext: review this report with your mentor before installation.\n'
    printf 'No MPI launcher was executed. No numerical package was imported.\n'
}

collect_report 2>&1 | tee "$task_record_path"
task_capture_status=$?
printf '\nSaved diagnostic: %s\n' "$task_record_path"
exit "$task_capture_status"

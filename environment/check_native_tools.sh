#!/usr/bin/env bash
# Native prerequisite selection check; no installation or MPI job launch.
# Run after selecting Homebrew Python 3.13 and Homebrew Open MPI in Terminal.
set -u
set -o pipefail

task_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
task_record_dir="$task_repo_root/environment/records"
mkdir -p "$task_record_dir" || exit 1
task_record_path="$task_record_dir/native-tools-$(date -u '+%Y%m%dT%H%M%SZ')-$$.txt"

collect_report() {
    printf 'Phase 0 native prerequisite selection check\n'
    printf 'UTC: %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    printf 'CONDA_SHLVL: %s\n' "${CONDA_SHLVL:-0}"
    if [ "${CONDA_SHLVL:-0}" != '0' ]; then
        printf 'CHECK FAILED: deactivate Conda before this checkpoint.\n'
        return 1
    fi

    local task_program task_selected_path
    for task_program in python3 mpicc mpiexec; do
        task_selected_path="$(command -v "$task_program")" || {
            printf 'CHECK FAILED: %s not found.\n' "$task_program"
            return 1
        }
        printf '%s: %s\n' "$task_program" "$task_selected_path"
        case "$task_selected_path" in
            /opt/homebrew/*) ;;
            *) printf 'CHECK FAILED: %s must select the intended Homebrew tool.\n' "$task_program"; return 1 ;;
        esac
    done

    printf '\nPython metadata\n'
    python3 - <<'PYTHON'
import os
from pathlib import Path
import platform
import sys

print("Executable:", sys.executable)
print("Version:", sys.version.split()[0])
print("Process architecture:", platform.machine())
print("Inside Python venv:", sys.prefix != sys.base_prefix)
print("CONDA_PREFIX:", os.environ.get("CONDA_PREFIX", "unset"))
if sys.version_info[:2] != (3, 13):
    raise SystemExit("CHECK FAILED: select the Homebrew Python 3.13 interpreter.")
if platform.machine() != "arm64":
    raise SystemExit("CHECK FAILED: native arm64 Python required for this route.")
if not str(Path(sys.executable).resolve()).startswith("/opt/homebrew/"):
    raise SystemExit("CHECK FAILED: Python resolves outside the intended Homebrew prefix.")
PYTHON
    if [ "$?" -ne 0 ]; then return 1; fi

    printf '\nInstalled native formula versions\n'
    HOMEBREW_NO_AUTO_UPDATE=1 HOMEBREW_NO_INSTALL_FROM_API=1 \
        /opt/homebrew/bin/brew list --versions python@3.13 open-mpi || return 1
    printf '\nMPI compiler command\n'
    mpicc --showme:command || return 1
    printf '\nMPI include settings\n'
    mpicc --showme:compile || return 1
    printf '\nMPI link settings\n'
    mpicc --showme:link || return 1
    printf '\nMPI launcher version\n'
    mpiexec --version || return 1
    printf '\nSELECTION CHECK PASSED. Numerical stack remains functionally unverified.\n'
    printf 'No MPI job was launched; no numerical package was imported.\n'
}

collect_report 2>&1 | tee "$task_record_path"
task_status=$?
printf '\nSaved record: %s\n' "$task_record_path"
exit "$task_status"

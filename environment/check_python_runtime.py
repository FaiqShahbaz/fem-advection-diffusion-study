#!/usr/bin/env python3
"""Researcher-run numerical imports/MPI/tableau and upstream Firedrake checks."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from configure_petsc import ROOT, digest, run_logged, validate_tools


def runtime_plan():
    review_path = ROOT / "environment/python_installation_review.json"
    review = json.loads(review_path.read_text())
    for path, sha in (
        (review["installation_manifest"], review["installation_manifest_sha256"]),
        (review["installation_transcript"], review["installation_transcript_sha256"]),
        (review["pip_report"], review["pip_report_sha256"]),
        (review["candidate_inventory"], review["candidate_inventory_sha256"]),
        (review["petsc_build_review"], review["petsc_build_review_sha256"]),
        (review["upstream_checker"], review["upstream_checker_sha256"]),
        ("environment/python_plan.json", review["python_plan_sha256"]),
        ("environment/requirements-phase0.in", review["requirements_sha256"]),
        ("environment/constraints-phase0-build.txt", review["build_constraints_sha256"]),
    ):
        if digest(ROOT / path) != sha:
            raise RuntimeError(f"Reviewed installation input or record changed: {path}")
    if review["checkpoint_exit_status"] != 0 or review["pip_check_exit_status"] != 0:
        raise RuntimeError("Accepted package installation is required.")
    build = json.loads((ROOT / review["petsc_build_review"]).read_text())
    if digest(ROOT / build["library_path"]) != build["library_sha256"]:
        raise RuntimeError("Compiled PETSc library changed after its accepted checks.")
    petsc = json.loads((ROOT / "environment/petsc_plan.json").read_text())
    venv = ROOT / ".venv"
    if json.loads((venv / "project-phase0-environment.json").read_text())["project_root"] != str(ROOT):
        raise RuntimeError("Project venv marker mismatch.")
    source = ROOT / petsc["planned_build_directory"]
    cache = ROOT / "environment/build/runtime-cache"
    env = os.environ.copy()
    env.update(PETSC_DIR=str(source), PETSC_ARCH=petsc["petsc_arch"],
               HDF5_DIR=str(source / petsc["petsc_arch"]), HDF5_MPI="ON",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               FIREDRAKE_CACHE_DIR=str(cache), PYOP2_CACHE_DIR=str(cache / "pyop2"),
               FIREDRAKE_TSFC_KERNEL_CACHE_DIR=str(cache / "tsfc"),
               XDG_CACHE_HOME=str(cache), TMPDIR=str(ROOT / "environment/build/runtime-tmp"),
               PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", FIREDRAKE_CI="1")
    env["PATH"] = os.pathsep.join((str(venv / "bin"), "/opt/homebrew/opt/open-mpi/bin",
                                   "/opt/homebrew/bin", "/opt/homebrew/sbin", env.get("PATH", "")))
    # Strip options that could silently select a different Python package or test set.
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTEST_ADDOPTS"):
        env.pop(key, None)
    return review_path, review, petsc, venv, env


def commands(venv, stem):
    python = str(venv / "bin/python")
    probe = str(ROOT / "environment/runtime_probe.py")
    mpi = "/opt/homebrew/opt/open-mpi/bin/mpiexec"
    return [
        ("serial_probe", [python, probe, "--expected-size", "1", "--output", str(stem) + "-serial.json"]),
        ("two_rank_probe", [mpi, "-n", "2", python, probe, "--expected-size", "2",
                             "--output", str(stem) + "-two-rank.json"]),
        ("firedrake_vendor_check", [python, str(venv / "bin/firedrake-check"),
                                   "--mpiexec", mpi + " -n"]),
    ]


def check(review_path, review, petsc, venv, env):
    tools = validate_tools(petsc)
    # Standard-library metadata only; numerical imports run in the separate probe.
    code = ("import importlib.metadata as m,json,sys; "
            "print(json.dumps({'prefix':sys.prefix,'versions':{n:m.version(n) for n in " +
            repr(list(review["installed_metadata"]["versions"])) + "}}))")
    metadata = json.loads(subprocess.check_output([str(venv / "bin/python"), "-c", code],
                                                 env=env, text=True))
    if metadata["prefix"] != str(venv) or metadata["versions"] != review["installed_metadata"]["versions"]:
        raise RuntimeError("Project interpreter or installed package versions changed.")
    inventory = subprocess.check_output([str(venv / "bin/python"), "-m", "pip", "freeze", "--all"],
                                        env=env, text=True)
    if inventory != (ROOT / review["candidate_inventory"]).read_text():
        raise RuntimeError("Resolved dependency inventory changed since the accepted installation.")
    for key in ("FIREDRAKE_CACHE_DIR", "PYOP2_CACHE_DIR", "FIREDRAKE_TSFC_KERNEL_CACHE_DIR", "TMPDIR"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = ROOT / "environment/records" / f"python-runtime-{stamp}-{os.getpid()}"
    transcript = stem.with_suffix(".txt")
    manifest_path = Path(str(stem) + "-manifest.json")
    steps = commands(venv, stem)
    state = {"started_utc": datetime.now(timezone.utc).isoformat(), "status": "started",
             "installation_review": str(review_path.relative_to(ROOT)),
             "installation_review_sha256": digest(review_path), "selected_tools": tools,
             "selected_environment": {key: env[key] for key in (
                 "PETSC_DIR", "PETSC_ARCH", "HDF5_DIR", "HDF5_MPI", "OPENBLAS_NUM_THREADS",
                 "OMP_NUM_THREADS", "FIREDRAKE_CACHE_DIR", "PYOP2_CACHE_DIR",
                 "FIREDRAKE_TSFC_KERNEL_CACHE_DIR", "XDG_CACHE_HOME", "TMPDIR",
                 "PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE", "FIREDRAKE_CI")},
             "inherited_library_settings": {key: env[key] for key in (
                 "DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH", "LD_LIBRARY_PATH") if key in env},
             "commands": {name: argv for name, argv in steps}, "completed_steps": [],
             "script_sha256": {name: digest(ROOT / "environment" / name) for name in
                               ("check_python_runtime.py", "runtime_probe.py", "configure_petsc.py")},
             "project_fem_or_time_step_checks_run": False,
             "vendor_process_counts": review["upstream_checker_process_counts"]}
    status = 1
    manifest_path.write_text(json.dumps(state, indent=2) + "\n")
    with transcript.open("w") as log:
        try:
            for name, argv in steps:
                run_logged(argv, ROOT, env, log)
                state["completed_steps"].append(name)
            state["status"] = "imports, MPI, tableaux, and vendor checks passed; project FEM verification pending"
            status = 0
            message = "PYTHON RUNTIME CHECKPOINT PASSED. Explained project FEM/Irksome time-step checks remain pending."
            print(message, flush=True)
            log.write(message + "\n")
        except (RuntimeError, OSError, subprocess.SubprocessError, KeyboardInterrupt) as error:
            state["status"] = "failed or interrupted; review required"
            state["error"] = str(error) or "Interrupted by researcher"
            message = f"STOP: {state['error']}"
            print(message, flush=True)
            log.write(message + "\n")
        finally:
            state["probe_records"] = {}
            for suffix in ("serial", "two-rank"):
                path = Path(str(stem) + f"-{suffix}.json")
                if path.is_file():
                    state["probe_records"][suffix] = {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}
            state["finished_utc"] = datetime.now(timezone.utc).isoformat()
            state["checkpoint_exit_status"] = status
            manifest_path.write_text(json.dumps(state, indent=2) + "\n")
    print(f"Saved transcript: {transcript}")
    print(f"Saved manifest: {manifest_path}")
    for record in state["probe_records"].values():
        print(f"Saved probe record: {ROOT / record['path']}")
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-commands", action="store_true",
                        help="Print planned steps without imports, MPI, or vendor tests.")
    args = parser.parse_args()
    review_path, review, petsc, venv, env = runtime_plan()
    if args.show_commands:
        for key in ("PETSC_DIR", "PETSC_ARCH", "FIREDRAKE_CACHE_DIR", "TMPDIR"):
            print(f"{key}={env[key]}")
        stem = ROOT / "environment/records/python-runtime-<UTC timestamp>-<process id>"
        for name, argv in commands(venv, stem):
            print(f"{name}: {shlex.join(argv)}")
        print("Display only: no numerical imports, MPI launch, vendor check, or project FEM solve run.")
        return 0
    return check(review_path, review, petsc, venv, env)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, OSError, subprocess.SubprocessError, KeyError, ValueError) as error:
        print(f"STOP before runtime checks: {error}", file=sys.stderr)
        sys.exit(1)

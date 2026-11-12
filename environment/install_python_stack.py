#!/usr/bin/env python3
"""Researcher-run isolated package installation; functional checks come later."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from configure_petsc import ROOT, digest, run_logged, validate_tools


def installation_plan():
    plan = json.loads((ROOT / "environment/python_plan.json").read_text())
    review_path = ROOT / plan["petsc_build_review"]
    if digest(review_path) != plan["petsc_build_review_sha256"]:
        raise RuntimeError("PETSc build review changed after installation planning.")
    review = json.loads(review_path.read_text())
    if digest(ROOT / review["build_manifest"]) != review["build_manifest_sha256"]:
        raise RuntimeError("Reviewed PETSc build manifest changed.")
    if review["check_exit_status"] != 0:
        raise RuntimeError("Successful reviewed PETSc checks are required.")
    if digest(ROOT / review["library_path"]) != review["library_sha256"]:
        raise RuntimeError("PETSc library changed since the reviewed checks.")
    petsc_plan = json.loads((ROOT / "environment/petsc_plan.json").read_text())
    source = ROOT / petsc_plan["planned_build_directory"]
    venv = ROOT / plan["venv_directory"]
    env = os.environ.copy()
    env.update(PETSC_DIR=str(source), PETSC_ARCH=petsc_plan["petsc_arch"],
               HDF5_DIR=str(source / petsc_plan["petsc_arch"]), HDF5_MPI="ON",
               CC="/opt/homebrew/opt/open-mpi/bin/mpicc",
               MPI4PY_BUILD_MPICC="/opt/homebrew/opt/open-mpi/bin/mpicc",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               CMAKE_BUILD_PARALLEL_LEVEL="4",
               TMPDIR=str(ROOT / "environment/build/python-tmp"),
               PIP_CACHE_DIR=str(ROOT / "environment/build/pip-cache"))
    env["PATH"] = str(venv / "bin") + os.pathsep + env.get("PATH", "")
    python = str(venv / "bin/python")
    install = [python, "-m", "pip", "install", "--no-cache-dir",
               "--no-binary", ",".join(plan["source_build_packages"]),
               "--build-constraint", str(ROOT / plan["build_constraints_file"]),
               "-r", str(ROOT / plan["requirements_file"])]
    return plan, petsc_plan, review, venv, env, install


def install_stack(plan, petsc_plan, review, venv, env, install):
    tools = validate_tools(petsc_plan)
    marker = venv / "project-phase0-environment.json"
    if venv.exists():
        if not marker.is_file() or json.loads(marker.read_text()).get("project_root") != str(ROOT):
            raise RuntimeError("Existing .venv is not recognized as this project checkpoint; nothing replaced.")
    for name in ("TMPDIR", "PIP_CACHE_DIR"):
        Path(env[name]).mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = ROOT / "environment/records" / f"python-install-{stamp}-{os.getpid()}"
    transcript = stem.with_suffix(".txt")
    manifest_path = Path(str(stem) + "-manifest.json")
    pip_report = Path(str(stem) + "-pip-report.json")
    inventory = Path(str(stem) + "-packages.txt")
    python = str(venv / "bin/python")
    create = [sys.executable, "-m", "venv", "--prompt", "fem-advection-diffusion", str(venv)]
    install = [*install, "--report", str(pip_report)]
    state = {
        "started_utc": datetime.now(timezone.utc).isoformat(), "status": "started",
        "python_plan_sha256": digest(ROOT / "environment/python_plan.json"),
        "requirements_sha256": digest(ROOT / plan["requirements_file"]),
        "build_constraints_sha256": digest(ROOT / plan["build_constraints_file"]),
        "petsc_build_review": plan["petsc_build_review"],
        "petsc_build_review_sha256": plan["petsc_build_review_sha256"],
        "selected_native_tools": tools, "venv": str(venv),
        "commands": {"create_venv_if_absent": create, "install": install,
                     "metadata_check": [python, "-m", "pip", "check"]},
        "selected_environment": {key: env[key] for key in (
            "PETSC_DIR", "PETSC_ARCH", "HDF5_DIR", "HDF5_MPI", "CC", "MPI4PY_BUILD_MPICC",
            "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "CMAKE_BUILD_PARALLEL_LEVEL",
            "TMPDIR", "PIP_CACHE_DIR")},
        "functional_import_or_fem_checks_run": False
    }
    manifest_path.write_text(json.dumps(state, indent=2) + "\n")
    status = 1
    with transcript.open("w") as log:
        try:
            if not venv.exists():
                run_logged(create, ROOT, env, log)
                marker.write_text(json.dumps({"project_root": str(ROOT),
                    "base_interpreter": sys.executable}, indent=2) + "\n")
            run_logged([python, "-m", "pip", "--version"], ROOT, env, log)
            run_logged(install, ROOT, env, log)
            run_logged([python, "-m", "pip", "check"], ROOT, env, log)
            state["pip_check_exit_status"] = 0
            # Standard-library metadata only; do not import/initialize numerical packages here.
            probe = ("import importlib.metadata as m,json,sys,platform; "
                     "print(json.dumps({'executable':sys.executable,'prefix':sys.prefix,"
                     "'base_prefix':sys.base_prefix,'architecture':platform.machine(),"
                     "'versions':{name:m.version(name) for name in " +
                     repr(list(plan["pinned_packages"])) + "}}))")
            installed = json.loads(subprocess.check_output([python, "-c", probe], env=env, text=True))
            if installed["prefix"] != str(venv) or installed["architecture"] != "arm64":
                raise RuntimeError("Installed environment is not the intended native project venv.")
            if installed["versions"] != plan["pinned_packages"]:
                raise RuntimeError("Installed top-level versions differ from the reviewed input pins.")
            state["installed_metadata"] = installed
            frozen = subprocess.check_output([python, "-m", "pip", "freeze", "--all"], env=env, text=True)
            inventory.write_text(frozen)
            state["candidate_inventory"] = str(inventory.relative_to(ROOT))
            state["candidate_inventory_sha256"] = digest(inventory)
            state["status"] = "installation and metadata check passed; functional verification pending"
            status = 0
            message = "PYTHON PACKAGE INSTALLATION CHECKPOINT PASSED. Imports and FEM/MPI checks remain pending."
            print(message, flush=True)
            log.write(message + "\n")
        except (RuntimeError, OSError, subprocess.SubprocessError, KeyboardInterrupt) as error:
            state["status"] = "failed or interrupted; review required"
            state["error"] = str(error) or "Interrupted by researcher"
            message = f"STOP: {state['error']}"
            print(message, flush=True)
            log.write(message + "\n")
        finally:
            if pip_report.is_file():
                state["pip_report"] = str(pip_report.relative_to(ROOT))
                state["pip_report_sha256"] = digest(pip_report)
            state["finished_utc"] = datetime.now(timezone.utc).isoformat()
            state["checkpoint_exit_status"] = status
            manifest_path.write_text(json.dumps(state, indent=2) + "\n")
    print(f"Saved transcript: {transcript}")
    print(f"Saved manifest: {manifest_path}")
    for path in (pip_report, inventory):
        if path.is_file():
            print(f"Saved record: {path}")
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-commands", action="store_true",
                        help="Read the reviewed plan and print steps; do not create a venv or install.")
    args = parser.parse_args()
    plan, petsc_plan, review, venv, env, install = installation_plan()
    if args.show_commands:
        print(f"Project venv: {venv}")
        print(shlex.join([sys.executable, "-m", "venv", "--prompt", "fem-advection-diffusion", str(venv)]))
        for key in ("PETSC_DIR", "PETSC_ARCH", "HDF5_DIR", "HDF5_MPI", "CC", "MPI4PY_BUILD_MPICC"):
            print(f"{key}={env[key]}")
        print(shlex.join(install))
        print(shlex.join([str(venv / "bin/python"), "-m", "pip", "check"]))
        print("Display only: no venv created, package installed, or numerical package imported.")
        return 0
    return install_stack(plan, petsc_plan, review, venv, env, install)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, OSError, subprocess.SubprocessError) as error:
        print(f"STOP before installation: {error}", file=sys.stderr)
        sys.exit(1)

#!/usr/bin/env python3
"""Researcher-run first BE assembly/solve checkpoint with serial/MPI comparison."""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

from check_python_runtime import runtime_plan
from configure_petsc import ROOT, digest, run_logged, validate_tools

CONFIG = ROOT / "configs/phase0_be_smoke.json"
SCRIPT = ROOT / "src/phase0_be_smoke.py"


def plan():
    review_path = ROOT / "environment/python_runtime_review.json"
    review = json.loads(review_path.read_text())
    for path, expected in (
        (review["runtime_manifest"], review["runtime_manifest_sha256"]),
        (review["runtime_transcript"], review["runtime_transcript_sha256"]),
        (review["installation_review"], review["installation_review_sha256"]),
        *((record["path"],record["sha256"]) for record in review["probe_records"].values()),
    ):
        if digest(ROOT / path) != expected:
            raise RuntimeError(f"Accepted runtime evidence changed: {path}")
    if review["checkpoint_exit_status"] != 0:
        raise RuntimeError("Accepted runtime checks are required.")
    _, installation, petsc, venv, env = runtime_plan()
    return review_path, review, installation, petsc, venv, env


def commands(venv, stem):
    python = str(venv / "bin/python")
    def probe(size, output):
        return [python, str(SCRIPT), "--config", str(CONFIG), "--expected-size", str(size),
                "--output", str(output)]
    return [
        ("serial", probe(1, str(stem) + "-serial.json")),
        ("two_rank", ["/opt/homebrew/opt/open-mpi/bin/mpiexec", "-n", "2",
                      *probe(2, str(stem) + "-two-rank.json")]),
    ]


def compare_records(serial, parallel, config):
    if serial["status"] != "passed" or parallel["status"] != "passed":
        raise RuntimeError("Both individual assembly/solve checks must pass.")
    for key in ("config", "total_dofs", "expected_free_dofs", "expected_cells", "h", "final_time", "method"):
        if serial[key] != parallel[key]:
            raise RuntimeError(f"Serial/MPI problem specification differs: {key}")
    checks = config["checks"]
    compared = {}
    for key in ("domain_area", "initial_l2_error", "final_l2_error", "final_h1_seminorm_error", "solution_l2_norm"):
        a, b = serial["metrics"][key], parallel["metrics"][key]
        if not math.isclose(a, b, rel_tol=checks["serial_mpi_comparison_rtol"],
                            abs_tol=checks["serial_mpi_comparison_atol"]):
            raise RuntimeError(f"Serial/two-rank global quantity differs beyond tolerance: {key}")
        compared[key] = {"serial":a, "two_rank":b, "absolute_difference":abs(a-b)}
    return {"status":"passed", "compared_global_quantities":compared,
            "relative_tolerance":checks["serial_mpi_comparison_rtol"],
            "absolute_tolerance":checks["serial_mpi_comparison_atol"],
            "limit":"Comparison of selected scalar quantities, not a pointwise field-equality proof or scaling measurement"}


def run(review_path, review, installation, petsc, venv, env):
    tools = validate_tools(petsc)
    inventory = subprocess.check_output([str(venv / "bin/python"),"-m","pip","freeze","--all"],
                                        env=env,text=True)
    if inventory != (ROOT / installation["candidate_inventory"]).read_text():
        raise RuntimeError("Installed dependency inventory changed since acceptance.")
    for key in ("FIREDRAKE_CACHE_DIR","PYOP2_CACHE_DIR","FIREDRAKE_TSFC_KERNEL_CACHE_DIR","TMPDIR"):
        Path(env[key]).mkdir(parents=True,exist_ok=True)
    config = json.loads(CONFIG.read_text())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = ROOT / "environment/records" / f"be-smoke-{stamp}-{os.getpid()}"
    transcript = stem.with_suffix(".txt")
    manifest_path = Path(str(stem)+"-manifest.json")
    steps = commands(venv,stem)
    state = {"started_utc":datetime.now(timezone.utc).isoformat(),"status":"started",
        "runtime_review":str(review_path.relative_to(ROOT)),"runtime_review_sha256":digest(review_path),
        "config":str(CONFIG.relative_to(ROOT)),"config_sha256":digest(CONFIG),
        "script_sha256":{str(path.relative_to(ROOT)):digest(path) for path in
                         (SCRIPT,Path(__file__).resolve(),ROOT/"environment/check_python_runtime.py")},
        "code_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "working_tree_status_before_run":subprocess.check_output(["git","status","--short"],cwd=ROOT,text=True),
        "selected_tools":tools,
        "selected_environment":{key:env[key] for key in (
            "PETSC_DIR","PETSC_ARCH","OPENBLAS_NUM_THREADS","OMP_NUM_THREADS",
            "FIREDRAKE_CACHE_DIR","PYOP2_CACHE_DIR","FIREDRAKE_TSFC_KERNEL_CACHE_DIR",
            "XDG_CACHE_HOME","TMPDIR","PYTHONNOUSERSITE","PYTHONDONTWRITEBYTECODE")},
        "inherited_options_or_library_settings":{key:env[key] for key in (
            "PETSC_OPTIONS","DYLD_LIBRARY_PATH","DYLD_FALLBACK_LIBRARY_PATH","LD_LIBRARY_PATH") if key in env},
        "commands":{name:argv for name,argv in steps},"completed_steps":[],
        "performance_measurement":False,"irksome_time_step_run":False}
    status = 1
    manifest_path.write_text(json.dumps(state,indent=2)+"\n")
    with transcript.open("w") as log:
        try:
            for name,argv in steps:
                run_logged(argv,ROOT,env,log)
                state["completed_steps"].append(name)
            serial = json.loads(Path(str(stem)+"-serial.json").read_text())
            parallel = json.loads(Path(str(stem)+"-two-rank.json").read_text())
            state["serial_mpi_comparison"] = compare_records(serial,parallel,config)
            state["status"] = "manual BE one-step serial/MPI checks passed; review and Irksome cross-check pending"
            status = 0
            message = "MANUAL BE SMOKE CHECKPOINT PASSED. Review before Irksome or convergence experiments."
            print(message,flush=True);log.write(message+"\n")
        except (RuntimeError,OSError,subprocess.SubprocessError,KeyError,ValueError,KeyboardInterrupt) as error:
            state["status"] = "failed or interrupted; review required"
            state["error"] = str(error) or "Interrupted by researcher"
            message = f"STOP: {state['error']}"
            print(message,flush=True);log.write(message+"\n")
        finally:
            state["solve_records"] = {}
            for name in ("serial","two-rank"):
                path = Path(str(stem)+f"-{name}.json")
                if path.is_file():
                    state["solve_records"][name] = {"path":str(path.relative_to(ROOT)),"sha256":digest(path)}
            state["finished_utc"] = datetime.now(timezone.utc).isoformat()
            state["checkpoint_exit_status"] = status
            manifest_path.write_text(json.dumps(state,indent=2)+"\n")
    print(f"Saved transcript: {transcript}")
    print(f"Saved manifest: {manifest_path}")
    for record in state["solve_records"].values():
        print(f"Saved solve record: {ROOT/record['path']}")
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-commands",action="store_true",
                        help="Print the configuration and commands without imports, assembly, solves, or MPI.")
    args = parser.parse_args()
    parameters = plan()
    if args.show_commands:
        print(CONFIG.read_text())
        stem = ROOT / "environment/records/be-smoke-<UTC timestamp>-<process id>"
        for name,argv in commands(parameters[4],stem):
            print(f"{name}: {shlex.join(argv)}")
        print("Display only: no numerical imports, FEM assembly, solve, or MPI launch run.")
        return 0
    return run(*parameters)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError,OSError,subprocess.SubprocessError,KeyError,ValueError) as error:
        print(f"STOP before BE smoke check: {error}",file=sys.stderr)
        sys.exit(1)

#!/usr/bin/env python3
"""Researcher-run coupled Radau checkpoint against the accepted tested environment."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from check_irksome_be import plan as previous_environment_plan
from check_be_smoke import compare_records
from configure_petsc import ROOT, digest, run_logged, validate_tools

CONFIG = ROOT / "configs/phase0_radau_smoke.json"
SCRIPT = ROOT / "src/phase0_radau_smoke.py"
SNAPSHOT = ROOT / "environment/verified_environment.json"


def plan():
    _, _, installation, petsc, venv, env = previous_environment_plan()
    review_path = ROOT / "environment/irksome_be_review.json"
    review = json.loads(review_path.read_text())
    pairs = [(review[key],review[key+"_sha256"]) for key in
             ("irksome_be_manifest","irksome_be_transcript","config","comparison_config","manual_be_review")]
    pairs.extend((record["path"],record["sha256"]) for record in review["solve_records"].values())
    pairs.extend(review["run_script_sha256"].items())
    for path, expected in pairs:
        if digest(ROOT/path) != expected:
            raise RuntimeError(f"Accepted Irksome BE evidence or source changed: {path}")
    if review["checkpoint_exit_status"] != 0:
        raise RuntimeError("Accepted manual/Irksome BE equivalence is required.")
    snapshot = json.loads(SNAPSHOT.read_text())
    for path, expected in snapshot["evidence_sha256"].items():
        if digest(ROOT/path) != expected:
            raise RuntimeError(f"Tested environment evidence changed: {path}")
    inventory = ROOT / snapshot["resolved_python_inventory"]
    if digest(inventory) != snapshot["resolved_python_inventory_sha256"]:
        raise RuntimeError("Recorded tested package inventory changed.")
    if digest(ROOT/snapshot["petsc"]["library_path"]) != snapshot["petsc"]["library_sha256"]:
        raise RuntimeError("Compiled PETSc library changed since the tested snapshot.")
    return review_path, review, installation, petsc, venv, env


def commands(venv, stem):
    def probe(size,output):
        return [str(venv/"bin/python"),str(SCRIPT),"--config",str(CONFIG),
                "--expected-size",str(size),"--output",str(output)]
    return [("serial",probe(1,str(stem)+"-serial.json")),
            ("two_rank",["/opt/homebrew/opt/open-mpi/bin/mpiexec","-n","2",
                         *probe(2,str(stem)+"-two-rank.json")])]


def run(review_path,review,installation,petsc,venv,env):
    tools = validate_tools(petsc)
    snapshot = json.loads(SNAPSHOT.read_text())
    inventory = subprocess.check_output([str(venv/"bin/python"),"-m","pip","--no-cache-dir","freeze","--all"],
                                        env=env,text=True)
    if inventory != (ROOT/snapshot["resolved_python_inventory"]).read_text():
        raise RuntimeError("Installed package inventory differs from the tested snapshot.")
    for key in ("FIREDRAKE_CACHE_DIR","PYOP2_CACHE_DIR","FIREDRAKE_TSFC_KERNEL_CACHE_DIR","TMPDIR"):
        Path(env[key]).mkdir(parents=True,exist_ok=True)
    config = json.loads(CONFIG.read_text())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = ROOT/"environment/records"/f"radau-smoke-{stamp}-{os.getpid()}"
    transcript = stem.with_suffix(".txt")
    manifest_path = Path(str(stem)+"-manifest.json")
    steps = commands(venv,stem)
    source_paths = (SCRIPT,ROOT/"src/phase0_be_smoke.py",ROOT/"src/phase0_irksome_be.py",
        Path(__file__).resolve(),ROOT/"environment/check_irksome_be.py",ROOT/"environment/check_be_smoke.py",
        ROOT/"environment/check_python_runtime.py",ROOT/"environment/configure_petsc.py")
    state = {"started_utc":datetime.now(timezone.utc).isoformat(),"status":"started",
        "irksome_be_review":str(review_path.relative_to(ROOT)),"irksome_be_review_sha256":digest(review_path),
        "verified_environment":str(SNAPSHOT.relative_to(ROOT)),"verified_environment_sha256":digest(SNAPSHOT),
        "config":str(CONFIG.relative_to(ROOT)),"config_sha256":digest(CONFIG),
        "script_sha256":{str(path.relative_to(ROOT)):digest(path) for path in source_paths},
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
        "performance_measurement":False,"accuracy_order_verified":False,"radau_time_step_run":False}
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
            if serial["stage_total_dofs"] != parallel["stage_total_dofs"]:
                raise RuntimeError("Serial/MPI stage dimensions differ.")
            for record in (serial,parallel):
                if not record["irksome_time_step_run"] or len(record["stages"]) != 2:
                    raise RuntimeError("Individual two-stage Radau check did not finish.")
            state["radau_time_step_run"] = True
            state["status"] = "Radau coupled-block, update, boundary, and selected serial/MPI checks passed; mentor review and convergence verification pending"
            status = 0
            message = "RADAU COUPLED CHECKPOINT PASSED. Review before convergence or timing experiments."
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
                        help="Print the configuration and commands without numerical imports, assembly, solve, or MPI.")
    args = parser.parse_args()
    parameters = plan()
    if args.show_commands:
        print(f"Tested environment: {SNAPSHOT}")
        print(CONFIG.read_text())
        stem = ROOT/"environment/records/radau-smoke-<UTC timestamp>-<process id>"
        for name,argv in commands(parameters[4],stem):
            print(f"{name}: {shlex.join(argv)}")
        print("Display only: no numerical imports, FEM assembly, solve, or MPI launch run.")
        return 0
    return run(*parameters)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError,OSError,subprocess.SubprocessError,KeyError,ValueError) as error:
        print(f"STOP before coupled Radau check: {error}",file=sys.stderr)
        sys.exit(1)

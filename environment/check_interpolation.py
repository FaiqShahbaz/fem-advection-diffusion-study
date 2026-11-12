#!/usr/bin/env python3
"""Researcher-run initial P1/P2 interpolation study with one distributed P2 probe."""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

from check_radau_smoke import plan as previous_environment_plan
from configure_petsc import ROOT, digest, run_logged, validate_tools

CONFIG = ROOT / "configs/phase1_interpolation.json"
SCRIPT = ROOT / "src/phase1_interpolation.py"
SNAPSHOT = ROOT / "environment/verified_environment.json"


def plan():
    _,_,installation,petsc,venv,env = previous_environment_plan()
    review_path = ROOT/"environment/radau_smoke_review.json"
    review = json.loads(review_path.read_text())
    pairs = [(review[key],review[key+"_sha256"]) for key in
             ("radau_manifest","radau_transcript","config","verified_environment","irksome_be_review")]
    pairs.extend((record["path"],record["sha256"]) for record in review["solve_records"].values())
    pairs.extend(review["run_script_sha256"].items())
    for path,expected in pairs:
        if digest(ROOT/path) != expected:
            raise RuntimeError(f"Accepted Radau evidence or source changed: {path}")
    if review["checkpoint_exit_status"] != 0:
        raise RuntimeError("Accepted Radau construction checks are required.")
    return review_path,review,installation,petsc,venv,env


def commands(venv,stem):
    def probe(group,size,output):
        return [str(venv/"bin/python"),str(SCRIPT),"--config",str(CONFIG),
                "--group",group,"--expected-size",str(size),"--output",str(output)]
    return [("serial",probe("all",1,str(stem)+"-serial.json")),
            ("two_rank_P2_probe",["/opt/homebrew/opt/open-mpi/bin/mpiexec","-n","2",
               *probe("mpi_probe",2,str(stem)+"-two-rank.json")])]


def compare_probe(serial,parallel,config):
    if serial["status"] != "passed" or parallel["status"] != "passed":
        raise RuntimeError("Both individual interpolation checks must pass.")
    if serial["config"] != config or parallel["config"] != config:
        raise RuntimeError("Serial/MPI input configuration differs.")
    if not serial["interpolation_trends_checked"] or len(serial["measurements"]) != 6 or len(serial["rates"]) != 8:
        raise RuntimeError("The complete serial P1/P2 refinement was not recorded.")
    if len(parallel["measurements"]) != 1 or parallel["group"] != "mpi_probe":
        raise RuntimeError("Expected one distributed P2 probe.")
    probe = parallel["measurements"][0]
    matching = [row for row in serial["measurements"] if row["case_id"] == probe["case_id"]]
    if len(matching) != 1:
        raise RuntimeError("MPI probe lacks a unique serial counterpart.")
    reference = matching[0]
    for key in ("element_degree","mesh_subdivisions","total_dofs","expected_total_dofs","expected_free_dofs","expected_cells","h"):
        if reference[key] != probe[key]:
            raise RuntimeError(f"Serial/MPI P2 specification differs: {key}")
    compared = {}
    checks = config["checks"]
    for metric_group in ("metrics","control_metrics"):
        for metric,a in reference[metric_group].items():
            b = probe[metric_group][metric]
            if not math.isclose(a,b,rel_tol=checks["serial_mpi_comparison_rtol"],
                                abs_tol=checks["serial_mpi_comparison_atol"]):
                raise RuntimeError(f"Serial/MPI P2 quantity differs beyond tolerance: {metric_group}/{metric}")
            compared[metric_group+"/"+metric] = {"serial":a,"two_rank":b,"absolute_difference":abs(a-b)}
    return {"status":"passed","case_id":probe["case_id"],"compared_global_quantities":compared,
        "relative_tolerance":checks["serial_mpi_comparison_rtol"],
        "absolute_tolerance":checks["serial_mpi_comparison_atol"],
        "limit":"One selected P2 case; agreement of integrated quantities, not all coefficients or MPI scaling"}


def compare_initial_baseline(serial,config):
    reference = json.loads((ROOT/"environment/be_smoke_review.json").read_text())["serial_observations"]["metrics"]["initial_l2_error"]
    rows = [row for row in serial["measurements"] if row["case_id"] == "P1-n8"]
    if len(rows) != 1:
        raise RuntimeError("Missing original P1/n=8 initial-function counterpart.")
    value = rows[0]["metrics"]["l2_error"]
    checks = config["checks"]
    if not math.isclose(value,reference,rel_tol=checks["serial_mpi_comparison_rtol"],
                        abs_tol=checks["serial_mpi_comparison_atol"]):
        raise RuntimeError("Initial P1/n=8 error differs from the accepted original initialization.")
    return {"status":"passed","case_id":"P1-n8","accepted_initial_l2_error":reference,
            "measured_initial_l2_error":value,"absolute_difference":abs(value-reference),
            "reference_review":"environment/be_smoke_review.json"}


def run(review_path,review,installation,petsc,venv,env):
    tools = validate_tools(petsc)
    snapshot = json.loads(SNAPSHOT.read_text())
    inventory = subprocess.check_output([str(venv/"bin/python"),"-m","pip","--no-cache-dir","freeze","--all"],env=env,text=True)
    if inventory != (ROOT/snapshot["resolved_python_inventory"]).read_text():
        raise RuntimeError("Installed package inventory differs from the tested snapshot.")
    for key in ("FIREDRAKE_CACHE_DIR","PYOP2_CACHE_DIR","FIREDRAKE_TSFC_KERNEL_CACHE_DIR","TMPDIR"):
        Path(env[key]).mkdir(parents=True,exist_ok=True)
    (ROOT/"results/raw").mkdir(parents=True,exist_ok=True)
    config = json.loads(CONFIG.read_text())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = ROOT/"results/raw"/f"interpolation-{stamp}-{os.getpid()}"
    transcript = stem.with_suffix(".txt")
    manifest_path = Path(str(stem)+"-manifest.json")
    steps = commands(venv,stem)
    sources = (SCRIPT,ROOT/"src/phase0_be_smoke.py",Path(__file__).resolve(),
       ROOT/"environment/check_radau_smoke.py",ROOT/"environment/check_irksome_be.py",
       ROOT/"environment/check_be_smoke.py",ROOT/"environment/check_python_runtime.py",ROOT/"environment/configure_petsc.py")
    state = {"started_utc":datetime.now(timezone.utc).isoformat(),"status":"started",
        "measurement_kind":"initial nodal interpolation refinement",
        "radau_review":str(review_path.relative_to(ROOT)),"radau_review_sha256":digest(review_path),
        "verified_environment":str(SNAPSHOT.relative_to(ROOT)),"verified_environment_sha256":digest(SNAPSHOT),
        "config":str(CONFIG.relative_to(ROOT)),"config_sha256":digest(CONFIG),
        "script_sha256":{str(path.relative_to(ROOT)):digest(path) for path in sources},
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
        "time_integration_run":False,"project_linear_solve_run":False,
        "transient_fem_convergence_verified":False,"performance_measurement":False}
    status = 1
    manifest_path.write_text(json.dumps(state,indent=2)+"\n")
    with transcript.open("w") as log:
        try:
            for name,argv in steps:
                run_logged(argv,ROOT,env,log)
                state["completed_steps"].append(name)
            serial = json.loads(Path(str(stem)+"-serial.json").read_text())
            parallel = json.loads(Path(str(stem)+"-two-rank.json").read_text())
            state["initial_baseline_comparison"] = compare_initial_baseline(serial,config)
            state["serial_mpi_comparison"] = compare_probe(serial,parallel,config)
            state["status"] = "Initial P1/P2 interpolation, analytic norms, quadrature controls, trends, and one MPI probe passed; mentor review pending"
            status = 0
            message = "INITIAL INTERPOLATION CHECKPOINT PASSED. Review before transient spatial or temporal experiments."
            print(message,flush=True);log.write(message+"\n")
        except (RuntimeError,OSError,subprocess.SubprocessError,KeyError,ValueError,KeyboardInterrupt) as error:
            state["status"] = "failed or interrupted; review required"
            state["error"] = str(error) or "Interrupted by researcher"
            message = f"STOP: {state['error']}"
            print(message,flush=True);log.write(message+"\n")
        finally:
            state["measurement_records"] = {}
            for name in ("serial","two-rank"):
                path = Path(str(stem)+f"-{name}.json")
                if path.is_file():
                    state["measurement_records"][name] = {"path":str(path.relative_to(ROOT)),"sha256":digest(path)}
            state["finished_utc"] = datetime.now(timezone.utc).isoformat()
            state["checkpoint_exit_status"] = status
            manifest_path.write_text(json.dumps(state,indent=2)+"\n")
    print(f"Saved transcript: {transcript}")
    print(f"Saved manifest: {manifest_path}")
    for record in state["measurement_records"].values():
        print(f"Saved measurement record: {ROOT/record['path']}")
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-commands",action="store_true",help="Print inputs/commands without numerical imports, interpolation, integration, solves, or MPI.")
    args = parser.parse_args()
    parameters = plan()
    if args.show_commands:
        print(CONFIG.read_text())
        stem = ROOT/"results/raw/interpolation-<UTC timestamp>-<process id>"
        for name,argv in commands(parameters[4],stem):
            print(f"{name}: {shlex.join(argv)}")
        print("Display only: no numerical imports, interpolation, integration, solve, or MPI launch run.")
        return 0
    return run(*parameters)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError,OSError,subprocess.SubprocessError,KeyError,ValueError) as error:
        print(f"STOP before interpolation verification: {error}",file=sys.stderr)
        sys.exit(1)

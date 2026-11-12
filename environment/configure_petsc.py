#!/usr/bin/env python3
"""Researcher-run PETSc source/configuration checkpoint; not the main build."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configuration(plan):
    """Expand the pinned helper options as argv; never evaluate shell text."""
    helper = ROOT / "environment/vendor/firedrake-configure"
    if digest(helper) != plan["helper_sha256"]:
        raise RuntimeError("Pinned Firedrake helper hash does not match the plan.")
    raw = subprocess.check_output(
        [sys.executable, str(helper), "--os", plan["helper_os_selection"],
         "--show-petsc-configure-options"], text=True
    ).strip()
    if raw != plan["upstream_configure_options"]:
        raise RuntimeError("Helper options differ from the reviewed planning record.")
    upstream = shlex.split(raw)
    removed = plan["project_removed_upstream_options"]
    if any(upstream.count(option) != 1 for option in removed):
        raise RuntimeError("Expected upstream option for replacement was not found once.")
    options = [option for option in upstream if option not in removed]
    return options + plan["project_configure_options"]


def validate_tools(plan):
    if sys.version_info[:2] != (3, 13) or platform.machine() != "arm64":
        raise RuntimeError("Use the selected native Homebrew Python 3.13.")
    expected_python = Path("/opt/homebrew/opt/python@3.13/bin/python3.13")
    if Path(sys.executable).resolve() != expected_python.resolve():
        raise RuntimeError("Interpreter does not match Homebrew Python 3.13.")
    if os.environ.get("CONDA_SHLVL", "0") not in ("", "0") or sys.prefix != sys.base_prefix:
        raise RuntimeError("Run this checkpoint with Conda and Python venv inactive.")
    tools = {"python": sys.executable}
    for command in ("mpicc", "mpicxx", "mpif90", "mpiexec"):
        selected = shutil.which(command)
        expected = Path("/opt/homebrew/opt/open-mpi/bin") / command
        if not selected or Path(selected).resolve() != expected.resolve():
            raise RuntimeError(f"{command} does not select the expected Homebrew MPI.")
        tools[command] = selected
    for path in ("/opt/homebrew/bin/gmake",
                 "/opt/homebrew/opt/openblas/lib/libopenblas.dylib",
                 "/opt/homebrew/opt/zlib/include/zlib.h",
                 "/opt/homebrew/opt/zlib/lib/libz.dylib"):
        if not Path(path).exists():
            raise RuntimeError(f"Required build tool/library is absent: {path}")
    report = ROOT / plan["prerequisite_record"]
    if "BUILD PREREQUISITE CHECK PASSED" not in report.read_text():
        raise RuntimeError("The reviewed successful prerequisite report is absent.")
    return tools


def run_logged(argv, cwd, env, log):
    display = shlex.join(argv)
    print(f"\nRunning: {display}", flush=True)
    log.write(f"\nWorking directory: {cwd}\nRunning: {display}\n")
    log.flush()
    with subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        status = process.wait()
    log.write(f"Exit status: {status}\n")
    log.flush()
    if status:
        raise RuntimeError(f"Command failed with exit status {status}; review the saved log.")


def configure(plan, options):
    tools = validate_tools(plan)
    source = ROOT / plan["planned_build_directory"]
    records = ROOT / "environment/records"
    records.mkdir(exist_ok=True)
    source.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = records / f"petsc-configure-{stamp}-{os.getpid()}"
    transcript = stem.with_suffix(".txt")
    manifest_path = Path(str(stem) + "-manifest.json")
    full_log = Path(str(stem) + "-configure.log")
    env = os.environ.copy()
    env.update(PETSC_DIR=str(source), PETSC_ARCH=plan["petsc_arch"],
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    argv = [sys.executable, str(source / "configure"), *options]
    manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "started", "main_petsc_build_run": False,
        "plan_sha256": digest(ROOT / "environment/petsc_plan.json"),
        "prerequisite_record": plan["prerequisite_record"],
        "prerequisite_sha256": digest(ROOT / plan["prerequisite_record"]),
        "helper_sha256": plan["helper_sha256"],
        "source_url": plan["petsc_source_url"], "source_tag": plan["petsc_tag"],
        "expected_source_commit": plan["petsc_source_commit"],
        "cwd": str(source), "configure_argv": argv,
        "selected_tools": tools,
        "selected_environment": {key: env[key] for key in (
            "PETSC_DIR", "PETSC_ARCH", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
        "inherited_build_settings": {key: env[key] for key in (
            "CC", "CXX", "FC", "CFLAGS", "CXXFLAGS", "FFLAGS", "CPPFLAGS",
            "LDFLAGS", "LIBS", "PKG_CONFIG_PATH", "DYLD_LIBRARY_PATH") if key in env}
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    status = 1
    with transcript.open("w") as log:
        try:
            log.write(f"PETSc configuration checkpoint, UTC {stamp}\n")
            if not source.exists():
                run_logged(["git", "clone", "--depth", "1", "--branch", plan["petsc_tag"],
                            plan["petsc_source_url"], str(source)], ROOT, env, log)
            if not (source / ".git").is_dir():
                raise RuntimeError("Existing source directory is not a PETSc Git clone; nothing removed.")
            actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
            manifest["actual_source_commit"] = actual
            if actual != plan["petsc_source_commit"]:
                raise RuntimeError("PETSc source commit does not match the reviewed pin.")
            run_logged(["git", "diff", "--exit-code", "HEAD"], source, env, log)
            run_logged(argv, source, env, log)
            status = 0
            manifest["status"] = "configure command passed; main build and checks pending"
            message = "PETSC CONFIGURATION COMMAND PASSED. Main PETSc build/checks remain pending."
            print(message, flush=True)
            log.write(message + "\n")
        except (RuntimeError, OSError, subprocess.SubprocessError, KeyboardInterrupt) as error:
            manifest["status"] = "failed or interrupted; review required"
            manifest["error"] = str(error) or "Interrupted by researcher"
            message = f"STOP: {manifest['error']}"
            print(message, flush=True)
            log.write(message + "\n")
        finally:
            # PETSc's complete diagnostic log includes compiler/library probe details.
            candidates = [source / plan["petsc_arch"] / "lib/petsc/conf/configure.log",
                          source / "configure.log"]
            for candidate in candidates:
                if candidate.is_file():
                    shutil.copyfile(candidate, full_log)
                    manifest["full_configure_log"] = str(full_log.relative_to(ROOT))
                    break
            manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
            manifest["checkpoint_exit_status"] = status
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved transcript: {transcript}")
    print(f"Saved manifest: {manifest_path}")
    if full_log.exists():
        print(f"Saved full configure log: {full_log}")
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-command", action="store_true",
                        help="Read and display the planned command; do not clone or configure.")
    args = parser.parse_args()
    plan = json.loads((ROOT / "environment/petsc_plan.json").read_text())
    options = configuration(plan)
    if args.show_command:
        source = ROOT / plan["planned_build_directory"]
        print(f"Planned source: {plan['petsc_tag']} at {plan['petsc_source_commit']}")
        print(f"Planned working directory: {source}")
        print(shlex.join([sys.executable, str(source / "configure"), *options]))
        print("Display only: no source clone, configuration, or build was run.")
        return 0
    return configure(plan, options)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, OSError, subprocess.SubprocessError) as error:
        print(f"STOP before configuration: {error}", file=sys.stderr)
        sys.exit(1)

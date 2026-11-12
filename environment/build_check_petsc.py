#!/usr/bin/env python3
"""Researcher-run PETSc main build followed by supplied installation checks."""

import argparse
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from configure_petsc import ROOT, digest, run_logged, validate_tools


def reviewed_configuration():
    plan = json.loads((ROOT / "environment/petsc_plan.json").read_text())
    review = json.loads((ROOT / "environment/petsc_configuration_review.json").read_text())
    manifest_path = ROOT / review["configuration_manifest"]
    if digest(manifest_path) != review["configuration_manifest_sha256"]:
        raise RuntimeError("Configuration manifest changed since mentor review.")
    configured = json.loads(manifest_path.read_text())
    if digest(ROOT / "environment/petsc_plan.json") != configured["plan_sha256"]:
        raise RuntimeError("Planning options changed since the successful configuration.")
    if configured["checkpoint_exit_status"] != 0:
        raise RuntimeError("A successful reviewed configuration is required.")
    source = ROOT / plan["planned_build_directory"]
    for relative, expected in review["configured_files"].items():
        if digest(source / relative) != expected:
            raise RuntimeError(f"Configured file changed since review: {relative}")
    common = ["/opt/homebrew/bin/gmake", f"PETSC_DIR={source}",
              f"PETSC_ARCH={plan['petsc_arch']}"]
    # Sequential commands: the runtime check starts only if the build passes.
    commands = {"build": [*common, "-j4", "all"],
                "check": [*common, "-j1", "check"]}
    return plan, review, source, commands


def archive_log(source, destination):
    if not source.is_file():
        return None
    with destination.open("wb") as out:
        with gzip.GzipFile(filename="", mode="wb", fileobj=out, mtime=0) as gz:
            gz.write(source.read_bytes())
    return {"path": str(destination.relative_to(ROOT)), "sha256": digest(destination),
            "uncompressed_sha256": digest(source), "uncompressed_bytes": source.stat().st_size}


def build_and_check(plan, review, source, commands):
    tools = validate_tools(plan)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if actual != plan["petsc_source_commit"]:
        raise RuntimeError("PETSc source commit differs from the reviewed pin.")
    env = os.environ.copy()
    env.update(PETSC_DIR=str(source), PETSC_ARCH=plan["petsc_arch"],
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = ROOT / "environment/records" / f"petsc-build-check-{stamp}-{os.getpid()}"
    transcript = stem.with_suffix(".txt")
    manifest_path = Path(str(stem) + "-manifest.json")
    state = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "started", "build_status": "pending", "vendor_check_status": "pending",
        "source_commit": actual, "cwd": str(source), "commands": commands,
        "configuration_manifest": review["configuration_manifest"],
        "configuration_manifest_sha256": review["configuration_manifest_sha256"],
        "configuration_review_sha256": digest(ROOT / "environment/petsc_configuration_review.json"),
        "selected_tools": tools,
        "selected_environment": {key: env[key] for key in (
            "PETSC_DIR", "PETSC_ARCH", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
        "inherited_check_settings": {key: env[key] for key in (
            "MAKEFLAGS", "PETSC_OPTIONS", "PETSC_TEST_OPTIONS", "EXTRA_OPTIONS") if key in env},
        "log_archives": {}
    }
    manifest_path.write_text(json.dumps(state, indent=2) + "\n")
    status = 1
    phase = "source validation"
    with transcript.open("w") as log:
        try:
            run_logged(["git", "diff", "--exit-code", "HEAD"], source, env, log)
            phase = "main build"
            state["build_status"] = "running"
            run_logged(commands["build"], source, env, log)
            state["build_status"] = "passed"
            library = source / plan["petsc_arch"] / "lib/libpetsc.dylib"
            if not library.is_file():
                raise RuntimeError("Build returned success but libpetsc.dylib is absent.")
            inspection = subprocess.check_output(["/usr/bin/file", str(library)], text=True)
            if "arm64" not in inspection:
                raise RuntimeError("Built PETSc library does not identify as arm64.")
            linkage = subprocess.check_output(["/usr/bin/otool", "-L", str(library)], text=True)
            state["library_inspection"] = inspection.strip()
            state["library_linkage"] = linkage.splitlines()
            print(inspection + linkage, end="", flush=True)
            log.write(inspection + linkage)
            phase = "vendor check"
            state["vendor_check_status"] = "running"
            run_logged(commands["check"], source, env, log)
            state["vendor_check_command_exit_status"] = 0
            check_log = source / plan["petsc_arch"] / "lib/petsc/conf/check.log"
            checked = check_log.read_text()
            required = ["C/C++ example src/snes/tutorials/ex19 run successfully with 1 MPI process",
                        "C/C++ example src/snes/tutorials/ex19 run successfully with 2 MPI processes"]
            state["required_example_success_lines"] = [line for line in required if line in checked]
            if any(line not in checked for line in required):
                raise RuntimeError("Expected serial/two-process success lines are absent; review check.log.")
            state["vendor_check_status"] = "passed; mentor log review pending"
            state["status"] = "build/check checkpoint passed; Python stack remains unverified"
            status = 0
            message = "PETSC BUILD AND CHECK CHECKPOINT PASSED. Python-stack verification remains pending."
            print(message, flush=True)
            log.write(message + "\n")
        except (RuntimeError, OSError, subprocess.SubprocessError, KeyboardInterrupt) as error:
            state["status"] = "failed or interrupted; review required"
            state["failed_phase"] = phase
            state["error"] = str(error) or "Interrupted by researcher"
            if phase == "main build":
                state["build_status"] = "failed or incomplete; review required"
            elif phase == "vendor check":
                state["vendor_check_status"] = "failed or incomplete; review required"
            message = f"STOP during {phase}: {state['error']}"
            print(message, flush=True)
            log.write(message + "\n")
        finally:
            for name in ("make", "check"):
                archived = archive_log(source / plan["petsc_arch"] / f"lib/petsc/conf/{name}.log",
                                       Path(str(stem) + f"-{name}.log.gz"))
                if archived:
                    state["log_archives"][name] = archived
            state["finished_utc"] = datetime.now(timezone.utc).isoformat()
            state["checkpoint_exit_status"] = status
            manifest_path.write_text(json.dumps(state, indent=2) + "\n")
    print(f"Saved transcript: {transcript}")
    print(f"Saved manifest: {manifest_path}")
    for archived in state["log_archives"].values():
        print(f"Saved full log archive: {ROOT / archived['path']}")
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-commands", action="store_true",
                        help="Validate reviewed records and display commands without building or launching MPI.")
    args = parser.parse_args()
    plan, review, source, commands = reviewed_configuration()
    if args.show_commands:
        print(f"Working directory: {source}")
        for phase, argv in commands.items():
            print(f"{phase}: {shlex.join(argv)}")
        print("Display only: no PETSc build or MPI check was run.")
        return 0
    return build_and_check(plan, review, source, commands)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, OSError, subprocess.SubprocessError) as error:
        print(f"STOP before main build: {error}", file=sys.stderr)
        sys.exit(1)

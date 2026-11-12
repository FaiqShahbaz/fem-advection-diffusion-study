#!/usr/bin/env python3
"""Researcher-run binding/tableau probe; no project FEM assembly or time stepping."""

import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def probe(comm, expected_size):
    # A collective with an independently known answer checks the Python MPI binding.
    rank, size = comm.Get_rank(), comm.Get_size()
    require(size == expected_size, "MPI process count differs from the requested probe.")
    total = comm.allreduce(rank + 1)
    require(total == size * (size + 1) // 2, "MPI collective returned an incorrect sum.")

    # Let Firedrake initialize PETSc before accessing the bindings.
    import firedrake
    from firedrake.petsc import PETSc
    import numpy as np
    import mpi4py
    import petsc4py
    import h5py
    import irksome
    from irksome.backend import get_backend
    from mpi4py import MPI

    require(Path(sys.prefix).resolve() == VENV.resolve(), "Use the project venv interpreter.")
    require(platform.machine() == "arm64", "Expected native ARM execution.")
    expected = json.loads((ROOT / "environment/python_installation_review.json").read_text())
    versions = {name: metadata.version(name) for name in expected["installed_metadata"]["versions"]}
    require(versions == expected["installed_metadata"]["versions"], "Package versions changed.")
    modules = {"mpi4py": mpi4py, "petsc4py": petsc4py, "PETSc": PETSc,
               "h5py": h5py, "firedrake": firedrake, "irksome": irksome}
    paths = {name: str(Path(module.__file__).resolve()) for name, module in modules.items()}
    require(all(Path(path).is_relative_to(VENV.resolve()) for path in paths.values()),
            "A numerical module loaded from outside the project venv.")
    config = petsc4py.get_config()
    require(Path(config["PETSC_DIR"]).resolve() == (ROOT / "environment/build/petsc").resolve(),
            "petsc4py was built against a different PETSc source directory.")
    require(config["PETSC_ARCH"] == "arch-firedrake-default", "Different PETSc architecture.")
    require(PETSc.Sys.getVersion() == (3, 26, 0), "Different loaded PETSc version.")
    require(np.dtype(PETSc.ScalarType) == np.dtype("float64"), "Expected real double PETSc.")
    require(np.dtype(PETSc.IntType) == np.dtype("int32"), "Expected 32-bit PETSc indices.")
    require(PETSc.COMM_WORLD.getSize() == size and PETSc.COMM_WORLD.getRank() == rank,
            "PETSc and mpi4py communicator sizes/ranks differ.")
    require(PETSc.COMM_WORLD.tompi4py().Compare(comm) in (MPI.IDENT, MPI.CONGRUENT),
            "PETSc and mpi4py world communicators are incompatible.")
    vendor, vendor_version = MPI.get_vendor()
    require(vendor == "Open MPI" and tuple(vendor_version) == (5, 0, 11),
            "Loaded MPI differs from the reviewed Open MPI 5.0.11.")
    require(h5py.get_config().mpi, "h5py does not report MPI support.")
    backend = get_backend("firedrake")
    require(backend.Function is firedrake.Function, "Irksome Firedrake backend mismatch.")

    be, radau = irksome.BackwardEuler(), irksome.RadauIIA(2)
    for scheme, A, b, c, order in (
        (be, [[1.0]], [1.0], [1.0], 1),
        (radau, [[5/12, -1/12], [3/4, 1/4]], [3/4, 1/4], [1/3, 1.0], 3),
    ):
        for actual, wanted in ((scheme.A, A), (scheme.b, b), (scheme.c, c)):
            require(np.shape(actual) == np.shape(wanted) and
                    np.allclose(actual, wanted, atol=5e-14, rtol=0),
                    "Irksome coefficients differ from the derived tableau.")
        require(scheme.order == order and scheme.num_stages == len(b),
                "Unexpected tableau order/stage metadata.")
        require(np.allclose(scheme.A @ np.ones(len(b)), scheme.c, atol=5e-14, rtol=0),
                "Tableau row sums differ from its stage times.")
    # Independent order-three algebraic identities, not a measured convergence rate.
    for actual, wanted in ((radau.b.sum(), 1), (radau.b @ radau.c, 1/2),
                           (radau.b @ (radau.c**2), 1/3),
                           (radau.b @ (radau.A @ radau.c), 1/6)):
        require(abs(actual - wanted) < 5e-14, "Radau order condition failed.")
    require(np.allclose(radau.A[-1], radau.b, atol=5e-14, rtol=0),
            "Radau stiff accuracy identity failed.")
    print(f"Rank {rank}/{size}: imports, bindings, MPI sum {total}, and BE/Radau coefficients passed.",
          flush=True)
    return {"rank": rank, "size": size, "collective_sum": total,
            "python": sys.executable, "architecture": platform.machine(),
            "versions": versions, "module_paths": paths, "petsc_build_config": config,
            "petsc_version": list(PETSc.Sys.getVersion()),
            "petsc_scalar_dtype": str(np.dtype(PETSc.ScalarType)),
            "petsc_index_dtype": str(np.dtype(PETSc.IntType)),
            "mpi_vendor": vendor, "mpi_vendor_version": list(vendor_version),
            "mpi_library_version": MPI.Get_library_version().strip(),
            "hdf5_mpi_enabled": bool(h5py.get_config().mpi),
            "hdf5_runtime_version": h5py.version.hdf5_version,
            "radau_A": radau.A.tolist(), "radau_b": radau.b.tolist(),
            "radau_c": radau.c.tolist(), "radau_order_conditions_passed": True,
            "project_fem_assembly_or_solve_run": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-size", type=int, choices=(1, 2), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # This probe accepts only application arguments, not PETSc command-line options.
    sys.argv = [sys.argv[0]]
    comm = None
    try:
        output = args.output.resolve()
        require(output.parent == ROOT / "environment/records", "Probe records must stay in the project.")
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        record = probe(comm, args.expected_size)
        records = comm.gather(record, root=0)
        if comm.Get_rank() == 0:
            output.write_text(json.dumps({"status": "passed", "ranks": records}, indent=2) + "\n")
        comm.Barrier()
        return 0
    except Exception:
        traceback.print_exc()
        # Terminate the group if a rank fails before another rank enters a collective.
        if comm is not None and comm.Get_size() > 1:
            comm.Abort(1)
        return 1


if __name__ == "__main__":
    sys.exit(main())

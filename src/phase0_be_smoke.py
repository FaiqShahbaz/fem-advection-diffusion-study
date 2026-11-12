#!/usr/bin/env python3
"""Researcher-run one-step BE for the agreed MMS; no performance experiment."""

import argparse
import json
import math
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def solve_one_step(config, expected_size, fd, PETSc, *, return_context=False):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    require(comm.size == expected_size, "Unexpected MPI process count.")
    require(config["element_family"] == "CG" and config["element_degree"] == 1,
            "This first teaching checkpoint uses P1 only.")
    require(config["steps"] == 1 and config["dt"] > 0 and config["diffusion"] > 0,
            "This checkpoint requires one positive time step and positive diffusion.")
    require(config["velocity"] == [1.0, 1.0] and config["mesh_diagonal"] == "left",
            "Keep the agreed velocity and stated mesh convention.")
    n = config["mesh_subdivisions"]
    require(isinstance(n, int) and n >= 2, "Use an integer mesh subdivision count >=2.")
    dt, kappa = config["dt"], config["diffusion"]

    # Domain and FE space: u_h(x,t) = sum_j U_j(t) phi_j(x).
    mesh = fd.UnitSquareMesh(n, n, diagonal="left", comm=comm)
    V = fd.FunctionSpace(mesh, "CG", 1)
    w, v = fd.TrialFunction(V), fd.TestFunction(V)
    x, y = fd.SpatialCoordinate(mesh)
    beta = fd.as_vector(config["velocity"])
    dxq = fd.dx(domain=mesh, metadata={"quadrature_degree": config["quadrature_degree"]})
    dsq = fd.ds(domain=mesh, metadata={"quadrature_degree": config["quadrature_degree"]})
    bc = fd.DirichletBC(V, 0.0, "on_boundary")

    # Initial nodal interpolation, then enforce exact zero boundary coefficients.
    initial = fd.sin(fd.pi*x) * fd.sin(fd.pi*y)
    u0 = fd.Function(V, name="u_initial").interpolate(initial)
    bc.apply(u0)
    u1 = fd.Function(V, name="u_after_one_BE_step")
    exact = fd.exp(-dt) * initial
    forcing = fd.exp(-dt) * (
        (2*kappa*fd.pi**2 - 1) * initial
        + fd.pi*fd.cos(fd.pi*x)*fd.sin(fd.pi*y)
        + fd.pi*fd.sin(fd.pi*x)*fd.cos(fd.pi*y))

    # Independent symbolic spatial differentiation checks the manufactured source.
    pde_residual = -exact + fd.dot(beta, fd.grad(exact)) - kappa*fd.div(fd.grad(exact)) - forcing
    source_residual = math.sqrt(max(0.0, float(fd.assemble(pde_residual**2 * dxq))))
    require(source_residual <= config["checks"]["source_residual_l2_max"],
            "Manufactured source fails the strong PDE identity.")

    # The matrix represents M + dt L on the free coefficients.
    mass = w*v*dxq
    transport_diffusion = (fd.dot(beta, fd.grad(w))*v
                           + kappa*fd.inner(fd.grad(w), fd.grad(v))) * dxq
    a = mass + dt*transport_diffusion
    rhs_form = (u0*v + dt*forcing*v) * dxq
    B = fd.assemble(a, bcs=[bc], mat_type="aij")
    rhs = fd.assemble(rhs_form, bcs=[bc])
    solver = fd.LinearSolver(B, solver_parameters=config["solver_parameters"],
                             options_prefix="phase0_be")
    history = []
    solver.ksp.setMonitor(lambda ksp, iteration, norm: history.append([iteration, float(norm)]))
    solver.solve(u1, rhs)
    ksp = solver.ksp
    require(ksp.getConvergedReason() > 0, "GMRES did not report convergence.")
    require(ksp.getType() == "gmres" and ksp.getPC().getType() == "jacobi",
            "Actual Krylov method or preconditioner differs from the teaching configuration.")
    require(ksp.getPCSide() == PETSc.PC.Side.RIGHT and
            ksp.getNormType() == PETSc.KSP.NormType.UNPRECONDITIONED,
            "Expected right preconditioning and an unpreconditioned stopping norm.")

    # Recompute the true global Euclidean residual using the assembled matrix and RHS.
    r = B.petscmat.createVecLeft()
    with u1.dat.vec_ro as solution, rhs.dat.vec_ro as right_hand_side:
        B.petscmat.mult(solution, r)
        r.axpy(-1.0, right_hand_side)
        residual_norm = float(r.norm())
        rhs_norm = float(right_hand_side.norm())
    r.destroy()
    rtol, atol, divtol, max_it = ksp.getTolerances()
    requested = config["solver_parameters"]
    require(rtol == requested["ksp_rtol"] and atol == requested["ksp_atol"] and
            max_it == requested["ksp_max_it"] and not ksp.getInitialGuessNonzero(),
            "Actual tolerances or initial-guess policy differ from the saved configuration.")
    target = max(atol, rtol*rhs_norm)
    require(residual_norm <= config["checks"]["residual_tolerance_factor"]*target,
            "Recomputed true residual exceeds the stated tolerance allowance.")

    # Errors against the continuum MMS mix spatial, temporal, quadrature, and solver error.
    error = u1 - exact
    metrics = {
        "domain_area": float(fd.assemble(fd.Constant(1.0)*dxq)),
        "source_residual_l2": source_residual,
        "initial_l2_error": math.sqrt(max(0.0, float(fd.assemble((u0-initial)**2*dxq)))),
        "final_l2_error": math.sqrt(max(0.0, float(fd.assemble(error**2*dxq)))),
        "final_h1_seminorm_error": math.sqrt(max(0.0, float(fd.assemble(fd.inner(fd.grad(error),fd.grad(error))*dxq)))),
        "solution_l2_norm": math.sqrt(max(0.0, float(fd.assemble(u1**2*dxq)))),
        "boundary_trace_l2": math.sqrt(max(0.0, float(fd.assemble(u1**2*dsq)))),
        "true_residual_norm": residual_norm,
        "rhs_norm": rhs_norm,
        "true_relative_residual": residual_norm/rhs_norm if rhs_norm else 0.0,
    }
    require(all(math.isfinite(value) for value in metrics.values()), "Nonfinite check metric.")
    require(abs(metrics["domain_area"]-1.0) <= 1e-12, "Mesh domain area differs from one.")
    require(metrics["solution_l2_norm"] > 0, "Unexpected zero solution.")
    require(metrics["boundary_trace_l2"] <= config["checks"]["boundary_trace_l2_max"],
            "Computed boundary trace is not zero to the stated tolerance.")
    require(V.dim() == (n+1)**2 and B.petscmat.getSize() == ((n+1)**2, (n+1)**2),
            "P1 dimension or assembled matrix size differs from the structured-mesh formula.")
    ksp.view()  # Record actual restart/PC/solver details in the checkpoint transcript.
    if comm.rank == 0:
        print(f"One BE step: P1, {V.dim()} total DOFs, dt={dt}, kappa={kappa}.", flush=True)
        print(f"GMRES iterations={ksp.getIterationNumber()}, true relative residual={metrics['true_relative_residual']:.6e}.", flush=True)
        print(f"L2 error={metrics['final_l2_error']:.6e}; this single value is not an observed order.", flush=True)
    result = {
        "status": "passed", "ranks": comm.size, "config": config,
        "total_dofs": V.dim(), "expected_free_dofs": (n-1)**2,
        "expected_cells": 2*n*n, "h": math.sqrt(2)/n,
        "h_definition": "Maximum diameter of the right triangles in the stated structured mesh",
        "final_time": dt, "method": "manual backward Euler, one step",
        "metrics": metrics,
        "actual_solver": {"ksp_type":ksp.getType(), "pc_type":ksp.getPC().getType(),
            "preconditioning_side":"right", "stopping_norm":"unpreconditioned",
            "rtol":rtol, "atol":atol, "divtol":divtol, "max_it":max_it,
            "iterations":ksp.getIterationNumber(), "converged_reason":int(ksp.getConvergedReason()),
            "reported_residual_norm":float(ksp.getResidualNorm()), "residual_history":history,
            "true_residual_target":target, "true_residual_allowance_factor":config["checks"]["residual_tolerance_factor"]},
        "accuracy_order_or_cost_claim": False, "irksome_time_step_run": False,
    }
    if return_context:
        return result, {"initial": u0, "solution": u1, "space": V, "mesh": mesh,
                        "operator": B, "rhs": rhs, "bc": bc, "dx": dxq, "ds": dsq,
                        "forcing_end": forcing}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-size", type=int, choices=(1,2), required=True)
    args = parser.parse_args()
    # All application flags are parsed here; solver options come only from the saved config.
    sys.argv = [sys.argv[0]]
    comm = None
    try:
        require(args.output.resolve().parent == ROOT/"environment/records", "Keep checkpoint records in the project.")
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        # Let Firedrake initialize PETSc, then access that initialized instance.
        import firedrake as fd
        from firedrake.petsc import PETSc
        config = json.loads(args.config.read_text())
        result = solve_one_step(config, args.expected_size, fd, PETSc)
        if comm.rank == 0:
            args.output.write_text(json.dumps(result, indent=2)+"\n")
        comm.Barrier()
        return 0
    except Exception:
        traceback.print_exc()
        if comm is not None and comm.size > 1:
            comm.Abort(1)
        return 1


if __name__ == "__main__":
    sys.exit(main())

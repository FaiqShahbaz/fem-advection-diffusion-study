#!/usr/bin/env python3
"""Researcher-run same-mesh manual/Irksome BE equivalence check, one step."""

import argparse
import copy
import json
import math
from pathlib import Path
import sys
import traceback

from phase0_be_smoke import require, solve_one_step

ROOT = Path(__file__).resolve().parents[1]


def true_residual(B, solution, rhs):
    """Global Euclidean norm of B*solution-rhs, recomputed independently."""
    residual = B.petscmat.createVecLeft()
    with solution.dat.vec_ro as x, rhs.dat.vec_ro as b:
        B.petscmat.mult(x, residual)
        residual.axpy(-1.0, b)
        value, rhs_norm = float(residual.norm()), float(b.norm())
    residual.destroy()
    return {"norm":value, "rhs_norm":rhs_norm,
            "relative_norm":value/rhs_norm if rhs_norm else 0.0}


def crosscheck(config, comparison, expected_size, fd, PETSc, irk, AI):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    require(comparison["time_method"] == "BackwardEuler" and comparison["stage_type"] == "deriv"
            and comparison["bc_type"] == "DAE" and comparison["splitting"] == "AI",
            "This checkpoint uses BE with classical derivative stages and zero-boundary DAE constraints.")
    manual, context = solve_one_step(config, expected_size, fd, PETSc, return_context=True)
    V, mesh = context["space"], context["mesh"]
    dxq, dsq = context["dx"], context["ds"]
    initial, manual_solution = context["initial"], context["solution"]
    state = fd.Function(V, name="irksome_BE_solution").assign(initial)
    require(state is not manual_solution, "Irksome must start from the original initial approximation.")
    t, dt = fd.Constant(0.0), fd.Constant(config["dt"])
    w, v = fd.TrialFunction(V), fd.TestFunction(V)
    x, y = fd.SpatialCoordinate(mesh)
    beta, kappa = fd.as_vector(config["velocity"]), config["diffusion"]
    sinusoid = fd.sin(fd.pi*x)*fd.sin(fd.pi*y)
    source = fd.exp(-t)*((2*kappa*fd.pi**2-1)*sinusoid
             + fd.pi*fd.cos(fd.pi*x)*fd.sin(fd.pi*y)
             + fd.pi*fd.sin(fd.pi*x)*fd.cos(fd.pi*y))

    # Irksome receives the semi-discrete residual, not a pre-written BE step.
    F = (irk.Dt(w)*v + fd.dot(beta,fd.grad(w))*v
         + kappa*fd.inner(fd.grad(w),fd.grad(v)) - source*v)*dxq
    tableau = irk.BackwardEuler()
    stage_parameters = {**config["solver_parameters"], "mat_type":"aij"}
    stepper = irk.TimeStepper(F, tableau, t, dt, state, bcs=[context["bc"]],
        stage_type="deriv", splitting=AI, bc_type="DAE", constant_jacobian=True,
        solver_parameters=stage_parameters, options_prefix="phase0_irksome_be")
    require(stepper.num_stages == 1, "BE must have one stage.")
    stage_ksp = stepper.solver.snes.ksp
    history = []
    stage_ksp.setMonitor(lambda ksp, iteration, norm: history.append([iteration,float(norm)]))
    stepper.advance()  # Mutates state, but the caller owns the time variable.
    require(float(t) == 0.0, "Unexpected automatic time increment; inspect the installed API.")
    require(stage_ksp.getConvergedReason() > 0, "Irksome stage solve did not converge.")
    require(stage_ksp.getType() == "gmres" and stage_ksp.getPC().getType() == "jacobi"
            and stage_ksp.getPCSide() == PETSc.PC.Side.RIGHT
            and stage_ksp.getNormType() == PETSc.KSP.NormType.UNPRECONDITIONED,
            "Actual stage solver configuration differs from the stated baseline.")
    rtol,atol,divtol,max_it = stage_ksp.getTolerances()
    require(rtol == config["solver_parameters"]["ksp_rtol"]
            and atol == config["solver_parameters"]["ksp_atol"]
            and max_it == config["solver_parameters"]["ksp_max_it"]
            and not stage_ksp.getInitialGuessNonzero(), "Unexpected stage tolerances/initial-guess policy.")

    # For BE/AI, the stored single stage is the physical slope Z.
    slope = stepper.stages.subfunctions[0]
    stage_rhs_form = (context["forcing_end"]*v - fd.dot(beta,fd.grad(initial))*v
                     - kappa*fd.inner(fd.grad(initial),fd.grad(v)))*dxq
    stage_rhs = fd.assemble(stage_rhs_form,bcs=[context["bc"]])
    stage_residual = true_residual(context["operator"],slope,stage_rhs)
    final_residual = true_residual(context["operator"],state,context["rhs"])
    allowance = config["checks"]["residual_tolerance_factor"]
    require(stage_residual["norm"] <= allowance*max(atol,rtol*stage_residual["rhs_norm"]),
            "Independent stage-equation residual is too large.")
    require(final_residual["norm"] <= allowance*max(atol,rtol*final_residual["rhs_norm"]),
            "Irksome update fails the common manually assembled BE equation.")

    def l2(expression):
        value = float(fd.assemble(expression**2*dxq))
        require(math.isfinite(value), "Nonfinite integrated squared error.")
        require(value >= -1e-14, "Unexpected negative integrated squared error.")
        return math.sqrt(max(0.0,value))

    difference = l2(state-manual_solution)
    reference_norm = l2(manual_solution)
    limit = comparison["field_difference_atol"] + comparison["field_difference_rtol"]*reference_norm
    require(difference <= limit, "Manual and Irksome BE fields differ beyond the declared tolerance.")
    update_defect = l2(state-(initial+config["dt"]*slope))
    require(update_defect <= comparison["update_identity_l2_max"], "BE slope-to-solution update identity failed.")
    exact = fd.exp(-config["dt"])*sinusoid
    error = state-exact
    metrics = {
        "domain_area":manual["metrics"]["domain_area"],
        "source_residual_l2":manual["metrics"]["source_residual_l2"],
        "initial_l2_error":manual["metrics"]["initial_l2_error"],
        "final_l2_error":l2(error),
        "final_h1_seminorm_error":math.sqrt(max(0.0,float(fd.assemble(fd.inner(fd.grad(error),fd.grad(error))*dxq)))),
        "solution_l2_norm":l2(state),
        "boundary_trace_l2":math.sqrt(max(0.0,float(fd.assemble(state**2*dsq)))),
        "true_residual_norm":final_residual["norm"], "rhs_norm":final_residual["rhs_norm"],
        "true_relative_residual":final_residual["relative_norm"],
    }
    require(all(math.isfinite(value) for value in metrics.values()), "Nonfinite Irksome metric.")
    require(metrics["boundary_trace_l2"] <= config["checks"]["boundary_trace_l2_max"],
            "Irksome boundary trace is not zero to the stated tolerance.")
    stats = stepper.solver_stats()
    require(stats[0] == 1, "Expected exactly one Irksome step.")
    stage_ksp.view()
    t.assign(config["dt"])  # Explicit caller update after checking the t=0 stage equation.
    result = copy.deepcopy(manual)
    result.update(method="manual/Irksome backward Euler one-step cross-check",
                  metrics=metrics, manual_reference=manual, irksome_time_step_run=True)
    result.pop("actual_solver")
    result["comparison_config"] = comparison
    result["field_comparison"] = {"difference_l2":difference,
        "reference_l2_norm":reference_norm,"relative_difference_l2":difference/reference_norm,
        "allowed_difference_l2":limit,"update_identity_l2":update_defect,"status":"passed"}
    result["irksome_stage"] = {
        "tableau":"BackwardEuler", "stage_type":"deriv", "splitting":"AI", "bc_type":"DAE",
        "t_before":0.0,"stage_time":config["dt"],"t_after_caller_update":float(t),
        "stage_equation_residual":stage_residual, "common_BE_equation_residual":final_residual,
        "slope_space_dofs":slope.function_space().dim(), "solver_stats":list(stats),
        "actual_solver":{"ksp_type":stage_ksp.getType(),"pc_type":stage_ksp.getPC().getType(),
            "preconditioning_side":"right","stopping_norm":"unpreconditioned","rtol":rtol,"atol":atol,
            "divtol":divtol,"max_it":max_it,"iterations":stage_ksp.getIterationNumber(),
            "converged_reason":int(stage_ksp.getConvergedReason()),
            "reported_residual_norm":float(stage_ksp.getResidualNorm()),"residual_history":history},
        "solver_parameters":stage_parameters,
    }
    if comm.rank == 0:
        print(f"Manual/Irksome BE L2 field difference={difference:.6e}, allowed={limit:.6e}.",flush=True)
        print(f"Irksome common BE true relative residual={final_residual['relative_norm']:.6e}.",flush=True)
        print("Same method/same-mesh equivalence check passed; no accuracy-order or cost comparison claimed.",flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--comparison-config",type=Path,required=True)
    parser.add_argument("--expected-size",type=int,choices=(1,2),required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    sys.argv = [sys.argv[0]]
    comm = None
    try:
        require(args.output.resolve().parent == ROOT/"environment/records", "Keep checkpoint records in the project.")
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        import firedrake as fd
        from firedrake.petsc import PETSc
        # Irksome must register Dt before any forms are compiled or assembled.
        import irksome as irk
        from irksome.tools import AI
        config = json.loads(args.config.read_text())
        comparison = json.loads(args.comparison_config.read_text())
        require(args.config.resolve() == (ROOT/comparison["base_config"]).resolve(), "Comparison base configuration mismatch.")
        result = crosscheck(config,comparison,args.expected_size,fd,PETSc,irk,AI)
        if comm.rank == 0:
            args.output.write_text(json.dumps(result,indent=2)+"\n")
        comm.Barrier()
        return 0
    except Exception:
        traceback.print_exc()
        if comm is not None and comm.size > 1:
            comm.Abort(1)
        return 1


if __name__ == "__main__":
    sys.exit(main())

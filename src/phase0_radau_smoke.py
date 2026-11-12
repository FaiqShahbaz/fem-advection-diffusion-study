#!/usr/bin/env python3
"""Researcher-run one-step Radau IIA check against an independent block system."""

import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import sys
import traceback

from phase0_be_smoke import require
from phase0_irksome_be import true_residual

ROOT = Path(__file__).resolve().parents[1]
EXACT_TABLEAU = {"A":[["5/12","-1/12"],["3/4","1/4"]],
                 "b":["3/4","1/4"], "c":["1/3","1"]}


def tableau_coefficients(config):
    require(config["time_method"] == "RadauIIA" and config["num_stages"] == 2
            and config["stage_type"] == "deriv" and config["splitting"] == "AI"
            and config["bc_type"] == "DAE", "Keep the explained two-stage Radau/AI/DAE formulation.")
    require(config["tableau_rational"] == EXACT_TABLEAU, "Unexpected Radau tableau specification.")
    A = [[float(Fraction(value)) for value in row] for row in EXACT_TABLEAU["A"]]
    b = [float(Fraction(value)) for value in EXACT_TABLEAU["b"]]
    c = [float(Fraction(value)) for value in EXACT_TABLEAU["c"]]
    return A, b, c


def solve_one_step(config, expected_size, fd, PETSc, irk, AI):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    require(comm.size == expected_size, "Unexpected MPI process count.")
    require(config["element_family"] == "CG" and config["element_degree"] == 1
            and config["steps"] == 1 and config["dt"] > 0 and config["diffusion"] > 0,
            "This teaching checkpoint uses one positive step and P1.")
    require(config["velocity"] == [1.0,1.0] and config["mesh_diagonal"] == "left",
            "Keep the agreed velocity and mesh convention.")
    n = config["mesh_subdivisions"]
    require(isinstance(n,int) and n >= 2, "Invalid mesh subdivision count.")
    A, b, c = tableau_coefficients(config)
    mesh = fd.UnitSquareMesh(n,n,diagonal="left",comm=comm)
    V = fd.FunctionSpace(mesh,"CG",1)
    x, y = fd.SpatialCoordinate(mesh)
    beta, kappa, dt_value = fd.as_vector(config["velocity"]), config["diffusion"], config["dt"]
    dxq = fd.dx(domain=mesh,metadata={"quadrature_degree":config["quadrature_degree"]})
    dsq = fd.ds(domain=mesh,metadata={"quadrature_degree":config["quadrature_degree"]})
    bc = fd.DirichletBC(V,0.0,"on_boundary")
    sinusoid = fd.sin(fd.pi*x)*fd.sin(fd.pi*y)
    initial = fd.Function(V,name="u_initial").interpolate(sinusoid)
    bc.apply(initial)
    state = fd.Function(V,name="u_after_one_Radau_step").assign(initial)
    t, dt = fd.Constant(0.0), fd.Constant(dt_value)

    def forcing(time):
        return fd.exp(-time)*((2*kappa*fd.pi**2-1)*sinusoid
            + fd.pi*fd.cos(fd.pi*x)*fd.sin(fd.pi*y)
            + fd.pi*fd.sin(fd.pi*x)*fd.cos(fd.pi*y))

    def transport_diffusion(w,v):
        return fd.dot(beta,fd.grad(w))*v + kappa*fd.inner(fd.grad(w),fd.grad(v))

    def l2(expression,measure=dxq):
        value = float(fd.assemble(expression**2*measure))
        require(math.isfinite(value) and value >= -1e-14, "Invalid integrated squared norm.")
        return math.sqrt(max(0.0,value))

    w, v = fd.TrialFunction(V), fd.TestFunction(V)
    F = (irk.Dt(w)*v + transport_diffusion(w,v) - forcing(t)*v)*dxq
    tableau = irk.RadauIIA(2)
    coefficient_tolerance = config["checks"]["tableau_coefficient_atol"]
    require(tableau.num_stages == 2, "Expected two Radau stages.")
    require(all(abs(float(tableau.A[i,j])-A[i][j]) <= coefficient_tolerance
                for i in range(2) for j in range(2))
            and all(abs(float(tableau.b[i])-b[i]) <= coefficient_tolerance
                    and abs(float(tableau.c[i])-c[i]) <= coefficient_tolerance for i in range(2)),
            "Installed tableau differs from the independently stated rational coefficients.")
    parameters = {**config["solver_parameters"],"mat_type":"aij"}
    stepper = irk.TimeStepper(F,tableau,t,dt,state,bcs=[bc],stage_type="deriv",
        splitting=AI,bc_type="DAE",constant_jacobian=True,
        solver_parameters=parameters,options_prefix="phase0_radau")
    ksp = stepper.solver.snes.ksp
    history = []
    ksp.setMonitor(lambda solver,iteration,norm: history.append([iteration,float(norm)]))
    stepper.advance()
    require(float(t) == 0.0, "The caller must control the time update.")
    require(ksp.getConvergedReason() > 0, "Coupled Radau stage solve did not converge.")
    require(ksp.getType() == "gmres" and ksp.getPC().getType() == "jacobi"
            and ksp.getPCSide() == PETSc.PC.Side.RIGHT
            and ksp.getNormType() == PETSc.KSP.NormType.UNPRECONDITIONED,
            "Actual coupled solver differs from the baseline.")
    rtol,atol,divtol,max_it = ksp.getTolerances()
    requested = config["solver_parameters"]
    require(rtol == requested["ksp_rtol"] and atol == requested["ksp_atol"]
            and max_it == requested["ksp_max_it"] and not ksp.getInitialGuessNonzero(),
            "Unexpected coupled solver tolerances/initial-guess policy.")

    # Assemble the textbook block equations independently of Irksome's getForm.
    # B_ij = delta_ij M + dt A_ij L; g_i = F(c_i dt) - L U0.
    W = stepper.stages.function_space()
    z, q = fd.TrialFunctions(W), fd.TestFunctions(W)
    independent_bcs = [fd.DirichletBC(W.sub(i),0.0,"on_boundary") for i in range(2)]
    block_form = sum(z[i]*q[i] + dt_value*sum(A[i][j]*transport_diffusion(z[j],q[i])
                     for j in range(2)) for i in range(2))*dxq
    load_form = sum(forcing(c[i]*dt_value)*q[i] - transport_diffusion(initial,q[i])
                    for i in range(2))*dxq
    block = fd.assemble(block_form,bcs=independent_bcs,mat_type="aij")
    load = fd.assemble(load_form,bcs=independent_bcs)
    residual = true_residual(block,stepper.stages,load)
    target = max(atol,rtol*residual["rhs_norm"])
    allowance = config["checks"]["residual_tolerance_factor"]
    require(residual["norm"] <= allowance*target,
            "Irksome slopes fail the independently assembled coupled Radau equations.")
    require(V.dim() == (n+1)**2 and W.dim() == 2*V.dim()
            and block.petscmat.getSize() == (W.dim(),W.dim()), "Unexpected stage-system dimensions.")

    slopes = stepper.stages.subfunctions
    require(len(slopes) == 2, "Expected two scalar slope functions.")
    stage_values = [initial + dt_value*sum(A[i][j]*slopes[j] for j in range(2)) for i in range(2)]
    update_defect = l2(state-(initial+dt_value*sum(b[j]*slopes[j] for j in range(2))))
    stiff_defect = l2(state-stage_values[1])
    require(update_defect <= config["checks"]["update_identity_l2_max"], "Radau weighted update failed.")
    require(stiff_defect <= config["checks"]["stiff_accuracy_l2_max"], "Radau stiff-accuracy identity failed.")
    stage_records = []
    for i in range(2):
        stage_time = c[i]*dt_value
        exact = fd.exp(-stage_time)*sinusoid
        source_residual = l2(-exact + fd.dot(beta,fd.grad(exact))
                             - kappa*fd.div(fd.grad(exact)) - forcing(stage_time))
        boundary = l2(slopes[i],dsq)
        require(source_residual <= config["checks"]["source_residual_l2_max"], "Manufactured source failed at a stage time.")
        require(boundary <= config["checks"]["stage_boundary_trace_l2_max"], "A slope boundary trace is nonzero.")
        stage_records.append({"index":i+1,"time":stage_time,"slope_l2_norm":l2(slopes[i]),
            "stage_value_l2_norm":l2(stage_values[i]),"slope_boundary_trace_l2":boundary,
            "source_residual_l2":source_residual})
    error = state-fd.exp(-dt_value)*sinusoid
    metrics = {
        "domain_area":float(fd.assemble(fd.Constant(1.0)*dxq)),
        "source_residual_l2":max(record["source_residual_l2"] for record in stage_records),
        "initial_l2_error":l2(initial-sinusoid),"final_l2_error":l2(error),
        "final_h1_seminorm_error":math.sqrt(max(0.0,float(fd.assemble(fd.inner(fd.grad(error),fd.grad(error))*dxq)))),
        "solution_l2_norm":l2(state),"boundary_trace_l2":l2(state,dsq),
        "true_residual_norm":residual["norm"],"rhs_norm":residual["rhs_norm"],
        "true_relative_residual":residual["relative_norm"],
    }
    require(all(math.isfinite(value) for value in metrics.values()), "Nonfinite Radau metric.")
    require(abs(metrics["domain_area"]-1.0) <= 1e-12 and metrics["solution_l2_norm"] > 0,
            "Invalid area or zero solution.")
    require(metrics["boundary_trace_l2"] <= config["checks"]["boundary_trace_l2_max"], "Nonzero final boundary trace.")
    stats = stepper.solver_stats()
    require(stats[0] == 1, "Expected exactly one completed Radau step.")
    ksp.view()
    t.assign(dt_value)
    if comm.rank == 0:
        print(f"Radau IIA: two coupled slopes, {W.dim()} total stage coefficients, dt={dt_value}.",flush=True)
        print(f"GMRES iterations={ksp.getIterationNumber()}, independent block relative residual={residual['relative_norm']:.6e}.",flush=True)
        print(f"Update/stiff-accuracy L2 defects={update_defect:.6e}/{stiff_defect:.6e}.",flush=True)
        print(f"Final continuum L2 error={metrics['final_l2_error']:.6e}; order/cost verification remains future work.",flush=True)
    return {
        "status":"passed","ranks":comm.size,"config":config,"total_dofs":V.dim(),
        "expected_free_dofs":(n-1)**2,"expected_cells":2*n*n,"stage_total_dofs":W.dim(),
        "stage_expected_free_dofs":2*(n-1)**2,"h":math.sqrt(2)/n,
        "h_definition":"Maximum diameter of the right triangles in the stated structured mesh",
        "final_time":float(t),"method":"Irksome two-stage Radau IIA, one step, derivative stages/AI/DAE",
        "metrics":metrics,"independent_block_residual":residual,
        "update_identity_l2":update_defect,"stiff_accuracy_l2":stiff_defect,"stages":stage_records,
        "tableau":{"rational":EXACT_TABLEAU,"installed_A":tableau.A.tolist(),
                   "installed_b":tableau.b.tolist(),"installed_c":tableau.c.tolist()},
        "solver_parameters":parameters,"solver_stats":list(stats),
        "actual_solver":{"ksp_type":ksp.getType(),"pc_type":ksp.getPC().getType(),
            "preconditioning_side":"right","stopping_norm":"unpreconditioned","rtol":rtol,"atol":atol,
            "divtol":divtol,"max_it":max_it,"iterations":ksp.getIterationNumber(),
            "converged_reason":int(ksp.getConvergedReason()),"reported_residual_norm":float(ksp.getResidualNorm()),
            "residual_history":history,"true_residual_target":target,"true_residual_allowance_factor":allowance},
        "accuracy_order_or_cost_claim":False,"irksome_time_step_run":True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--expected-size",type=int,choices=(1,2),required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    sys.argv = [sys.argv[0]]
    comm = None
    try:
        require(args.output.resolve().parent == ROOT/"environment/records", "Keep records inside the project.")
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        import firedrake as fd
        from firedrake.petsc import PETSc
        # Register Irksome's Dt before any UFL forms are built or assembled.
        import irksome as irk
        from irksome.tools import AI
        config = json.loads(args.config.read_text())
        result = solve_one_step(config,args.expected_size,fd,PETSc,irk,AI)
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

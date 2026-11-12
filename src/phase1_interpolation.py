#!/usr/bin/env python3
"""Researcher-run initial P1/P2 interpolation refinement with quadrature controls."""

import argparse
import json
import math
from pathlib import Path
import sys
import traceback

from phase0_be_smoke import require

ROOT = Path(__file__).resolve().parents[1]


def cases(config, group):
    require(config["element_family"] == "CG" and config["element_degrees"] == [1,2]
            and config["mesh_subdivisions"] == [8,16,32] and config["mesh_diagonal"] == "left"
            and config["evaluation_time"] == 0.0 and config["exact_solution"] == "sin(pi*x)*sin(pi*y)"
            and config["boundary_values"] == "zero", "Keep the explained interpolation study.")
    require(config["quadrature_control_degree"] > config["quadrature_degree"] >= 4,
            "Use a higher-degree quadrature control.")
    all_cases = [(degree,n) for degree in config["element_degrees"] for n in config["mesh_subdivisions"]]
    probe = (config["mpi_probe"]["element_degree"],config["mpi_probe"]["mesh_subdivisions"])
    require(probe in all_cases, "MPI probe must match a serial case.")
    return all_cases if group == "all" else [probe]


def observed_rate(coarse, fine, metric):
    a,b = coarse["metrics"][metric],fine["metrics"][metric]
    if not (0 < b < a and coarse["h"] > fine["h"]):
        return None
    return math.log(a/b)/math.log(coarse["h"]/fine["h"])


def measure_case(config, degree, n, fd, comm):
    mesh = fd.UnitSquareMesh(n,n,diagonal="left",comm=comm)
    V = fd.FunctionSpace(mesh,"CG",degree)
    x,y = fd.SpatialCoordinate(mesh)
    exact = fd.sin(fd.pi*x)*fd.sin(fd.pi*y)
    exact_gradient = fd.as_vector((fd.pi*fd.cos(fd.pi*x)*fd.sin(fd.pi*y),
                                   fd.pi*fd.sin(fd.pi*x)*fd.cos(fd.pi*y)))
    interpolant = fd.Function(V,name=f"P{degree}_initial_interpolant").interpolate(exact)
    fd.DirichletBC(V,0.0,"on_boundary").apply(interpolant)
    error = interpolant-exact
    gradient_error = fd.grad(interpolant)-exact_gradient

    def root_norm(value):
        require(math.isfinite(value) and value >= -1e-14, "Invalid integrated squared norm.")
        return math.sqrt(max(0.0,value))

    def metrics(qdegree):
        dxq = fd.dx(domain=mesh,metadata={"quadrature_degree":qdegree})
        dsq = fd.ds(domain=mesh,metadata={"quadrature_degree":qdegree})
        return {
            "domain_area":float(fd.assemble(fd.Constant(1.0)*dxq)),
            "l2_error":root_norm(float(fd.assemble(error**2*dxq))),
            "h1_seminorm_error":root_norm(float(fd.assemble(fd.inner(gradient_error,gradient_error)*dxq))),
            "boundary_trace_l2":root_norm(float(fd.assemble(interpolant**2*dsq))),
            "exact_l2_norm":root_norm(float(fd.assemble(exact**2*dxq))),
            "exact_h1_seminorm":root_norm(float(fd.assemble(fd.inner(exact_gradient,exact_gradient)*dxq))),
            "interpolant_l2_norm":root_norm(float(fd.assemble(interpolant**2*dxq))),
        }

    primary = metrics(config["quadrature_degree"])
    control = metrics(config["quadrature_control_degree"])
    checks = config["checks"]
    total_expected = (degree*n+1)**2
    free_expected = (degree*n-1)**2
    validation = {"dimension":V.dim() == total_expected,"positive_errors":
                  primary["l2_error"] > 0 and primary["h1_seminorm_error"] > 0}
    quadrature_comparison = {}
    for label,values in (("primary",primary),("control",control)):
        validation[label+"_finite"] = all(math.isfinite(value) for value in values.values())
        validation[label+"_area"] = abs(values["domain_area"]-1.0) <= checks["domain_area_atol"]
        validation[label+"_boundary"] = values["boundary_trace_l2"] <= checks["boundary_trace_l2_max"]
        validation[label+"_exact_l2"] = math.isclose(values["exact_l2_norm"],0.5,
            rel_tol=checks["analytic_norm_rtol"],abs_tol=checks["analytic_norm_atol"])
        validation[label+"_exact_h1"] = math.isclose(values["exact_h1_seminorm"],math.pi/math.sqrt(2),
            rel_tol=checks["analytic_norm_rtol"],abs_tol=checks["analytic_norm_atol"])
    for metric in ("l2_error","h1_seminorm_error"):
        difference = abs(primary[metric]-control[metric])
        limit = checks["quadrature_comparison_atol"] + checks["quadrature_comparison_rtol"]*abs(control[metric])
        validation[metric+"_quadrature"] = difference <= limit
        quadrature_comparison[metric] = {"absolute_difference":difference,"allowed_difference":limit}
    result = {"case_id":f"P{degree}-n{n}","element_degree":degree,"mesh_subdivisions":n,
        "total_dofs":V.dim(),"expected_total_dofs":total_expected,"expected_free_dofs":free_expected,
        "expected_cells":2*n*n,"h":math.sqrt(2)/n,
        "h_definition":"Maximum diameter of the right triangles in the stated structured mesh",
        "metrics":primary,"control_metrics":control,"quadrature_comparison":quadrature_comparison,
        "checks":validation,"status":"passed" if all(validation.values()) else "failed"}
    if comm.rank == 0:
        print(f"{result['case_id']}: {V.dim()} DOFs, L2={primary['l2_error']:.9e}, H1 seminorm={primary['h1_seminorm_error']:.9e}, checks={result['status']}.",flush=True)
    return result


def study(config, group, expected_size, fd, comm):
    require(comm.size == expected_size, "Unexpected MPI process count.")
    require((group == "all" and expected_size == 1) or (group == "mpi_probe" and expected_size == 2),
            "Run the full study in serial and the selected P2 probe on two ranks.")
    measurements = [measure_case(config,degree,n,fd,comm) for degree,n in cases(config,group)]
    rates = []
    if group == "all":
        for degree in config["element_degrees"]:
            rows = [record for record in measurements if record["element_degree"] == degree]
            for coarse,fine in zip(rows,rows[1:]):
                for metric in ("l2_error","h1_seminorm_error"):
                    value = observed_rate(coarse,fine,metric)
                    expected = config["expected_orders"][str(degree)][metric]
                    passed = value is not None and abs(value-expected) <= config["checks"]["observed_order_atol"]
                    rates.append({"element_degree":degree,"metric":metric,
                        "coarse_case":coarse["case_id"],"fine_case":fine["case_id"],
                        "observed_order":value,"theoretical_expected_order":expected,
                        "allowed_order_difference":config["checks"]["observed_order_atol"],
                        "status":"passed" if passed else "failed"})
                    if comm.rank == 0:
                        print(f"P{degree} {metric}, n={coarse['mesh_subdivisions']}->{fine['mesh_subdivisions']}: observed order={value}, expected={expected}, checks={'passed' if passed else 'failed'}.",flush=True)
    passed = all(record["status"] == "passed" for record in measurements+rates)
    return {"status":"passed" if passed else "failed","ranks":comm.size,"group":group,
        "config":config,"measurements":measurements,"rates":rates,
        "interpolation_trends_checked":group == "all", "evaluation_time":0.0,
        "analytic_exact_norms":{"l2":0.5,"h1_seminorm":math.pi/math.sqrt(2)},
        "project_linear_solve_run":False,"time_integration_run":False,
        "transient_fem_convergence_verified":False,"performance_measurement":False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--group",choices=("all","mpi_probe"),required=True)
    parser.add_argument("--expected-size",type=int,choices=(1,2),required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    sys.argv = [sys.argv[0]]
    comm = None
    try:
        require(args.output.resolve().parent == ROOT/"results/raw", "Keep verification data in the project results/raw folder.")
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        import firedrake as fd
        config = json.loads(args.config.read_text())
        result = study(config,args.group,args.expected_size,fd,comm)
        if comm.rank == 0:
            args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
        comm.Barrier()
        return 0 if result["status"] == "passed" else 1
    except Exception:
        traceback.print_exc()
        if comm is not None and comm.size > 1:
            comm.Abort(1)
        return 1


if __name__ == "__main__":
    sys.exit(main())

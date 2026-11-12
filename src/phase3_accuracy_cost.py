#!/usr/bin/env python3
"""Researcher-run targeted temporal-accuracy/cost comparison; fixed mesh and Jacobi."""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys
from time import perf_counter
import traceback

from phase0_be_smoke import require
from phase1_spatial import step_count, error_metrics, run_trajectory, sanitize
from phase2_temporal import (radau_config, run_be, measured_difference, scaled_control,
                             observed_time_order, METRICS)
from phase3_cost_pilot import timed_trajectory, EVENTS

ROOT = Path(__file__).resolve().parents[1]
CASES = [dict(case_id='BE_loose',method='manual_backward_euler',dt=0.0125),
         dict(case_id='Radau_coarse',method='RadauIIA',dt=0.1),
         dict(case_id='BE_fine',method='manual_backward_euler',dt=0.0005),
         dict(case_id='Radau_fine',method='RadauIIA',dt=0.05)]


def validate_config(config):
    accepted = json.loads((ROOT/'configs/phase2_temporal.json').read_text())
    keys = ('mesh_subdivisions','element_degree','element_family','mesh_diagonal','velocity',
            'diffusion','final_time','quadrature_degree','quadrature_control_degree',
            'solver_parameters','radau_formulation','reference','solver_control','expected_temporal_orders','time_methods')
    require(all(config[key]==accepted[key] for key in keys),'Keep the accepted fixed problem, references and solver.')
    require(config['cases']==CASES and config['warmups_per_case']==1 and config['repetitions_per_case']==3,
            'Keep the explained four-case comparison.')
    ids = [case['case_id'] for case in CASES]
    require(config['measured_case_order']==[ids,list(reversed(ids)),ids]
            and config['target_time_errors_l2']==[1e-4,3e-6] and config['required_power_source']=='AC Power',
            'Keep the declared order, targets and AC condition.')
    require(all(config['checks'][key]==value for key,value in accepted['checks'].items())
            and config['checks']['equivalence_atol']==1e-12 and config['checks']['equivalence_rtol']==1e-10
            and config['checks']['time_error_transfer_fraction']==0.01,'Keep all explained accuracy controls.')
    base = sum(step_count(1,case['dt']) for case in CASES)
    references = step_count(1,config['reference']['coarse_dt'])+2*step_count(1,config['reference']['fine_dt'])
    tighter = sum(step_count(1,min(case['dt'] for case in CASES if case['method']==method)) for method in config['time_methods'])
    return 5*base+references+tighter


def error_transfer(baseline_errors, timed_record, checks):
    """Reverse triangle inequality for errors measured against the same numerical reference."""
    norms = {}
    for metric in METRICS:
        error = baseline_errors[metric]
        difference = timed_record['difference_measurement']['metrics'][metric]
        allowed = checks['field_control_atol']+checks['time_error_transfer_fraction']*error
        norms[metric] = dict(verified_baseline_time_error=error, measured_timed_baseline_distance=difference,
            lower=max(0.0,error-difference), upper=error+difference, allowed_transfer_distance=allowed,
            status='passed' if difference<=allowed else 'failed')
    return dict(status='passed' if all(row['status']=='passed' for row in norms.values()) else 'failed',norms=norms,
        interpretation='Baseline error paired with measured timed-baseline field distance; interval about finite same-mesh reference, not a direct timed-reference norm or certified exact-time error bound')


def summarize_timings(rows):
    summary = dict(repetitions=len(rows))
    for key in ('construction_seconds','first_step_seconds','remaining_steps_seconds','trajectory_wall_seconds'):
        values = [row['timing'][key] for row in rows]
        summary[key] = dict(values=values,minimum=min(values),median=statistics.median(values),maximum=max(values))
    return summary


def choose_candidates(config, curve):
    selections = []
    for target in config['target_time_errors_l2']:
        for method in config['time_methods']:
            candidates = [row for row in curve if row['method']==method
                          and row['timed_error_envelope']['l2_error']['upper']<=target]
            chosen = min(candidates,key=lambda row:row['timing_summary']['trajectory_wall_seconds']['median']) if candidates else None
            selections.append(dict(target_l2_time_error=target,method=method,
                eligible_cases=[row['case_id'] for row in candidates],selected_case=chosen['case_id'] if chosen else None,
                status='candidate available' if chosen else 'no eligible measured candidate',
                interpretation='Least measured median among the declared candidates only; no continuous optimum or total-PDE error claim'))
    return selections


def study(config,result,fd,PETSc,irk,AI,comm):
    require(comm.size==1 and validate_config(config)==15770,'Keep the serial 25-trajectory/15770-step study.')
    started = perf_counter()
    n = config['mesh_subdivisions']
    mesh = fd.UnitSquareMesh(n,n,diagonal='left',comm=comm)
    V = fd.FunctionSpace(mesh,'CG',config['element_degree'])
    require(V.dim()==1089,'Unexpected P2/n16 space.')
    x,y = fd.SpatialCoordinate(mesh)
    sinusoid = fd.sin(fd.pi*x)*fd.sin(fd.pi*y)
    gradient = fd.as_vector((fd.pi*fd.cos(fd.pi*x)*fd.sin(fd.pi*y),fd.pi*fd.sin(fd.pi*x)*fd.cos(fd.pi*y)))
    beta,kappa = fd.as_vector(config['velocity']),config['diffusion']
    bc = fd.DirichletBC(V,0.0,'on_boundary')
    initial = fd.Function(V,name='shared_cost_initial').interpolate(sinusoid)
    bc.apply(initial)
    context = dict(mesh=mesh,space=V,initial=initial,sinusoid=sinusoid,exact_gradient=gradient,beta=beta,bc=bc,
        dx=fd.dx(domain=mesh,metadata={'quadrature_degree':config['quadrature_degree']}),
        ds=fd.ds(domain=mesh,metadata={'quadrature_degree':config['quadrature_degree']}),
        spatial=lambda w,v:fd.dot(beta,fd.grad(w))*v+kappa*fd.inner(fd.grad(w),fd.grad(v)),
        forcing=lambda time:fd.exp(-time)*((2*kappa*fd.pi**2-1)*sinusoid+gradient[0]+gradient[1]))
    result.update(shared_problem_construction_seconds=perf_counter()-started,total_dofs=V.dim(),
        expected_free_dofs=961,expected_cells=512,h=math.sqrt(2)/n,h_definition='Maximum triangle diameter')
    initial_metrics = error_metrics(initial,0.0,context,config['quadrature_degree'],fd)
    accepted = json.loads((ROOT/'results/processed/interpolation_review.json').read_text())['serial_observations']['measurements']
    baseline_initial = next(row['metrics'] for row in accepted if row['case_id']=='P2-n16')
    result['initial_baseline_comparison'] = {metric:math.isclose(initial_metrics[metric],baseline_initial[metric],
        rel_tol=config['checks']['initial_baseline_rtol'],abs_tol=config['checks']['initial_baseline_atol']) for metric in METRICS}
    require(all(result['initial_baseline_comparison'].values()),'Initialization changed.')
    def checked(method,dt,factor,label,role):
        record = dict(status='prepared',method=method,dt=dt,role=role)
        result['trajectories'][label] = record
        result['active_trajectory'] = label
        if method=='manual_backward_euler':
            state = run_be(config,context,dt,factor,label,record,fd,PETSc)
        else:
            state = run_trajectory(radau_config(config),context,dt,factor,label,record,fd,PETSc,irk,AI)
        require(record['status']=='passed','Checked trajectory failed.')
        result.pop('active_trajectory',None)
        return state
    refs = config['reference']
    reference_coarse = checked('RadauIIA',refs['coarse_dt'],1.0,'reference_coarse','reference')
    reference_fine = checked('RadauIIA',refs['fine_dt'],1.0,'reference_fine','reference')
    reference_tight = checked('RadauIIA',refs['fine_dt'],config['solver_control']['tolerance_factor'],'reference_fine_tighter','reference_control')
    result['reference_gap'] = measured_difference(reference_coarse,reference_fine,config,context,fd)
    result['reference_solver_difference'] = measured_difference(reference_fine,reference_tight,config,context,fd)
    baseline_fields = {}
    previous_review = json.loads((ROOT/'results/processed/temporal_review.json').read_text())
    for case in config['cases']:
        case_id,method,dt = case['case_id'],case['method'],case['dt']
        field = checked(method,dt,1.0,case_id+'_verified','verified_baseline')
        baseline_fields[case_id] = field
        difference = measured_difference(field,reference_fine,config,context,fd)
        row = dict(**case,trajectory=case_id+'_verified',time_errors=difference['metrics'],difference_measurement=difference,
                   status='passed' if difference['status']=='passed' and all(v>0 for v in difference['metrics'].values()) else 'failed')
        previous = next((r for r in previous_review['curve'] if r['method']==method and r['dt']==dt),None)
        row['previous_temporal_error_comparison'] = None if previous is None else {
            metric:math.isclose(row['time_errors'][metric],previous['time_errors'][metric],
                rel_tol=config['checks']['initial_baseline_rtol'],abs_tol=config['checks']['initial_baseline_atol']) for metric in METRICS}
        require(previous is None or all(row['previous_temporal_error_comparison'].values()),'Previously measured time errors changed.')
        result['curve'].append(row)
        print(f'{case_id}: checked baseline time errors {row["time_errors"]}',flush=True)
    checks = config['checks']
    minimum_error = {metric:min(row['time_errors'][metric] for row in result['curve']) for metric in METRICS}
    norms = {}
    for metric in METRICS:
        gap,change = result['reference_gap']['metrics'][metric],result['reference_solver_difference']['metrics'][metric]
        allowed = checks['field_control_atol']+checks['reference_fraction']*minimum_error[metric]
        norms[metric] = dict(refinement_gap=gap,tolerance_change=change,combined_sensitivity=gap+change,
            allowed_combined_sensitivity=allowed,status='passed' if gap+change<=allowed else 'failed')
    result['reference_resolution'] = dict(norms=norms,status='passed' if all(r['status']=='passed' for r in norms.values()) else 'failed',
        interpretation='Measured reference sensitivity, not a certified absolute reference error bound')
    for method in config['time_methods']:
        cases = sorted([case for case in config['cases'] if case['method']==method],key=lambda c:c['dt'],reverse=True)
        finest = cases[-1]
        tight = checked(method,finest['dt'],config['solver_control']['tolerance_factor'],method+'_finest_tighter','solver_control')
        difference = measured_difference(tight,baseline_fields[finest['case_id']],config,context,fd)
        scale = next(row['time_errors'] for row in result['curve'] if row['case_id']==finest['case_id'])
        result['curve_solver_controls'][method] = scaled_control(difference,scale,checks['solver_control_fraction'],checks['field_control_atol'])
        coarse,fine = [next(row for row in result['curve'] if row['case_id']==case['case_id']) for case in cases]
        for metric in METRICS:
            order = observed_time_order(coarse,fine,metric)
            expected = config['expected_temporal_orders'][method]
            result['rates'].append(dict(method=method,metric=metric,coarse_dt=coarse['dt'],fine_dt=fine['dt'],
                observed_order=order,status='passed' if order is not None and abs(order-expected)<=checks['observed_order_atol'] else 'failed'))
    require(all(result[key]['status']=='passed' for key in ('reference_gap','reference_solver_difference','reference_resolution'))
            and all(r['status']=='passed' for r in result['curve']+result['rates'])
            and all(r['status']=='passed' for r in result['curve_solver_controls'].values()),
            'Accuracy/reference/solver control failed before timing.')
    PETSc.Log.begin()
    stage = PETSc.Log.Stage('phase3_accuracy_cost_core')
    events = {name:PETSc.Log.Event(name) for name in EVENTS}
    cases_by_id = {case['case_id']:case for case in config['cases']}
    def timed(case_id,role,repetition):
        case = cases_by_id[case_id]
        label = f'{case_id}_{role}{repetition}'
        record = dict(status='prepared',case_id=case_id,method=case['method'],role=role,repetition=repetition)
        result['trajectories'][label] = record
        result['active_trajectory'] = label
        result['execution_order'].append(label)
        # Reuse the accepted pilot function unchanged, including all timer boundaries.
        timed_trajectory({**config,'dt':case['dt']},context,case['method'],label,record,baseline_fields[case_id],stage,events,fd,PETSc,irk,AI)
        scale = next(row['time_errors'] for row in result['curve'] if row['case_id']==case_id)
        record['time_error_transfer'] = error_transfer(scale,record,checks)
        require(record['time_error_transfer']['status']=='passed','Timed-baseline discrepancy is too large relative to time error.')
        result.pop('active_trajectory',None)
    for case in config['cases']:
        timed(case['case_id'],'warmup',0)
    for repetition,order in enumerate(config['measured_case_order'],start=1):
        for case_id in order:
            timed(case_id,'measured',repetition)
    for row in result['curve']:
        repeats = [result['trajectories'][f'{row["case_id"]}_measured{i}'] for i in (1,2,3)]
        row['timing_summary'] = summarize_timings(repeats)
        row['measured_trajectories'] = [f'{row["case_id"]}_measured{i}' for i in (1,2,3)]
        row['timed_error_envelope'] = {metric:dict(lower=min(r['time_error_transfer']['norms'][metric]['lower'] for r in repeats),
            upper=max(r['time_error_transfer']['norms'][metric]['upper'] for r in repeats)) for metric in METRICS}
    result['candidate_selections'] = choose_candidates(config,result['curve'])
    result.update(status='passed',completed_study=True,completed_trajectories=len(result['trajectories']),
                  completed_steps=sum(row['completed_steps'] for row in result['trajectories'].values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    sys.argv = [sys.argv[0]]
    result = dict(status='failed',completed_study=False,ranks=1,trajectories={},curve=[],rates=[],
        curve_solver_controls={},execution_order=[],candidate_selections=[],performance_measurement=True,
        error_definition='Temporal error about the same-mesh finite Radau reference; continuum errors are diagnostics',
        timing_scope='Accepted pilot construction/step boundaries; warm cache, fresh operators/solvers, one thread/process',
        limitations='Four declared candidates; fixed Galerkin system; not total-PDE accuracy, an optimum or broad robustness')
    allowed = args.output.resolve().parent==ROOT/'results/raw'
    try:
        require(allowed,'Keep raw records in the Projects repository.')
        config = json.loads(args.config.read_text()); result['config'] = config
        from mpi4py import MPI
        require(MPI.COMM_WORLD.size==1,'This comparison is serial.')
        import firedrake as fd
        from firedrake.petsc import PETSc
        import irksome as irk
        from irksome.tools import AI
        study(config,result,fd,PETSc,irk,AI,MPI.COMM_WORLD)
    except (Exception,KeyboardInterrupt) as error:
        result.update(status='failed',error=str(error) or 'Interrupted by researcher')
        label = result.get('active_trajectory')
        if label: result['trajectories'][label]['status'] = 'failed'
        traceback.print_exc()
    finally:
        if allowed: args.output.write_text(json.dumps(sanitize(result),indent=2,allow_nan=False)+'\n')
    return 0 if result['status']=='passed' else 1


if __name__=='__main__':
    sys.exit(main())

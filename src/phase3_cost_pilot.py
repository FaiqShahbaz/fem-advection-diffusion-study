#!/usr/bin/env python3
"""Researcher-run timing/equivalence pilot; one fixed problem, no cost ranking."""
import argparse
import gc
import json
import math
from pathlib import Path
import statistics
import sys
from time import perf_counter
import traceback

from phase0_be_smoke import require
from phase0_irksome_be import true_residual
from phase0_radau_smoke import tableau_coefficients
from phase1_spatial import (step_count, safe_sqrt, norm_pair, error_metrics, check_metrics,
                            run_trajectory, capture_failed_solve, sanitize)
from phase2_temporal import radau_config, run_be, measured_difference, METRICS

ROOT = Path(__file__).resolve().parents[1]
METHODS = ['manual_backward_euler', 'RadauIIA']
EVENTS = ('KSPSolve', 'KSPSetUp', 'PCSetUp', 'PCApply')


def validate_config(config):
    accepted = json.loads((ROOT/'configs/phase2_temporal.json').read_text())
    common = ('mesh_subdivisions', 'element_degree', 'element_family', 'mesh_diagonal',
              'velocity', 'diffusion', 'final_time', 'quadrature_degree',
              'quadrature_control_degree', 'solver_parameters', 'radau_formulation')
    require(all(config[key] == accepted[key] for key in common),
            'Keep the accepted fixed P2/n16 problem, solver and stage formulation.')
    require(config['time_methods'] == METHODS and config['dt'] == 0.05
            and config['warmups_per_method'] == 1 and config['repetitions_per_method'] == 3,
            'Keep the bounded one-step-size timing pilot.')
    require(config['measured_order'] == [METHODS, list(reversed(METHODS)), METHODS],
            'Keep the explained alternating method order.')
    require(all(config['checks'][key] == value for key, value in accepted['checks'].items()),
            'Keep accepted numerical checks.')
    require(config['checks']['equivalence_atol'] == 1e-12
            and config['checks']['equivalence_rtol'] == 1e-10,
            'Keep the declared actual-field equivalence budget.')
    tableau_coefficients(radau_config(config))
    return 10*step_count(config['final_time'], config['dt'])


def event_snapshot(events, stage):
    snapshots = {name:event.getPerfInfo(stage.id) for name,event in events.items()}
    return {name:{key:info[key] for key in ('count', 'time', 'flops')}
            for name,info in snapshots.items()}


def event_difference(before, after):
    result = {name: {key: after[name][key]-before[name][key]
                    for key in ('count', 'time', 'flops')} for name in EVENTS}
    require(all(math.isfinite(value) and value >= 0 for row in result.values() for value in row.values()),
            'Invalid PETSc profiling delta; retain records for review.')
    return result


def check_actual_solver(ksp, parameters, PETSc):
    rtol, atol, divtol, max_it = ksp.getTolerances()
    require(ksp.getType() == 'gmres' and ksp.getPC().getType() == 'jacobi'
            and ksp.getPCSide() == PETSc.PC.Side.RIGHT
            and ksp.getNormType() == PETSc.KSP.NormType.UNPRECONDITIONED
            and not ksp.getInitialGuessNonzero()
            and rtol == parameters['ksp_rtol'] and atol == parameters['ksp_atol']
            and max_it == parameters['ksp_max_it'], 'Actual solver changed.')
    ksp.view()  # Actual restart is recorded here; this petsc4py has no restart getter.
    return dict(ksp_type=ksp.getType(), pc_type=ksp.getPC().getType(), side='right',
                norm='unpreconditioned', rtol=rtol, atol=atol, divtol=divtol, max_it=max_it,
                requested_restart=parameters['ksp_gmres_restart'],
                actual_restart_evidence='KSP view in transcript')


def timed_trajectory(config, context, method, label, record, baseline, stage, events, fd, PETSc, irk, AI):
    """Time construction and stepping; perform independent validation afterward."""
    V, initial, dxq, dsq, bc = (context[k] for k in ('space', 'initial', 'dx', 'ds', 'bc'))
    spatial, forcing = context['spatial'], context['forcing']
    dt_value = config['dt']
    count = step_count(config['final_time'], dt_value)
    A, b, c = tableau_coefficients(radau_config(config))
    checks = config['checks']
    record.update(status='running', method=method, dt=dt_value, planned_steps=count,
                  completed_steps=0, step_records=[],
                  reuse_policy='Fresh solver/operator per trajectory; constant operator within trajectory',
                  initial_guess='Physical solution or derivative stages zeroed each step; KSP zero guess')
    print(f'{label}: {method}, dt={dt_value}, {count} steps', flush=True)
    # Collect old solver cycles before starting the next wall-clock interval.
    gc.collect()
    with stage:
        before = event_snapshot(events, stage)
        started = perf_counter()
        state = fd.Function(V, name=label+'_solution').assign(initial)
        old = fd.Function(V, name=label+'_old').assign(initial)
        t = fd.Constant(0.0)
        w, v = fd.TrialFunction(V), fd.TestFunction(V)
        parameters = dict(config['solver_parameters'])
        if method == 'manual_backward_euler':
            B = fd.assemble((w*v+dt_value*spatial(w, v))*dxq, bcs=[bc], mat_type='aij')
            endpoint = fd.Constant(dt_value)
            rhs_form = (old*v+dt_value*forcing(endpoint)*v)*dxq
            solver = fd.LinearSolver(B, solver_parameters=parameters, options_prefix='phase3_'+label)
            ksp = solver.ksp
        else:
            parameters.update(snes_type='ksponly', mat_type='aij')
            F = (irk.Dt(w)*v+spatial(w, v)-forcing(t)*v)*dxq
            stepper = irk.TimeStepper(F, irk.RadauIIA(2), t, fd.Constant(dt_value), state,
                bcs=[bc], stage_type='deriv', splitting=AI, bc_type='DAE', constant_jacobian=True,
                solver_parameters=parameters, options_prefix='phase3_'+label)
            ksp = stepper.solver.snes.ksp
            slopes = stepper.stages.subfunctions
        construction_seconds = perf_counter()-started
        for index in range(count):
            step_started = perf_counter()
            old.assign(state)
            try:
                old_time = float(t)
                require(math.isclose(old_time,index*dt_value,rel_tol=0,abs_tol=1e-14),'Incorrect pre-step clock.')
                if method == 'manual_backward_euler':
                    state.assign(0.0)
                    endpoint.assign((index+1)*dt_value)
                    rhs = fd.assemble(rhs_form, bcs=[bc])
                    solver.solve(state, rhs)
                    source_times = [float(endpoint)]
                else:
                    for slope in slopes:
                        slope.assign(0.0)
                    stepper.advance()
                    require(float(t)==old_time,'Irksome changed the caller-owned clock.')
                    source_times = [old_time+ci*dt_value for ci in c]
                # Small counters/checks are included; no Krylov Python monitor or independent assembly.
                reason = int(ksp.getConvergedReason())
                iterations = ksp.getIterationNumber()
                require(reason > 0 and 0 < iterations <= parameters['ksp_max_it'],
                        'A timed solve did not converge.')
                t.assign((index+1)*dt_value)
                row = dict(step=index+1, t_start=index*dt_value, t_end=float(t),
                           converged_reason=reason, iterations=iterations, source_times=source_times,
                           reported_residual_norm=float(ksp.getResidualNorm()),
                           wall_seconds=perf_counter()-step_started)
                record['step_records'].append(row)
                record['completed_steps'] = index+1
            except Exception as error:
                capture_failed_solve(record, index, index*dt_value, dt_value, ksp, [], error)
                raise
        trajectory_seconds = perf_counter()-started
        after = event_snapshot(events, stage)
    # Everything below is outside the timed construction/step interval.
    profile = event_difference(before, after)
    record['timing'] = dict(construction_seconds=construction_seconds,
        first_step_seconds=record['step_records'][0]['wall_seconds'],
        remaining_steps_seconds=sum(row['wall_seconds'] for row in record['step_records'][1:]),
        trajectory_wall_seconds=trajectory_seconds,
        loop_bookkeeping_seconds=trajectory_seconds-construction_seconds
            -sum(row['wall_seconds'] for row in record['step_records']),
        petsc_event_deltas=profile,
        petsc_event_interpretation='Inclusive nested event times; do not add them or add to wall time')
    require(all(math.isfinite(value) and value >= 0 for key, value in record['timing'].items()
                if key.endswith('_seconds')), 'Invalid timing partition.')
    require(profile['KSPSolve']['count'] == count and profile['PCSetUp']['count'] >= 1
            and profile['PCApply']['count'] > 0, 'Missing solve/setup/application profiling evidence.')
    record['actual_solver'] = check_actual_solver(ksp, parameters, PETSc)
    require(math.isclose(float(t), config['final_time'], rel_tol=0, abs_tol=1e-14), 'Incorrect final time.')
    if method == 'manual_backward_euler':
        require(B.petscmat.getSize() == (V.dim(), V.dim()), 'Unexpected BE system dimension.')
        residual = true_residual(B, state, rhs)
        final_check = dict(equation='(M+dt L) U_new = M U_old + dt F(T)', true_residual=residual,
                           source_times=[float(endpoint)])
        record['system_dofs'] = V.dim()
    else:
        W = stepper.stages.function_space()
        z, q = fd.TrialFunctions(W), fd.TestFunctions(W)
        independent_bcs = [fd.DirichletBC(W.sub(i), 0.0, 'on_boundary') for i in range(2)]
        block_form = sum(z[i]*q[i]+dt_value*sum(A[i][j]*spatial(z[j], q[i]) for j in range(2))
                         for i in range(2))*dxq
        # Caller clock is already T: reconstruct the FINAL step's old time explicitly.
        last_start = (count-1)*dt_value
        load_form = sum(forcing(last_start+c[i]*dt_value)*q[i]-spatial(old, q[i]) for i in range(2))*dxq
        block = fd.assemble(block_form, bcs=independent_bcs, mat_type='aij')
        load = fd.assemble(load_form, bcs=independent_bcs)
        require(W.dim() == 2*V.dim() and block.petscmat.getSize() == (W.dim(), W.dim()),
                'Unexpected stage system dimension.')
        residual = true_residual(block, stepper.stages, load)
        l2 = lambda expression, measure=dxq: safe_sqrt(float(fd.assemble(expression**2*measure)))
        final_check = dict(equation='(delta_ij M+dt A_ij L) Z_j = F(t_old+c_i dt)-L U_old',
            true_residual=residual, source_times=[last_start+ci*dt_value for ci in c],
            update_identity_l2=l2(state-old-dt_value*sum(b[j]*slopes[j] for j in range(2))),
            stiff_accuracy_l2=l2(state-old-dt_value*sum(A[1][j]*slopes[j] for j in range(2))),
            slope_boundary_trace_l2=[l2(slope, dsq) for slope in slopes])
        require(final_check['update_identity_l2'] <= checks['update_identity_l2_max']
                and final_check['stiff_accuracy_l2'] <= checks['stiff_accuracy_l2_max']
                and all(value <= checks['stage_boundary_trace_l2_max'] for value in final_check['slope_boundary_trace_l2']),
                'Final Radau update or boundary identity failed.')
        stats = list(stepper.solver_stats())
        require(stats[0] == count and stats[2] == sum(row['iterations'] for row in record['step_records']),
                'Irksome iteration accounting disagrees.')
        record.update(system_dofs=W.dim(), solver_stats=stats)
    target = max(parameters['ksp_atol'], parameters['ksp_rtol']*residual['rhs_norm'])
    require(all(math.isfinite(value) for value in residual.values())
            and residual['norm'] <= checks['residual_tolerance_factor']*target,
            'Independent final-step equation residual failed.')
    final_check.update(nominal_target=target, allowance_factor=checks['residual_tolerance_factor'], status='passed')
    primary = error_metrics(state, config['final_time'], context, config['quadrature_degree'], fd)
    control = error_metrics(state, config['final_time'], context, config['quadrature_control_degree'], fd)
    validation, quadrature = check_metrics(primary, control, config['final_time'], checks)
    difference = measured_difference(state, baseline, config, context, fd)
    baseline_norms = norm_pair(baseline, fd,
        fd.dx(domain=context['mesh'], metadata={'quadrature_degree':config['quadrature_control_degree']}))
    equivalence = {metric:dict(actual_field_difference=difference['metrics'][metric],
        baseline_field_norm=baseline_norms[metric],
        allowed_difference=checks['equivalence_atol']+checks['equivalence_rtol']*baseline_norms[metric])
        for metric in METRICS}
    validation['actual_field_equivalence'] = difference['status'] == 'passed' and all(
        row['actual_field_difference'] <= row['allowed_difference'] for row in equivalence.values())
    record.update(status='passed' if all(validation.values()) else 'failed', checks=validation,
        final_time=float(t), final_equation_check=final_check, equivalence=equivalence,
        difference_measurement=difference, metrics=primary, control_metrics=control,
        quadrature_comparison=quadrature, total_iterations=sum(row['iterations'] for row in record['step_records']),
        maximum_step_iterations=max(row['iterations'] for row in record['step_records']))
    print(f'{label}: field checks {record["status"]}; core wall={trajectory_seconds:.6f} s; '
          f'PC setups={profile["PCSetUp"]["count"]}', flush=True)
    require(record['status'] == 'passed', 'Timed implementation differs from verified baseline.')


def study(config, result, fd, PETSc, irk, AI, comm):
    require(comm.size == 1 and validate_config(config) == 200, 'Serial, fixed 200-step pilot required.')
    started = perf_counter()
    n = config['mesh_subdivisions']
    mesh = fd.UnitSquareMesh(n, n, diagonal='left', comm=comm)
    V = fd.FunctionSpace(mesh, 'CG', config['element_degree'])
    require(V.dim() == 1089, 'Unexpected P2/n16 dimension.')
    x, y = fd.SpatialCoordinate(mesh)
    sinusoid = fd.sin(fd.pi*x)*fd.sin(fd.pi*y)
    gradient = fd.as_vector((fd.pi*fd.cos(fd.pi*x)*fd.sin(fd.pi*y), fd.pi*fd.sin(fd.pi*x)*fd.cos(fd.pi*y)))
    beta, kappa = fd.as_vector(config['velocity']), config['diffusion']
    bc = fd.DirichletBC(V, 0.0, 'on_boundary')
    initial = fd.Function(V, name='shared_pilot_initial').interpolate(sinusoid)
    bc.apply(initial)
    context = dict(mesh=mesh, space=V, initial=initial, sinusoid=sinusoid, exact_gradient=gradient,
        beta=beta, bc=bc, dx=fd.dx(domain=mesh, metadata={'quadrature_degree':config['quadrature_degree']}),
        ds=fd.ds(domain=mesh, metadata={'quadrature_degree':config['quadrature_degree']}),
        spatial=lambda w,v:fd.dot(beta, fd.grad(w))*v+kappa*fd.inner(fd.grad(w), fd.grad(v)),
        forcing=lambda time:fd.exp(-time)*((2*kappa*fd.pi**2-1)*sinusoid+gradient[0]+gradient[1]))
    result.update(shared_problem_construction_seconds=perf_counter()-started, total_dofs=V.dim(),
                  expected_free_dofs=961, expected_cells=512, h=math.sqrt(2)/n,
                  h_definition='Maximum triangle diameter')
    initial_metrics = error_metrics(initial, 0.0, context, config['quadrature_degree'], fd)
    accepted_initial = json.loads((ROOT/'results/processed/interpolation_review.json').read_text())['serial_observations']['measurements']
    initial_baseline = next(row['metrics'] for row in accepted_initial if row['case_id'] == 'P2-n16')
    result['initial_baseline_comparison'] = {metric:math.isclose(initial_metrics[metric], initial_baseline[metric],
        rel_tol=config['checks']['initial_baseline_rtol'], abs_tol=config['checks']['initial_baseline_atol']) for metric in METRICS}
    require(all(result['initial_baseline_comparison'].values()), 'Initialization changed.')
    accepted = json.loads((ROOT/'results/processed/temporal_review.json').read_text())
    baselines = {}
    for method in METHODS:
        label = method+'_verified'
        record = dict(status='prepared', method=method, role='verified_baseline')
        result['trajectories'][label] = record
        result['active_trajectory'] = label
        if method == METHODS[0]:
            baselines[method] = run_be(config, context, config['dt'], 1.0, label, record, fd, PETSc)
        else:
            baselines[method] = run_trajectory(radau_config(config), context, config['dt'], 1.0, label, record, fd, PETSc, irk, AI)
        previous = accepted['trajectory_summaries'][method+'_dt1']['control_metrics']
        record['accepted_temporal_metric_comparison'] = {metric:math.isclose(record['control_metrics'][metric], previous[metric],
            rel_tol=config['checks']['initial_baseline_rtol'], abs_tol=config['checks']['initial_baseline_atol']) for metric in METRICS}
        require(record['status'] == 'passed' and all(record['accepted_temporal_metric_comparison'].values()),
                'Replayed verified baseline changed; inspect before timing.')
    PETSc.Log.begin()
    events = {name:PETSc.Log.Event(name) for name in EVENTS}
    stage = PETSc.Log.Stage('phase3_pilot_core')
    def execute(method, role, repetition):
        label = f'{method}_{role}{repetition}'
        record = dict(status='prepared', method=method, role=role, repetition=repetition)
        result['trajectories'][label] = record
        result['active_trajectory'] = label
        result['execution_order'].append(label)
        timed_trajectory(config, context, method, label, record, baselines[method], stage, events, fd, PETSc, irk, AI)
    for method in METHODS:
        execute(method, 'warmup', 0)
    for repetition, methods in enumerate(config['measured_order'], start=1):
        for method in methods:
            execute(method, 'measured', repetition)
    result.pop('active_trajectory', None)
    for method in METHODS:
        rows = [row for row in result['trajectories'].values() if row['method'] == method and row['role'] == 'measured']
        summary = {'repetitions':len(rows)}
        for key in ('construction_seconds', 'first_step_seconds', 'remaining_steps_seconds', 'trajectory_wall_seconds'):
            values = [row['timing'][key] for row in rows]
            summary[key] = dict(values=values, minimum=min(values), median=statistics.median(values), maximum=max(values))
        result['timing_summary'][method] = summary
    result.update(status='passed', completed_study=True, completed_trajectories=len(result['trajectories']),
                  completed_steps=sum(row['completed_steps'] for row in result['trajectories'].values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.argv = [sys.argv[0]]
    result = dict(status='failed', completed_study=False, ranks=1, trajectories={}, execution_order=[],
        timing_summary={}, performance_measurement=True, measurement_kind='Timing-instrumentation/equivalence pilot',
        timing_scope='Warm code/cache; fresh operators and solvers per trajectory; construction plus all steps; '
            'excludes imports/shared mesh/verification/printing/I-O/destruction; includes profiling and small step counters',
        limitations='One same-dt tiny serial case; no accuracy-cost ranking, publication speedup or cold-start application timing')
    allowed = args.output.resolve().parent == ROOT/'results/raw'
    try:
        require(allowed, 'Keep all records in the Projects repository.')
        config = json.loads(args.config.read_text())
        result['config'] = config
        from mpi4py import MPI
        require(MPI.COMM_WORLD.size == 1, 'This pilot is serial.')
        import firedrake as fd
        from firedrake.petsc import PETSc
        import irksome as irk
        from irksome.tools import AI
        study(config, result, fd, PETSc, irk, AI, MPI.COMM_WORLD)
    except (Exception, KeyboardInterrupt) as error:
        result.update(status='failed', error=str(error) or 'Interrupted by researcher')
        label = result.get('active_trajectory')
        if label:
            result['trajectories'][label]['status'] = 'failed'
        traceback.print_exc()
    finally:
        if allowed:
            args.output.write_text(json.dumps(sanitize(result), indent=2, allow_nan=False)+'\n')
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    sys.exit(main())

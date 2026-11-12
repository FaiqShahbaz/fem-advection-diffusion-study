#!/usr/bin/env python3
"""Researcher-run transient P1/P2 spatial verification with explicit error controls."""
import argparse
import json
import math
from pathlib import Path
import sys
import traceback

from phase0_be_smoke import require
from phase0_irksome_be import true_residual
from phase0_radau_smoke import tableau_coefficients
from phase1_interpolation import observed_rate

ROOT = Path(__file__).resolve().parents[1]


def step_count(final_time, dt):
    require(final_time > 0 and dt > 0, 'Use positive final time and step size.')
    count = round(final_time/dt)
    require(count > 0 and math.isclose(count*dt,final_time,rel_tol=0.0,abs_tol=1e-14),
            'The fixed time step must divide the final time exactly to roundoff.')
    return count


def validate_config(config):
    require(config['element_family'] == 'CG' and config['element_degrees'] == [1,2]
            and config['mesh_subdivisions'] == [8,16,32] and config['mesh_diagonal'] == 'left',
            'Keep the explained P1/P2 three-mesh study.')
    require(config['velocity'] == [1.0,1.0] and config['diffusion'] == 1.0
            and config['final_time'] == 1.0, 'Keep the agreed verification PDE and T=1.')
    require(config['dt'] == 2*config['control_dt'] and config['quadrature_control_degree'] > config['quadrature_degree'] >= 4,
            'Use halved time steps and higher-degree quadrature controls.')
    step_count(config['final_time'],config['dt'])
    step_count(config['final_time'],config['control_dt'])
    require(config['solver_control'] == {'element_degree':2,'mesh_subdivisions':32,'tolerance_factor':0.1},
            'Use the explained ten-times-tighter finest-P2 solver control.')
    return tableau_coefficients(config)


def safe_sqrt(value):
    require(math.isfinite(value) and value >= -1e-14, 'Invalid integrated squared norm.')
    return math.sqrt(max(0.0,value))


def norm_pair(expression, fd, dxq):
    return {'l2_error':safe_sqrt(float(fd.assemble(expression**2*dxq))),
            'h1_seminorm_error':safe_sqrt(float(fd.assemble(fd.inner(fd.grad(expression),fd.grad(expression))*dxq)))}


def error_metrics(state, time, context, qdegree, fd):
    mesh, sinusoid, gradient = context['mesh'],context['sinusoid'],context['exact_gradient']
    exact = fd.exp(-time)*sinusoid
    exact_gradient = fd.exp(-time)*gradient
    dxq = fd.dx(domain=mesh,metadata={'quadrature_degree':qdegree})
    dsq = fd.ds(domain=mesh,metadata={'quadrature_degree':qdegree})
    squared = lambda expression: safe_sqrt(float(fd.assemble(expression**2*dxq)))
    gradient_error = fd.grad(state)-exact_gradient
    return {'domain_area':float(fd.assemble(fd.Constant(1.0)*dxq)),
        'l2_error':squared(state-exact),
        'h1_seminorm_error':safe_sqrt(float(fd.assemble(fd.inner(gradient_error,gradient_error)*dxq))),
        'solution_l2_norm':squared(state),'exact_l2_norm':squared(exact),
        'exact_h1_seminorm':safe_sqrt(float(fd.assemble(fd.inner(exact_gradient,exact_gradient)*dxq))),
        'boundary_trace_l2':safe_sqrt(float(fd.assemble(state**2*dsq)))}


def check_metrics(primary,control,time,checks):
    validation, quadrature = {},{}
    for label,values in (('primary',primary),('control',control)):
        validation[label+'_finite_positive'] = all(math.isfinite(x) for x in values.values()) and all(values[k] > 0 for k in ('l2_error','h1_seminorm_error','solution_l2_norm'))
        validation[label+'_area'] = abs(values['domain_area']-1.0) <= checks['domain_area_atol']
        validation[label+'_boundary'] = values['boundary_trace_l2'] <= checks['boundary_trace_l2_max']
        for metric,expected in (('exact_l2_norm',.5*math.exp(-time)),('exact_h1_seminorm',math.pi/math.sqrt(2)*math.exp(-time))):
            validation[label+'_'+metric] = math.isclose(values[metric],expected,
                rel_tol=checks['analytic_norm_rtol'],abs_tol=checks['analytic_norm_atol'])
    for metric in ('l2_error','h1_seminorm_error'):
        difference = abs(primary[metric]-control[metric])
        allowed = checks['quadrature_comparison_atol'] + checks['quadrature_comparison_rtol']*control[metric]
        validation[metric+'_quadrature'] = difference <= allowed
        quadrature[metric] = {'absolute_difference':difference,'allowed_difference':allowed}
    return validation,quadrature


def capture_failed_solve(record, index, old_time, dt_value, ksp, history, error):
    """Retain diagnostics for an uncommitted step; never treat its iterate as a result."""
    diagnostic = {'step':index+1,'t_start':old_time,'dt':dt_value,'error':str(error),
                  'reported_residual_history':list(history),'step_committed':False}
    record['status'] = 'failed'
    record['failed_step'] = diagnostic
    try:
        rtol,atol,divtol,max_it = ksp.getTolerances()
        diagnostic.update(iterations=ksp.getIterationNumber(),converged_reason=int(ksp.getConvergedReason()),
            reported_residual_norm=float(ksp.getResidualNorm()),rtol=rtol,atol=atol,divtol=divtol,max_it=max_it)
        print(f"  FAILED step {index+1}: GMRES iterations={diagnostic['iterations']}, reason={diagnostic['converged_reason']}, reported residual={diagnostic['reported_residual_norm']:.6e}",flush=True)
        # PETSc's current KSP system is the SNES linear correction equation.
        # Firedrake has not copied the failed correction into the physical slopes.
        vector = ksp.buildResidual()
        try:
            value,rhs_norm = float(vector.norm()),float(ksp.getRhs().norm())
        finally:
            vector.destroy()
        diagnostic['petsc_correction_residual'] = {'norm':value,'rhs_norm':rhs_norm,
            'relative_norm':value/rhs_norm if rhs_norm else None,
            'nominal_target':max(atol,rtol*rhs_norm),
            'meaning':'Recomputed using the PETSc KSP correction operator; not the independent physical-stage check of a completed step'}
        ksp.view()
    except Exception as diagnostic_error:
        diagnostic['diagnostic_error'] = str(diagnostic_error)


def run_trajectory(config,context,dt_value,tolerance_factor,label,record,fd,PETSc,irk,AI):
    V, initial, dxq, dsq = (context[k] for k in ('space','initial','dx','ds'))
    A,b,c = tableau_coefficients(config)
    checks = config['checks']
    count = step_count(config['final_time'],dt_value)
    state = fd.Function(V,name=label+'_solution').assign(initial)
    old = fd.Function(V,name=label+'_previous_solution').assign(initial)
    t,dt = fd.Constant(0.0),fd.Constant(dt_value)
    parameters = {**config['solver_parameters'],'mat_type':'aij'}
    parameters['ksp_rtol'] *= tolerance_factor
    parameters['ksp_atol'] *= tolerance_factor
    w,v = fd.TrialFunction(V),fd.TestFunction(V)
    spatial,forcing = context['spatial'],context['forcing']
    F = (irk.Dt(w)*v+spatial(w,v)-forcing(t)*v)*dxq
    tableau = irk.RadauIIA(2)
    tolerance = checks['tableau_coefficient_atol']
    require(all(abs(float(tableau.A[i,j])-A[i][j])<=tolerance for i in range(2) for j in range(2))
            and all(abs(float(tableau.b[i])-b[i])<=tolerance and abs(float(tableau.c[i])-c[i])<=tolerance for i in range(2)),
            'Installed tableau differs from the independently specified coefficients.')
    stepper = irk.TimeStepper(F,tableau,t,dt,state,bcs=[context['bc']],stage_type='deriv',
        splitting=AI,bc_type='DAE',constant_jacobian=True,solver_parameters=parameters,
        options_prefix='phase1_spatial_'+label)
    ksp = stepper.solver.snes.ksp
    reported_history = []
    ksp.setMonitor(lambda solver,iteration,norm: reported_history.append([iteration,float(norm)]))
    W = stepper.stages.function_space()
    z,q = fd.TrialFunctions(W),fd.TestFunctions(W)
    independent_bcs = [fd.DirichletBC(W.sub(i),0.0,'on_boundary') for i in range(2)]
    # Fixed matrix; old and t in the load retain the pre-step values during validation.
    block_form = sum(z[i]*q[i]+dt_value*sum(A[i][j]*spatial(z[j],q[i]) for j in range(2)) for i in range(2))*dxq
    load_form = sum(forcing(t+c[i]*dt_value)*q[i]-spatial(old,q[i]) for i in range(2))*dxq
    block = fd.assemble(block_form,bcs=independent_bcs,mat_type='aij')
    require(W.dim()==2*V.dim() and block.petscmat.getSize()==(W.dim(),W.dim()),'Unexpected coupled-system dimensions.')
    record.update(status='running',dt=dt_value,planned_steps=count,completed_steps=0,
        solver_parameters=parameters,stage_total_dofs=W.dim(),step_records=[],
        physical_stage_initial_guess='zeroed before each step',ksp_initial_guess_nonzero=False,
        reuse_policy='Fresh stepper per trajectory; constant stage Jacobian within each fixed-dt trajectory')
    print(f"  {label}: dt={dt_value}, steps={count}, rtol={parameters['ksp_rtol']:.1e}",flush=True)
    l2 = lambda expression,measure=dxq: safe_sqrt(float(fd.assemble(expression**2*measure)))
    slopes = stepper.stages.subfunctions
    require(len(slopes)==2,'Expected two scalar slopes.')
    for index in range(count):
        old_time = index*dt_value
        require(math.isclose(float(t),old_time,rel_tol=0.0,abs_tol=1e-14),'Incorrect pre-step clock.')
        old.assign(state)
        for slope in slopes:
            slope.assign(0.0)
        reported_history.clear()
        try:
            stepper.advance()
        except Exception as error:
            capture_failed_solve(record,index,old_time,dt_value,ksp,reported_history,error)
            raise
        require(float(t)==old_time,'Irksome changed the caller-owned time variable.')
        rtol,atol,divtol,max_it = ksp.getTolerances()
        require(ksp.getType()=='gmres' and ksp.getPC().getType()=='jacobi'
            and ksp.getPCSide()==PETSc.PC.Side.RIGHT and ksp.getNormType()==PETSc.KSP.NormType.UNPRECONDITIONED
            and not ksp.getInitialGuessNonzero() and rtol==parameters['ksp_rtol'] and atol==parameters['ksp_atol']
            and max_it==parameters['ksp_max_it'],
            'Actual solver differs from the stated baseline.')
        load = fd.assemble(load_form,bcs=independent_bcs)
        residual = true_residual(block,stepper.stages,load)
        target = max(atol,rtol*residual['rhs_norm'])
        update_defect = l2(state-(old+dt_value*sum(b[j]*slopes[j] for j in range(2))))
        stiff_defect = l2(state-(old+dt_value*sum(A[1][j]*slopes[j] for j in range(2))))
        row = {'step':index+1,'t_start':old_time,'t_end':(index+1)*dt_value,
            'stage_times':[old_time+c[i]*dt_value for i in range(2)],
            'iterations':ksp.getIterationNumber(),'converged_reason':int(ksp.getConvergedReason()),
            'reported_residual_norm':float(ksp.getResidualNorm()),'true_block_residual':residual,
            'true_residual_target':target,'true_residual_allowance_factor':checks['residual_tolerance_factor'],
            'update_identity_l2':update_defect,'stiff_accuracy_l2':stiff_defect}
        record['step_records'].append(row)
        require(row['converged_reason']>0,'A coupled Radau solve did not converge.')
        require(all(math.isfinite(x) for x in residual.values()) and residual['norm']<=checks['residual_tolerance_factor']*target,
                'Independent coupled residual exceeds its declared allowance.')
        require(update_defect<=checks['update_identity_l2_max'] and stiff_defect<=checks['stiff_accuracy_l2_max'],
                'Weighted update or stiff accuracy failed.')
        t.assign((index+1)*dt_value)
        record['completed_steps'] = index+1
        if (index+1)%100==0 or index+1==count:
            print(f"    {label}: {index+1}/{count} steps checked, t={float(t):.6f}",flush=True)
    require(math.isclose(float(t),config['final_time'],rel_tol=0.0,abs_tol=1e-14),'Incorrect final time.')
    ksp.view()  # Installed petsc4py has no GMRES restart getter; the view records actual restart.
    record['last_step_reported_residual_history'] = list(reported_history)
    stats = list(stepper.solver_stats())
    require(stats[0]==count and stats[2]==sum(row['iterations'] for row in record['step_records']),
            'Recorded work differs from Irksome solver statistics.')
    primary = error_metrics(state,float(t),context,config['quadrature_degree'],fd)
    control = error_metrics(state,float(t),context,config['quadrature_control_degree'],fd)
    validation,quadrature = check_metrics(primary,control,float(t),checks)
    # Check the MMS independently at both final-step stage times, plus endpoints.
    source_checks = []
    for value in (0.0,config['final_time'],(count-1+c[0])*dt_value,(count-1+c[1])*dt_value):
        exact = fd.exp(-value)*context['sinusoid']
        defect = l2(-exact+fd.dot(context['beta'],fd.grad(exact))-config['diffusion']*fd.div(fd.grad(exact))-forcing(value))
        source_checks.append({'time':value,'source_residual_l2':defect})
    validation['source'] = all(row['source_residual_l2']<=checks['source_residual_l2_max'] for row in source_checks)
    stage_boundary = [l2(slope,dsq) for slope in slopes]
    validation['final_slope_boundaries'] = all(value<=checks['stage_boundary_trace_l2_max'] for value in stage_boundary)
    record.update(status='passed' if all(validation.values()) else 'failed',final_time=float(t),
        metrics=primary,control_metrics=control,checks=validation,quadrature_comparison=quadrature,
        source_checks=source_checks,final_slope_boundary_trace_l2=stage_boundary,solver_stats=stats,
        total_iterations=sum(row['iterations'] for row in record['step_records']),
        maximum_step_iterations=max(row['iterations'] for row in record['step_records']),
        maximum_true_relative_residual=max(row['true_block_residual']['relative_norm'] for row in record['step_records']),
        actual_solver={'ksp_type':ksp.getType(),'pc_type':ksp.getPC().getType(),'side':'right','norm':'unpreconditioned',
           'rtol':rtol,'atol':atol,'divtol':divtol,'max_it':max_it,
           'requested_restart':parameters['ksp_gmres_restart'],'actual_restart_evidence':'KSP view in transcript'})
    print(f"  {label}: final L2={control['l2_error']:.9e}, H1 seminorm={control['h1_seminorm_error']:.9e}, checks={record['status']}",flush=True)
    return state


def compare_fields(a,b,reference_metrics,fraction,checks,fd,dxq):
    differences = norm_pair(a-b,fd,dxq)
    result = {}
    for metric,difference in differences.items():
        limit = checks['field_control_atol']+fraction*reference_metrics[metric]
        result[metric] = {'actual_field_difference':difference,'reference_continuum_error':reference_metrics[metric],
            'fraction_of_error':difference/reference_metrics[metric] if reference_metrics[metric] else None,
            'allowed_difference':limit,'status':'passed' if difference<=limit else 'failed'}
    return {'status':'passed' if all(v['status']=='passed' for v in result.values()) else 'failed','norms':result}


def study(config,result,fd,PETSc,irk,AI,comm):
    require(comm.size==1,'Run this controlled accuracy study in serial.')
    validate_config(config)
    baseline = json.loads((ROOT/'results/processed/interpolation_review.json').read_text())['serial_observations']['measurements']
    for degree in config['element_degrees']:
        for n in config['mesh_subdivisions']:
            case_id = f'P{degree}-n{n}'
            print(f'{case_id}: transient spatial verification at T=1',flush=True)
            mesh = fd.UnitSquareMesh(n,n,diagonal='left',comm=comm)
            V = fd.FunctionSpace(mesh,'CG',degree)
            x,y = fd.SpatialCoordinate(mesh)
            beta,kappa = fd.as_vector(config['velocity']),config['diffusion']
            sinusoid = fd.sin(fd.pi*x)*fd.sin(fd.pi*y)
            gradient = fd.as_vector((fd.pi*fd.cos(fd.pi*x)*fd.sin(fd.pi*y),fd.pi*fd.sin(fd.pi*x)*fd.cos(fd.pi*y)))
            bc = fd.DirichletBC(V,0.0,'on_boundary')
            initial = fd.Function(V,name='initial_'+case_id).interpolate(sinusoid)
            bc.apply(initial)
            forcing = lambda time: fd.exp(-time)*((2*kappa*fd.pi**2-1)*sinusoid+gradient[0]+gradient[1])
            spatial = lambda w,v: fd.dot(beta,fd.grad(w))*v+kappa*fd.inner(fd.grad(w),fd.grad(v))
            context = dict(mesh=mesh,space=V,initial=initial,sinusoid=sinusoid,exact_gradient=gradient,
                beta=beta,bc=bc,forcing=forcing,spatial=spatial,
                dx=fd.dx(domain=mesh,metadata={'quadrature_degree':config['quadrature_degree']}),
                ds=fd.ds(domain=mesh,metadata={'quadrature_degree':config['quadrature_degree']}))
            require(V.dim()==(degree*n+1)**2,'Unexpected spatial dimension.')
            case = {'case_id':case_id,'element_degree':degree,'mesh_subdivisions':n,'h':math.sqrt(2)/n,
                'h_definition':'Maximum triangle diameter','total_dofs':V.dim(),'expected_free_dofs':(degree*n-1)**2,
                'expected_cells':2*n*n,'status':'running','trajectories':{}}
            result['measurements'].append(case)
            initial_metrics = error_metrics(initial,0.0,context,config['quadrature_degree'],fd)
            original = next(row for row in baseline if row['case_id']==case_id)
            case['initial_metrics'] = initial_metrics
            case['initial_baseline_comparison'] = {metric:math.isclose(initial_metrics[metric],original['metrics'][metric],
                rel_tol=config['checks']['initial_baseline_rtol'],abs_tol=config['checks']['initial_baseline_atol']) for metric in ('l2_error','h1_seminorm_error')}
            require(all(case['initial_baseline_comparison'].values()),'Initialization differs from accepted interpolation.')
            for label in ('primary','half_dt'):
                case['trajectories'][label] = {'status':'prepared'}
            primary = run_trajectory(config,context,config['dt'],1.0,'primary',case['trajectories']['primary'],fd,PETSc,irk,AI)
            refined = run_trajectory(config,context,config['control_dt'],1.0,'half_dt',case['trajectories']['half_dt'],fd,PETSc,irk,AI)
            dxcontrol = fd.dx(domain=mesh,metadata={'quadrature_degree':config['quadrature_control_degree']})
            half_record = case['trajectories']['half_dt']
            case['metrics'] = half_record['control_metrics']
            case['time_control'] = compare_fields(primary,refined,case['metrics'],config['checks']['time_control_fraction'],config['checks'],fd,dxcontrol)
            if degree==config['solver_control']['element_degree'] and n==config['solver_control']['mesh_subdivisions']:
                case['trajectories']['tighter_solver'] = {'status':'prepared'}
                tight = run_trajectory(config,context,config['control_dt'],config['solver_control']['tolerance_factor'],
                    'tighter_solver',case['trajectories']['tighter_solver'],fd,PETSc,irk,AI)
                case['solver_control'] = compare_fields(tight,refined,case['metrics'],config['checks']['solver_control_fraction'],config['checks'],fd,dxcontrol)
            validations = [record['status']=='passed' for record in case['trajectories'].values()]+[case['time_control']['status']=='passed']
            if 'solver_control' in case:
                validations.append(case['solver_control']['status']=='passed')
            case['status'] = 'passed' if all(validations) else 'failed'
            print(f"{case_id}: controls {case['status']}",flush=True)
    for degree in config['element_degrees']:
        rows = [row for row in result['measurements'] if row['element_degree']==degree]
        for coarse,fine in zip(rows,rows[1:]):
            for metric in ('l2_error','h1_seminorm_error'):
                value = observed_rate(coarse,fine,metric)
                expected = config['expected_orders'][str(degree)][metric]
                passed = value is not None and abs(value-expected)<=config['checks']['observed_order_atol']
                result['rates'].append({'element_degree':degree,'metric':metric,'coarse_case':coarse['case_id'],
                    'fine_case':fine['case_id'],'observed_order':value,'expected_order':expected,
                    'allowed_order_difference':config['checks']['observed_order_atol'],'status':'passed' if passed else 'failed'})
                print(f"P{degree} {metric}, n={coarse['mesh_subdivisions']}->{fine['mesh_subdivisions']}: order={value}, checks={'passed' if passed else 'failed'}",flush=True)
    result['status'] = 'passed' if all(row['status']=='passed' for row in result['measurements']+result['rates']) else 'failed'
    result['completed_study'] = True


def sanitize(value):
    # Preserve failed/incomplete data as valid JSON even if a numerical value is nonfinite.
    if isinstance(value,float) and not math.isfinite(value):
        return str(value)
    if isinstance(value,dict):
        return {k:sanitize(v) for k,v in value.items()}
    if isinstance(value,list):
        return [sanitize(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    sys.argv = [sys.argv[0]]
    result = {'status':'failed','completed_study':False,'ranks':1,'measurements':[],'rates':[],
        'error_curve':'half_dt baseline trajectory, continuum norms at quadrature degree 14',
        'time_control_interpretation':'Sensitivity to one step halving, not verified temporal order or an absolute error bound',
        'solver_control_interpretation':'Sensitivity to ten-times-tighter tolerances at finest P2 case',
        'temporal_order_verified':False,'performance_measurement':False}
    output_allowed = args.output.resolve().parent == ROOT/'results/raw'
    try:
        require(output_allowed,'Keep raw data in the Projects repository results/raw folder.')
        result['config'] = json.loads(args.config.read_text())
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        require(comm.size==1,'This verification is serial.')
        import firedrake as fd
        from firedrake.petsc import PETSc
        import irksome as irk  # Register Dt before constructing or compiling any form.
        from irksome.tools import AI
        study(result['config'],result,fd,PETSc,irk,AI,comm)
    except (Exception,KeyboardInterrupt) as error:
        result['status'] = 'failed'
        result['error'] = str(error) or 'Interrupted by researcher'
        if result['measurements'] and result['measurements'][-1]['status']=='running':
            result['measurements'][-1]['status'] = 'failed'
        traceback.print_exc()
    finally:
        if output_allowed:
            args.output.write_text(json.dumps(sanitize(result),indent=2,allow_nan=False)+'\n')
    return 0 if result['status']=='passed' else 1


if __name__=='__main__':
    sys.exit(main())

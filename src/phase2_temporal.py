#!/usr/bin/env python3
"""Researcher-run same-mesh BE/Radau temporal verification with controlled references."""
import argparse
import json
import math
from pathlib import Path
import sys
import traceback

from phase0_be_smoke import require
from phase0_irksome_be import true_residual
from phase0_radau_smoke import tableau_coefficients
from phase1_spatial import (step_count, safe_sqrt, norm_pair, error_metrics, check_metrics,
                            run_trajectory, capture_failed_solve, sanitize)

ROOT = Path(__file__).resolve().parents[1]
METRICS = ('l2_error','h1_seminorm_error')


def radau_config(config):
    return {**config,**config['radau_formulation'],
            'solver_parameters':{**config['solver_parameters'],'snes_type':'ksponly'}}


def validate_config(config):
    require(config['mesh_subdivisions']==16 and config['element_degree']==2
        and config['element_family']=='CG' and config['mesh_diagonal']=='left'
        and config['velocity']==[1.0,1.0] and config['diffusion']==1.0 and config['final_time']==1.0,
        'Keep the explained fixed P2/n16 verification at T=1.')
    require(config['time_methods']==['manual_backward_euler','RadauIIA']
        and config['time_step_sizes']==[0.1,0.05,0.025,0.0125], 'Keep the bounded four-step-size cut.')
    reference = config['reference']
    require(reference=={'method':'RadauIIA','coarse_dt':1/640,'fine_dt':1/1280},'Keep the two explained reference steps.')
    require(config['solver_control']=={'tolerance_factor':0.1,'curves':'finest_dt_each_method','reference':'fine_dt'},
        'Keep the explained three tolerance controls.')
    require(config['quadrature_control_degree']>config['quadrature_degree']>=4,'Use a higher-degree norm control.')
    tableau_coefficients(radau_config(config))
    for dt in [*config['time_step_sizes'],reference['coarse_dt'],reference['fine_dt']]:
        step_count(config['final_time'],dt)
    return 2*sum(step_count(1,dt) for dt in config['time_step_sizes'])+640+1280+1280+2*80


def run_be(config,context,dt_value,tolerance_factor,label,record,fd,PETSc):
    V,dxq,initial,bc = (context[key] for key in ('space','dx','initial','bc'))
    spatial,forcing = context['spatial'],context['forcing']
    count = step_count(config['final_time'],dt_value)
    state = fd.Function(V,name=label).assign(initial)
    old = fd.Function(V,name=label+'_old').assign(initial)
    end_time = fd.Constant(dt_value)
    w,v = fd.TrialFunction(V),fd.TestFunction(V)
    # This is the same manual BE equation as the accepted one-step check.
    B = fd.assemble((w*v+dt_value*spatial(w,v))*dxq,bcs=[bc],mat_type='aij')
    rhs_form = (old*v+dt_value*forcing(end_time)*v)*dxq
    parameters = dict(config['solver_parameters'])
    parameters['ksp_rtol'] *= tolerance_factor
    parameters['ksp_atol'] *= tolerance_factor
    solver = fd.LinearSolver(B,solver_parameters=parameters,options_prefix='phase2_'+label)
    ksp = solver.ksp
    history = []
    ksp.setMonitor(lambda solver,iteration,norm: history.append([iteration,float(norm)]))
    require(B.petscmat.getSize()==(V.dim(),V.dim()),'Unexpected BE matrix dimensions.')
    record.update(status='running',method='manual_backward_euler',dt=dt_value,planned_steps=count,
        completed_steps=0,step_records=[],solver_parameters=parameters,total_dofs=V.dim(),
        physical_solution_initial_guess='zeroed before each solve',
        reuse_policy='One assembled BE matrix and LinearSolver per fixed-dt trajectory')
    print(f'{label}: manual BE dt={dt_value}, steps={count}, rtol={parameters["ksp_rtol"]:.1e}',flush=True)
    for index in range(count):
        old.assign(state)
        state.assign(0.0)
        end_time.assign((index+1)*dt_value)
        rhs = fd.assemble(rhs_form,bcs=[bc])
        history.clear()
        try:
            solver.solve(state,rhs)
        except Exception as error:
            capture_failed_solve(record,index,index*dt_value,dt_value,ksp,history,error)
            if 'petsc_correction_residual' in record.get('failed_step',{}):
                record['failed_step']['petsc_correction_residual']['meaning'] = 'Recomputed PETSc linear correction residual for the manual BE solve; the failed solution was not copied into the output field or committed'
            raise
        rtol,atol,divtol,max_it = ksp.getTolerances()
        require(ksp.getType()=='gmres' and ksp.getPC().getType()=='jacobi'
            and ksp.getPCSide()==PETSc.PC.Side.RIGHT and ksp.getNormType()==PETSc.KSP.NormType.UNPRECONDITIONED
            and not ksp.getInitialGuessNonzero() and rtol==parameters['ksp_rtol'] and atol==parameters['ksp_atol']
            and max_it==parameters['ksp_max_it'],'Unexpected actual BE solver configuration.')
        residual = true_residual(B,state,rhs)
        target = max(atol,rtol*residual['rhs_norm'])
        row = {'step':index+1,'t_start':index*dt_value,'t_end':float(end_time),
            'source_time':float(end_time),'iterations':ksp.getIterationNumber(),
            'converged_reason':int(ksp.getConvergedReason()),'reported_residual_norm':float(ksp.getResidualNorm()),
            'true_BE_residual':residual,'true_residual_target':target,
            'true_residual_allowance_factor':config['checks']['residual_tolerance_factor']}
        record['step_records'].append(row)
        require(row['converged_reason']>0 and all(math.isfinite(x) for x in residual.values())
            and residual['norm']<=config['checks']['residual_tolerance_factor']*target,'BE equation residual failed.')
        record['completed_steps'] = index+1
        if (index+1)%100==0 or index+1==count:
            print(f'  {label}: {index+1}/{count} checked steps',flush=True)
    require(math.isclose(float(end_time),config['final_time'],rel_tol=0.0,abs_tol=1e-14),'Incorrect BE final time.')
    ksp.view()
    primary = error_metrics(state,config['final_time'],context,config['quadrature_degree'],fd)
    control = error_metrics(state,config['final_time'],context,config['quadrature_control_degree'],fd)
    validation,quadrature = check_metrics(primary,control,config['final_time'],config['checks'])
    source_checks = []
    for time in (0.0,config['final_time']):
        exact = fd.exp(-time)*context['sinusoid']
        expression = -exact+fd.dot(context['beta'],fd.grad(exact))-config['diffusion']*fd.div(fd.grad(exact))-forcing(time)
        defect = safe_sqrt(float(fd.assemble(expression**2*dxq)))
        source_checks.append({'time':time,'source_residual_l2':defect})
    validation['source'] = all(row['source_residual_l2']<=config['checks']['source_residual_l2_max'] for row in source_checks)
    record.update(status='passed' if all(validation.values()) else 'failed',final_time=float(end_time),
        metrics=primary,control_metrics=control,checks=validation,quadrature_comparison=quadrature,
        source_checks=source_checks,total_iterations=sum(row['iterations'] for row in record['step_records']),
        maximum_step_iterations=max(row['iterations'] for row in record['step_records']),
        last_step_reported_residual_history=list(history),
        actual_solver={'ksp_type':ksp.getType(),'pc_type':ksp.getPC().getType(),'side':'right','norm':'unpreconditioned',
            'rtol':rtol,'atol':atol,'divtol':divtol,'max_it':max_it,'requested_restart':parameters['ksp_gmres_restart'],
            'actual_restart_evidence':'KSP view in transcript'})
    return state


def measured_difference(a,b,config,context,fd):
    def measure(q):
        dxq = fd.dx(domain=context['mesh'],metadata={'quadrature_degree':q})
        return norm_pair(a-b,fd,dxq)
    primary,control = measure(config['quadrature_degree']),measure(config['quadrature_control_degree'])
    comparison = {}
    for metric in METRICS:
        difference = abs(primary[metric]-control[metric])
        allowance = config['checks']['quadrature_comparison_atol']+config['checks']['quadrature_comparison_rtol']*control[metric]
        comparison[metric] = {'absolute_difference':difference,'allowed_difference':allowance,
                             'status':'passed' if difference<=allowance else 'failed'}
    return {'metrics':control,'primary_metrics':primary,'quadrature_comparison':comparison,
            'status':'passed' if all(row['status']=='passed' for row in comparison.values()) else 'failed'}


def scaled_control(difference,scale,fraction,atol):
    norms = {metric:{'actual_field_difference':difference['metrics'][metric],'temporal_error_scale':scale[metric],
        'allowed_difference':atol+fraction*scale[metric],
        'status':'passed' if difference['metrics'][metric]<=atol+fraction*scale[metric] else 'failed'} for metric in METRICS}
    return {'status':'passed' if difference['status']=='passed' and all(row['status']=='passed' for row in norms.values()) else 'failed',
            'norms':norms,'measured_difference':difference}


def observed_time_order(coarse,fine,metric):
    a,b = coarse['time_errors'][metric],fine['time_errors'][metric]
    return math.log(a/b)/math.log(coarse['dt']/fine['dt']) if 0<b<a and coarse['dt']>fine['dt'] else None


def study(config,result,fd,PETSc,irk,AI,comm):
    require(comm.size==1,'Run this accuracy verification in serial.')
    require(validate_config(config)==3660,'Unexpected fixed verification schedule.')
    n,degree = config['mesh_subdivisions'],config['element_degree']
    mesh = fd.UnitSquareMesh(n,n,diagonal='left',comm=comm)
    V = fd.FunctionSpace(mesh,'CG',degree)
    require(V.dim()==1089,'Unexpected P2/n16 space dimension.')
    x,y = fd.SpatialCoordinate(mesh)
    sinusoid = fd.sin(fd.pi*x)*fd.sin(fd.pi*y)
    gradient = fd.as_vector((fd.pi*fd.cos(fd.pi*x)*fd.sin(fd.pi*y),fd.pi*fd.sin(fd.pi*x)*fd.cos(fd.pi*y)))
    beta,kappa = fd.as_vector(config['velocity']),config['diffusion']
    bc = fd.DirichletBC(V,0.0,'on_boundary')
    initial = fd.Function(V,name='shared_initial').interpolate(sinusoid)
    bc.apply(initial)
    context = dict(mesh=mesh,space=V,initial=initial,sinusoid=sinusoid,exact_gradient=gradient,beta=beta,bc=bc,
        dx=fd.dx(domain=mesh,metadata={'quadrature_degree':config['quadrature_degree']}),
        ds=fd.ds(domain=mesh,metadata={'quadrature_degree':config['quadrature_degree']}),
        spatial=lambda w,v:fd.dot(beta,fd.grad(w))*v+kappa*fd.inner(fd.grad(w),fd.grad(v)),
        forcing=lambda time:fd.exp(-time)*((2*kappa*fd.pi**2-1)*sinusoid+gradient[0]+gradient[1]))
    result.update(total_dofs=V.dim(),expected_free_dofs=961,expected_cells=512,h=math.sqrt(2)/n,
                  h_definition='Maximum triangle diameter',initial_metrics=error_metrics(initial,0.0,context,config['quadrature_degree'],fd))
    accepted = json.loads((ROOT/'results/processed/interpolation_review.json').read_text())['serial_observations']['measurements']
    baseline = next(row for row in accepted if row['case_id']=='P2-n16')['metrics']
    result['initial_baseline_comparison'] = {metric:math.isclose(result['initial_metrics'][metric],baseline[metric],
        rel_tol=config['checks']['initial_baseline_rtol'],abs_tol=config['checks']['initial_baseline_atol']) for metric in METRICS}
    require(all(result['initial_baseline_comparison'].values()),'Initial function differs from accepted initialization.')
    rkconfig = radau_config(config)
    def trajectory(method,dt,factor,label):
        record = {'status':'prepared','method':method,'dt':dt}
        result['trajectories'][label] = record
        result['active_trajectory'] = label
        if method=='RadauIIA':
            state = run_trajectory(rkconfig,context,dt,factor,label,record,fd,PETSc,irk,AI)
        else:
            state = run_be(config,context,dt,factor,label,record,fd,PETSc)
        result.pop('active_trajectory',None)
        return state
    reference_coarse = trajectory('RadauIIA',config['reference']['coarse_dt'],1.0,'reference_coarse')
    reference_fine = trajectory('RadauIIA',config['reference']['fine_dt'],1.0,'reference_fine')
    reference_tight = trajectory('RadauIIA',config['reference']['fine_dt'],config['solver_control']['tolerance_factor'],'reference_fine_tighter')
    result['reference_gap'] = measured_difference(reference_coarse,reference_fine,config,context,fd)
    result['reference_solver_difference'] = measured_difference(reference_fine,reference_tight,config,context,fd)
    finest_states = {}
    for method in config['time_methods']:
        for index,dt in enumerate(config['time_step_sizes']):
            label = f'{method}_dt{index}'
            state = trajectory(method,dt,1.0,label)
            difference = measured_difference(state,reference_fine,config,context,fd)
            row = {'method':method,'dt':dt,'trajectory':label,'time_errors':difference['metrics'],
                   'reference_trajectory':'reference_fine','difference_measurement':difference,
                   'status':'passed' if difference['status']=='passed' and all(difference['metrics'][metric]>0 for metric in METRICS) else 'failed'}
            result['curve'].append(row)
            print(f'{method}: dt={dt}, same-mesh L2={row["time_errors"]["l2_error"]:.9e}, H1 seminorm={row["time_errors"]["h1_seminorm_error"]:.9e}',flush=True)
            if index==len(config['time_step_sizes'])-1:
                finest_states[method] = state
    minimum_error = {metric:min(row['time_errors'][metric] for row in result['curve']) for metric in METRICS}
    checks = config['checks']
    result['reference_resolution'] = {'status':'passed','minimum_curve_errors':minimum_error,'norms':{},
        'interpretation':'Combined refinement/tolerance sensitivity; not a certified absolute reference error bound'}
    for metric in METRICS:
        gap = result['reference_gap']['metrics'][metric]
        algebraic_change = result['reference_solver_difference']['metrics'][metric]
        allowed = checks['field_control_atol']+checks['reference_fraction']*minimum_error[metric]
        passed = gap+algebraic_change<=allowed
        result['reference_resolution']['norms'][metric] = {'refinement_gap':gap,'tolerance_change':algebraic_change,
            'combined_sensitivity':gap+algebraic_change,'allowed_combined_sensitivity':allowed,'status':'passed' if passed else 'failed'}
        if not passed:
            result['reference_resolution']['status'] = 'failed'
    for method in config['time_methods']:
        dt = config['time_step_sizes'][-1]
        tight = trajectory(method,dt,config['solver_control']['tolerance_factor'],method+'_finest_tighter')
        difference = measured_difference(tight,finest_states[method],config,context,fd)
        row = next(row for row in result['curve'] if row['method']==method and row['dt']==dt)
        result['curve_solver_controls'][method] = scaled_control(difference,row['time_errors'],checks['solver_control_fraction'],checks['field_control_atol'])
    for method in config['time_methods']:
        rows = [row for row in result['curve'] if row['method']==method]
        for coarse,fine in zip(rows,rows[1:]):
            for metric in METRICS:
                value = observed_time_order(coarse,fine,metric)
                expected = config['expected_temporal_orders'][method]
                passed = value is not None and abs(value-expected)<=checks['observed_order_atol']
                result['rates'].append({'method':method,'metric':metric,'coarse_dt':coarse['dt'],'fine_dt':fine['dt'],
                    'observed_order':value,'expected_order':expected,'allowed_order_difference':checks['observed_order_atol'],
                    'status':'passed' if passed else 'failed'})
                print(f'{method} {metric}, dt={coarse["dt"]}->{fine["dt"]}: order={value}, checks={"passed" if passed else "failed"}',flush=True)
    validations = [row['status']=='passed' for row in result['trajectories'].values()]
    validations += [row['status']=='passed' for row in result['curve']+result['rates']]
    validations += [result[key]['status']=='passed' for key in ('reference_gap','reference_solver_difference','reference_resolution')]
    validations += [row['status']=='passed' for row in result['curve_solver_controls'].values()]
    result.update(status='passed' if all(validations) else 'failed',completed_study=True,
                  completed_trajectories=len(result['trajectories']),completed_steps=sum(t['completed_steps'] for t in result['trajectories'].values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    sys.argv = [sys.argv[0]]
    result = {'status':'failed','completed_study':False,'ranks':1,'trajectories':{},'curve':[],'rates':[],
              'curve_solver_controls':{},'temporal_error_definition':'Same-mesh final-field difference from baseline fine Radau reference',
              'continuum_errors_role':'Diagnostic only; not the temporal-order curve','performance_measurement':False}
    allowed = args.output.resolve().parent==ROOT/'results/raw'
    try:
        require(allowed,'Keep raw records in the Projects repository.')
        config = json.loads(args.config.read_text())
        result['config'] = config
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        require(comm.size==1,'This verification is serial.')
        import firedrake as fd
        from firedrake.petsc import PETSc
        import irksome as irk
        from irksome.tools import AI
        study(config,result,fd,PETSc,irk,AI,comm)
    except (Exception,KeyboardInterrupt) as error:
        result.update(status='failed',error=str(error) or 'Interrupted by researcher')
        label = result.get('active_trajectory')
        if label and result['trajectories'][label]['status'] in ('running','prepared'):
            result['trajectories'][label]['status'] = 'failed'
        traceback.print_exc()
    finally:
        if allowed:
            args.output.write_text(json.dumps(sanitize(result),indent=2,allow_nan=False)+'\n')
    return 0 if result['status']=='passed' else 1


if __name__=='__main__':
    sys.exit(main())

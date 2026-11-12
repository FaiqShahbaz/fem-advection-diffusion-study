#!/usr/bin/env python3
"""Researcher-run full Radau trajectories with fixed ASM/local ILU and reuse checks."""
import argparse
import json
import math
from pathlib import Path
import sys
import traceback

from phase0_be_smoke import require
from phase0_irksome_be import true_residual
from phase0_radau_smoke import tableau_coefficients
from phase1_spatial import (run_trajectory,step_count,safe_sqrt,error_metrics,check_metrics,
                            capture_failed_solve,sanitize)
from phase2_temporal import radau_config,METRICS
from phase3_asm_construction import (validate_config as validate_construction,
    inspect_domains,compare,additive_action_check)

ROOT=Path(__file__).resolve().parents[1]
EVENTS=('KSPSolve','KSPSetUp','PCSetUp','PCApply')


def validate_config(config):
    construction=json.loads((ROOT/'configs/phase3_asm_construction.json').read_text())
    validate_construction(construction)
    assert_keys=set(construction)-{'purpose','checkpoint_steps'}
    require(all(config[k]==construction[k] for k in assert_keys),'Keep the accepted PDE, tolerances and ASM configuration.')
    require(config['checkpoint_steps']==step_count(config['final_time'],config['dt'])==10
        and config['trajectory_roles']==['irksome_jacobi','manual_jacobi_reference','manual_jacobi','manual_asm']
        and config['performance_measurement'] is False,'Keep these four serial verification trajectories.')


def event_counts(events,stage):
    return {name:int(event.getPerfInfo(stage.id)['count']) for name,event in events.items()}


def count_difference(before,after):
    result={key:after[key]-before[key] for key in before}
    require(all(value>=0 for value in result.values()),'Invalid event-count difference.')
    return result


def make_context(config,fd,comm):
    mesh=fd.UnitSquareMesh(config['mesh_subdivisions'],config['mesh_subdivisions'],diagonal='left',comm=comm)
    V=fd.FunctionSpace(mesh,'CG',config['element_degree']);x,y=fd.SpatialCoordinate(mesh)
    sinusoid=fd.sin(fd.pi*x)*fd.sin(fd.pi*y)
    gradient=fd.as_vector((fd.pi*fd.cos(fd.pi*x)*fd.sin(fd.pi*y),fd.pi*fd.sin(fd.pi*x)*fd.cos(fd.pi*y)))
    beta,kappa=fd.as_vector(config['velocity']),config['diffusion'];bc=fd.DirichletBC(V,0.,'on_boundary')
    initial=fd.Function(V,name='shared_initial').interpolate(sinusoid);bc.apply(initial)
    return dict(mesh=mesh,space=V,initial=initial,sinusoid=sinusoid,exact_gradient=gradient,beta=beta,bc=bc,
        dx=fd.dx(domain=mesh,metadata={'quadrature_degree':config['quadrature_degree']}),
        ds=fd.ds(domain=mesh,metadata={'quadrature_degree':config['quadrature_degree']}),
        spatial=lambda w,v:fd.dot(beta,fd.grad(w))*v+kappa*fd.inner(fd.grad(w),fd.grad(v)),
        forcing=lambda time:fd.exp(-time)*((2*kappa*fd.pi**2-1)*sinusoid+gradient[0]+gradient[1]))


def manual_trajectory(config,context,parameters,label,record,fd,PETSc,references=None):
    from petsctools import inserted_options
    V=context['space'];W=V*V;dt=config['dt'];count=step_count(config['final_time'],dt)
    A,b,c=tableau_coefficients(radau_config(config));spatial,forcing=context['spatial'],context['forcing']
    state=fd.Function(V,name=label+'_state').assign(context['initial'])
    old=fd.Function(V,name=label+'_old');slopes=fd.Function(W,name=label+'_slopes')
    t=fd.Constant(0.0);z,q=fd.TrialFunctions(W),fd.TestFunctions(W)
    bcs=[fd.DirichletBC(W.sub(i),0.,'on_boundary') for i in range(2)]
    block_form=sum(z[i]*q[i]+dt*sum(A[i][j]*spatial(z[j],q[i]) for j in range(2)) for i in range(2))*context['dx']
    load_form=sum(forcing(t+c[i]*dt)*q[i]-spatial(old,q[i]) for i in range(2))*context['dx']
    block=fd.assemble(block_form,bcs=bcs,mat_type='aij')
    require(W.dim()==2178 and block.petscmat.getSize()==(2178,2178) and block.petscmat.getBlockSize()==1,
            'Unexpected coupled layout.')
    prefix='phase3_trajectory_'+label+'_';block.petscmat.setOptionsPrefix(prefix)
    ksp=PETSc.KSP().create(comm=context['mesh'].comm)
    ksp.setOptionsPrefix(prefix);ksp.setOperators(block.petscmat)
    history=[];ksp.setMonitor(lambda solver,iteration,norm:history.append([iteration,float(norm)]))
    work=[];subksps=[];phase={'name':'outer_solve'};snapshots=[]
    stage=PETSc.Log.Stage('phase3_asm_trajectory_'+label)
    events={name:PETSc.Log.Event(name) for name in EVENTS}
    record.update(status='running',method='RadauIIA',dt=dt,planned_steps=count,completed_steps=0,
        solver_parameters=parameters,stage_total_dofs=W.dim(),step_records=[],matrix_assemblies=1,
        physical_stage_initial_guess='zeroed before every solve',reuse_policy='One fixed B and KSP per trajectory; reuse only while B is unchanged',
        performance_measurement=False)
    try:
        with stage:
            start_counts=event_counts(events,stage)
            with inserted_options(parameters=parameters,options_prefix=prefix):
                ksp.setFromOptions();pc=ksp.getPC();ksp.setUp()
                record['initial_setup_event_counts']=count_difference(start_counts,event_counts(events,stage))
                if parameters['pc_type']=='asm':
                    options=PETSc.Options(prefix)
                    observed={key:options.getString(key) for key in ('pc_asm_type','pc_asm_local_type','mat_partitioning_type',
                        'sub_ksp_type','sub_pc_type','sub_pc_factor_mat_ordering_type','sub_pc_factor_mat_solver_type','sub_pc_factor_shift_type')}
                    observed.update({key:options.getInt(key) for key in ('pc_asm_blocks','pc_asm_overlap','sub_pc_factor_levels')})
                    require(all(observed[key]==parameters[key] for key in observed),'ASM options changed.')
                    record['options_observed_in_scope']=observed;record['domains']=inspect_domains(pc,block.petscmat)
                    subksps=pc.getASMSubKSP();require(len(subksps)==4,'Expected four local KSPs.')
                    record['local_work']=work
                    def postsolve(sub,rhs,solution,index):
                        iterations=sub.getIterationNumber();reason=int(sub.getConvergedReason())
                        require(sub.getType()=='preonly' and sub.getPC().getType()=='ilu' and iterations==1 and reason>0,
                                'Unexpected local action.')
                        counts=work[index][phase['name']];counts['calls']+=1;counts['reported_iterations']+=iterations
                    for i,sub in enumerate(subksps):
                        work.append(dict(index=i,outer_solve=dict(calls=0,reported_iterations=0),action_check=dict(calls=0,reported_iterations=0)))
                        sub.setPostSolve(postsolve,args=(i,))
                record['matrix_options_prefix']=prefix
                for index in range(count):
                    old_time=index*dt;t.assign(old_time);old.assign(state);slopes.assign(0.)
                    load=fd.assemble(load_form,bcs=bcs);history.clear()
                    before=event_counts(events,stage);prior=[w['outer_solve']['calls'] for w in work]
                    try:
                        with load.dat.vec_ro as rhs,slopes.dat.vec_wo as solution:ksp.solve(rhs,solution)
                    except Exception as error:
                        capture_failed_solve(record,index,old_time,dt,ksp,history,error)
                        diagnostic=record.get('failed_step',{}).get('petsc_correction_residual')
                        if diagnostic is not None:diagnostic['meaning']='Direct physical B Z = g residual of unaccepted partial slopes'
                        raise
                    counts=count_difference(before,event_counts(events,stage))
                    rtol,atol,divtol,max_it=ksp.getTolerances()
                    require(ksp.getType()=='gmres' and pc.getType()==parameters['pc_type'] and ksp.getPCSide()==PETSc.PC.Side.RIGHT
                        and ksp.getNormType()==PETSc.KSP.NormType.UNPRECONDITIONED and not ksp.getInitialGuessNonzero()
                        and rtol==parameters['ksp_rtol'] and atol==parameters['ksp_atol'] and max_it==1000,'Actual solver changed.')
                    residual=true_residual(block,slopes,load);target=max(atol,rtol*residual['rhs_norm'])
                    components=slopes.subfunctions
                    state.interpolate(old+dt*sum(b[j]*components[j] for j in range(2)))
                    l2=lambda expr,measure=context['dx']:safe_sqrt(float(fd.assemble(expr**2*measure)))
                    row=dict(step=index+1,t_start=old_time,t_end=(index+1)*dt,stage_times=[old_time+ci*dt for ci in c],
                        iterations=ksp.getIterationNumber(),converged_reason=int(ksp.getConvergedReason()),
                        reported_residual_norm=float(ksp.getResidualNorm()),true_block_residual=residual,true_residual_target=target,
                        event_counts=counts,update_identity_l2=l2(state-(old+dt*sum(b[j]*components[j] for j in range(2)))),
                        stiff_accuracy_l2=l2(state-(old+dt*sum(A[1][j]*components[j] for j in range(2)))),
                        stage_boundary_trace_l2=[l2(slope,context['ds']) for slope in components],
                        local_calls_per_domain=[w['outer_solve']['calls']-p for w,p in zip(work,prior)])
                    record['step_records'].append(row)
                    require(row['converged_reason']>0 and all(math.isfinite(v) for v in residual.values())
                        and residual['norm']<=config['checks']['residual_tolerance_factor']*target,'Independent physical residual failed.')
                    require(row['update_identity_l2']<=1e-12 and row['stiff_accuracy_l2']<=1e-12
                        and all(v<=1e-12 for v in row['stage_boundary_trace_l2']),'Update/boundary check failed.')
                    require(index==0 or counts['PCSetUp']==0,'Unexpected PC/factor rebuild after the first step.')
                    if work:require(min(row['local_calls_per_domain'])>0 and len(set(row['local_calls_per_domain']))==1,'Local work mismatch.')
                    source=[]
                    for value in row['stage_times']:
                        exact=fd.exp(-value)*context['sinusoid']
                        defect=l2(-exact+fd.dot(context['beta'],fd.grad(exact))-config['diffusion']*fd.div(fd.grad(exact))-forcing(value))
                        source.append(dict(time=value,source_residual_l2=defect))
                    row['source_checks']=source
                    require(all(s['source_residual_l2']<=config['checks']['source_residual_l2_max'] for s in source),'MMS source check failed.')
                    if references is not None:
                        ref_state,ref_slopes=references[index]
                        row['reference_comparisons']={'update':compare(state,ref_state,config,context,fd)}
                        for i in range(2):row['reference_comparisons']['slope'+str(i+1)]=compare(components[i],ref_slopes.subfunctions[i],config,context,fd)
                        require(all(r['status']=='passed' for r in row['reference_comparisons'].values()),'Step field agreement failed.')
                    snapshots.append((fd.Function(V).assign(state),fd.Function(W).assign(slopes)))
                    record['completed_steps']=index+1
                record['operator_event_counts']=count_difference(start_counts,event_counts(events,stage))
                require(record['operator_event_counts']['PCSetUp']>=1,'No recorded PC setup.')
                record['last_step_reported_residual_history']=list(history)
                ksp.view()
                if work:
                    record['local_solvers']=[dict(index=i,ksp_type=sub.getType(),pc_type=sub.getPC().getType(),
                        factor_solver_type=sub.getPC().getFactorSolverType(),matrix_size=list(sub.getOperators()[0].getSize()),
                        factor_nonzeros=float(sub.getPC().getFactorMatrix().getInfo()['nz_used'])) for i,sub in enumerate(subksps)]
                    require(all(s['factor_solver_type']=='petsc' for s in record['local_solvers']),'Local backend changed.')
                    phase['name']='action_check'
                    record['additive_action_check']=additive_action_check(pc,subksps,record['domains'],block.petscmat,config['checks'],PETSc)
                    require(record['additive_action_check']['status']=='passed' and all(w['action_check']['calls']==1 for w in work),'Final PC action check failed.')
        primary=error_metrics(state,config['final_time'],context,config['quadrature_degree'],fd)
        control=error_metrics(state,config['final_time'],context,config['quadrature_control_degree'],fd)
        checks,quadrature=check_metrics(primary,control,config['final_time'],config['checks'])
        require(all(checks.values()),'Final MMS/quadrature checks failed.')
        record.update(status='passed',final_time=config['final_time'],metrics=primary,control_metrics=control,checks=checks,
            quadrature_comparison=quadrature,total_iterations=sum(s['iterations'] for s in record['step_records']),
            maximum_step_iterations=max(s['iterations'] for s in record['step_records']),
            actual_solver=dict(ksp_type=ksp.getType(),pc_type=pc.getType(),side='right',norm='unpreconditioned',
                rtol=rtol,atol=atol,divtol=divtol,max_it=max_it,requested_restart=50,restart_evidence='Actual KSP view in transcript'))
        print(f"  {label}: {count} steps passed; total GMRES iterations={record['total_iterations']}",flush=True)
        return state,snapshots
    finally:
        ksp.destroy()


def study(config,result,fd,PETSc,irk,AI,comm):
    require(comm.size==1,'Serial trajectory check required.');validate_config(config)
    context=make_context(config,fd,comm);require(context['space'].dim()==1089,'Unexpected space.')
    result.update(total_dofs=1089,stage_total_dofs=2178)
    records=result['trajectories'];fields={}
    baseline=dict(status='prepared',method='RadauIIA');records['irksome_jacobi']=baseline;result['active_trajectory']='irksome_jacobi'
    fields['irksome_jacobi']=run_trajectory(radau_config(config),context,config['dt'],1.,'asm_trajectory_irksome',baseline,fd,PETSc,irk,AI)
    require(baseline['status']=='passed','Irksome baseline failed.')
    # Reproduction check against the accepted same-configuration trajectory; actual new fields are compared below.
    review=json.loads((ROOT/'results/processed/accuracy_cost_review.json').read_text())
    accepted=json.loads((ROOT/review['measurement_records']['serial']['path']).read_text())['trajectories']['Radau_coarse_verified']
    require([s['iterations'] for s in baseline['step_records']]==[s['iterations'] for s in accepted['step_records']],
        'Accepted baseline iteration sequence changed; retain records for review.')
    result['baseline_reproduction']={metric:math.isclose(baseline['control_metrics'][metric],accepted['control_metrics'][metric],
        rel_tol=1e-9,abs_tol=1e-12) for metric in METRICS}
    require(all(result['baseline_reproduction'].values()),'Accepted baseline norm reproduction failed.')
    reference_params=dict(config['solver_parameters'])
    for key in ('ksp_rtol','ksp_atol'):reference_params[key]*=config['manual_reference_tolerance_factor']
    references=None
    for label,params in [('manual_jacobi_reference',reference_params),('manual_jacobi',config['solver_parameters']),('manual_asm',config['asm_solver_parameters'])]:
        row=dict(status='prepared',role=label);records[label]=row;result['active_trajectory']=label
        state,saved=manual_trajectory(config,context,params,label,row,fd,PETSc,references)
        fields[label]=state
        if label=='manual_jacobi_reference':references=saved
        result['comparisons'][label+'_vs_irksome']=compare(state,fields['irksome_jacobi'],config,context,fd)
        require(result['comparisons'][label+'_vs_irksome']['status']=='passed','Final manual/Irksome field check failed.')
    result['comparisons']['asm_vs_manual_jacobi']=compare(fields['manual_asm'],fields['manual_jacobi'],config,context,fd)
    require(result['comparisons']['asm_vs_manual_jacobi']['status']=='passed','Matched-path field check failed.')
    curve=next(c for c in json.loads((ROOT/review['measurement_records']['serial']['path']).read_text())['curve'] if c['case_id']=='Radau_coarse')
    for label in ('manual_jacobi','manual_asm'):
        transfer={};comparison=result['comparisons'][label+'_vs_irksome']
        for metric in METRICS:
            E=curve['time_errors'][metric];D=comparison['norms'][metric]['actual_field_difference']
            allowed=config['checks']['field_control_atol']+config['checks']['time_error_transfer_fraction']*E
            transfer[metric]=dict(accepted_baseline_time_error=E,actual_field_distance=D,allowed_distance=allowed)
            require(D<=allowed,'PC trajectory difference is too large for the temporal error budget.')
        records[label]['accepted_time_error_budget_check']=transfer
    result.pop('active_trajectory',None)
    result.update(status='passed',completed_study=True,completed_trajectories=4,completed_steps=40)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();sys.argv=[sys.argv[0]]
    result=dict(status='failed',completed_study=False,ranks=1,trajectories={},comparisons={},performance_measurement=False)
    allowed=args.output.resolve().parent==ROOT/'results/raw'
    try:
        require(allowed,'Keep records in the Projects repository.')
        config=json.loads(args.config.read_text());result['config']=config
        from mpi4py import MPI
        import firedrake as fd
        from firedrake.petsc import PETSc
        import irksome as irk
        from irksome.tools import AI
        PETSc.Log.begin()
        study(config,result,fd,PETSc,irk,AI,MPI.COMM_WORLD)
    except (Exception,KeyboardInterrupt) as error:
        result.update(status='failed',error=str(error) or 'Interrupted by researcher')
        if result.get('active_trajectory'):result['trajectories'][result['active_trajectory']]['status']='failed'
        traceback.print_exc()
    finally:
        if allowed:args.output.write_text(json.dumps(sanitize(result),indent=2,allow_nan=False)+'\n')
    return 0 if result['status']=='passed' else 1

if __name__=='__main__':
    sys.exit(main())

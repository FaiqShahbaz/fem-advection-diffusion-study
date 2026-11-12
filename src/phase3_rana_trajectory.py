#!/usr/bin/env python3
"""Researcher-run full Radau trajectories with installed RanaLD and fixed auxiliary LU."""
import argparse
import json
import math
from pathlib import Path
import sys
import traceback

from phase0_be_smoke import require
from phase0_irksome_be import true_residual
from phase0_radau_smoke import tableau_coefficients
from phase1_spatial import run_trajectory,step_count,safe_sqrt,error_metrics,check_metrics,capture_failed_solve,sanitize
from phase2_temporal import radau_config,METRICS
from phase3_asm_construction import compare
from phase3_asm_trajectory import make_context,manual_trajectory,event_counts,count_difference,EVENTS as CORE_EVENTS
from phase3_rana_construction import validate_config as validate_construction,exact_factors,matrix_comparison,auxiliary_action

ROOT=Path(__file__).resolve().parents[1]
EVENTS=CORE_EVENTS+('MatSolve','MatLUFactorSym','MatLUFactorNum')


def validate_config(config):
    construction=json.loads((ROOT/'configs/phase3_rana_construction.json').read_text())
    validate_construction(construction)
    require(all(config[k]==v for k,v in construction.items() if k not in ('purpose','checkpoint_steps')),
        'Keep the accepted Rana construction problem, PC and thresholds.')
    require(config['checkpoint_steps']==step_count(config['final_time'],config['dt'])==10
        and config['trajectory_roles']==['irksome_jacobi','manual_jacobi_reference','residual_jacobi','rana_ld']
        and config['performance_measurement'] is False,'Keep these four serial verification trajectories.')


def residual_parameters(config,label):
    if label=='rana_ld':return dict(config['rana_solver_parameters'])
    require(label=='residual_jacobi','Unknown matched residual trajectory.')
    return {**config['solver_parameters'],'mat_type':'aij','snes_type':'ksponly'}


def matrix_token(matrix):
    return dict(object_id=int(matrix.getId()),state=int(matrix.stateGet()))


def irksome_trajectory(config,context,label,record,references,fd,PETSc,irk,AI):
    V=context['space'];dt_value=config['dt'];count=step_count(config['final_time'],dt_value)
    A,b,c=tableau_coefficients(radau_config(config));At=exact_factors(config)
    state=fd.Function(V,name=label+'_state').assign(context['initial']);old=fd.Function(V,name=label+'_old')
    t,dt=fd.Constant(0.),fd.Constant(dt_value);v=fd.TestFunction(V)
    spatial,forcing=context['spatial'],context['forcing'];parameters=residual_parameters(config,label)
    F=(irk.Dt(state)*v+spatial(state,v)-forcing(t)*v)*context['dx']
    argument_count=len(F.arguments());require(argument_count==1,'Installed Rana requires a one-test-argument residual.')
    record.update(status='running',method='RadauIIA',dt=dt_value,planned_steps=count,completed_steps=0,
        solver_parameters=parameters,stage_total_dofs=2178,step_records=[],performance_measurement=False,
        weak_form_argument_count=argument_count,weak_form_representation='state_function_residual',
        physical_stage_initial_guess='zeroed before each advance',constant_jacobian=True,
        reuse_policy='One TimeStepper per trajectory; unchanged physical/auxiliary matrices and no new PC setup after first step')
    stage=PETSc.Log.Stage('phase3_rana_trajectory_'+label);events={name:PETSc.Log.Event(name) for name in EVENTS}
    with stage:
        start=event_counts(events,stage)
        stepper=irk.TimeStepper(F,irk.RadauIIA(2),t,dt,state,bcs=[context['bc']],stage_type='deriv',splitting=AI,
            bc_type='DAE',constant_jacobian=True,solver_parameters=parameters,options_prefix='phase3_rana_trajectory_'+label)
        record['construction_event_counts']=count_difference(start,event_counts(events,stage))
        ksp=stepper.solver.snes.ksp;pc=ksp.getPC();history=[]
        ksp.setMonitor(lambda solver,iteration,norm:history.append([iteration,float(norm)]))
        W=stepper.stages.function_space();require(W.dim()==2178,'Unexpected stage dimension.')
        z,q=fd.TrialFunctions(W),fd.TestFunctions(W)
        bcs=[fd.DirichletBC(W.sub(i),0.,'on_boundary') for i in range(2)]
        block_form=sum(z[i]*q[i]+dt_value*sum(A[i][j]*spatial(z[j],q[i]) for j in range(2)) for i in range(2))*context['dx']
        auxiliary_form=sum(z[i]*q[i]+dt_value*sum(At[i][j]*spatial(z[j],q[i]) for j in range(2)) for i in range(2))*context['dx']
        load_form=sum(forcing(t+c[i]*dt_value)*q[i]-spatial(old,q[i]) for i in range(2))*context['dx']
        block=fd.assemble(block_form,bcs=bcs,mat_type='aij')
        auxiliary=fd.assemble(auxiliary_form,bcs=bcs,mat_type='aij') if label=='rana_ld' else None
        record['independent_check_matrix_assemblies']=dict(physical_B=1,auxiliary_P=int(auxiliary is not None))
        first_tokens=None;auxctx=None
        for index in range(count):
            old_time=index*dt_value;t.assign(old_time);old.assign(state);history.clear()
            for slope in stepper.stages.subfunctions:slope.assign(0.)
            before=event_counts(events,stage)
            try:stepper.advance()
            except Exception as error:
                if ksp.getIterationNumber()==0 and int(ksp.getConvergedReason())==0:
                    record.update(status='failed',failed_step=dict(step=index+1,t_start=old_time,dt=dt_value,
                        failure_phase='setup_before_first_GMRES_iteration',iterations=0,converged_reason=0,
                        reported_residual_history=list(history),step_committed=False,error=str(error),
                        residual_diagnostic='Unavailable before Krylov initialization; no physical residual or iterate accepted'))
                else:capture_failed_solve(record,index,old_time,dt_value,ksp,history,error)
                raise
            counts=count_difference(before,event_counts(events,stage))
            require(abs(float(t)-old_time)<=1e-14 and float(dt)==dt_value,'Caller clock or dt changed during advance.')
            load=fd.assemble(load_form,bcs=bcs)
            residual=true_residual(block,stepper.stages,load);rtol,atol,divtol,max_it=ksp.getTolerances()
            target=max(atol,rtol*residual['rhs_norm'])
            require(ksp.getType()=='gmres' and pc.getType()==parameters['pc_type'] and stepper.solver.snes.getType()=='ksponly'
                and ksp.getPCSide()==PETSc.PC.Side.RIGHT and ksp.getNormType()==PETSc.KSP.NormType.UNPRECONDITIONED
                and not ksp.getInitialGuessNonzero() and rtol==parameters['ksp_rtol'] and atol==parameters['ksp_atol'] and max_it==1000,
                'Actual residual-form solver changed.')
            slopes=stepper.stages.subfunctions;l2=lambda expr,measure=context['dx']:safe_sqrt(float(fd.assemble(expr**2*measure)))
            row=dict(step=index+1,t_start=old_time,t_end=(index+1)*dt_value,stage_times=[old_time+ci*dt_value for ci in c],
                iterations=ksp.getIterationNumber(),converged_reason=int(ksp.getConvergedReason()),reported_residual_norm=float(ksp.getResidualNorm()),
                true_block_residual=residual,true_residual_target=target,event_counts=counts,
                reported_residual_history=list(history),
                update_identity_l2=l2(state-(old+dt_value*sum(b[j]*slopes[j] for j in range(2)))),
                stiff_accuracy_l2=l2(state-(old+dt_value*sum(A[1][j]*slopes[j] for j in range(2)))),
                stage_boundary_trace_l2=[l2(slope,context['ds']) for slope in slopes])
            record['step_records'].append(row)
            require(row['converged_reason']>0 and all(math.isfinite(value) for value in residual.values())
                and residual['norm']<=config['checks']['residual_tolerance_factor']*target,'Independent physical Radau residual failed.')
            require(row['update_identity_l2']<=1e-12 and row['stiff_accuracy_l2']<=1e-12
                and all(value<=1e-12 for value in row['stage_boundary_trace_l2']),'Stage/update boundary check failed.')
            row['matrix_checks']=dict(physical_B=matrix_comparison(ksp.getOperators()[0],block.petscmat,config['checks']))
            tokens=dict(physical_B=matrix_token(ksp.getOperators()[0]))
            require(all(abs(float(stepper.butcher_tableau.A[i,j])-A[i][j])<=1e-14 for i in range(2) for j in range(2)),
                'Physical Radau tableau changed.')
            if label=='rana_ld':
                auxctx=pc.getPythonContext();require(isinstance(auxctx,irk.RanaLD),'Installed RanaLD context changed.')
                require(auxctx.pc.getType()=='lu' and auxctx.pc.getFactorSolverType()=='petsc','Fixed auxiliary LU changed.')
                actual_at=auxctx.getAtilde(stepper.butcher_tableau.A)
                require(all(abs(float(actual_at[i,j])-At[i][j])<=1e-14 for i in range(2) for j in range(2)),'Auxiliary tableau changed.')
                row['matrix_checks'].update(auxiliary_P=matrix_comparison(auxctx.P.petscmat,auxiliary.petscmat,config['checks']),
                    auxiliary_PC_Pmat=matrix_comparison(auxctx.pc.getOperators()[1],auxiliary.petscmat,config['checks']))
                tokens.update(auxiliary_P=matrix_token(auxctx.P.petscmat),auxiliary_factor=matrix_token(auxctx.pc.getFactorMatrix()))
                require(counts['MatSolve']==counts['PCApply'] and counts['MatSolve']>0,
                    'Actual auxiliary LU action count differs from outer PC applications.')
                row['auxiliary_factor_nonzeros']=float(auxctx.pc.getFactorMatrix().getInfo()['nz_used'])
            require(all(check['status']=='passed' for check in row['matrix_checks'].values()),'Physical/auxiliary matrix comparison failed.')
            row['matrix_reuse_tokens']=tokens
            if first_tokens is None:first_tokens=tokens
            row['reuse_matches_first_step']=tokens==first_tokens
            require(row['reuse_matches_first_step'] and (index==0 or all(counts[event]==0 for event in ('PCSetUp','MatLUFactorSym','MatLUFactorNum'))),
                'Matrix/factor state or PC setup changed after the first step; preserve and review before timing.')
            row['source_checks']=[]
            for value in (old_time,(index+1)*dt_value,*row['stage_times']):
                exact=fd.exp(-value)*context['sinusoid']
                defect=l2(-exact+fd.dot(context['beta'],fd.grad(exact))-config['diffusion']*fd.div(fd.grad(exact))-forcing(value))
                row['source_checks'].append(dict(time=value,source_residual_l2=defect))
            require(all(s['source_residual_l2']<=1e-10 for s in row['source_checks']),'Manufactured source check failed.')
            ref_state,ref_slopes=references[index]
            row['reference_comparisons']=dict(update=compare(state,ref_state,config,context,fd))
            for i in range(2):row['reference_comparisons']['slope'+str(i+1)]=compare(slopes[i],ref_slopes.subfunctions[i],config,context,fd)
            require(all(r['status']=='passed' for r in row['reference_comparisons'].values()),'Actual per-step field agreement failed.')
            record['completed_steps']=index+1
        record['operator_event_counts']=count_difference(start,event_counts(events,stage))
        require(record['operator_event_counts']['KSPSolve']==count and record['operator_event_counts']['PCSetUp']>=1,
            'Missing trajectory solve/setup evidence.')
        if label=='rana_ld':
            require(all(record['operator_event_counts'][event]>=1 for event in ('MatLUFactorSym','MatLUFactorNum')),'Missing auxiliary LU factorization evidence.')
            before=event_counts(events,stage)
            record['auxiliary_action_check']=auxiliary_action(pc,auxiliary.petscmat,config['checks'])
            record['diagnostic_event_counts']=count_difference(before,event_counts(events,stage))
            require(record['auxiliary_action_check']['status']=='passed' and record['diagnostic_event_counts']['MatSolve']==1
                and all(record['diagnostic_event_counts'][event]==0 for event in ('KSPSolve','PCSetUp','MatLUFactorSym','MatLUFactorNum')),
                'Final inverse probe failed or unexpectedly rebuilt its factors.')
            record['auxiliary_solver']=dict(pc_type=auxctx.pc.getType(),factor_solver_type=auxctx.pc.getFactorSolverType(),
                factor_nonzeros=float(auxctx.pc.getFactorMatrix().getInfo()['nz_used']),inner_krylov_solve=False,
                interpretation='One fixed sparse LU of full auxiliary P; no scalable stage-subsolve or timing claim')
        ksp.view()
    primary=error_metrics(state,config['final_time'],context,config['quadrature_degree'],fd)
    control=error_metrics(state,config['final_time'],context,config['quadrature_control_degree'],fd)
    checks,quadrature=check_metrics(primary,control,config['final_time'],config['checks'])
    require(all(checks.values()),'Final manufactured-solution/quadrature check failed.')
    record.update(status='passed',final_time=config['final_time'],metrics=primary,control_metrics=control,checks=checks,
        quadrature_comparison=quadrature,solver_stats=list(stepper.solver_stats()),
        total_iterations=sum(s['iterations'] for s in record['step_records']),maximum_step_iterations=max(s['iterations'] for s in record['step_records']),
        actual_solver=dict(ksp_type=ksp.getType(),pc_type=pc.getType(),snes_type=stepper.solver.snes.getType(),side='right',norm='unpreconditioned',
            rtol=rtol,atol=atol,divtol=divtol,max_it=max_it,requested_restart=50,restart_evidence='Actual KSP view in transcript'))
    if label=='rana_ld':record['actual_solver']['python_context']=type(auxctx).__module__+'.'+type(auxctx).__name__
    require(record['solver_stats']==[count,count,record['total_iterations']],'Unexpected SNES/KSP work counters.')
    print(f"  {label}: {count} steps checked; total outer GMRES iterations={record['total_iterations']}",flush=True)
    return state


def study(config,result,fd,PETSc,irk,AI,comm):
    require(comm.size==1,'Serial trajectory verification required.');validate_config(config)
    context=make_context(config,fd,comm);require(context['space'].dim()==1089,'Unexpected spatial dimension.')
    result.update(total_dofs=1089,stage_total_dofs=2178)
    records=result['trajectories'];fields={}
    baseline=dict(status='prepared');records['irksome_jacobi']=baseline;result['active_trajectory']='irksome_jacobi'
    fields['irksome_jacobi']=run_trajectory(radau_config(config),context,config['dt'],1.,'rana_trajectory_original_jacobi',baseline,fd,PETSc,irk,AI)
    accepted_review=json.loads((ROOT/'results/processed/asm_trajectory_review.json').read_text())
    accepted=json.loads((ROOT/accepted_review['measurement_records']['serial']['path']).read_text())['trajectories']['irksome_jacobi']
    result['baseline_reproduction']={metric:math.isclose(baseline['control_metrics'][metric],accepted['control_metrics'][metric],rel_tol=1e-9,abs_tol=1e-12) for metric in METRICS}
    result['baseline_iteration_sequence_matches']=[s['iterations'] for s in baseline['step_records']]==[s['iterations'] for s in accepted['step_records']]
    require(all(result['baseline_reproduction'].values()) and result['baseline_iteration_sequence_matches'],'Accepted baseline reproduction failed.')
    parameters=dict(config['solver_parameters'])
    for key in ('ksp_rtol','ksp_atol'):parameters[key]*=config['manual_reference_tolerance_factor']
    reference=dict(status='prepared');records['manual_jacobi_reference']=reference;result['active_trajectory']='manual_jacobi_reference'
    fields['manual_jacobi_reference'],references=manual_trajectory(config,context,parameters,'rana_trajectory_reference',reference,fd,PETSc)
    for label in ('residual_jacobi','rana_ld'):
        row=dict(status='prepared');records[label]=row;result['active_trajectory']=label
        fields[label]=irksome_trajectory(config,context,label,row,references,fd,PETSc,irk,AI)
    for label in ('manual_jacobi_reference','residual_jacobi','rana_ld'):
        result['comparisons'][label+'_vs_original_irksome']=compare(fields[label],fields['irksome_jacobi'],config,context,fd)
    result['comparisons']['rana_vs_residual_jacobi']=compare(fields['rana_ld'],fields['residual_jacobi'],config,context,fd)
    require(all(r['status']=='passed' for r in result['comparisons'].values()),'Final actual field agreement failed.')
    cost_review=json.loads((ROOT/'results/processed/accuracy_cost_review.json').read_text())
    curve=next(c for c in json.loads((ROOT/cost_review['measurement_records']['serial']['path']).read_text())['curve'] if c['case_id']=='Radau_coarse')
    for label in ('residual_jacobi','rana_ld'):
        row=records[label];row['accepted_time_error_budget_check']={}
        for metric in METRICS:
            E=curve['time_errors'][metric];D=result['comparisons'][label+'_vs_original_irksome']['norms'][metric]['actual_field_difference']
            allowed=config['checks']['field_control_atol']+config['checks']['time_error_transfer_fraction']*E
            row['accepted_time_error_budget_check'][metric]=dict(accepted_baseline_time_error=E,actual_field_distance=D,allowed_distance=allowed)
            require(D<=allowed,'Field distance exceeds the inherited same-mesh temporal-error budget.')
    result.pop('active_trajectory',None);result.update(status='passed',completed_study=True,completed_trajectories=4,completed_steps=40)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();sys.argv=[sys.argv[0]]
    result=dict(status='failed',completed_study=False,ranks=1,trajectories={},comparisons={},performance_measurement=False)
    allowed=args.output.resolve().parent==ROOT/'results/raw'
    try:
        require(allowed,'Keep all records in the Projects repository.');config=json.loads(args.config.read_text());result['config']=config
        from mpi4py import MPI
        import firedrake as fd
        from firedrake.petsc import PETSc
        import irksome as irk
        from irksome.tools import AI
        PETSc.Log.begin();study(config,result,fd,PETSc,irk,AI,MPI.COMM_WORLD)
    except (Exception,KeyboardInterrupt) as error:
        result.update(status='failed',error=str(error) or 'Interrupted by researcher')
        if result.get('active_trajectory'):result['trajectories'][result['active_trajectory']]['status']='failed'
        traceback.print_exc()
    finally:
        if allowed:args.output.write_text(json.dumps(sanitize(result),indent=2,allow_nan=False)+'\n')
    return 0 if result['status']=='passed' else 1

if __name__=='__main__':sys.exit(main())

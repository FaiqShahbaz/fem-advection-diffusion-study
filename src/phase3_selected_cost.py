#!/usr/bin/env python3
"""Researcher-run matched warm core timing at three selected verified parameter points."""
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
from phase1_spatial import step_count,safe_sqrt,error_metrics,check_metrics,capture_failed_solve,sanitize
from phase2_temporal import radau_config,METRICS
from phase3_asm_construction import compare,inspect_domains,additive_action_check
from phase3_asm_trajectory import make_context,event_counts,count_difference
from phase3_rana_construction import exact_factors,auxiliary_action
from phase3_rana_trajectory import EVENTS,matrix_token
from phase3_solver_sensitivity import point_metadata,matrix_comparison,parameters_for
from phase3_selected_trajectories import (validate_config as validate_trajectory,final_metrics,equation_checks,
    independent_forms,reference_trajectory,actual_solver,REBUILD_EVENTS)
from phase3_cost_pilot import event_snapshot
from phase3_rana_cost import profiling_difference,field_equivalence

ROOT=Path(__file__).resolve().parents[1]


def expected_orders():
    return [
        ['baseline_Jacobi','baseline_ASM','baseline_RanaLD','mesh_refined_ASM','mesh_refined_RanaLD','lower_diffusion_Jacobi','lower_diffusion_ASM','lower_diffusion_RanaLD'],
        ['mesh_refined_RanaLD','mesh_refined_ASM','lower_diffusion_ASM','lower_diffusion_RanaLD','lower_diffusion_Jacobi','baseline_ASM','baseline_RanaLD','baseline_Jacobi'],
        ['lower_diffusion_RanaLD','lower_diffusion_Jacobi','lower_diffusion_ASM','baseline_RanaLD','baseline_Jacobi','baseline_ASM','mesh_refined_ASM','mesh_refined_RanaLD']]


def validate_config(config):
    accepted=json.loads((ROOT/'configs/phase3_selected_trajectories.json').read_text());validate_trajectory(accepted)
    extra={'cases','warmups_per_case','repetitions_per_case','measured_case_order','required_power_source','timing_failure_policy'}
    require(set(config)==set(accepted)|extra and all(config[k]==v for k,v in accepted.items() if k not in ('purpose','performance_measurement')),
        'Keep the accepted selected points, physics, solver parameters, references and thresholds.')
    cases=[dict(case_id=p['point_id']+'_'+s,point_id=p['point_id'],solver=s) for p in accepted['points'] for s in accepted['solver_cases'] if not(p['point_id']=='mesh_refined' and s=='Jacobi')]
    require(config['cases']==cases and config['warmups_per_case']==1 and config['repetitions_per_case']==3
        and config['measured_case_order']==expected_orders() and config['performance_measurement'] is True and config['required_power_source']=='AC Power'
        and config['timing_failure_policy']=='Any fresh checked or timed failure stops the cost comparison; preserve records without accepting a cost ranking',
        'Keep eight accepted candidates, one warmup and three declared repeat orders; retain the pilot exclusion.')


def checked_trajectory(config,context,case,record,block,references,fd,PETSc,irk,AI):
    V=context['space'];dt_value=config['dt'];count=config['checkpoint_steps'];A,b,c=tableau_coefficients(radau_config(config));At=exact_factors(config)
    state=fd.Function(V,name=case+'_state').assign(context['initial']);old=fd.Function(V)
    t,dt=fd.Constant(0.),fd.Constant(dt_value);v=fd.TestFunction(V);spatial=context['spatial']
    F=(irk.Dt(state)*v+spatial(state,v)-context['forcing'](t)*v)*context['dx'];require(len(F.arguments())==1,'Expected one-test-argument residual.')
    parameters=parameters_for(config,case);prefix='phase3_selected_cost_checked_'+config['point_id']+'_'+case+'_'
    stage=PETSc.Log.Stage(prefix);events={name:PETSc.Log.Event(name) for name in EVENTS};history=[]
    record.update(status='running',solver_parameters=parameters,planned_steps=count,completed_steps=0,step_records=[],weak_form_argument_count=1,
        performance_measurement=False,constant_jacobian=True,physical_stage_initial_guess='Zeroed before every advance',
        reuse_policy='One TimeStepper per trajectory; fixed B, fixed PC/factors after first step; no cross-point reuse')
    with stage:
        start=event_counts(events,stage)
        stepper=irk.TimeStepper(F,irk.RadauIIA(2),t,dt,state,bcs=[context['bc']],stage_type='deriv',splitting=AI,bc_type='DAE',constant_jacobian=True,
            solver_parameters=parameters,options_prefix=prefix)
        record['construction_event_counts']=count_difference(start,event_counts(events,stage));ksp=stepper.solver.snes.ksp;pc=ksp.getPC()
        ksp.setMonitor(lambda solver,iteration,norm:history.append([iteration,float(norm)]))
        W=stepper.stages.function_space();require(W.dim()==2*V.dim() and block.petscmat.getSize()==(W.dim(),W.dim()),'Stage dimensions changed.')
        record['stage_total_dofs']=W.dim()
        _,load_form,bcs=independent_forms(config,context,W,old,t,fd);auxiliary=None;first_tokens=None;subksps=[]
        if case=='RanaLD':
            z,q=fd.TrialFunctions(W),fd.TestFunctions(W)
            form=sum(z[i]*q[i]+dt_value*sum(At[i][j]*spatial(z[j],q[i]) for j in range(2)) for i in range(2))*context['dx']
            auxiliary=fd.assemble(form,bcs=bcs,mat_type='aij')
        record['independent_check_matrix_assemblies']=dict(shared_physical_B=0,auxiliary_P=int(auxiliary is not None))
        physical_totals=dict(record['construction_event_counts'])
        for index in range(count):
            old_time=index*dt_value;t.assign(old_time);old.assign(state);history.clear()
            for slope in stepper.stages.subfunctions:slope.assign(0.)
            before=event_counts(events,stage)
            try:stepper.advance()
            except Exception as error:
                if not (int(ksp.getConvergedReason())==int(PETSc.KSP.ConvergedReason.DIVERGED_MAX_IT) and ksp.getIterationNumber()==parameters['ksp_max_it']):
                    raise
                after=event_counts(events,stage);counts=count_difference(before,after)
                for key in physical_totals:physical_totals[key]+=counts[key]
                record['actual_solver']=actual_solver(stepper,parameters,PETSc)
                record['failed_physical_matrix_check']=matrix_comparison(ksp.getOperators()[0],block.petscmat,config['checks'])
                require(record['failed_physical_matrix_check']['status']=='passed','Physical matrix failed on an uncommitted attempt.')
                if int(ksp.getConvergedReason())==int(PETSc.KSP.ConvergedReason.DIVERGED_MAX_IT) and ksp.getIterationNumber()==parameters['ksp_max_it']:
                    capture_failed_solve(record,index,old_time,dt_value,ksp,history,error)
                    record['failure_diagnostic_event_counts']=count_difference(after,event_counts(events,stage))
                    change=safe_sqrt(float(fd.assemble((state-old)**2*context['dx'])))
                    require(change<=1e-12,'Capped advance unexpectedly committed its physical update.')
                    record.update(status='iteration_limit',operator_event_counts=physical_totals,failed_advance_event_counts=counts,
                        step_committed=False,failed_step_state_change_l2=change,
                        total_iterations=sum(r['iterations'] for r in record['step_records'])+ksp.getIterationNumber(),
                        interpretation='Prior checked steps retained; failed step uncommitted; no T=1 field or performance result accepted')
                    ksp.view();print(f"  {config['point_id']} / {case}: capped at step {index+1}; prior steps={index}",flush=True);return
                raise
            counts=count_difference(before,event_counts(events,stage))
            for key in physical_totals:physical_totals[key]+=counts[key]
            observed=actual_solver(stepper,parameters,PETSc);record['actual_solver']=observed
            require(ksp.getConvergedReason()>0 and abs(float(t)-old_time)<=1e-14 and float(dt)==dt_value,'Candidate solve or caller clock failed.')
            # Its own old field is essential: the candidate and physical-LU reference have independent algebraic histories.
            load=fd.assemble(load_form,bcs=bcs)
            row=equation_checks(state,old,stepper.stages,block,load,index,config,context,observed['rtol'],observed['atol'],fd)
            row.update(iterations=ksp.getIterationNumber(),converged_reason=int(ksp.getConvergedReason()),reported_residual_history=list(history),
                reported_residual_norm=float(ksp.getResidualNorm()),event_counts=counts,
                matrix_checks=dict(physical_B=matrix_comparison(ksp.getOperators()[0],block.petscmat,config['checks'])))
            tokens=dict(physical_B=matrix_token(ksp.getOperators()[0]))
            require(all(abs(float(stepper.butcher_tableau.A[i,j])-A[i][j])<=1e-14 for i in range(2) for j in range(2)),'Physical tableau changed.')
            if case=='RanaLD':
                ctx=pc.getPythonContext();require(isinstance(ctx,irk.RanaLD) and ctx.pc.getType()=='lu' and ctx.pc.getFactorSolverType()=='petsc','Rana backend changed.')
                actual_at=ctx.getAtilde(stepper.butcher_tableau.A)
                require(all(abs(float(actual_at[i,j])-At[i][j])<=1e-14 for i in range(2) for j in range(2)),'Auxiliary tableau changed.')
                row['matrix_checks'].update(auxiliary_P=matrix_comparison(ctx.P.petscmat,auxiliary.petscmat,config['checks']),
                    auxiliary_PC_Pmat=matrix_comparison(ctx.pc.getOperators()[1],auxiliary.petscmat,config['checks']))
                tokens.update(auxiliary_P=matrix_token(ctx.P.petscmat),auxiliary_factor=matrix_token(ctx.pc.getFactorMatrix()))
                require(counts['MatSolve']==counts['PCApply'] and counts['MatSolve']>0,'Rana auxiliary work mismatch.')
            elif case=='ASM':
                if index==0:
                    record['domains']=inspect_domains(pc,ksp.getOperators()[0]);subksps=pc.getASMSubKSP();require(len(subksps)==4,'Expected four ASM subsolvers.')
                    record['local_solvers']=[dict(ksp_type=s.getType(),pc_type=s.getPC().getType(),factor_solver_type=s.getPC().getFactorSolverType(),
                        matrix_size=list(s.getOperators()[0].getSize()),factor_nonzeros=float(s.getPC().getFactorMatrix().getInfo()['nz_used'])) for s in subksps]
                require(all(s.getType()=='preonly' and s.getPC().getType()=='ilu' and s.getPC().getFactorSolverType()=='petsc' for s in subksps),'ASM local solver changed.')
                for i,s in enumerate(subksps):
                    tokens['local_matrix_'+str(i)]=matrix_token(s.getOperators()[0]);tokens['local_factor_'+str(i)]=matrix_token(s.getPC().getFactorMatrix())
                require(counts['MatSolve']==4*counts['PCApply'] and counts['PCApply']>0,'ASM actual local MatSolve count differs from fixed composition.')
                row['local_work_derived']=dict(per_domain=counts['PCApply'],total=4*counts['PCApply'],basis='Fixed four-domain PREONLY/ILU composition, corroborated by actual MatSolve; no per-domain callbacks')
            require(all(v['status']=='passed' for v in row['matrix_checks'].values()),'Full matrix coefficients changed.')
            if first_tokens is None:first_tokens=tokens
            require(tokens==first_tokens and (index==0 or all(counts[e]==0 for e in REBUILD_EVENTS)),'PC/matrix/factor rebuilt after the first step.')
            row.update(matrix_reuse_tokens=tokens,reuse_matches_first_step=True)
            ref_state,ref_slopes=references[index];row['reference_comparisons']=dict(update=compare(state,ref_state,config,context,fd))
            for i in range(2):row['reference_comparisons']['slope'+str(i+1)]=compare(stepper.stages.subfunctions[i],ref_slopes.subfunctions[i],config,context,fd)
            require(all(v['status']=='passed' for v in row['reference_comparisons'].values()),'Actual per-step field agreement failed.')
            record['step_records'].append(row);record['completed_steps']=index+1
        record['operator_event_counts']=physical_totals
        require(physical_totals['KSPSolve']==count and physical_totals['PCSetUp']>=1,'Missing physical solve/setup evidence.')
        if case=='RanaLD':
            require(all(physical_totals[e]==1 for e in ('MatLUFactorSym','MatLUFactorNum')),'Expected one auxiliary LU factorization per trajectory.')
            before=event_counts(events,stage);record['auxiliary_action_check']=auxiliary_action(pc,auxiliary.petscmat,config['checks'])
            record['diagnostic_event_counts']=count_difference(before,event_counts(events,stage))
            require(record['auxiliary_action_check']['status']=='passed' and record['diagnostic_event_counts']['PCApply']==record['diagnostic_event_counts']['MatSolve']==1
                and all(record['diagnostic_event_counts'][e]==0 for e in ('KSPSolve',*REBUILD_EVENTS)),'Final Rana probe failed or rebuilt factors.')
            record['auxiliary_solver']=dict(pc_type=ctx.pc.getType(),factor_solver_type=ctx.pc.getFactorSolverType(),
                factor_nonzeros=float(ctx.pc.getFactorMatrix().getInfo()['nz_used']),inner_krylov_solve=False)
        elif case=='ASM':
            require(physical_totals['MatLUFactorSym']==0 and physical_totals['MatLUFactorNum']==4,'Expected four local ILU numeric events in step one.')
            before=event_counts(events,stage)
            record['additive_action_check']=additive_action_check(pc,subksps,record['domains'],ksp.getOperators()[0],config['checks'],PETSc)
            record['diagnostic_event_counts']=count_difference(before,event_counts(events,stage))
            require(record['additive_action_check']['status']=='passed' and record['diagnostic_event_counts']['KSPSolve']==4
                and record['diagnostic_event_counts']['MatSolve']==8 and all(record['diagnostic_event_counts'][e]==0 for e in REBUILD_EVENTS),
                'Final ASM action probe failed or rebuilt factors.')
        record.update(status='passed',total_iterations=sum(r['iterations'] for r in record['step_records']),solver_stats=list(stepper.solver_stats()),
            maximum_step_iterations=max(r['iterations'] for r in record['step_records']),**final_metrics(state,config,context,fd))
        require(record['solver_stats']==[count,count,record['total_iterations']],'Unexpected Irksome solver statistics.')
        ksp.view()
    print(f"  {config['point_id']} / {case}: ten steps verified; outer iterations={record['total_iterations']}",flush=True)

    return state


def timed_trajectory(config,context,case,label,record,baseline,baseline_record,stage,events,fd,PETSc,irk,AI):
    dt_value=config['dt'];count=step_count(config['final_time'],dt_value);V=context['space']
    A,b,c=tableau_coefficients(radau_config(config));At=exact_factors(config)
    parameters=parameters_for(config,case['solver']);spatial,forcing=context['spatial'],context['forcing']
    record.update(status='running',method='RadauIIA',dt=dt_value,planned_steps=count,completed_steps=0,step_records=[],
        solver_parameters=parameters,stage_total_dofs=2*V.dim(),timed_python_monitors=False,timed_local_callbacks=False,
        weak_form_representation='state_function_residual',constant_jacobian=True,
        reuse_policy='Fresh TimeStepper per repetition; constant physical/auxiliary matrices and factors within ten steps')
    gc.collect()  # Deliberate collection is outside the wall-clock boundary.
    with stage:
        before=event_snapshot(events,stage);start=perf_counter()
        state=fd.Function(V,name=label+'_state').assign(context['initial']);old=fd.Function(V,name=label+'_old')
        t,dt=fd.Constant(0.),fd.Constant(dt_value);v=fd.TestFunction(V)
        F=(irk.Dt(state)*v+spatial(state,v)-forcing(t)*v)*context['dx']
        argument_count=len(F.arguments());require(argument_count==1,'Expected a one-test-argument residual.')
        stepper=irk.TimeStepper(F,irk.RadauIIA(2),t,dt,state,bcs=[context['bc']],stage_type='deriv',splitting=AI,
            bc_type='DAE',constant_jacobian=True,solver_parameters=parameters,options_prefix='phase3_selected_cost_'+label)
        ksp=stepper.solver.snes.ksp;components=stepper.stages.subfunctions
        construction_end=perf_counter()
        for index in range(count):
            step_start=perf_counter();old_time=index*dt_value;t.assign(old_time);old.assign(state)
            for slope in components:slope.assign(0.)
            try:stepper.advance()
            except Exception as error:
                if ksp.getIterationNumber()==0 and int(ksp.getConvergedReason())==0:
                    record.update(status='failed',failed_step=dict(step=index+1,t_start=old_time,dt=dt_value,
                        failure_phase='setup_before_first_GMRES_iteration',iterations=0,converged_reason=0,
                        step_committed=False,error=str(error),residual_diagnostic='Unavailable before Krylov initialization; no accepted physical residual'))
                else:capture_failed_solve(record,index,old_time,dt_value,ksp,[],error)
                raise
            iterations,reason=ksp.getIterationNumber(),int(ksp.getConvergedReason())
            require(reason>0,'Timed physical solve did not converge.')
            step_end=perf_counter()
            record['step_records'].append(dict(step=index+1,t_start=old_time,t_end=(index+1)*dt_value,
                stage_times=[old_time+ci*dt_value for ci in c],iterations=iterations,converged_reason=reason,wall_seconds=step_end-step_start))
            record['completed_steps']=index+1
        end=perf_counter()
        # All independent numerical diagnostics are outside the core timer and
        # outside the recorded physical PETSc event delta from this point on.
        after=event_snapshot(events,stage)
        wall=end-start;construction=construction_end-start;step_walls=[s['wall_seconds'] for s in record['step_records']]
        record['timing']=dict(construction_seconds=construction,first_step_seconds=step_walls[0],remaining_steps_seconds=sum(step_walls[1:]),
            trajectory_wall_seconds=wall,loop_bookkeeping_seconds=wall-construction-sum(step_walls),petsc_event_deltas=profiling_difference(before,after),
            interpretation='Construction plus ten advances; setup/assembly may occur lazily in the first advance; nested event times are inclusive, not additive')
        require(all(math.isfinite(v) and v>=0 for k,v in record['timing'].items() if k.endswith('_seconds')),'Invalid wall timing partition.')
        prof=record['timing']['petsc_event_deltas']
        require(all(prof[e]['count']==baseline_record['operator_event_counts'][e] for e in EVENTS),'Timed work/reuse differs from checked trajectory.')
        require([s['iterations'] for s in record['step_records']]==[s['iterations'] for s in baseline_record['step_records']],
            'Timed iteration sequence differs from the checked path.')
        require(abs(float(t)-(count-1)*dt_value)<=1e-14 and float(dt)==dt_value,'Last caller-owned clock or dt changed.')
        record['weak_form_argument_count']=argument_count;record['solver_stats']=list(stepper.solver_stats())
        require(record['solver_stats']==[count,count,sum(s['iterations'] for s in record['step_records'])],'SNES/KSP work counters differ.')
        rtol,atol,divtol,max_it=ksp.getTolerances();pc=ksp.getPC()
        require(ksp.getType()=='gmres' and pc.getType()==parameters['pc_type'] and stepper.solver.snes.getType()=='ksponly'
            and ksp.getPCSide()==PETSc.PC.Side.RIGHT and ksp.getNormType()==PETSc.KSP.NormType.UNPRECONDITIONED
            and not ksp.getInitialGuessNonzero() and rtol==parameters['ksp_rtol'] and atol==parameters['ksp_atol'] and max_it==1000,'Actual timed solver changed.')
        record['actual_solver']=dict(ksp_type=ksp.getType(),pc_type=pc.getType(),snes_type=stepper.solver.snes.getType(),side='right',norm='unpreconditioned',
            rtol=rtol,atol=atol,divtol=divtol,max_it=max_it,requested_restart=50,restart_evidence='Actual KSP view in transcript')
        ksp.view();W=stepper.stages.function_space();require(W.dim()==2*V.dim(),'Unexpected stage dimension.')
        z,q=fd.TrialFunctions(W),fd.TestFunctions(W);bcs=[fd.DirichletBC(W.sub(i),0.,'on_boundary') for i in range(2)]
        block_form=sum(z[i]*q[i]+dt_value*sum(A[i][j]*spatial(z[j],q[i]) for j in range(2)) for i in range(2))*context['dx']
        auxiliary_form=sum(z[i]*q[i]+dt_value*sum(At[i][j]*spatial(z[j],q[i]) for j in range(2)) for i in range(2))*context['dx']
        load_form=sum(forcing(t+c[i]*dt_value)*q[i]-spatial(old,q[i]) for i in range(2))*context['dx']
        block=fd.assemble(block_form,bcs=bcs,mat_type='aij');load=fd.assemble(load_form,bcs=bcs)
        residual=true_residual(block,stepper.stages,load);target=max(atol,rtol*residual['rhs_norm'])
        l2=lambda expr,measure=context['dx']:safe_sqrt(float(fd.assemble(expr**2*measure)))
        final=dict(status='passed',true_block_residual=residual,true_residual_target=target,stage_times=[float(t)+ci*dt_value for ci in c],
            update_identity_l2=l2(state-(old+dt_value*sum(b[j]*components[j] for j in range(2)))),
            stiff_accuracy_l2=l2(state-(old+dt_value*sum(A[1][j]*components[j] for j in range(2)))),
            stage_boundary_trace_l2=[l2(slope,context['ds']) for slope in components])
        record['final_equation_check']=final
        require(all(math.isfinite(v) for v in residual.values()) and residual['norm']<=10*target
            and final['update_identity_l2']<=1e-12 and final['stiff_accuracy_l2']<=1e-12
            and all(v<=1e-12 for v in final['stage_boundary_trace_l2']),'Timed final physical equation/update/boundary check failed.')
        record['matrix_checks']=dict(physical_B=matrix_comparison(ksp.getOperators()[0],block.petscmat,config['checks']))
        require(all(abs(float(stepper.butcher_tableau.A[i,j])-A[i][j])<=1e-14 for i in range(2) for j in range(2)),'Physical Radau tableau changed.')
        if case['solver']=='RanaLD':
            ctx=pc.getPythonContext();require(isinstance(ctx,irk.RanaLD) and ctx.pc.getType()=='lu' and ctx.pc.getFactorSolverType()=='petsc','Auxiliary Rana/LU changed.')
            actual_at=ctx.getAtilde(stepper.butcher_tableau.A)
            require(all(abs(float(actual_at[i,j])-At[i][j])<=1e-14 for i in range(2) for j in range(2)),'Auxiliary tableau changed.')
            P=fd.assemble(auxiliary_form,bcs=bcs,mat_type='aij')
            record['matrix_checks'].update(auxiliary_P=matrix_comparison(ctx.P.petscmat,P.petscmat,config['checks']),
                auxiliary_PC_Pmat=matrix_comparison(ctx.pc.getOperators()[1],P.petscmat,config['checks']))
            record['actual_solver']['python_context']=type(ctx).__module__+'.'+type(ctx).__name__
            record['auxiliary_solver']=dict(pc_type=ctx.pc.getType(),factor_solver_type=ctx.pc.getFactorSolverType(),
                factor_nonzeros=float(ctx.pc.getFactorMatrix().getInfo()['nz_used']),inner_krylov_solve=False)
            require(record['auxiliary_solver']['factor_nonzeros']==baseline_record['auxiliary_solver']['factor_nonzeros'],'Auxiliary factor structure changed.')
            diagnostic_before=event_snapshot(events,stage)
            record['auxiliary_action_check']=auxiliary_action(pc,P.petscmat,config['checks'])
            record['diagnostic_event_deltas']=profiling_difference(diagnostic_before,event_snapshot(events,stage))
            require(record['auxiliary_action_check']['status']=='passed' and record['diagnostic_event_deltas']['MatSolve']['count']==1
                and all(record['diagnostic_event_deltas'][e]['count']==0 for e in ('KSPSolve','PCSetUp','MatLUFactorSym','MatLUFactorNum')),
                'Untimed inverse probe failed or rebuilt its factors.')
        elif case['solver']=='ASM':
            domains=inspect_domains(pc,ksp.getOperators()[0]);subksps=pc.getASMSubKSP()
            require(domains==baseline_record['domains'] and len(subksps)==4,'Timed ASM graph domains changed.')
            local=[dict(ksp_type=s.getType(),pc_type=s.getPC().getType(),factor_solver_type=s.getPC().getFactorSolverType(),
                matrix_size=list(s.getOperators()[0].getSize()),factor_nonzeros=float(s.getPC().getFactorMatrix().getInfo()['nz_used'])) for s in subksps]
            require(local==baseline_record['local_solvers'],'Timed ASM local solver structure changed.')
            record.update(domains=domains,local_solvers=local,local_work_derived=dict(per_domain=prof['PCApply']['count'],total=4*prof['PCApply']['count'],
                basis='Fixed four-domain PREONLY/ILU composition, corroborated by physical MatSolve; no timed callbacks'))
            require(prof['MatSolve']['count']==record['local_work_derived']['total'],'Timed actual ASM local work mismatch.')
            diagnostic_before=event_snapshot(events,stage)
            record['additive_action_check']=additive_action_check(pc,subksps,domains,ksp.getOperators()[0],config['checks'],PETSc)
            record['diagnostic_event_deltas']=profiling_difference(diagnostic_before,event_snapshot(events,stage))
            require(record['additive_action_check']['status']=='passed' and record['diagnostic_event_deltas']['MatSolve']['count']==8
                and record['diagnostic_event_deltas']['KSPSolve']['count']==4 and all(record['diagnostic_event_deltas'][e]['count']==0 for e in REBUILD_EVENTS),
                'Untimed additive probe failed or rebuilt local factors.')
        require(all(r['status']=='passed' for r in record['matrix_checks'].values()),'Timed physical/auxiliary matrix check failed.')
    record['source_checks']=[]
    for value in (0.,1.,*final['stage_times']):
        exact=fd.exp(-value)*context['sinusoid']
        defect=l2(-exact+fd.dot(context['beta'],fd.grad(exact))-config['diffusion']*fd.div(fd.grad(exact))-forcing(value))
        record['source_checks'].append(dict(time=value,source_residual_l2=defect))
    require(all(s['source_residual_l2']<=1e-10 for s in record['source_checks']),'Timed source check failed.')
    primary=error_metrics(state,1.,context,config['quadrature_degree'],fd);control=error_metrics(state,1.,context,config['quadrature_control_degree'],fd)
    checks,quadrature=check_metrics(primary,control,1.,config['checks']);require(all(checks.values()),'Timed final MMS/quadrature check failed.')
    difference,equivalence=field_equivalence(state,baseline,config,context,fd)
    record.update(status='passed',final_time=1.,metrics=primary,control_metrics=control,checks=checks,quadrature_comparison=quadrature,
        difference_measurement=difference,equivalence=equivalence,total_iterations=sum(s['iterations'] for s in record['step_records']),
        maximum_step_iterations=max(s['iterations'] for s in record['step_records']))
    print(f"  {label}: checked core wall={wall:.6f} s; GMRES total={record['total_iterations']}",flush=True)


def study(config,result,fd,PETSc,irk,AI,comm):
    require(comm.size==1,'Keep the selected timing study serial.');validate_config(config)
    review=json.loads((ROOT/'results/processed/selected_trajectories_review.json').read_text())
    accepted=json.loads((ROOT/review['measurement_records']['serial']['path']).read_text())
    cfg0=accepted['config'];contexts={};fields={}
    checked=dict(status='running',completed_study=False,ranks=1,points={},performance_measurement=False,config=cfg0)
    result['checked_study']=checked
    for point in cfg0['points']:
        name=point['point_id'];cfg={**cfg0,**point};context=make_context(cfg,fd,comm);contexts[name]=context
        meta=point_metadata(point,cfg0);meta['checkpoint_end_time']=cfg0['final_time'];require(context['space'].dim()==meta['total_dofs'],'Spatial dimension changed.')
        row=dict(status='running',metadata=meta,reference={},candidates={});checked['points'][name]=row;result['active_trajectory']=name+'/reference'
        block,references=reference_trajectory(cfg,context,row['reference'],fd,PETSc)
        for solver in cfg0['solver_cases']:
            if name=='mesh_refined' and solver=='Jacobi':
                row['candidates'][solver]=dict(accepted['points'][name]['candidates'][solver]);continue
            case=next(c for c in config['cases'] if c['point_id']==name and c['solver']==solver)
            candidate=dict(status='prepared');row['candidates'][solver]=candidate;result['active_trajectory']=case['case_id']+'/verified'
            fields[case['case_id']]=checked_trajectory(cfg,context,solver,candidate,block,references,fd,PETSc,irk,AI)
            require(candidate['status']=='passed' and fields[case['case_id']] is not None,'A previously accepted candidate did not complete; no cost comparison accepted.')
            previous=accepted['points'][name]['candidates'][solver]
            require([s['iterations'] for s in candidate['step_records']]==[s['iterations'] for s in previous['step_records']]
                and candidate['operator_event_counts']==previous['operator_event_counts'],'Accepted work/reuse changed; preserve for review.')
            candidate['accepted_metric_reproduction']={m:math.isclose(candidate['control_metrics'][m],previous['control_metrics'][m],rel_tol=1e-9,abs_tol=1e-12) for m in METRICS}
            require(all(candidate['accepted_metric_reproduction'].values()),'Accepted final norm reproduction failed.')
        row['status']='collected'
    checked.update(status='collected',completed_study=True,reference_trajectories=3,reference_completed_steps=30,candidate_attempts=8,
        completed_candidate_trajectories=8,capped_candidate_trajectories=0,predeclared_skipped_candidates=1,candidate_completed_steps=80)
    stage=PETSc.Log.Stage('phase3_selected_cost_core');events={name:PETSc.Log.Event(name) for name in EVENTS}
    sequence=[(c['case_id'],'warmup0') for c in config['cases']]
    sequence += [(case_id,f'measured{i}') for i,ids in enumerate(config['measured_case_order'],1) for case_id in ids]
    for case_id,suffix in sequence:
        case=next(c for c in config['cases'] if c['case_id']==case_id);label=case_id+'_'+suffix;point=next(p for p in cfg0['points'] if p['point_id']==case['point_id'])
        cfg={**config,**point};record=dict(status='prepared',role='warmup' if suffix=='warmup0' else 'measured',case_id=case_id,point_id=case['point_id'],solver=case['solver'])
        result['trajectories'][label]=record;result['active_trajectory']=label;result['execution_order'].append(label)
        baseline_record=checked['points'][case['point_id']]['candidates'][case['solver']]
        timed_trajectory(cfg,contexts[case['point_id']],case,label,record,fields[case_id],baseline_record,stage,events,fd,PETSc,irk,AI)
    for case in config['cases']:
        rows=[result['trajectories'][case['case_id']+'_measured'+str(i)] for i in (1,2,3)];summary={}
        for key in ('construction_seconds','first_step_seconds','remaining_steps_seconds','trajectory_wall_seconds'):
            values=[r['timing'][key] for r in rows];summary[key]=dict(values=values,minimum=min(values),median=statistics.median(values),maximum=max(values))
        result['timing_summary'][case['case_id']]=summary
    result.pop('active_trajectory',None);result.update(status='passed',completed_study=True,completed_trajectories=43,completed_steps=430,
        predeclared_exclusions=cfg0['planned_skips'],interpretation='Compare matched core costs within each selected point; no elapsed cost accepted for excluded finer-mesh Jacobi')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();sys.argv=[sys.argv[0]]
    result=dict(status='failed',completed_study=False,ranks=1,trajectories={},execution_order=[],timing_summary={},performance_measurement=True)
    allowed=args.output.resolve().parent==ROOT/'results/raw'
    try:
        require(allowed,'Keep records in the Projects repository.');config=json.loads(args.config.read_text());result['config']=config
        from mpi4py import MPI
        import firedrake as fd
        from firedrake.petsc import PETSc
        import irksome as irk
        from irksome.tools import AI
        PETSc.Log.begin();study(config,result,fd,PETSc,irk,AI,MPI.COMM_WORLD)
    except (Exception,KeyboardInterrupt) as error:
        result.update(status='failed',error=str(error) or 'Interrupted by researcher')
        if result.get('active_trajectory') in result['trajectories']:result['trajectories'][result['active_trajectory']]['status']='failed'
        traceback.print_exc()
    finally:
        if allowed:args.output.write_text(json.dumps(sanitize(result),indent=2,allow_nan=False)+'\n')
    return 0 if result['status']=='passed' else 1

if __name__=='__main__':sys.exit(main())

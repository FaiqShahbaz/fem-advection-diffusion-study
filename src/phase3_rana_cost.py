#!/usr/bin/env python3
"""Researcher-run matched residual-form Radau Jacobi/RanaLD warm timing and verification."""
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
from phase1_spatial import step_count,safe_sqrt,norm_pair,error_metrics,check_metrics,capture_failed_solve,sanitize
from phase2_temporal import radau_config,measured_difference,METRICS
from phase3_asm_trajectory import make_context,manual_trajectory
from phase3_rana_construction import exact_factors,matrix_comparison,auxiliary_action
from phase3_rana_trajectory import validate_config as validate_trajectory,irksome_trajectory,residual_parameters,EVENTS
from phase3_cost_pilot import event_snapshot

ROOT=Path(__file__).resolve().parents[1]


def validate_config(config):
    accepted=json.loads((ROOT/'configs/phase3_rana_trajectory.json').read_text());validate_trajectory(accepted)
    require(all(config[k]==v for k,v in accepted.items() if k not in ('purpose','performance_measurement','trajectory_roles')),
        'Keep the accepted fixed problem, references, PCs and all numerical thresholds.')
    require(config['cases']==[dict(case_id='Jacobi',accepted_role='residual_jacobi'),dict(case_id='RanaLD',accepted_role='rana_ld')]
        and config['warmups_per_case']==1 and config['repetitions_per_case']==3
        and config['measured_case_order']==[['Jacobi','RanaLD'],['RanaLD','Jacobi'],['Jacobi','RanaLD']]
        and config['performance_measurement'] is True and config['required_power_source']=='AC Power',
        'Keep the bounded matched two-PC, three-repeat warm timing comparison.')
    require(step_count(config['final_time'],config['dt'])==10,'Keep ten steps per trajectory.')


def profiling_difference(before,after):
    # The accepted pilot's event_difference intentionally covers only four
    # events. Extend locally, preserving that accepted helper unchanged.
    result={name:{key:after[name][key]-before[name][key] for key in ('count','time','flops')} for name in EVENTS}
    require(all(math.isfinite(v) and v>=0 for row in result.values() for v in row.values()),'Invalid profiling delta.')
    return result


def field_equivalence(state,baseline,config,context,fd):
    difference=measured_difference(state,baseline,config,context,fd)
    norms=norm_pair(baseline,fd,fd.dx(domain=context['mesh'],metadata={'quadrature_degree':config['quadrature_control_degree']}))
    rows={m:dict(actual_field_difference=difference['metrics'][m],baseline_field_norm=norms[m],
        allowed_difference=config['checks']['equivalence_atol']+config['checks']['equivalence_rtol']*norms[m]) for m in METRICS}
    require(difference['status']=='passed' and all(r['actual_field_difference']<=r['allowed_difference'] for r in rows.values()),
        'Actual fields differ beyond the declared equivalence budget.')
    return difference,rows


def timed_trajectory(config,context,case,label,record,baseline,baseline_record,stage,events,fd,PETSc,irk,AI):
    dt_value=config['dt'];count=step_count(config['final_time'],dt_value);V=context['space']
    A,b,c=tableau_coefficients(radau_config(config));At=exact_factors(config)
    parameters=residual_parameters(config,case['accepted_role']);spatial,forcing=context['spatial'],context['forcing']
    record.update(status='running',method='RadauIIA',dt=dt_value,planned_steps=count,completed_steps=0,step_records=[],
        solver_parameters=parameters,stage_total_dofs=2178,timed_python_monitors=False,timed_local_callbacks=False,
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
            bc_type='DAE',constant_jacobian=True,solver_parameters=parameters,options_prefix='phase3_rana_cost_'+label)
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
        ksp.view();W=stepper.stages.function_space();require(W.dim()==2178,'Unexpected stage dimension.')
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
        if case['accepted_role']=='rana_ld':
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
    require(comm.size==1,'Keep this serial timing comparison.');validate_config(config)
    context=make_context(config,fd,comm);require(context['space'].dim()==1089,'Unexpected spatial space.')
    result.update(total_dofs=1089,stage_total_dofs=2178)
    review=json.loads((ROOT/'results/processed/rana_trajectory_review.json').read_text())
    accepted=json.loads((ROOT/review['measurement_records']['serial']['path']).read_text())['trajectories']
    records=result['trajectories'];fields={}
    params=dict(config['solver_parameters'])
    for key in ('ksp_rtol','ksp_atol'):params[key]*=config['manual_reference_tolerance_factor']
    ref=dict(status='prepared',role='reference');records['manual_jacobi_reference']=ref;result['active_trajectory']='manual_jacobi_reference'
    _,references=manual_trajectory(config,context,params,'rana_cost_reference',ref,fd,PETSc)
    require([s['iterations'] for s in ref['step_records']]==[s['iterations'] for s in accepted['manual_jacobi_reference']['step_records']],
        'Tighter reference work changed; preserve for review.')
    for case in config['cases']:
        label=case['case_id']+'_verified';row=dict(status='prepared',role='verified',case_id=case['case_id']);records[label]=row;result['active_trajectory']=label
        fields[case['case_id']]=irksome_trajectory(config,context,case['accepted_role'],row,references,fd,PETSc,irk,AI)
        previous=accepted[case['accepted_role']]
        require([s['iterations'] for s in row['step_records']]==[s['iterations'] for s in previous['step_records']]
            and row['operator_event_counts']==previous['operator_event_counts'],'Accepted work/reuse sequence changed.')
        row['accepted_metric_reproduction']={m:math.isclose(row['control_metrics'][m],previous['control_metrics'][m],rel_tol=1e-9,abs_tol=1e-12) for m in METRICS}
        require(all(row['accepted_metric_reproduction'].values()),'Accepted norm reproduction failed.')
    result['verified_pair_difference'],result['verified_pair_equivalence']=field_equivalence(fields['RanaLD'],fields['Jacobi'],config,context,fd)
    stage=PETSc.Log.Stage('phase3_rana_cost_core');events={name:PETSc.Log.Event(name) for name in EVENTS}
    sequence=[(case['case_id'],'warmup0') for case in config['cases']]
    sequence += [(case_id,f'measured{i}') for i,ids in enumerate(config['measured_case_order'],1) for case_id in ids]
    for case_id,suffix in sequence:
        case=next(c for c in config['cases'] if c['case_id']==case_id);label=case_id+'_'+suffix
        row=dict(status='prepared',role='warmup' if suffix=='warmup0' else 'measured',case_id=case_id)
        records[label]=row;result['active_trajectory']=label;result['execution_order'].append(label)
        timed_trajectory(config,context,case,label,row,fields[case_id],records[case_id+'_verified'],stage,events,fd,PETSc,irk,AI)
    for case in config['cases']:
        rows=[records[case['case_id']+'_measured'+str(i)] for i in (1,2,3)];summary={}
        for key in ('construction_seconds','first_step_seconds','remaining_steps_seconds','trajectory_wall_seconds'):
            values=[r['timing'][key] for r in rows];summary[key]=dict(values=values,minimum=min(values),median=statistics.median(values),maximum=max(values))
        result['timing_summary'][case['case_id']]=summary
    result['jacobi_over_rana_median_core_ratio']=result['timing_summary']['Jacobi']['trajectory_wall_seconds']['median']/result['timing_summary']['RanaLD']['trajectory_wall_seconds']['median']
    result.pop('active_trajectory',None);result.update(status='passed',completed_study=True,completed_trajectories=11,completed_steps=110)


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
        if result.get('active_trajectory'):result['trajectories'][result['active_trajectory']]['status']='failed'
        traceback.print_exc()
    finally:
        if allowed:args.output.write_text(json.dumps(sanitize(result),indent=2,allow_nan=False)+'\n')
    return 0 if result['status']=='passed' else 1

if __name__=='__main__':sys.exit(main())

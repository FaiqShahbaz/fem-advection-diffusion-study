#!/usr/bin/env python3
"""Researcher-run selected full trajectories; retain predeclared skips and any new iteration limits."""
import argparse
import json
import math
from pathlib import Path
import sys
import traceback

from phase0_be_smoke import require
from phase0_irksome_be import true_residual
from phase0_radau_smoke import tableau_coefficients
from phase1_spatial import step_count,safe_sqrt,error_metrics,check_metrics,capture_failed_solve,sanitize
from phase2_temporal import radau_config
from phase3_asm_construction import compare,inspect_domains,additive_action_check
from phase3_asm_trajectory import make_context,event_counts,count_difference
from phase3_rana_construction import exact_factors,auxiliary_action
from phase3_rana_trajectory import EVENTS,matrix_token
from phase3_solver_sensitivity import point_metadata,matrix_comparison,parameters_for,validate_config as validate_pilot

ROOT=Path(__file__).resolve().parents[1]
REBUILD_EVENTS=('PCSetUp','MatLUFactorSym','MatLUFactorNum')


def validate_config(config):
    pilot=json.loads((ROOT/'configs/phase3_solver_sensitivity.json').read_text());validate_pilot(pilot)
    require(set(config)==set(pilot)|{'planned_skips'} and all(config[k]==v for k,v in pilot.items() if k not in ('purpose','checkpoint_steps')),
        'Keep the accepted pilot points, equations, parameters and thresholds.')
    require(config['checkpoint_steps']==step_count(config['final_time'],config['dt'])==10
        and config['planned_skips']==[dict(point_id='mesh_refined',solver='Jacobi',reason='Accepted first-step iteration limit; no full trajectory or timing accepted')],
        'Keep ten steps and the recorded finer-mesh Jacobi exclusion; no new sweep or relaxed cap.')


def final_metrics(state,config,context,fd):
    time=config['final_time'];primary=error_metrics(state,time,context,config['quadrature_degree'],fd)
    control=error_metrics(state,time,context,config['quadrature_control_degree'],fd)
    checks,quadrature=check_metrics(primary,control,time,config['checks']);require(all(checks.values()),'Final norm/quadrature check failed.')
    return dict(final_time=time,metrics=primary,control_metrics=control,checks=checks,quadrature_comparison=quadrature)


def equation_checks(state,old,slopes,block,load,index,config,context,rtol,atol,fd):
    dt=config['dt'];A,b,c=tableau_coefficients(radau_config(config));start=index*dt
    r=true_residual(block,slopes,load);target=max(atol,rtol*r['rhs_norm'])
    l2=lambda expr,measure=context['dx']:safe_sqrt(float(fd.assemble(expr**2*measure)))
    parts=slopes.subfunctions
    row=dict(step=index+1,t_start=start,t_end=(index+1)*dt,stage_times=[start+ci*dt for ci in c],
        true_block_residual=r,true_residual_target=target,
        update_identity_l2=l2(state-(old+dt*sum(b[j]*parts[j] for j in range(2)))),
        stiff_accuracy_l2=l2(state-(old+dt*sum(A[1][j]*parts[j] for j in range(2)))),
        stage_boundary_trace_l2=[l2(s,context['ds']) for s in parts],source_checks=[])
    require(all(math.isfinite(v) and v>=0 for v in r.values()) and r['rhs_norm']>0 and r['norm']<=config['checks']['residual_tolerance_factor']*target,
        'Independent physical residual failed.')
    require(row['update_identity_l2']<=1e-12 and row['stiff_accuracy_l2']<=1e-12 and all(v<=1e-12 for v in row['stage_boundary_trace_l2']),
        'Physical update/stiff-accuracy/boundary check failed.')
    for time in (start,(index+1)*dt,*row['stage_times']):
        exact=fd.exp(-time)*context['sinusoid']
        defect=l2(-exact+fd.dot(context['beta'],fd.grad(exact))-config['diffusion']*fd.div(fd.grad(exact))-context['forcing'](time))
        row['source_checks'].append(dict(time=time,source_residual_l2=defect))
    require(all(r['source_residual_l2']<=1e-10 for r in row['source_checks']),'Point-specific manufactured source failed.')
    return row


def independent_forms(config,context,W,old,t,fd):
    z,q=fd.TrialFunctions(W),fd.TestFunctions(W);A,b,c=tableau_coefficients(radau_config(config));dt=config['dt'];spatial=context['spatial']
    block=sum(z[i]*q[i]+dt*sum(A[i][j]*spatial(z[j],q[i]) for j in range(2)) for i in range(2))*context['dx']
    load=sum(context['forcing'](t+c[i]*dt)*q[i]-spatial(old,q[i]) for i in range(2))*context['dx']
    return block,load,[fd.DirichletBC(W.sub(i),0.,'on_boundary') for i in range(2)]


def reference_trajectory(config,context,record,fd,PETSc):
    from petsctools import inserted_options
    V=context['space'];W=V*V;dt=config['dt'];count=config['checkpoint_steps'];A,b,c=tableau_coefficients(radau_config(config))
    state=fd.Function(V,name='physical_reference_state').assign(context['initial']);old=fd.Function(V)
    slopes=fd.Function(W);t=fd.Constant(0.);form,load_form,bcs=independent_forms(config,context,W,old,t,fd)
    block=fd.assemble(form,bcs=bcs,mat_type='aij');prefix='phase3_selected_reference_'+config['point_id']+'_'
    ksp=PETSc.KSP().create(comm=context['mesh'].comm);ksp.setOptionsPrefix(prefix);ksp.setOperators(block.petscmat)
    stage=PETSc.Log.Stage(prefix);events={name:PETSc.Log.Event(name) for name in EVENTS};snapshots=[];first_tokens=None
    record.update(status='running',planned_steps=count,completed_steps=0,step_records=[],stage_total_dofs=W.dim(),matrix_assemblies=1,
        solver_parameters=config['reference_solver_parameters'],performance_measurement=False,
        reference_residual_rtol=config['reference_residual_rtol'],reference_residual_atol=config['reference_residual_atol'],
        residual_budget_meaning='Independent external physical-residual check; PREONLY does not iterate to its displayed default KSP tolerances')
    try:
        with stage:
            start=event_counts(events,stage)
            with inserted_options(parameters=config['reference_solver_parameters'],options_prefix=prefix):
                ksp.setFromOptions();pc=ksp.getPC()
                record['construction_event_counts']=count_difference(start,event_counts(events,stage))
                for index in range(count):
                    t.assign(index*dt);old.assign(state);slopes.assign(0.);load=fd.assemble(load_form,bcs=bcs)
                    before=event_counts(events,stage)
                    with load.dat.vec_ro as rhs,slopes.dat.vec_wo as solution:ksp.solve(rhs,solution)
                    counts=count_difference(before,event_counts(events,stage))
                    require(ksp.getType()=='preonly' and pc.getType()=='lu' and pc.getFactorSolverType()=='petsc' and ksp.getConvergedReason()>0,
                        'Independent physical LU reference failed.')
                    state.interpolate(old+dt*sum(b[j]*slopes.subfunctions[j] for j in range(2)))
                    row=equation_checks(state,old,slopes,block,load,index,config,context,config['reference_residual_rtol'],config['reference_residual_atol'],fd)
                    tokens=dict(physical_B=matrix_token(block.petscmat),physical_factor=matrix_token(pc.getFactorMatrix()))
                    if first_tokens is None:first_tokens=tokens
                    require(tokens==first_tokens and (index==0 or all(counts[key]==0 for key in REBUILD_EVENTS)),'Physical reference rebuilt its fixed factors.')
                    row.update(event_counts=counts,matrix_reuse_tokens=tokens,reuse_matches_first_step=True,
                        converged_reason=int(ksp.getConvergedReason()),iterations=ksp.getIterationNumber())
                    record['step_records'].append(row);record['completed_steps']=index+1
                    snapshots.append((fd.Function(V).assign(state),fd.Function(W).assign(slopes)))
                record.update(operator_event_counts=count_difference(start,event_counts(events,stage)),
                    actual_solver=dict(ksp_type=ksp.getType(),pc_type=pc.getType(),factor_solver_type=pc.getFactorSolverType()),
                    factor_nonzeros=float(pc.getFactorMatrix().getInfo()['nz_used']))
                require(record['operator_event_counts']['KSPSolve']==count and all(record['operator_event_counts'][e]==1 for e in ('MatLUFactorSym','MatLUFactorNum')),
                    'Expected one physical LU factorization and ten reference solves.')
                ksp.view()
        record.update(status='passed',**final_metrics(state,config,context,fd))
        print(f"  {config['point_id']} / physical LU reference: ten steps verified",flush=True)
        return block,snapshots
    finally:ksp.destroy()


def actual_solver(stepper,parameters,PETSc):
    ksp=stepper.solver.snes.ksp;pc=ksp.getPC();rtol,atol,divtol,max_it=ksp.getTolerances()
    require(ksp.getType()=='gmres' and pc.getType()==parameters['pc_type'] and stepper.solver.snes.getType()=='ksponly'
        and ksp.getPCSide()==PETSc.PC.Side.RIGHT and ksp.getNormType()==PETSc.KSP.NormType.UNPRECONDITIONED
        and not ksp.getInitialGuessNonzero() and rtol==parameters['ksp_rtol'] and atol==parameters['ksp_atol'] and max_it==1000,
        'Actual candidate parameters changed.')
    return dict(ksp_type=ksp.getType(),pc_type=pc.getType(),snes_type=stepper.solver.snes.getType(),side='right',norm='unpreconditioned',
        rtol=rtol,atol=atol,divtol=divtol,max_it=max_it,requested_restart=50,restart_evidence='Actual KSP view in transcript')


def candidate_trajectory(config,context,case,record,block,references,fd,PETSc,irk,AI):
    V=context['space'];dt_value=config['dt'];count=config['checkpoint_steps'];A,b,c=tableau_coefficients(radau_config(config));At=exact_factors(config)
    state=fd.Function(V,name=case+'_state').assign(context['initial']);old=fd.Function(V)
    t,dt=fd.Constant(0.),fd.Constant(dt_value);v=fd.TestFunction(V);spatial=context['spatial']
    F=(irk.Dt(state)*v+spatial(state,v)-context['forcing'](t)*v)*context['dx'];require(len(F.arguments())==1,'Expected one-test-argument residual.')
    parameters=parameters_for(config,case);prefix='phase3_selected_'+config['point_id']+'_'+case+'_'
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


def study(config,result,fd,PETSc,irk,AI,comm):
    require(comm.size==1,'Serial selected trajectory collection required.');validate_config(config)
    pilot_review=json.loads((ROOT/'results/processed/solver_sensitivity_review.json').read_text())
    pilot=json.loads((ROOT/pilot_review['measurement_records']['serial']['path']).read_text())
    for point in config['points']:
        cfg={**config,**point};context=make_context(cfg,fd,comm);metadata=point_metadata(point,config)
        metadata.update(checkpoint_end_time=config['final_time']);require(context['space'].dim()==metadata['total_dofs'],'P2 dimension changed.')
        row=dict(status='running',metadata=metadata,reference={},candidates={});result['points'][point['point_id']]=row
        result['active_trajectory']=point['point_id']+'/physical_reference'
        block,references=reference_trajectory(cfg,context,row['reference'],fd,PETSc)
        for case in config['solver_cases']:
            result['active_trajectory']=point['point_id']+'/'+case;record=dict(status='prepared');row['candidates'][case]=record
            skip=next((s for s in config['planned_skips'] if s['point_id']==point['point_id'] and s['solver']==case),None)
            if skip:
                previous=pilot['points'][point['point_id']]['candidates'][case]
                require(previous['status']=='iteration_limit' and previous['converged_reason']==-3 and previous['completed_steps']==0,'Skip lacks the accepted pilot failure.')
                record.update(status='not_attempted',reason=skip['reason'],completed_steps=0,accepted_final_solution=False,
                    evidence=pilot_review['measurement_records']['serial'],pilot_iterations=previous['iterations'],pilot_converged_reason=previous['converged_reason'])
                continue
            candidate_trajectory(cfg,context,case,record,block,references,fd,PETSc,irk,AI)
        row['status']='collected'
    result.pop('active_trajectory',None);outcomes=[r for p in result['points'].values() for r in p['candidates'].values()]
    result.update(status='collected',completed_study=True,reference_trajectories=3,reference_completed_steps=30,candidate_attempts=8,
        completed_candidate_trajectories=sum(r['status']=='passed' for r in outcomes),capped_candidate_trajectories=sum(r['status']=='iteration_limit' for r in outcomes),
        predeclared_skipped_candidates=sum(r['status']=='not_attempted' for r in outcomes),candidate_completed_steps=sum(r['completed_steps'] for r in outcomes),
        performance_measurement=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();sys.argv=[sys.argv[0]];result=dict(status='failed',completed_study=False,ranks=1,points={},performance_measurement=False)
    allowed=args.output.resolve().parent==ROOT/'results/raw'
    try:
        require(allowed,'Keep all records in Projects.');config=json.loads(args.config.read_text());result['config']=config
        from mpi4py import MPI
        import firedrake as fd
        from firedrake.petsc import PETSc
        import irksome as irk
        from irksome.tools import AI
        PETSc.Log.begin();study(config,result,fd,PETSc,irk,AI,MPI.COMM_WORLD)
    except (Exception,KeyboardInterrupt) as error:
        result.update(status='failed',error=str(error) or 'Interrupted by researcher');traceback.print_exc()
    finally:
        if allowed:args.output.write_text(json.dumps(sanitize(result),indent=2,allow_nan=False)+'\n')
    return 0 if result['status']=='collected' else 1

if __name__=='__main__':sys.exit(main())

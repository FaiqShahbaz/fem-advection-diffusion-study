#!/usr/bin/env python3
"""Researcher-run first-step solver sensitivity pilot; retain iteration-limit outcomes."""
import argparse
import json
import math
from pathlib import Path
import sys
import traceback

from phase0_be_smoke import require
from phase0_irksome_be import true_residual
from phase0_radau_smoke import tableau_coefficients
from phase1_spatial import safe_sqrt,error_metrics,check_metrics,capture_failed_solve,sanitize
from phase2_temporal import radau_config
from phase3_asm_construction import compare,inspect_domains,additive_action_check
from phase3_asm_trajectory import make_context,event_counts,count_difference
from phase3_rana_construction import exact_factors,auxiliary_action
from phase3_rana_trajectory import EVENTS

ROOT=Path(__file__).resolve().parents[1]


def validate_config(config):
    accepted=json.loads((ROOT/'configs/phase3_rana_cost.json').read_text())
    keys=('element_degree','element_family','mesh_diagonal','velocity','final_time','quadrature_degree','quadrature_control_degree',
        'solver_parameters','radau_formulation','checks','dt','matrix_block_size_expected','rana_solver_parameters','rana_tableau_rational')
    require(all(config[k]==accepted[k] for k in keys),'Keep accepted physical formulation, PCs and thresholds.')
    asm=json.loads((ROOT/'configs/phase3_asm_cost.json').read_text())['asm_solver_parameters']
    require(config['asm_solver_parameters']=={**asm,'mat_type':'aij','snes_type':'ksponly'},'Keep the existing ASM/ILU configuration.')
    require(config['points']==[dict(point_id='baseline',mesh_subdivisions=16,diffusion=1.),dict(point_id='mesh_refined',mesh_subdivisions=32,diffusion=1.),dict(point_id='lower_diffusion',mesh_subdivisions=16,diffusion=.1)]
        and config['solver_cases']==['Jacobi','ASM','RanaLD'] and config['checkpoint_steps']==1 and config['performance_measurement'] is False,
        'Keep these three one-step diagnostic points; no Cartesian sweep.')
    require(config['reference_solver_parameters']==dict(ksp_type='preonly',pc_type='lu',pc_factor_mat_solver_type='petsc',pc_factor_mat_ordering_type='rcm',pc_factor_shift_type='none',ksp_error_if_not_converged=True,ksp_initial_guess_nonzero=False)
        and config['reference_residual_rtol']==1e-12 and config['reference_residual_atol']==1e-14,'Keep the independent physical LU reference and residual checks.')
    exact_factors(config)


def point_metadata(point,config):
    n=point['mesh_subdivisions'];h=math.sqrt(2)/n;speed=math.sqrt(sum(v*v for v in config['velocity']));kappa=point['diffusion'];dt=config['dt']
    return dict(mesh_subdivisions=n,diffusion=kappa,cells=2*n*n,total_dofs=(2*n+1)**2,free_dofs=(2*n-1)**2,
        stage_total_dofs=2*(2*n+1)**2,h_max=h,cell_peclet=speed*h/(2*kappa),advective_step_ratio=speed*dt/h,diffusive_step_ratio=kappa*dt/h**2,
        checkpoint_end_time=dt,research_final_time=config['final_time'])


def matrix_comparison(actual,expected,checks):
    n,m=expected.getSize();require(n==m and actual.getSize()==(n,n) and actual.getType()==expected.getType()=='seqaij'
        and actual.getBlockSize()==expected.getBlockSize()==1,'Expected compatible serial scalar AIJ matrices.')
    ap,aj,av=actual.getValuesCSR();bp,bj,bv=expected.getValuesCSR();squared=[];reference=[]
    for i in range(n):
        a={int(aj[k]):float(av[k]) for k in range(int(ap[i]),int(ap[i+1]))}
        b={int(bj[k]):float(bv[k]) for k in range(int(bp[i]),int(bp[i+1]))}
        squared.extend((a.get(k,0.)-b.get(k,0.))**2 for k in a.keys()|b.keys());reference.extend(v*v for v in b.values())
    defect=math.sqrt(math.fsum(squared));norm=math.sqrt(math.fsum(reference));allowed=checks['auxiliary_matrix_atol']+checks['auxiliary_matrix_rtol']*norm
    return dict(status='passed' if math.isfinite(defect) and math.isfinite(norm) and defect<=allowed else 'failed',
        frobenius_defect=defect,reference_frobenius_norm=norm,allowed_defect=allowed,
        meaning='Full CSR coefficients including boundary rows; absent and stored-zero entries equivalent')


def metrics(state,config,context,fd):
    primary=error_metrics(state,config['dt'],context,config['quadrature_degree'],fd)
    control=error_metrics(state,config['dt'],context,config['quadrature_control_degree'],fd)
    checks,quadrature=check_metrics(primary,control,config['dt'],config['checks']);require(all(checks.values()),'MMS/norm/quadrature check failed.')
    return dict(metrics=primary,control_metrics=control,checks=checks,quadrature_comparison=quadrature)


def direct_reference(config,context,block,load,W,record,fd,PETSc):
    from petsctools import inserted_options
    slopes=fd.Function(W,name='physical_LU_slopes');slopes.assign(0.)
    prefix='phase3_sensitivity_reference_';ksp=PETSc.KSP().create(comm=context['mesh'].comm);ksp.setOptionsPrefix(prefix);ksp.setOperators(block.petscmat)
    try:
        with inserted_options(parameters=config['reference_solver_parameters'],options_prefix=prefix):
            ksp.setFromOptions()
            with load.dat.vec_ro as rhs,slopes.dat.vec_wo as solution:ksp.solve(rhs,solution)
            pc=ksp.getPC();r=true_residual(block,slopes,load);target=max(config['reference_residual_atol'],config['reference_residual_rtol']*r['rhs_norm'])
            require(ksp.getType()=='preonly' and pc.getType()=='lu' and pc.getFactorSolverType()=='petsc' and ksp.getConvergedReason()>0
                and all(math.isfinite(v) for v in r.values()) and r['norm']<=10*target,'Independent physical LU reference failed.')
            A,b,c=tableau_coefficients(radau_config(config));state=fd.Function(context['space'],name='physical_LU_update')
            state.interpolate(context['initial']+config['dt']*sum(b[j]*slopes.subfunctions[j] for j in range(2)))
            l2=lambda expr,measure=context['dx']:safe_sqrt(float(fd.assemble(expr**2*measure)))
            record.update(update_identity_l2=l2(state-(context['initial']+config['dt']*sum(b[j]*slopes.subfunctions[j] for j in range(2)))),
                stiff_accuracy_l2=l2(state-(context['initial']+config['dt']*sum(A[1][j]*slopes.subfunctions[j] for j in range(2)))),
                stage_boundary_trace_l2=[l2(slope,context['ds']) for slope in slopes.subfunctions],stage_times=[ci*config['dt'] for ci in c])
            require(record['update_identity_l2']<=1e-12 and record['stiff_accuracy_l2']<=1e-12
                and all(v<=1e-12 for v in record['stage_boundary_trace_l2']),'Physical LU reference update/boundary failed.')
            record.update(status='passed',solver_parameters=config['reference_solver_parameters'],true_block_residual=r,true_residual_target=target,
                stage_total_dofs=W.dim(),actual_solver=dict(ksp_type=ksp.getType(),pc_type=pc.getType(),factor_solver_type=pc.getFactorSolverType()),
                factor_nonzeros=float(pc.getFactorMatrix().getInfo()['nz_used']),**metrics(state,config,context,fd))
            ksp.view();return state,slopes
    finally:ksp.destroy()


def parameters_for(config,case):
    if case=='RanaLD':return dict(config['rana_solver_parameters'])
    if case=='ASM':return dict(config['asm_solver_parameters'])
    require(case=='Jacobi','Unknown candidate.');return {**config['solver_parameters'],'mat_type':'aij','snes_type':'ksponly'}


def candidate(config,context,case,record,reference,block,load,fd,PETSc,irk,AI):
    V=context['space'];dt_value=config['dt'];A,b,c=tableau_coefficients(radau_config(config));At=exact_factors(config)
    state=fd.Function(V,name=case+'_state').assign(context['initial']);old=fd.Function(V).assign(state)
    t,dt=fd.Constant(0.),fd.Constant(dt_value);v=fd.TestFunction(V);spatial,forcing=context['spatial'],context['forcing']
    F=(irk.Dt(state)*v+spatial(state,v)-forcing(t)*v)*context['dx'];require(len(F.arguments())==1,'Expected one-test-argument residual.')
    params=parameters_for(config,case);history=[];stage=PETSc.Log.Stage('phase3_sensitivity_'+config['point_id']+'_'+case)
    events={name:PETSc.Log.Event(name) for name in EVENTS};record.update(status='running',solver_parameters=params,completed_steps=0,step_committed=False,weak_form_argument_count=1)
    with stage:
        before=event_counts(events,stage)
        stepper=irk.TimeStepper(F,irk.RadauIIA(2),t,dt,state,bcs=[context['bc']],stage_type='deriv',splitting=AI,bc_type='DAE',constant_jacobian=True,
            solver_parameters=params,options_prefix='phase3_sensitivity_'+config['point_id']+'_'+case)
        ksp=stepper.solver.snes.ksp;ksp.setMonitor(lambda solver,iteration,norm:history.append([iteration,float(norm)]))
        for slope in stepper.stages.subfunctions:slope.assign(0.)
        capped=False;after_advance=None
        try:stepper.advance()
        except Exception as error:
            if int(ksp.getConvergedReason())==int(PETSc.KSP.ConvergedReason.DIVERGED_MAX_IT) and ksp.getIterationNumber()==params['ksp_max_it']:
                after_advance=event_counts(events,stage)
                capture_failed_solve(record,0,0.,dt_value,ksp,history,error)
                record['failure_diagnostic_event_counts']=count_difference(after_advance,event_counts(events,stage));capped=True
            else:raise
        record['operator_event_counts']=count_difference(before,after_advance if after_advance is not None else event_counts(events,stage))
        pc=ksp.getPC();rtol,atol,divtol,max_it=ksp.getTolerances()
        require(ksp.getType()=='gmres' and pc.getType()==params['pc_type'] and stepper.solver.snes.getType()=='ksponly' and ksp.getPCSide()==PETSc.PC.Side.RIGHT
            and ksp.getNormType()==PETSc.KSP.NormType.UNPRECONDITIONED and not ksp.getInitialGuessNonzero()
            and rtol==params['ksp_rtol'] and atol==params['ksp_atol'] and max_it==1000,'Actual candidate solver changed.')
        record.update(iterations=ksp.getIterationNumber(),converged_reason=int(ksp.getConvergedReason()),reported_residual_history=history,
            actual_solver=dict(ksp_type=ksp.getType(),pc_type=pc.getType(),snes_type=stepper.solver.snes.getType(),side='right',norm='unpreconditioned',rtol=rtol,atol=atol,max_it=max_it,requested_restart=50),
            physical_matrix_check=matrix_comparison(ksp.getOperators()[0],block.petscmat,config['checks']))
        require(record['physical_matrix_check']['status']=='passed','Physical matrix changed.')
        require(all(abs(float(stepper.butcher_tableau.A[i,j])-A[i][j])<=1e-14 for i in range(2) for j in range(2)),'Physical Radau tableau changed.')
        ksp.view()
        if capped:
            # SNES did not complete its correction or Irksome its update. Its
            # correction diagnostic is not an accepted physical slope solution.
            change=safe_sqrt(float(fd.assemble((state-context['initial'])**2*context['dx'])))
            require(change<=1e-12,'Capped advance unexpectedly committed the physical update.')
            record.update(status='iteration_limit',step_committed=False,initial_state_change_l2=change,
                interpretation='Unchanged-cap nonconvergence; no candidate update/slope field or performance result accepted')
            print(f"  {config['point_id']} / {case}: iteration cap reached; retained as an unaccepted candidate",flush=True)
            return
        require(ksp.getConvergedReason()>0 and float(t)==0.,'Unexpected candidate convergence or clock.')
        r=true_residual(block,stepper.stages,load);target=max(atol,rtol*r['rhs_norm'])
        slopes=stepper.stages.subfunctions;l2=lambda expr,measure=context['dx']:safe_sqrt(float(fd.assemble(expr**2*measure)))
        record.update(true_block_residual=r,true_residual_target=target,
            update_identity_l2=l2(state-(old+dt_value*sum(b[j]*slopes[j] for j in range(2)))),
            stiff_accuracy_l2=l2(state-(old+dt_value*sum(A[1][j]*slopes[j] for j in range(2)))),
            stage_boundary_trace_l2=[l2(slope,context['ds']) for slope in slopes],stage_times=[ci*dt_value for ci in c])
        require(all(math.isfinite(v) for v in r.values()) and r['norm']<=10*target and record['update_identity_l2']<=1e-12
            and record['stiff_accuracy_l2']<=1e-12 and all(v<=1e-12 for v in record['stage_boundary_trace_l2']),'Physical candidate equation/update/boundary failed.')
        record['reference_comparisons']=dict(update=compare(state,reference[0],config,context,fd))
        for i in range(2):record['reference_comparisons']['slope'+str(i+1)]=compare(slopes[i],reference[1].subfunctions[i],config,context,fd)
        require(all(row['status']=='passed' for row in record['reference_comparisons'].values()),'Reference field agreement failed.')
        if case=='RanaLD':
            ctx=pc.getPythonContext();require(isinstance(ctx,irk.RanaLD) and ctx.pc.getType()=='lu' and ctx.pc.getFactorSolverType()=='petsc','Rana auxiliary solver changed.')
            W=stepper.stages.function_space();z,q=fd.TrialFunctions(W),fd.TestFunctions(W)
            form=sum(z[i]*q[i]+dt_value*sum(At[i][j]*spatial(z[j],q[i]) for j in range(2)) for i in range(2))*context['dx']
            bcs=[fd.DirichletBC(W.sub(i),0.,'on_boundary') for i in range(2)];P=fd.assemble(form,bcs=bcs,mat_type='aij')
            record['auxiliary_matrix_checks']=dict(P=matrix_comparison(ctx.P.petscmat,P.petscmat,config['checks']),Pmat=matrix_comparison(ctx.pc.getOperators()[1],P.petscmat,config['checks']))
            require(all(row['status']=='passed' for row in record['auxiliary_matrix_checks'].values()),'Auxiliary matrix changed.')
            start=event_counts(events,stage);record['auxiliary_action_check']=auxiliary_action(pc,P.petscmat,config['checks'])
            record['diagnostic_event_counts']=count_difference(start,event_counts(events,stage))
            require(record['auxiliary_action_check']['status']=='passed','Auxiliary inverse probe failed.')
            record['auxiliary_solver']=dict(pc_type=ctx.pc.getType(),factor_solver_type=ctx.pc.getFactorSolverType(),factor_nonzeros=float(ctx.pc.getFactorMatrix().getInfo()['nz_used']),inner_krylov_solve=False)
        elif case=='ASM':
            record['domains']=inspect_domains(pc,ksp.getOperators()[0]);subksps=pc.getASMSubKSP()
            record['local_solvers']=[dict(ksp_type=s.getType(),pc_type=s.getPC().getType(),factor_solver_type=s.getPC().getFactorSolverType(),matrix_size=list(s.getOperators()[0].getSize()),factor_nonzeros=float(s.getPC().getFactorMatrix().getInfo()['nz_used'])) for s in subksps]
            require(all(s['ksp_type']=='preonly' and s['pc_type']=='ilu' and s['factor_solver_type']=='petsc' for s in record['local_solvers']),'ASM local action changed.')
            start=event_counts(events,stage);record['additive_action_check']=additive_action_check(pc,subksps,record['domains'],ksp.getOperators()[0],config['checks'],PETSc)
            record['diagnostic_event_counts']=count_difference(start,event_counts(events,stage))
            require(record['additive_action_check']['status']=='passed','Actual additive action probe failed.')
            applications=record['operator_event_counts']['PCApply']
            record['local_work_derived']=dict(basis='Actual outer PCApply count times the fixed four-domain PREONLY/ILU additive composition',per_domain=applications,total=4*applications)
        record.update(status='passed',completed_steps=1,step_committed=True,**metrics(state,config,context,fd))
        print(f"  {config['point_id']} / {case}: first-step checks passed; GMRES iterations={record['iterations']}",flush=True)


def study(config,result,fd,PETSc,irk,AI,comm):
    require(comm.size==1,'Serial diagnostic required.');validate_config(config)
    for point in config['points']:
        cfg={**config,**point};metadata=point_metadata(point,config);ctx=make_context(cfg,fd,comm)
        require(ctx['space'].dim()==metadata['total_dofs'],'P2 mesh dimension differs.')
        row=dict(status='running',metadata=metadata,reference={},candidates={},source_checks=[]);result['points'][point['point_id']]=row
        V=ctx['space'];W=V*V;z,q=fd.TrialFunctions(W),fd.TestFunctions(W);A,b,c=tableau_coefficients(radau_config(cfg));dt=cfg['dt']
        require(W.dim()==metadata['stage_total_dofs'],'Coupled dimension differs.')
        spatial,forcing=ctx['spatial'],ctx['forcing'];initial=ctx['initial']
        block_form=sum(z[i]*q[i]+dt*sum(A[i][j]*spatial(z[j],q[i]) for j in range(2)) for i in range(2))*ctx['dx']
        load_form=sum(forcing(c[i]*dt)*q[i]-spatial(initial,q[i]) for i in range(2))*ctx['dx']
        bcs=[fd.DirichletBC(W.sub(i),0.,'on_boundary') for i in range(2)]
        block=fd.assemble(block_form,bcs=bcs,mat_type='aij');load=fd.assemble(load_form,bcs=bcs)
        for value in (0.,dt,*[ci*dt for ci in c]):
            exact=fd.exp(-value)*ctx['sinusoid'];defect=safe_sqrt(float(fd.assemble((-exact+fd.dot(ctx['beta'],fd.grad(exact))-cfg['diffusion']*fd.div(fd.grad(exact))-forcing(value))**2*ctx['dx'])))
            row['source_checks'].append(dict(time=value,source_residual_l2=defect))
        require(all(r['source_residual_l2']<=1e-10 for r in row['source_checks']),'Point-specific manufactured source failed.')
        ref=direct_reference(cfg,ctx,block,load,W,row['reference'],fd,PETSc)
        for case in config['solver_cases']:
            result['active_candidate']=point['point_id']+'/'+case;record=dict(status='prepared');row['candidates'][case]=record
            candidate(cfg,ctx,case,record,ref,block,load,fd,PETSc,irk,AI)
        row['status']='collected'
    result.pop('active_candidate',None)
    outcomes=[r['status'] for point in result['points'].values() for r in point['candidates'].values()]
    result.update(status='collected',completed_study=True,reference_solves=3,candidate_attempts=9,
        converged_candidates=outcomes.count('passed'),capped_candidates=outcomes.count('iteration_limit'),performance_measurement=False)


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

#!/usr/bin/env python3
"""Researcher-run first-step construction check of installed Irksome RanaLD."""
import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import sys
import traceback

from phase0_be_smoke import require
from phase0_irksome_be import true_residual
from phase0_radau_smoke import tableau_coefficients
from phase1_spatial import run_trajectory,safe_sqrt,error_metrics,check_metrics,capture_failed_solve,sanitize
from phase2_temporal import radau_config
from phase3_asm_construction import manual_solve,compare
from phase3_asm_trajectory import make_context,event_counts,count_difference,EVENTS

ROOT=Path(__file__).resolve().parents[1]


def exact_factors(config):
    A=[[Fraction(v) for v in row] for row in config['radau_formulation']['tableau_rational']['A']]
    lower=A[1][0]/A[0][0];second=A[1][1]-lower*A[0][1]
    L=[[Fraction(1),Fraction(0)],[lower,Fraction(1)]]
    D=[[A[0][0],Fraction(0)],[Fraction(0),second]]
    U=[[Fraction(1),A[0][1]/A[0][0]],[Fraction(0),Fraction(1)]]
    multiply=lambda X,Y:[[sum(X[i][k]*Y[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
    LD=multiply(L,D);require(multiply(LD,U)==A,'Independent exact LDU identity failed.')
    for name,actual in (('L',L),('D',D),('U',U),('Atilde',LD)):
        expected=[[Fraction(v) for v in row] for row in config['rana_tableau_rational'][name]]
        require(actual==expected,'Declared Rana '+name+' differs from exact algebra.')
    return [[float(v) for v in row] for row in LD]


def validate_config(config):
    accepted=json.loads((ROOT/'configs/phase3_asm_construction.json').read_text())
    keys=set(accepted)-{'purpose','asm_solver_parameters','checks'}
    require(all(config[k]==accepted[k] for k in keys),'Keep the accepted first-step problem and reference control.')
    require(all(config['checks'][k]==v for k,v in accepted['checks'].items()),'Keep accepted checks.')
    expected={**config['solver_parameters'],'pc_type':'python','pc_python_type':'irksome.RanaLD',
        'mat_type':'aij','snes_type':'ksponly','aux_mat_type':'aij','aux_pc_type':'lu',
        'aux_pc_factor_mat_solver_type':'petsc','aux_pc_factor_mat_ordering_type':'rcm','aux_pc_factor_shift_type':'none'}
    require(config['rana_solver_parameters']==expected,'Keep this fixed installed RanaLD/exact-auxiliary-LU construction.')
    require([config['checks'][k] for k in ('auxiliary_matrix_atol','auxiliary_matrix_rtol','auxiliary_action_atol','auxiliary_action_rtol')]
        ==[1e-11,1e-12,1e-11,1e-10],'Keep the declared matrix/action checks.')
    return exact_factors(config)


def matrix_comparison(actual,expected,checks):
    require(actual.getSize()==expected.getSize()==(2178,2178)
        and actual.getType()==expected.getType()=='seqaij' and actual.getBlockSize()==expected.getBlockSize()==1,'Expected compatible serial scalar AIJ matrices.')
    ap,aj,av=actual.getValuesCSR();bp,bj,bv=expected.getValuesCSR()
    squared=[];reference=[]
    for i in range(actual.getSize()[0]):
        a={int(aj[k]):float(av[k]) for k in range(int(ap[i]),int(ap[i+1]))}
        b={int(bj[k]):float(bv[k]) for k in range(int(bp[i]),int(bp[i+1]))}
        squared.extend((a.get(k,0.)-b.get(k,0.))**2 for k in a.keys()|b.keys())
        reference.extend(v*v for v in b.values())
    defect=math.sqrt(math.fsum(squared));norm=math.sqrt(math.fsum(reference))
    allowed=checks['auxiliary_matrix_atol']+checks['auxiliary_matrix_rtol']*norm
    return dict(status='passed' if math.isfinite(defect) and math.isfinite(norm) and defect<=allowed else 'failed',
        frobenius_defect=defect,reference_frobenius_norm=norm,allowed_defect=allowed,
        meaning='Full CSR coefficient comparison, treating absent entries as zero; includes imposed boundary rows')


def auxiliary_action(pc,matrix,checks):
    rhs=matrix.createVecLeft();answer=matrix.createVecRight();residual=rhs.duplicate()
    try:
        n=matrix.getSize()[0];rhs.setValues(list(range(n)),[math.sin((i+1)*.123)+.2*math.cos((i+1)*.059) for i in range(n)]);rhs.assemble()
        pc.apply(rhs,answer);matrix.mult(answer,residual);residual.axpy(-1.,rhs)
        value,norm=float(residual.norm()),float(rhs.norm());allowed=checks['auxiliary_action_atol']+checks['auxiliary_action_rtol']*norm
        return dict(status='passed' if math.isfinite(value) and value<=allowed else 'failed',
            residual_norm=value,rhs_norm=norm,relative_residual=value/norm,allowed_residual=allowed,
            diagnostic_outer_PC_applications=1,meaning='P_expected times actual RanaLD PC action minus probe RHS; exact auxiliary LU configuration')
    finally:
        rhs.destroy();answer.destroy();residual.destroy()


def rana_step(config,context,record,fd,PETSc,irk,AI):
    V=context['space'];dt_value=config['dt'];A,b,c=tableau_coefficients(radau_config(config));At=exact_factors(config)
    state=fd.Function(V,name='rana_solution').assign(context['initial']);old=fd.Function(V).assign(state)
    t,dt=fd.Constant(0.),fd.Constant(dt_value);v=fd.TestFunction(V)
    spatial,forcing=context['spatial'],context['forcing'];parameters=config['rana_solver_parameters']
    # Installed RanaLD unpacks one test argument from stepper.F. The unknown
    # must be a Function coefficient, not a second UFL TrialFunction argument.
    F=(irk.Dt(state)*v+spatial(state,v)-forcing(t)*v)*context['dx']
    argument_count=len(F.arguments())
    record.update(status='running',dt=dt_value,planned_steps=1,completed_steps=0,solver_parameters=parameters,performance_measurement=False,
        weak_form_argument_count=argument_count,weak_form_representation='state_function_residual')
    require(argument_count==1,'Installed RanaLD requires a one-test-argument residual F(state; v).')
    stage=PETSc.Log.Stage('phase3_rana_construction');events={name:PETSc.Log.Event(name) for name in EVENTS}
    with stage:
        before=event_counts(events,stage)
        stepper=irk.TimeStepper(F,irk.RadauIIA(2),t,dt,state,bcs=[context['bc']],stage_type='deriv',splitting=AI,
            bc_type='DAE',constant_jacobian=True,solver_parameters=parameters,options_prefix='phase3_rana_construction')
        ksp=stepper.solver.snes.ksp;history=[]
        ksp.setMonitor(lambda solver,iteration,norm:history.append([iteration,float(norm)]))
        for slope in stepper.stages.subfunctions:slope.assign(0.)
        try:stepper.advance()
        except Exception as error:
            if ksp.getIterationNumber()==0 and int(ksp.getConvergedReason())==0:
                # KSP has not built a Krylov basis; building its residual here
                # would produce a secondary diagnostic error that masks setup.
                record.update(status='failed',failed_step=dict(step=1,t_start=0.,dt=dt_value,
                    failure_phase='setup_before_first_GMRES_iteration',iterations=0,converged_reason=0,
                    reported_residual_history=list(history),step_committed=False,error=str(error),
                    residual_diagnostic='Unavailable before Krylov initialization; no physical residual or iterate accepted'))
            else:
                capture_failed_solve(record,0,0.,dt_value,ksp,history,error)
            raise
        record['operator_event_counts']=count_difference(before,event_counts(events,stage))
    require(float(t)==0.,'Irksome changed caller-owned time.')
    W=stepper.stages.function_space();require(W.dim()==2178,'Unexpected coupled dimension.')
    z,q=fd.TrialFunctions(W),fd.TestFunctions(W)
    bcs=[fd.DirichletBC(W.sub(i),0.,'on_boundary') for i in range(2)]
    block_form=sum(z[i]*q[i]+dt_value*sum(A[i][j]*spatial(z[j],q[i]) for j in range(2)) for i in range(2))*context['dx']
    auxiliary_form=sum(z[i]*q[i]+dt_value*sum(At[i][j]*spatial(z[j],q[i]) for j in range(2)) for i in range(2))*context['dx']
    load_form=sum(forcing(c[i]*dt_value)*q[i]-spatial(old,q[i]) for i in range(2))*context['dx']
    block=fd.assemble(block_form,bcs=bcs,mat_type='aij');P=fd.assemble(auxiliary_form,bcs=bcs,mat_type='aij');load=fd.assemble(load_form,bcs=bcs)
    residual=true_residual(block,stepper.stages,load);rtol,atol,divtol,max_it=ksp.getTolerances();target=max(atol,rtol*residual['rhs_norm'])
    pc=ksp.getPC();ctx=pc.getPythonContext()
    require(ksp.getType()=='gmres' and pc.getType()=='python' and isinstance(ctx,irk.RanaLD)
        and ksp.getPCSide()==PETSc.PC.Side.RIGHT and ksp.getNormType()==PETSc.KSP.NormType.UNPRECONDITIONED
        and not ksp.getInitialGuessNonzero() and rtol==parameters['ksp_rtol'] and atol==parameters['ksp_atol']
        and max_it==1000 and ksp.getConvergedReason()>0,'Actual outer solver or Rana class differs.')
    require(ctx.pc.getType()=='lu' and ctx.pc.getFactorSolverType()=='petsc','Expected fixed exact auxiliary PETSc LU.')
    actual_at=ctx.getAtilde(stepper.butcher_tableau.A)
    record['installed_Atilde']=[[float(actual_at[i,j]) for j in range(2)] for i in range(2)]
    require(all(abs(record['installed_Atilde'][i][j]-At[i][j])<=1e-14
        and abs(float(stepper.butcher_tableau.A[i,j])-A[i][j])<=1e-14 for i in range(2) for j in range(2)),
        'Auxiliary tableau or original physical tableau differs.')
    slopes=stepper.stages.subfunctions;l2=lambda expr,measure=context['dx']:safe_sqrt(float(fd.assemble(expr**2*measure)))
    record.update(completed_steps=1,stage_total_dofs=W.dim(),iterations=ksp.getIterationNumber(),converged_reason=int(ksp.getConvergedReason()),
        true_block_residual=residual,true_residual_target=target,stage_times=[ci*dt_value for ci in c],
        update_identity_l2=l2(state-(old+dt_value*sum(b[j]*slopes[j] for j in range(2)))),
        stiff_accuracy_l2=l2(state-(old+dt_value*sum(A[1][j]*slopes[j] for j in range(2)))),
        stage_boundary_trace_l2=[l2(slope,context['ds']) for slope in slopes],reported_residual_history=history,
        actual_solver=dict(ksp_type=ksp.getType(),pc_type=pc.getType(),python_context=type(ctx).__module__+'.'+type(ctx).__name__,
            side='right',norm='unpreconditioned',rtol=rtol,atol=atol,divtol=divtol,max_it=max_it,requested_restart=50,
            restart_evidence='Actual KSP view in transcript'),
        auxiliary_solver=dict(pc_type=ctx.pc.getType(),factor_solver_type=ctx.pc.getFactorSolverType(),
            factor_nonzeros=float(ctx.pc.getFactorMatrix().getInfo()['nz_used']),inner_krylov_solve=False,
            interpretation='Fixed auxiliary LU action; ordinary GMRES is appropriate'))
    require(all(math.isfinite(v) for v in residual.values()) and residual['norm']<=10*target,'True physical Radau equation failed.')
    require(record['update_identity_l2']<=1e-12 and record['stiff_accuracy_l2']<=1e-12
        and all(v<=1e-12 for v in record['stage_boundary_trace_l2']),'Update or stage boundary check failed.')
    record['matrix_checks']=dict(physical_B=matrix_comparison(ksp.getOperators()[0],block.petscmat,config['checks']),
        auxiliary_P=matrix_comparison(ctx.P.petscmat,P.petscmat,config['checks']),
        auxiliary_PC_Pmat=matrix_comparison(ctx.pc.getOperators()[1],P.petscmat,config['checks']))
    require(all(r['status']=='passed' for r in record['matrix_checks'].values()),'Installed physical/auxiliary matrix differs from independent construction.')
    record['auxiliary_action_check']=auxiliary_action(pc,P.petscmat,config['checks'])
    require(record['auxiliary_action_check']['status']=='passed','Actual Rana action does not solve independently assembled P for the probe.')
    record['source_checks']=[]
    for value in (0.,dt_value,*record['stage_times']):
        exact=fd.exp(-value)*context['sinusoid']
        defect=l2(-exact+fd.dot(context['beta'],fd.grad(exact))-config['diffusion']*fd.div(fd.grad(exact))-forcing(value))
        record['source_checks'].append(dict(time=value,source_residual_l2=defect))
    require(all(r['source_residual_l2']<=1e-10 for r in record['source_checks']),'MMS source check failed.')
    primary=error_metrics(state,dt_value,context,config['quadrature_degree'],fd)
    control=error_metrics(state,dt_value,context,config['quadrature_control_degree'],fd)
    checks,quadrature=check_metrics(primary,control,dt_value,config['checks']);require(all(checks.values()),'MMS/quadrature checks failed.')
    record.update(status='passed',final_time=dt_value,metrics=primary,control_metrics=control,checks=checks,quadrature_comparison=quadrature,
        solver_stats=list(stepper.solver_stats()))
    ksp.view()
    print(f"  RanaLD first step: physical/auxiliary matrices, action and fields checked; outer iterations={record['iterations']}",flush=True)
    return state,stepper.stages


def study(config,result,fd,PETSc,irk,AI,comm):
    require(comm.size==1,'This construction check is serial.');validate_config(config)
    context=make_context(config,fd,comm);V=context['space'];require(V.dim()==1089,'Unexpected space.')
    result.update(total_dofs=1089,stage_total_dofs=2178,checkpoint_end_time=.1,research_final_time=1.)
    baseline=dict(status='prepared');result['solves']['irksome_jacobi']=baseline;result['active_solve']='irksome_jacobi'
    baseline_state=run_trajectory(radau_config({**config,'final_time':.1}),context,.1,1.,'rana_irksome_jacobi',baseline,fd,PETSc,irk,AI)
    require(baseline['status']=='passed','Accepted first-step baseline failed.')
    A,b,c=tableau_coefficients(radau_config(config));W=V*V;z,q=fd.TrialFunctions(W),fd.TestFunctions(W)
    spatial,forcing=context['spatial'],context['forcing'];dt=config['dt'];initial=context['initial']
    block_form=sum(z[i]*q[i]+dt*sum(A[i][j]*spatial(z[j],q[i]) for j in range(2)) for i in range(2))*context['dx']
    load_form=sum(forcing(c[i]*dt)*q[i]-spatial(initial,q[i]) for i in range(2))*context['dx']
    bcs=[fd.DirichletBC(W.sub(i),0.,'on_boundary') for i in range(2)]
    block=fd.assemble(block_form,bcs=bcs,mat_type='aij');load=fd.assemble(load_form,bcs=bcs)
    parameters=dict(config['solver_parameters'])
    for k in ('ksp_rtol','ksp_atol'):parameters[k]*=config['manual_reference_tolerance_factor']
    reference=dict(status='prepared');result['solves']['manual_jacobi_reference']=reference;result['active_solve']='manual_jacobi_reference'
    ref_state,ref_slopes=manual_solve(config,context,block,load,W,parameters,'rana_manual_reference',reference,fd,PETSc)
    rana=dict(status='prepared');result['solves']['rana_ld']=rana;result['active_solve']='rana_ld'
    state,slopes=rana_step(config,context,rana,fd,PETSc,irk,AI)
    result['comparisons']['reference_update_vs_irksome']=compare(ref_state,baseline_state,config,context,fd)
    result['comparisons']['rana_update_vs_reference']=compare(state,ref_state,config,context,fd)
    for i in range(2):result['comparisons']['rana_slope'+str(i+1)+'_vs_reference']=compare(slopes.subfunctions[i],ref_slopes.subfunctions[i],config,context,fd)
    require(all(r['status']=='passed' for r in result['comparisons'].values()),'Actual Rana/reference field comparison failed.')
    result.pop('active_solve',None);result.update(status='passed',completed_study=True,completed_operator_solves=3)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();sys.argv=[sys.argv[0]]
    result=dict(status='failed',completed_study=False,ranks=1,solves={},comparisons={},performance_measurement=False)
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
        if result.get('active_solve'):result['solves'][result['active_solve']]['status']='failed'
        traceback.print_exc()
    finally:
        if allowed:args.output.write_text(json.dumps(sanitize(result),indent=2,allow_nan=False)+'\n')
    return 0 if result['status']=='passed' else 1

if __name__=='__main__':sys.exit(main())

#!/usr/bin/env python3
"""Researcher-run one-step standard ASM/local-ILU(0) construction and equivalence check."""
import argparse
import json
import math
from pathlib import Path
import sys
import traceback

from phase0_be_smoke import require
from phase0_irksome_be import true_residual
from phase0_radau_smoke import tableau_coefficients
from phase1_spatial import run_trajectory, norm_pair, safe_sqrt, capture_failed_solve, sanitize
from phase2_temporal import radau_config, measured_difference, METRICS

ROOT=Path(__file__).resolve().parents[1]


def validate_config(config):
    accepted=json.loads((ROOT/'configs/phase3_accuracy_cost.json').read_text())
    keys=('mesh_subdivisions','element_degree','element_family','mesh_diagonal','velocity','diffusion',
          'final_time','quadrature_degree','quadrature_control_degree','solver_parameters','radau_formulation')
    require(all(config[key]==accepted[key] for key in keys),'Keep the accepted fixed problem and baseline solver.')
    require(config['dt']==0.1 and config['checkpoint_steps']==1 and config['manual_reference_tolerance_factor']==0.1
            and config['matrix_block_size_expected']==1 and config['performance_measurement'] is False,
            'Keep this one-step algebraic construction check.')
    expected={**config['solver_parameters'],'pc_type':'asm','pc_asm_blocks':4,'pc_asm_overlap':1,
        'pc_asm_type':'basic','pc_asm_local_type':'additive','pc_asm_dm_subdomains':False,
        'mat_partitioning_type':'ptscotch','sub_ksp_type':'preonly','sub_pc_type':'ilu','sub_pc_factor_levels':0,
        'sub_pc_factor_mat_ordering_type':'rcm','sub_pc_factor_mat_solver_type':'petsc','sub_pc_factor_shift_type':'none'}
    require(config['asm_solver_parameters']==expected,'Keep the explained single ASM/ILU(0) choice.')
    require(all(config['checks'][key]==value for key,value in accepted['checks'].items())
        and config['checks']['pc_field_comparison_atol']==1e-11 and config['checks']['pc_field_comparison_rtol']==1e-8
        and config['checks']['additive_action_atol']==1e-12 and config['checks']['additive_action_rtol']==1e-11,
        'Keep the declared construction checks.')
    tableau_coefficients(radau_config(config))


def one_layer(core,rowptr,columns):
    result=set(core)
    for index in core:
        result.update(int(v) for v in columns[int(rowptr[index]):int(rowptr[index+1])])
    return sorted(result)


def inspect_domains(pc,matrix):
    count,outer,core=pc.getASMLocalSubdomains()
    require(count==4 and len(outer)==len(core)==4,'Expected four explicitly requested local subdomains.')
    rowptr,columns,_=matrix.getValuesCSR()
    rows=[];all_core=[]
    for index,(outside,inside) in enumerate(zip(outer,core)):
        full=[int(v) for v in outside.getIndices()];base=[int(v) for v in inside.getIndices()]
        require(base and len(full)==len(set(full)) and len(base)==len(set(base)),'Empty/duplicate domain indices.')
        require(sorted(full)==one_layer(base,rowptr,columns),'Actual overlap differs from one matrix-graph layer.')
        require(set(base)<=set(full) and all(0<=i<matrix.getSize()[0] for i in full),'Invalid domain restriction.')
        rows.append(dict(index=index,core_indices=base,overlap_indices=full,core_size=len(base),overlap_size=len(full),one_layer_check=True))
        all_core.extend(base)
    require(len(all_core)==matrix.getSize()[0] and sorted(all_core)==list(range(matrix.getSize()[0])),
            'Cores do not form a disjoint cover of all algebraic coefficients.')
    return rows


def compare(a,b,config,context,fd):
    measured=measured_difference(a,b,config,context,fd)
    reference=norm_pair(b,fd,fd.dx(domain=context['mesh'],metadata={'quadrature_degree':config['quadrature_control_degree']}))
    norms={}
    for metric in METRICS:
        difference=measured['metrics'][metric]
        allowed=config['checks']['pc_field_comparison_atol']+config['checks']['pc_field_comparison_rtol']*reference[metric]
        norms[metric]=dict(actual_field_difference=difference,reference_field_norm=reference[metric],allowed_difference=allowed,
                           status='passed' if difference<=allowed else 'failed')
    return dict(status='passed' if measured['status']=='passed' and all(r['status']=='passed' for r in norms.values()) else 'failed',
                norms=norms,difference_measurement=measured)


def additive_action_check(pc,subksps,domains,matrix,checks,PETSc):
    n=matrix.getSize()[0]
    values=[math.sin((i+1)*0.123)+0.2*math.cos((i+1)*0.059) for i in range(n)]
    probe=matrix.createVecRight();actual=probe.duplicate();expected=probe.duplicate();delta=probe.duplicate()
    try:
        probe.setValues(list(range(n)),values);probe.assemble();expected.set(0.0)
        pc.apply(probe,actual)
        for sub,domain in zip(subksps,domains):
            local_matrix=sub.getOperators()[0];indices=domain['overlap_indices']
            require(local_matrix.getSize()==(len(indices),len(indices)),'Local matrix/restriction size mismatch.')
            local_rhs=local_matrix.createVecRight();local_solution=local_rhs.duplicate()
            try:
                local_rhs.setValues(list(range(len(indices))),[values[i] for i in indices]);local_rhs.assemble()
                # PREONLY applies this fixed local ILU action once. Bypass its KSP for the independent sum.
                sub.getPC().apply(local_rhs,local_solution)
                expected.setValues(indices,local_solution.getArray(readonly=True).copy(),addv=PETSc.InsertMode.ADD_VALUES)
            finally:
                local_rhs.destroy();local_solution.destroy()
        expected.assemble();actual.copy(delta);delta.axpy(-1.0,expected)
        defect,norm=float(delta.norm()),float(expected.norm())
        allowed=checks['additive_action_atol']+checks['additive_action_rtol']*norm
        passed=math.isfinite(defect) and math.isfinite(norm) and norm>0 and defect<=allowed
        return dict(status='passed' if passed else 'failed',input_definition='sin((i+1)*0.123)+0.2*cos((i+1)*0.059)',
            defect_norm=defect,reference_action_norm=norm,relative_defect=defect/norm if norm>0 else None,allowed_defect=allowed,
            actual_global_PC_applications=1,independent_direct_local_PC_applications=4)
    finally:
        probe.destroy();actual.destroy();expected.destroy();delta.destroy()


def manual_solve(config,context,block,load,W,parameters,label,record,fd,PETSc):
    record.update(status='running',method='RadauIIA',role=label,dt=config['dt'],solver_parameters=parameters)
    slopes=fd.Function(W,name=label+'_slopes').assign(0.0)
    prefix='phase3_'+label+'_'
    # PCASM's internal matrix partitioner inherits the MATRIX prefix, not the KSP prefix.
    block.petscmat.setOptionsPrefix(prefix)
    # Solve the independent assembled stage system directly through PETSc.
    from petsctools import inserted_options
    ksp=PETSc.KSP().create(comm=context['mesh'].comm)
    ksp.setOptionsPrefix(prefix);ksp.setOperators(block.petscmat)
    history=[];ksp.setMonitor(lambda solver,iteration,norm:history.append([iteration,float(norm)]))
    subksps=[];domains=[];work=[];phase={'name':'outer_solve'}
    try:
        with inserted_options(parameters=parameters,options_prefix=prefix):
            ksp.setFromOptions();pc=ksp.getPC()
            if parameters['pc_type']=='asm':
                options=PETSc.Options(prefix)
                observed={key:options.getString(key) for key in ('pc_asm_type','pc_asm_local_type','mat_partitioning_type',
                    'sub_ksp_type','sub_pc_type','sub_pc_factor_mat_ordering_type','sub_pc_factor_mat_solver_type','sub_pc_factor_shift_type')}
                observed.update({key:options.getInt(key) for key in ('pc_asm_blocks','pc_asm_overlap','sub_pc_factor_levels')})
                require(all(observed[key]==parameters[key] for key in observed),'Prefixed ASM options were overridden.')
                ksp.setUp()
                require(pc.getType()=='asm' and block.petscmat.getOptionsPrefix()==prefix,'ASM or partitioner prefix changed.')
                domains=inspect_domains(pc,block.petscmat);subksps=pc.getASMSubKSP()
                require(len(subksps)==4,'Expected four local KSP objects.')
                record.update(options_observed_in_scope=observed,matrix_options_prefix=prefix,domains=domains,local_work=work)
                def postsolve(sub,rhs,solution,index):
                    reason,iterations=int(sub.getConvergedReason()),sub.getIterationNumber()
                    require(sub.getType()=='preonly' and sub.getPC().getType()=='ilu' and reason>0 and iterations==1,
                            'Local action differs from one successful PREONLY/ILU application.')
                    counts=work[index][phase['name']];counts['calls']+=1;counts['reported_iterations']+=iterations
                for index,sub in enumerate(subksps):
                    require(sub.getType()=='preonly' and sub.getPC().getType()=='ilu','Unexpected local solver types.')
                    work.append(dict(index=index,outer_solve=dict(calls=0,reported_iterations=0),
                                     action_check=dict(calls=0,reported_iterations=0)))
                    sub.setPostSolve(postsolve,args=(index,))
            with load.dat.vec_ro as rhs,slopes.dat.vec_wo as solution:
                ksp.solve(rhs,solution)
    except Exception as error:
        capture_failed_solve(record,0,0.0,config['dt'],ksp,history,error)
        diagnostic=record.get('failed_step',{}).get('petsc_correction_residual')
        if diagnostic is not None:
            diagnostic['meaning']='Recomputed residual of the direct physical B Z = g solve; partial failed slopes were not accepted'
        raise
    rtol,atol,divtol,max_it=ksp.getTolerances()
    require(ksp.getType()=='gmres' and pc.getType()==parameters['pc_type'] and ksp.getPCSide()==PETSc.PC.Side.RIGHT
        and ksp.getNormType()==PETSc.KSP.NormType.UNPRECONDITIONED and not ksp.getInitialGuessNonzero()
        and rtol==parameters['ksp_rtol'] and atol==parameters['ksp_atol'] and max_it==parameters['ksp_max_it']
        and ksp.getConvergedReason()>0,'Actual coupled solver differs from its specification.')
    residual=true_residual(block,slopes,load);target=max(atol,rtol*residual['rhs_norm'])
    record.update(completed_operator_solves=1,stage_total_dofs=W.dim(),iterations=ksp.getIterationNumber(),
        converged_reason=int(ksp.getConvergedReason()),true_block_residual=residual,true_residual_target=target,
        residual_allowance_factor=config['checks']['residual_tolerance_factor'],reported_residual_history=history)
    require(all(math.isfinite(v) for v in residual.values()) and residual['norm']<=config['checks']['residual_tolerance_factor']*target,
            'Independent physical stage residual failed.')
    ksp.view()
    if parameters['pc_type']=='asm':
        require(all(w['outer_solve']['calls']>0 for w in work)
                and len({w['outer_solve']['calls'] for w in work})==1,'Unequal or missing additive local work.')
        record['local_solvers']=[dict(index=i,ksp_type=sub.getType(),pc_type=sub.getPC().getType(),
            factor_solver_type=sub.getPC().getFactorSolverType(),matrix_size=list(sub.getOperators()[0].getSize()),
            factor_nonzeros=float(sub.getPC().getFactorMatrix().getInfo()['nz_used']),
            factor_levels_evidence='Zero fill and RCM ordering in actual KSP view; prefixed options recorded') for i,sub in enumerate(subksps)]
        require(all(r['factor_solver_type']=='petsc' for r in record['local_solvers']),'Local factor backend changed.')
        phase['name']='action_check'
        record['additive_action_check']=additive_action_check(pc,subksps,domains,block.petscmat,config['checks'],PETSc)
        require(record['additive_action_check']['status']=='passed','Actual PC action differs from sum R_i^T local_ILU_i R_i.')
        require(all(w['action_check']['calls']==1 for w in work),'Diagnostic action should invoke each local KSP once.')
        record['outer_preconditioner_applications_from_local_calls']=work[0]['outer_solve']['calls']
    A,b,c=tableau_coefficients(radau_config(config))
    components=slopes.subfunctions
    state=fd.Function(context['space'],name=label+'_update').interpolate(
        context['initial']+config['dt']*sum(b[j]*components[j] for j in range(2)))
    boundary=[safe_sqrt(float(fd.assemble(slope**2*context['ds']))) for slope in components]
    require(all(value<=config['checks']['stage_boundary_trace_l2_max'] for value in boundary),'Nonzero slope boundary.')
    record.update(status='passed',completed_operator_solves=1,stage_total_dofs=W.dim(),iterations=ksp.getIterationNumber(),
        converged_reason=int(ksp.getConvergedReason()),true_block_residual=residual,true_residual_target=target,
        residual_allowance_factor=config['checks']['residual_tolerance_factor'],stage_boundary_trace_l2=boundary,
        source_times=[ci*config['dt'] for ci in c],reported_residual_history=history,
        actual_solver=dict(ksp_type=ksp.getType(),pc_type=pc.getType(),side='right',norm='unpreconditioned',
            rtol=rtol,atol=atol,divtol=divtol,max_it=max_it,requested_restart=parameters['ksp_gmres_restart'],
            restart_evidence='Actual KSP view in transcript'))
    ksp.destroy()
    return state,slopes


def study(config,result,fd,PETSc,irk,AI,comm):
    require(comm.size==1,'This construction check is serial.');validate_config(config)
    n=config['mesh_subdivisions'];mesh=fd.UnitSquareMesh(n,n,diagonal='left',comm=comm)
    V=fd.FunctionSpace(mesh,'CG',config['element_degree']);x,y=fd.SpatialCoordinate(mesh)
    sinusoid=fd.sin(fd.pi*x)*fd.sin(fd.pi*y)
    gradient=fd.as_vector((fd.pi*fd.cos(fd.pi*x)*fd.sin(fd.pi*y),fd.pi*fd.sin(fd.pi*x)*fd.cos(fd.pi*y)))
    beta,kappa=fd.as_vector(config['velocity']),config['diffusion'];bc=fd.DirichletBC(V,0.,'on_boundary')
    initial=fd.Function(V,name='asm_shared_initial').interpolate(sinusoid);bc.apply(initial)
    context=dict(mesh=mesh,space=V,initial=initial,sinusoid=sinusoid,exact_gradient=gradient,beta=beta,bc=bc,
        dx=fd.dx(domain=mesh,metadata={'quadrature_degree':config['quadrature_degree']}),
        ds=fd.ds(domain=mesh,metadata={'quadrature_degree':config['quadrature_degree']}),
        spatial=lambda w,v:fd.dot(beta,fd.grad(w))*v+kappa*fd.inner(fd.grad(w),fd.grad(v)),
        forcing=lambda time:fd.exp(-time)*((2*kappa*fd.pi**2-1)*sinusoid+gradient[0]+gradient[1]))
    require(V.dim()==1089,'Unexpected P2/n16 dimension.')
    result.update(total_dofs=V.dim(),stage_total_dofs=2*V.dim(),checkpoint_end_time=config['dt'],research_final_time=1.0)
    baseline={'status':'prepared','role':'irksome_jacobi'};result['solves']['irksome_jacobi']=baseline;result['active_solve']='irksome_jacobi'
    baseline_config=radau_config({**config,'final_time':config['dt']})
    irksome_state=run_trajectory(baseline_config,context,config['dt'],1.0,'asm_irksome_jacobi',baseline,fd,PETSc,irk,AI)
    require(baseline['status']=='passed' and baseline['completed_steps']==1,'One-step Irksome baseline failed.')
    A,b,c=tableau_coefficients(radau_config(config));W=V*V
    z,q=fd.TrialFunctions(W),fd.TestFunctions(W);spatial,forcing=context['spatial'],context['forcing'];dt=config['dt']
    block_form=sum(z[i]*q[i]+dt*sum(A[i][j]*spatial(z[j],q[i]) for j in range(2)) for i in range(2))*context['dx']
    load_form=sum(forcing(c[i]*dt)*q[i]-spatial(initial,q[i]) for i in range(2))*context['dx']
    bcs=[fd.DirichletBC(W.sub(i),0.,'on_boundary') for i in range(2)]
    block=fd.assemble(block_form,bcs=bcs,mat_type='aij');load=fd.assemble(load_form,bcs=bcs)
    require(W.dim()==2178 and block.petscmat.getSize()==(2178,2178)
            and block.petscmat.getBlockSize()==1,'Unexpected coupled scalar AIJ layout.')
    reference_parameters=dict(config['solver_parameters'])
    for key in ('ksp_rtol','ksp_atol'):reference_parameters[key]*=config['manual_reference_tolerance_factor']
    reference={'status':'prepared'};result['solves']['manual_jacobi_reference']=reference;result['active_solve']='manual_jacobi_reference'
    reference_state,reference_slopes=manual_solve(config,context,block,load,W,reference_parameters,'manual_jacobi_reference',reference,fd,PETSc)
    asm={'status':'prepared'};result['solves']['manual_asm']=asm;result['active_solve']='manual_asm'
    asm_state,asm_slopes=manual_solve(config,context,block,load,W,config['asm_solver_parameters'],'manual_asm',asm,fd,PETSc)
    result.pop('active_solve',None)
    result['comparisons']['manual_reference_update_vs_irksome']=compare(reference_state,irksome_state,config,context,fd)
    result['comparisons']['asm_update_vs_manual_reference']=compare(asm_state,reference_state,config,context,fd)
    for i in range(2):
        result['comparisons'][f'asm_slope{i+1}_vs_manual_reference']=compare(asm_slopes.subfunctions[i],reference_slopes.subfunctions[i],config,context,fd)
    result.update(status='passed' if all(r['status']=='passed' for r in result['comparisons'].values()) else 'failed',
                  completed_study=True,completed_operator_solves=3)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();sys.argv=[sys.argv[0]]
    result=dict(status='failed',completed_study=False,ranks=1,solves={},comparisons={},performance_measurement=False,
                purpose='Standard ASM/local ILU construction on one Radau step; no timing or accuracy-cost comparison')
    allowed=args.output.resolve().parent==ROOT/'results/raw'
    try:
        require(allowed,'Keep all records in the Projects repository.')
        config=json.loads(args.config.read_text());result['config']=config
        from mpi4py import MPI
        require(MPI.COMM_WORLD.size==1,'Serial construction check required.')
        import firedrake as fd
        from firedrake.petsc import PETSc
        import irksome as irk
        from irksome.tools import AI
        study(config,result,fd,PETSc,irk,AI,MPI.COMM_WORLD)
    except (Exception,KeyboardInterrupt) as error:
        result.update(status='failed',error=str(error) or 'Interrupted by researcher')
        if result.get('active_solve'):result['solves'][result['active_solve']]['status']='failed'
        traceback.print_exc()
    finally:
        if allowed:args.output.write_text(json.dumps(sanitize(result),indent=2,allow_nan=False)+'\n')
    return 0 if result['status']=='passed' else 1


if __name__=='__main__':
    sys.exit(main())

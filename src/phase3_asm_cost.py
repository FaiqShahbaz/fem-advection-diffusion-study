#!/usr/bin/env python3
"""Researcher-run matched manual Radau Jacobi/ASM warm timing and field verification."""
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
from phase3_asm_construction import inspect_domains,additive_action_check
from phase3_asm_trajectory import make_context,manual_trajectory
from phase3_cost_pilot import EVENTS,event_snapshot,event_difference

ROOT=Path(__file__).resolve().parents[1]


def validate_config(config):
    accepted=json.loads((ROOT/'configs/phase3_asm_trajectory.json').read_text())
    keys=('mesh_subdivisions','element_degree','element_family','mesh_diagonal','velocity','diffusion','final_time',
        'dt','quadrature_degree','quadrature_control_degree','checks','solver_parameters','asm_solver_parameters',
        'matrix_block_size_expected','radau_formulation')
    require(all(config[k]==accepted[k] for k in keys),'Keep the accepted fixed trajectory, checks and both PCs.')
    require(config['cases']==[dict(case_id='Jacobi',accepted_role='manual_jacobi'),dict(case_id='ASM',accepted_role='manual_asm')]
        and config['warmups_per_case']==1 and config['repetitions_per_case']==3
        and config['measured_case_order']==[['Jacobi','ASM'],['ASM','Jacobi'],['Jacobi','ASM']]
        and config['performance_measurement'] is True and config['required_power_source']=='AC Power',
        'Keep the explained two-case, three-repeat warm timing comparison.')
    require(step_count(config['final_time'],config['dt'])==10,'Keep ten steps per trajectory.')


def parameters_for(config,case):
    return config['solver_parameters'] if case['case_id']=='Jacobi' else config['asm_solver_parameters']


def timed_trajectory(config,context,parameters,label,record,baseline,baseline_record,stage,events,fd,PETSc):
    from petsctools import inserted_options
    dt=config['dt'];count=step_count(config['final_time'],dt);A,b,c=tableau_coefficients(radau_config(config))
    prefix='phase3_asm_cost_'+label+'_';V=context['space'];spatial,forcing=context['spatial'],context['forcing']
    record.update(status='running',role=record['role'],method='RadauIIA',dt=dt,planned_steps=count,completed_steps=0,
        solver_parameters=parameters,step_records=[],stage_total_dofs=2178,matrix_assemblies=1,
        timed_python_monitors=False,timed_local_callbacks=False)
    ksp=None
    gc.collect()  # Deliberate collection lies outside the timing boundary.
    try:
        with stage:
            event_before=event_snapshot(events,stage)
            start=perf_counter()
            with inserted_options(parameters=parameters,options_prefix=prefix):
                W=V*V
                state=fd.Function(V,name=label+'_state').assign(context['initial'])
                old=fd.Function(V,name=label+'_old');slopes=fd.Function(W,name=label+'_slopes');t=fd.Constant(0.)
                z,q=fd.TrialFunctions(W),fd.TestFunctions(W)
                bcs=[fd.DirichletBC(W.sub(i),0.,'on_boundary') for i in range(2)]
                block_form=sum(z[i]*q[i]+dt*sum(A[i][j]*spatial(z[j],q[i]) for j in range(2)) for i in range(2))*context['dx']
                load_form=sum(forcing(t+c[i]*dt)*q[i]-spatial(old,q[i]) for i in range(2))*context['dx']
                block=fd.assemble(block_form,bcs=bcs,mat_type='aij');block.petscmat.setOptionsPrefix(prefix)
                ksp=PETSc.KSP().create(comm=context['mesh'].comm)
                ksp.setOptionsPrefix(prefix);ksp.setOperators(block.petscmat);ksp.setFromOptions();ksp.setUp()
                components=slopes.subfunctions
                construction_end=perf_counter()
                for index in range(count):
                    step_start=perf_counter()
                    old_time=index*dt;t.assign(old_time);old.assign(state);slopes.assign(0.)
                    load=fd.assemble(load_form,bcs=bcs)
                    try:
                        with load.dat.vec_ro as rhs,slopes.dat.vec_wo as solution:ksp.solve(rhs,solution)
                    except Exception as error:
                        capture_failed_solve(record,index,old_time,dt,ksp,[],error)
                        diagnostic=record.get('failed_step',{}).get('petsc_correction_residual')
                        if diagnostic is not None:diagnostic['meaning']='Direct physical B Z = g residual of unaccepted partial slopes'
                        raise
                    state.interpolate(old+dt*sum(b[j]*components[j] for j in range(2)))
                    iterations,reason=ksp.getIterationNumber(),int(ksp.getConvergedReason())
                    require(reason>0,'Timed physical solve did not converge.')
                    step_end=perf_counter()
                    record['step_records'].append(dict(step=index+1,t_start=old_time,t_end=(index+1)*dt,
                        stage_times=[old_time+ci*dt for ci in c],iterations=iterations,converged_reason=reason,
                        wall_seconds=step_end-step_start))
                    record['completed_steps']=index+1
                end=perf_counter()
                # Diagnostics from this point onward are outside the core timer.
                options=PETSc.Options(prefix)
                record['options_observed_in_scope']={k:options.getString(k) for k in parameters if isinstance(parameters[k],str)}
                record['options_observed_in_scope'].update({k:options.getBool(k) if isinstance(v,bool) else options.getInt(k) for k,v in parameters.items() if isinstance(v,(bool,int))})
            event_after=event_snapshot(events,stage)
        wall=end-start;construction=construction_end-start;step_walls=[r['wall_seconds'] for r in record['step_records']]
        record['timing']=dict(construction_seconds=construction,first_step_seconds=step_walls[0],
            remaining_steps_seconds=sum(step_walls[1:]),trajectory_wall_seconds=wall,
            loop_bookkeeping_seconds=wall-construction-sum(step_walls),petsc_event_deltas=event_difference(event_before,event_after),
            interpretation='Inclusive nested event times; do not add event times to each other or to wall time')
        require(all(math.isfinite(v) and v>=0 for k,v in record['timing'].items() if k.endswith('_seconds')),'Invalid timing partition.')
        rtol,atol,divtol,max_it=ksp.getTolerances();pc=ksp.getPC()
        require(ksp.getType()=='gmres' and pc.getType()==parameters['pc_type'] and ksp.getPCSide()==PETSc.PC.Side.RIGHT
            and ksp.getNormType()==PETSc.KSP.NormType.UNPRECONDITIONED and not ksp.getInitialGuessNonzero()
            and rtol==parameters['ksp_rtol'] and atol==parameters['ksp_atol'] and max_it==1000,'Actual timed solver changed.')
        require(W.dim()==2178 and block.petscmat.getSize()==(2178,2178) and block.petscmat.getBlockSize()==1,'Unexpected timed matrix.')
        require(all(v==parameters[k] for k,v in record['options_observed_in_scope'].items()),'Timed solver options changed.')
        record['actual_solver']=dict(ksp_type=ksp.getType(),pc_type=pc.getType(),side='right',norm='unpreconditioned',
            rtol=rtol,atol=atol,divtol=divtol,max_it=max_it,requested_restart=50,restart_evidence='Actual KSP view in transcript')
        ksp.view()
        prof=record['timing']['petsc_event_deltas']
        checked_events=baseline_record['operator_event_counts']
        require(all(prof[e]['count']==checked_events[e] for e in EVENTS),'Timed event/work counts differ from the checked path.')
        require([s['iterations'] for s in record['step_records']]==[s['iterations'] for s in baseline_record['step_records']],
                'Timed iteration sequence differs from checked trajectory.')
        # Reassemble the textbook RHS at the preserved LAST old time and state, not at T.
        final_load=fd.assemble(load_form,bcs=bcs);residual=true_residual(block,slopes,final_load)
        target=max(atol,rtol*residual['rhs_norm'])
        l2=lambda expr,measure=context['dx']:safe_sqrt(float(fd.assemble(expr**2*measure)))
        final=dict(status='passed',true_block_residual=residual,true_residual_target=target,
            stage_times=[float(t)+ci*dt for ci in c],
            update_identity_l2=l2(state-(old+dt*sum(b[j]*components[j] for j in range(2)))),
            stiff_accuracy_l2=l2(state-(old+dt*sum(A[1][j]*components[j] for j in range(2)))),
            stage_boundary_trace_l2=[l2(slope,context['ds']) for slope in components])
        record['final_equation_check']=final
        require(all(math.isfinite(v) for v in residual.values()) and residual['norm']<=10*target
            and final['update_identity_l2']<=1e-12 and final['stiff_accuracy_l2']<=1e-12
            and all(v<=1e-12 for v in final['stage_boundary_trace_l2']),'Timed final physical equation/update/boundary check failed.')
        record['source_checks']=[]
        for value in (0.,1.,*final['stage_times']):
            exact=fd.exp(-value)*context['sinusoid']
            defect=l2(-exact+fd.dot(context['beta'],fd.grad(exact))-config['diffusion']*fd.div(fd.grad(exact))-forcing(value))
            record['source_checks'].append(dict(time=value,source_residual_l2=defect))
        require(all(s['source_residual_l2']<=1e-10 for s in record['source_checks']),'Timed source check failed.')
        primary=error_metrics(state,1.,context,config['quadrature_degree'],fd)
        control=error_metrics(state,1.,context,config['quadrature_control_degree'],fd)
        checks,quadrature=check_metrics(primary,control,1.,config['checks'])
        require(all(checks.values()),'Timed final MMS/quadrature check failed.')
        difference=measured_difference(state,baseline,config,context,fd)
        norm=norm_pair(baseline,fd,fd.dx(domain=context['mesh'],metadata={'quadrature_degree':config['quadrature_control_degree']}))
        equivalence={metric:dict(actual_field_difference=difference['metrics'][metric],baseline_field_norm=norm[metric],
            allowed_difference=config['checks']['equivalence_atol']+config['checks']['equivalence_rtol']*norm[metric]) for metric in METRICS}
        record.update(metrics=primary,control_metrics=control,checks=checks,quadrature_comparison=quadrature,
            difference_measurement=difference,equivalence=equivalence)
        require(difference['status']=='passed' and all(r['actual_field_difference']<=r['allowed_difference'] for r in equivalence.values()),
                'Timed actual fields differ from the verified baseline.')
        if parameters['pc_type']=='asm':
            domains=inspect_domains(pc,block.petscmat);subksps=pc.getASMSubKSP()
            require(domains==baseline_record['domains'] and len(subksps)==4,'Timed ASM partition/overlap changed.')
            record['domains']=domains
            record['local_solvers']=[dict(index=i,ksp_type=sub.getType(),pc_type=sub.getPC().getType(),
                factor_solver_type=sub.getPC().getFactorSolverType(),matrix_size=list(sub.getOperators()[0].getSize()),
                factor_nonzeros=float(sub.getPC().getFactorMatrix().getInfo()['nz_used'])) for i,sub in enumerate(subksps)]
            require(record['local_solvers']==baseline_record['local_solvers'],'Timed local solvers/factors changed.')
            applications=int(prof['PCApply']['count'])
            require(all(w['outer_solve']['calls']==applications for w in baseline_record['local_work']),
                    'Checked callbacks do not support the fixed-work derivation.')
            record['local_work_derived']=dict(basis='Top-level PCApply count times fixed four-domain additive PREONLY/ILU composition; no timed Python callbacks',
                applications_per_domain=applications,total_local_applications=4*applications,domains=4)
            # A separate untimed action probe; excluded from event deltas and derived solve work.
            record['diagnostic_additive_action']=additive_action_check(pc,subksps,domains,block.petscmat,config['checks'],PETSc)
            require(record['diagnostic_additive_action']['status']=='passed','Timed solver final diagnostic action failed.')
        record.update(status='passed',final_time=1.,total_iterations=sum(s['iterations'] for s in record['step_records']),
            maximum_step_iterations=max(s['iterations'] for s in record['step_records']))
        print(f"  {label}: checked core wall={wall:.6f} s; GMRES total={record['total_iterations']}",flush=True)
    finally:
        if ksp is not None:ksp.destroy()


def study(config,result,fd,PETSc,comm):
    require(comm.size==1,'Keep this serial timing comparison.');validate_config(config)
    context=make_context(config,fd,comm);require(context['space'].dim()==1089,'Unexpected space.')
    result.update(total_dofs=1089,stage_total_dofs=2178)
    review=json.loads((ROOT/'results/processed/asm_trajectory_review.json').read_text())
    accepted=json.loads((ROOT/review['measurement_records']['serial']['path']).read_text())
    fields={};records=result['trajectories']
    for case in config['cases']:
        label=case['case_id']+'_verified';row=dict(status='prepared',role='verified',case_id=case['case_id']);records[label]=row;result['active_trajectory']=label
        state,_=manual_trajectory(config,context,parameters_for(config,case),'asm_cost_'+label,row,fd,PETSc)
        previous=accepted['trajectories'][case['accepted_role']]
        require(row['status']=='passed' and [s['iterations'] for s in row['step_records']]==[s['iterations'] for s in previous['step_records']],
                'Accepted verified work sequence changed.')
        require(row['operator_event_counts']==previous['operator_event_counts'],'Accepted profiling/work configuration changed.')
        if case['case_id']=='ASM':require(row['domains']==previous['domains'] and row['local_solvers']==previous['local_solvers'],'Accepted ASM construction changed.')
        row['accepted_metric_reproduction']={m:math.isclose(row['control_metrics'][m],previous['control_metrics'][m],rel_tol=1e-9,abs_tol=1e-12) for m in METRICS}
        require(all(row['accepted_metric_reproduction'].values()),'Accepted norm reproduction failed.')
        fields[case['case_id']]=state
    pair=measured_difference(fields['ASM'],fields['Jacobi'],config,context,fd)
    result['verified_pair_difference']=pair
    pair_norms=norm_pair(fields['Jacobi'],fd,fd.dx(domain=context['mesh'],metadata={'quadrature_degree':config['quadrature_control_degree']}))
    result['verified_pair_equivalence']={m:dict(actual_field_difference=pair['metrics'][m],baseline_field_norm=pair_norms[m],
        allowed_difference=config['checks']['equivalence_atol']+config['checks']['equivalence_rtol']*pair_norms[m]) for m in METRICS}
    require(pair['status']=='passed' and all(r['actual_field_difference']<=r['allowed_difference'] for r in result['verified_pair_equivalence'].values()),
            'Checked PC fields differ materially.')
    stage=PETSc.Log.Stage('phase3_asm_cost_core');events={name:PETSc.Log.Event(name) for name in EVENTS}
    sequence=[(case['case_id'],'warmup0') for case in config['cases']]
    sequence += [(case_id,f'measured{i}') for i,ids in enumerate(config['measured_case_order'],1) for case_id in ids]
    for case_id,suffix in sequence:
        case=next(c for c in config['cases'] if c['case_id']==case_id);label=case_id+'_'+suffix
        row=dict(status='prepared',role='warmup' if suffix=='warmup0' else 'measured',case_id=case_id)
        records[label]=row;result['active_trajectory']=label;result['execution_order'].append(label)
        timed_trajectory(config,context,parameters_for(config,case),label,row,fields[case_id],records[case_id+'_verified'],stage,events,fd,PETSc)
    for case in config['cases']:
        rows=[records[case['case_id']+'_measured'+str(i)] for i in (1,2,3)]
        summary={}
        for key in ('construction_seconds','first_step_seconds','remaining_steps_seconds','trajectory_wall_seconds'):
            values=[r['timing'][key] for r in rows]
            summary[key]=dict(values=values,minimum=min(values),median=statistics.median(values),maximum=max(values))
        result['timing_summary'][case['case_id']]=summary
    result['jacobi_over_asm_median_core_ratio']=result['timing_summary']['Jacobi']['trajectory_wall_seconds']['median']/result['timing_summary']['ASM']['trajectory_wall_seconds']['median']
    result.pop('active_trajectory',None)
    result.update(status='passed',completed_study=True,completed_trajectories=10,completed_steps=100)


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
        PETSc.Log.begin();study(config,result,fd,PETSc,MPI.COMM_WORLD)
    except (Exception,KeyboardInterrupt) as error:
        result.update(status='failed',error=str(error) or 'Interrupted by researcher')
        if result.get('active_trajectory'):result['trajectories'][result['active_trajectory']]['status']='failed'
        traceback.print_exc()
    finally:
        if allowed:args.output.write_text(json.dumps(sanitize(result),indent=2,allow_nan=False)+'\n')
    return 0 if result['status']=='passed' else 1

if __name__=='__main__':sys.exit(main())

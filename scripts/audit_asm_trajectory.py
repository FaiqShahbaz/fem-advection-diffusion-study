#!/usr/bin/env python3
"""Audit saved full ASM trajectory evidence using only the standard library."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
STEM=ROOT/'results/raw/asm-trajectory-20261019T115856Z-96535'
METRICS=('l2_error','h1_seminorm_error')

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def audit():
    sys.path.insert(0,str(ROOT/'environment'))
    import check_asm_trajectory as runner
    runner.plan()
    m=json.loads(Path(str(STEM)+'-manifest.json').read_text())
    raw=ROOT/m['measurement_records']['serial']['path'];d=json.loads(raw.read_text());config=d['config']
    assert m['checkpoint_exit_status']==0 and m['working_tree_status_before_run']=='' and m['completed_steps']==['serial']
    assert m['performance_measurement'] is False and m['inherited_options_or_library_settings']=={}
    assert m['selected_environment']['OMP_NUM_THREADS']==m['selected_environment']['OPENBLAS_NUM_THREADS']=='1'
    for key in ('asm_construction_review','verified_environment','config'):
        assert digest(ROOT/m[key])==m[key+'_sha256']
    for path,expected in m['script_sha256'].items():
        assert digest(ROOT/path)==expected,path
        assert hashlib.sha256(subprocess.check_output(['git','show',m['code_commit']+':'+path],cwd=ROOT)).hexdigest()==expected,path
    assert digest(raw)==m['measurement_records']['serial']['sha256'] and config==json.loads(runner.CONFIG.read_text())
    assert runner.verify_complete(d,config)==m['study_validation']
    extrema=dict(max_true_residual_over_nominal_target=0.,max_field_difference=0.,max_update_or_stiff_accuracy_l2=0.,max_quadrature_change=0.)
    def field_check(item):
        assert item['status']==item['difference_measurement']['status']=='passed'
        measured=item['difference_measurement']
        for metric,row in item['norms'].items():
            value=measured['metrics'][metric];primary=measured['primary_metrics'][metric]
            assert math.isfinite(value) and value>=0 and value==row['actual_field_difference']
            assert row['allowed_difference']==1e-11+1e-8*row['reference_field_norm'] and value<=row['allowed_difference']
            q=measured['quadrature_comparison'][metric]
            assert q['absolute_difference']==abs(primary-value) and q['allowed_difference']==1e-13+1e-6*value
            assert q['absolute_difference']<=q['allowed_difference']
            extrema['max_field_difference']=max(extrema['max_field_difference'],value)
            extrema['max_quadrature_change']=max(extrema['max_quadrature_change'],abs(primary-value))
    summary={};completed=0
    for label,row in d['trajectories'].items():
        params=row['solver_parameters'];actual=row['actual_solver']
        expected=dict(config['asm_solver_parameters'] if label=='manual_asm' else config['solver_parameters'])
        if label=='manual_jacobi_reference':
            for key in ('ksp_rtol','ksp_atol'):expected[key]*=.1
        assert all(params[k]==v for k,v in expected.items())
        assert actual['ksp_type']=='gmres' and actual['pc_type']==params['pc_type'] and actual['side']=='right' and actual['norm']=='unpreconditioned'
        assert actual['rtol']==params['ksp_rtol'] and actual['atol']==params['ksp_atol'] and actual['max_it']==1000 and actual['requested_restart']==50
        assert row['total_iterations']==sum(s['iterations'] for s in row['step_records'])
        assert row['maximum_step_iterations']==max(s['iterations'] for s in row['step_records'])
        assert len(row['step_records'])==row['completed_steps']==10 and row['final_time']==1.0 and all(row['checks'].values())
        for i,step in enumerate(row['step_records']):
            assert step['step']==i+1 and abs(step['t_start']-i*.1)<=1e-14 and abs(step['t_end']-(i+1)*.1)<=1e-14
            assert all(abs(a-b)<1e-14 for a,b in zip(step['stage_times'],[(i+1/3)*.1,(i+1)*.1]))
            assert step['converged_reason']>0 and 0<step['iterations']<=1000
            r=step['true_block_residual'];target=max(params['ksp_atol'],params['ksp_rtol']*r['rhs_norm'])
            assert all(math.isfinite(v) and v>=0 for v in r.values()) and r['rhs_norm']>0
            assert math.isclose(r['relative_norm'],r['norm']/r['rhs_norm'],rel_tol=1e-14)
            assert target==step['true_residual_target'] and r['norm']<=10*target
            extrema['max_true_residual_over_nominal_target']=max(extrema['max_true_residual_over_nominal_target'],r['norm']/target)
            for key in ('update_identity_l2','stiff_accuracy_l2'):
                assert 0<=step[key]<=1e-12
                extrema['max_update_or_stiff_accuracy_l2']=max(extrema['max_update_or_stiff_accuracy_l2'],step[key])
            if label!='irksome_jacobi':
                assert step['event_counts']['KSPSolve']==1 and step['event_counts']['KSPSetUp']==0 and step['event_counts']['PCSetUp']==0
                assert all(v<=1e-12 for v in step['stage_boundary_trace_l2'])
                assert [s['time'] for s in step['source_checks']]==step['stage_times'] and all(s['source_residual_l2']<=1e-10 for s in step['source_checks'])
                if label!='manual_jacobi_reference':
                    for item in step['reference_comparisons'].values():field_check(item)
        for values in (row['metrics'],row['control_metrics']):
            assert abs(values['domain_area']-1)<=1e-12 and values['boundary_trace_l2']<=1e-12
            assert math.isclose(values['exact_l2_norm'],math.exp(-1)/2,rel_tol=1e-10,abs_tol=1e-12)
            assert math.isclose(values['exact_h1_seminorm'],math.pi*math.exp(-1)/math.sqrt(2),rel_tol=1e-10,abs_tol=1e-12)
        for metric in METRICS:
            assert abs(row['metrics'][metric]-row['control_metrics'][metric])<=1e-13+1e-6*row['control_metrics'][metric]
        if label=='irksome_jacobi':
            assert row['solver_stats']==[10,10,row['total_iterations']]
            assert all(s['source_residual_l2']<=1e-10 for s in row['source_checks'])
        else:
            for event,count in row['operator_event_counts'].items():
                assert count==row['initial_setup_event_counts'][event]+sum(s['event_counts'][event] for s in row['step_records'])
            assert row['matrix_assemblies']==1 and row['operator_event_counts']['PCSetUp']==1
        summary[label]=dict(total_iterations=row['total_iterations'],maximum_step_iterations=row['maximum_step_iterations'],
            operator_event_counts=row.get('operator_event_counts'),continuum_metrics=row['control_metrics'])
        completed+=10
    assert completed==d['completed_steps']==40
    for item in d['comparisons'].values():field_check(item)
    asm=d['trajectories']['manual_asm'];domains=asm['domains']
    assert sorted(v for row in domains for v in row['core_indices'])==list(range(2178))
    assert all(v==config['asm_solver_parameters'][k] for k,v in asm['options_observed_in_scope'].items())
    for i,(domain,solver,work) in enumerate(zip(domains,asm['local_solvers'],asm['local_work'])):
        core,outer=domain['core_indices'],domain['overlap_indices']
        assert len(core)==domain['core_size'] and len(outer)==domain['overlap_size']==len(set(outer)) and domain['one_layer_check']
        assert set(core)<=set(outer) and solver['matrix_size']==[len(outer),len(outer)]
        assert work['index']==i and solver['ksp_type']=='preonly' and solver['pc_type']=='ilu' and solver['factor_solver_type']=='petsc'
        assert solver['factor_nonzeros']>0
        calls=sum(s['local_calls_per_domain'][i] for s in asm['step_records'])
        assert calls==asm['operator_event_counts']['PCApply'] and work['outer_solve']==dict(calls=calls,reported_iterations=calls)
        assert work['action_check']==dict(calls=1,reported_iterations=1)
    action=asm['additive_action_check'];assert action['allowed_defect']==1e-12+1e-11*action['reference_action_norm']
    assert action['defect_norm']<=action['allowed_defect'] and action['relative_defect']==action['defect_norm']/action['reference_action_norm']
    cost_review=json.loads((ROOT/'results/processed/accuracy_cost_review.json').read_text())
    cost=json.loads((ROOT/cost_review['measurement_records']['serial']['path']).read_text())
    accepted=cost['trajectories']['Radau_coarse_verified']
    assert [s['iterations'] for s in d['trajectories']['irksome_jacobi']['step_records']]==[s['iterations'] for s in accepted['step_records']]
    curve=next(c for c in cost['curve'] if c['case_id']=='Radau_coarse')
    for label in ('manual_jacobi','manual_asm'):
        for metric,budget in d['trajectories'][label]['accepted_time_error_budget_check'].items():
            assert budget['accepted_baseline_time_error']==curve['time_errors'][metric]
            assert budget['actual_field_distance']==d['comparisons'][label+'_vs_irksome']['norms'][metric]['actual_field_difference']
            assert budget['allowed_distance']==1e-12+.01*curve['time_errors'][metric] and budget['actual_field_distance']<=budget['allowed_distance']
    return dict(status='passed',trajectories=4,steps=40,trajectory_summary=summary,extrema=extrema,
        final_field_comparisons={k:{n:r['actual_field_difference'] for n,r in v['norms'].items()} for k,v in d['comparisons'].items()},
        local_calls_per_domain=[r['outer_solve']['calls'] for r in asm['local_work']],
        total_outer_local_calls=sum(r['outer_solve']['calls'] for r in asm['local_work']),
        additive_action_check=action,performance_measurement=False,
        audit_limit='Saved arithmetic, source/raw hashes and index consistency audited without rerunning numerical operations; reuse/action/field evidence was recorded by the researcher run.')

if __name__=='__main__':print(json.dumps(audit(),indent=2,allow_nan=False))

#!/usr/bin/env python3
"""Audit saved full RanaLD trajectory evidence without numerical imports or operations."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
STEM=ROOT/'results/raw/rana-trajectory-20261021T111718Z-20060'
METRICS=('l2_error','h1_seminorm_error')

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def audit():
    sys.path.insert(0,str(ROOT/'environment'))
    import check_rana_trajectory as runner
    runner.plan()
    m=json.loads(Path(str(STEM)+'-manifest.json').read_text());raw=ROOT/m['measurement_records']['serial']['path']
    d=json.loads(raw.read_text());config=d['config']
    assert m['checkpoint_exit_status']==0 and m['working_tree_status_before_run']=='' and m['completed_steps']==['serial']
    assert m['performance_measurement'] is False and m['accuracy_cost_ranking_established'] is False
    assert m['inherited_options_or_library_settings']=={} and m['selected_environment']['OMP_NUM_THREADS']==m['selected_environment']['OPENBLAS_NUM_THREADS']=='1'
    for key in ('rana_construction_review','verified_environment','config','installed_source_inspection'):
        assert digest(ROOT/m[key])==m[key+'_sha256'],key
    for path,expected in m['script_sha256'].items():
        assert digest(ROOT/path)==expected,path
        assert hashlib.sha256(subprocess.check_output(['git','show',m['code_commit']+':'+path],cwd=ROOT)).hexdigest()==expected,path
    for path,expected in m['inspected_installed_source_sha256'].items():assert digest(ROOT/path)==expected,path
    assert digest(raw)==m['measurement_records']['serial']['sha256'] and config==json.loads(runner.CONFIG.read_text())
    assert runner.verify_complete(d,config)==m['study_validation']
    extrema=dict(max_true_residual_over_nominal_target=0.,max_field_difference=0.,max_update_or_stiff_accuracy_l2=0.,max_quadrature_change=0.,max_matrix_frobenius_defect=0.)
    def field_check(item):
        measured=item['difference_measurement'];assert measured['status']==item['status']=='passed'
        for metric,row in item['norms'].items():
            value=measured['metrics'][metric];primary=measured['primary_metrics'][metric]
            assert math.isfinite(value) and value>=0 and row['actual_field_difference']==value
            assert row['allowed_difference']==1e-11+1e-8*row['reference_field_norm'] and value<=row['allowed_difference']
            q=measured['quadrature_comparison'][metric]
            assert q['absolute_difference']==abs(primary-value) and q['allowed_difference']==1e-13+1e-6*value and q['absolute_difference']<=q['allowed_difference']
            extrema['max_field_difference']=max(extrema['max_field_difference'],value)
            extrema['max_quadrature_change']=max(extrema['max_quadrature_change'],abs(primary-value))
    summary={}
    for label,row in d['trajectories'].items():
        assert row['total_iterations']==sum(s['iterations'] for s in row['step_records'])
        assert row['maximum_step_iterations']==max(s['iterations'] for s in row['step_records'])
        assert row['completed_steps']==len(row['step_records'])==10 and all(row['checks'].values())
        expected=dict(config['rana_solver_parameters'] if label=='rana_ld' else config['solver_parameters'])
        if label=='manual_jacobi_reference':
            for key in ('ksp_rtol','ksp_atol'):expected[key]*=config['manual_reference_tolerance_factor']
        if label=='residual_jacobi':expected.update(mat_type='aij',snes_type='ksponly')
        assert all(row['solver_parameters'][k]==v for k,v in expected.items())
        actual=row['actual_solver']
        assert actual['ksp_type']=='gmres' and actual['pc_type']==expected['pc_type'] and actual['side']=='right' and actual['norm']=='unpreconditioned'
        assert actual['rtol']==expected['ksp_rtol'] and actual['atol']==expected['ksp_atol'] and actual['max_it']==1000 and actual['requested_restart']==50
        for i,step in enumerate(row['step_records']):
            assert step['step']==i+1 and abs(step['t_start']-i*.1)<=1e-14 and abs(step['t_end']-(i+1)*.1)<=1e-14
            assert all(abs(a-b)<=1e-14 for a,b in zip(step['stage_times'],[(i+1/3)*.1,(i+1)*.1]))
            r=step['true_block_residual'];target=max(expected['ksp_atol'],expected['ksp_rtol']*r['rhs_norm'])
            assert all(math.isfinite(v) and v>=0 for v in r.values()) and r['rhs_norm']>0 and math.isclose(r['relative_norm'],r['norm']/r['rhs_norm'],rel_tol=1e-14)
            assert target==step['true_residual_target'] and r['norm']<=10*target and step['converged_reason']>0 and 0<step['iterations']<=1000
            extrema['max_true_residual_over_nominal_target']=max(extrema['max_true_residual_over_nominal_target'],r['norm']/target)
            for key in ('update_identity_l2','stiff_accuracy_l2'):
                assert 0<=step[key]<=1e-12
                extrema['max_update_or_stiff_accuracy_l2']=max(extrema['max_update_or_stiff_accuracy_l2'],step[key])
            if label!='irksome_jacobi':
                assert all(v<=1e-12 for v in step['stage_boundary_trace_l2']) and all(s['source_residual_l2']<=1e-10 for s in step['source_checks'])
                if i>0:assert step['event_counts']['PCSetUp']==0
            if label in ('residual_jacobi','rana_ld'):
                assert step['matrix_reuse_tokens']==row['step_records'][0]['matrix_reuse_tokens'] and step['reuse_matches_first_step']
                assert step['event_counts']['KSPSolve']==1 and step['event_counts']['KSPSetUp']==1
                for token in step['matrix_reuse_tokens'].values():assert token['object_id']>0 and token['state']>=0
                for matrix in step['matrix_checks'].values():
                    assert matrix['status']=='passed' and 0<=matrix['frobenius_defect']<=matrix['allowed_defect']==1e-11+1e-12*matrix['reference_frobenius_norm']
                    extrema['max_matrix_frobenius_defect']=max(extrema['max_matrix_frobenius_defect'],matrix['frobenius_defect'])
                for item in step['reference_comparisons'].values():field_check(item)
                history=step['reported_residual_history'];assert history[0][0]==0 and history[-1][0]==step['iterations'] and all(math.isfinite(v) and v>=0 for _,v in history)
                if i>0:assert step['event_counts']['MatLUFactorSym']==step['event_counts']['MatLUFactorNum']==0
                if label=='rana_ld':assert step['event_counts']['MatSolve']==step['event_counts']['PCApply']>0 and step['auxiliary_factor_nonzeros']==167432.
        for values in (row['metrics'],row['control_metrics']):
            assert abs(values['domain_area']-1)<=1e-12 and values['boundary_trace_l2']<=1e-12
            assert math.isclose(values['exact_l2_norm'],math.exp(-1)/2,rel_tol=1e-10,abs_tol=1e-12)
            assert math.isclose(values['exact_h1_seminorm'],math.pi*math.exp(-1)/math.sqrt(2),rel_tol=1e-10,abs_tol=1e-12)
        for metric in METRICS:assert abs(row['metrics'][metric]-row['control_metrics'][metric])<=1e-13+1e-6*row['control_metrics'][metric]
        if label in ('irksome_jacobi','residual_jacobi','rana_ld'):assert row['solver_stats']==[10,10,row['total_iterations']]
        if label in ('residual_jacobi','rana_ld'):
            for event,count in row['operator_event_counts'].items():assert count==row['construction_event_counts'][event]+sum(s['event_counts'][event] for s in row['step_records'])
        summary[label]=dict(total_iterations=row['total_iterations'],step_iterations=[s['iterations'] for s in row['step_records']],
            operator_event_counts=row.get('operator_event_counts'),control_metrics=row['control_metrics'])
    for item in d['comparisons'].values():field_check(item)
    old_review=json.loads((ROOT/'results/processed/asm_trajectory_review.json').read_text())
    old=json.loads((ROOT/old_review['measurement_records']['serial']['path']).read_text())['trajectories']['irksome_jacobi']
    assert [s['iterations'] for s in d['trajectories']['irksome_jacobi']['step_records']]==[s['iterations'] for s in old['step_records']]
    cost_review=json.loads((ROOT/'results/processed/accuracy_cost_review.json').read_text())
    curve=next(c for c in json.loads((ROOT/cost_review['measurement_records']['serial']['path']).read_text())['curve'] if c['case_id']=='Radau_coarse')
    for label in ('residual_jacobi','rana_ld'):
        for metric,budget in d['trajectories'][label]['accepted_time_error_budget_check'].items():
            assert budget['accepted_baseline_time_error']==curve['time_errors'][metric]
            assert budget['actual_field_distance']==d['comparisons'][label+'_vs_original_irksome']['norms'][metric]['actual_field_difference']
            assert budget['allowed_distance']==1e-12+.01*curve['time_errors'][metric] and budget['actual_field_distance']<=budget['allowed_distance']
    rana=d['trajectories']['rana_ld'];probe=rana['auxiliary_action_check'];events=rana['operator_event_counts']
    assert events['MatSolve']==events['PCApply']==80 and events['MatLUFactorSym']==events['MatLUFactorNum']==1
    assert events['PCSetUp']==2 and events['KSPSetUp']==events['KSPSolve']==10
    assert rana['diagnostic_event_counts']==dict(KSPSolve=0,KSPSetUp=0,PCSetUp=0,PCApply=1,MatSolve=1,MatLUFactorSym=0,MatLUFactorNum=0)
    assert probe['allowed_residual']==1e-11+1e-10*probe['rhs_norm'] and probe['residual_norm']<=probe['allowed_residual'] and probe['relative_residual']==probe['residual_norm']/probe['rhs_norm']
    transcript=Path(str(STEM)+'.txt').read_text();assert 'RANA TRAJECTORY CHECKPOINT PASSED.' in transcript and 'Python: irksome.RanaLD' in transcript
    return dict(status='passed',trajectories=4,steps=40,trajectory_summary=summary,extrema=extrema,
        final_field_comparisons={k:{n:r['actual_field_difference'] for n,r in v['norms'].items()} for k,v in d['comparisons'].items()},
        rana_auxiliary_action_check=probe,rana_diagnostic_event_counts=rana['diagnostic_event_counts'],performance_measurement=False,
        audit_limit='Saved arithmetic, source/raw/installed-interface hashes and profile consistency audited without numerical imports. Actual FEM matrix/field/probe operations and reuse states were checked during the researcher run and were not rerun by this auditor.')

if __name__=='__main__':print(json.dumps(audit(),indent=2,allow_nan=False))

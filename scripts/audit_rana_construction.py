#!/usr/bin/env python3
"""Audit saved researcher-run RanaLD construction evidence using the standard library."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
STEM=ROOT/'results/raw/rana-construction-20261021T110146Z-18747'
METRICS=('l2_error','h1_seminorm_error')

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def audit():
    sys.path.insert(0,str(ROOT/'environment'))
    import check_rana_construction as runner
    runner.plan()  # Read-only accepted evidence and installed-source hashes.
    m=json.loads(Path(str(STEM)+'-manifest.json').read_text())
    raw=ROOT/m['measurement_records']['serial']['path'];d=json.loads(raw.read_text());config=d['config'];checks=config['checks']
    assert m['checkpoint_exit_status']==0 and m['working_tree_status_before_run']=='' and m['completed_steps']==['serial']
    assert m['performance_measurement'] is False and m['accuracy_cost_ranking_established'] is False
    assert m['inherited_options_or_library_settings']=={}
    assert m['selected_environment']['OMP_NUM_THREADS']==m['selected_environment']['OPENBLAS_NUM_THREADS']=='1'
    for key in ('asm_cost_review','verified_environment','config','installed_source_inspection'):
        assert digest(ROOT/m[key])==m[key+'_sha256'],key
    for path,expected in m['script_sha256'].items():
        assert digest(ROOT/path)==expected,path
        assert hashlib.sha256(subprocess.check_output(['git','show',m['code_commit']+':'+path],cwd=ROOT)).hexdigest()==expected,path
    for path,expected in m['inspected_installed_source_sha256'].items():assert digest(ROOT/path)==expected,path
    assert digest(raw)==m['measurement_records']['serial']['sha256'] and config==json.loads(runner.CONFIG.read_text())
    assert runner.verify_complete(d,config)==m['study_validation']
    extrema=dict(max_true_residual_over_nominal_target=0.,max_field_difference=0.,max_quadrature_change=0.)
    summary={}
    for label,row in d['solves'].items():
        expected=dict(config['rana_solver_parameters'] if label=='rana_ld' else config['solver_parameters'])
        if label=='manual_jacobi_reference':
            for key in ('ksp_rtol','ksp_atol'):expected[key]*=config['manual_reference_tolerance_factor']
        assert all(row['solver_parameters'][k]==v for k,v in expected.items())
        actual=row['actual_solver']
        assert actual['ksp_type']=='gmres' and actual['pc_type']==expected['pc_type'] and actual['side']=='right' and actual['norm']=='unpreconditioned'
        assert actual['rtol']==expected['ksp_rtol'] and actual['atol']==expected['ksp_atol'] and actual['max_it']==1000 and actual['requested_restart']==50
        step=row['step_records'][0] if label=='irksome_jacobi' else row
        r=step['true_block_residual'];target=max(expected['ksp_atol'],expected['ksp_rtol']*r['rhs_norm'])
        assert all(math.isfinite(v) and v>=0 for v in r.values()) and r['rhs_norm']>0
        assert math.isclose(r['relative_norm'],r['norm']/r['rhs_norm'],rel_tol=1e-14)
        assert target==step['true_residual_target'] and r['norm']<=checks['residual_tolerance_factor']*target
        assert step['converged_reason']>0 and 0<step['iterations']<=1000
        history=row['last_step_reported_residual_history'] if label=='irksome_jacobi' else row['reported_residual_history']
        assert history[0][0]==0 and history[-1][0]==step['iterations'] and all(math.isfinite(v) and v>=0 for _,v in history)
        assert math.isclose(history[0][1],r['rhs_norm'],rel_tol=1e-14)
        if label=='manual_jacobi_reference':
            assert row['source_times']==[.1/3,.1] and all(v<=1e-12 for v in row['stage_boundary_trace_l2'])
        else:
            assert row['solver_stats']==[1,1,step['iterations']]
            assert all(row['checks'].values()) and all(s['source_residual_l2']<=1e-10 for s in row['source_checks'])
            assert step['stage_times']==[.1/3,.1]
            assert 0<=step['update_identity_l2']<=1e-12 and 0<=step['stiff_accuracy_l2']<=1e-12
            for values in (row['metrics'],row['control_metrics']):
                assert abs(values['domain_area']-1)<=1e-12 and values['boundary_trace_l2']<=1e-12
                assert math.isclose(values['exact_l2_norm'],math.exp(-.1)/2,rel_tol=1e-10,abs_tol=1e-12)
                assert math.isclose(values['exact_h1_seminorm'],math.pi*math.exp(-.1)/math.sqrt(2),rel_tol=1e-10,abs_tol=1e-12)
            for metric in METRICS:
                assert abs(row['metrics'][metric]-row['control_metrics'][metric])<=1e-13+1e-6*row['control_metrics'][metric]
        extrema['max_true_residual_over_nominal_target']=max(extrema['max_true_residual_over_nominal_target'],r['norm']/target)
        summary[label]=dict(iterations=step['iterations'],converged_reason=step['converged_reason'],true_block_residual=r,nominal_target=target)
    fields={}
    for label,item in d['comparisons'].items():
        measured=item['difference_measurement'];assert measured['status']==item['status']=='passed'
        fields[label]={}
        for metric,row in item['norms'].items():
            value=measured['metrics'][metric];primary=measured['primary_metrics'][metric]
            allowed=checks['pc_field_comparison_atol']+checks['pc_field_comparison_rtol']*row['reference_field_norm']
            assert math.isfinite(value) and value>=0 and row['actual_field_difference']==value and row['allowed_difference']==allowed and value<=allowed
            q=measured['quadrature_comparison'][metric]
            assert q['absolute_difference']==abs(primary-value) and q['allowed_difference']==1e-13+1e-6*value and q['absolute_difference']<=q['allowed_difference']
            extrema['max_field_difference']=max(extrema['max_field_difference'],value)
            extrema['max_quadrature_change']=max(extrema['max_quadrature_change'],abs(primary-value))
            fields[label][metric]=dict(difference=value,allowed=allowed)
    rana=d['solves']['rana_ld']
    assert rana['weak_form_argument_count']==1 and rana['weak_form_representation']=='state_function_residual'
    expected=[[5/12,0.],[3/4,2/5]]
    assert all(abs(rana['installed_Atilde'][i][j]-expected[i][j])<=1e-14 for i in range(2) for j in range(2))
    for row in rana['matrix_checks'].values():
        assert 0<=row['frobenius_defect']<=row['allowed_defect']==1e-11+1e-12*row['reference_frobenius_norm']
    probe=rana['auxiliary_action_check']
    assert probe['rhs_norm']>0 and probe['allowed_residual']==1e-11+1e-10*probe['rhs_norm'] and probe['residual_norm']<=probe['allowed_residual']
    assert probe['relative_residual']==probe['residual_norm']/probe['rhs_norm'] and probe['diagnostic_outer_PC_applications']==1
    events=rana['operator_event_counts']
    assert set(events)=={'KSPSolve','KSPSetUp','PCSetUp','PCApply'} and events['KSPSolve']==events['KSPSetUp']==1 and events['PCSetUp']>=1 and events['PCApply']>=rana['iterations']
    assert rana['auxiliary_solver']['factor_nonzeros']>0 and rana['auxiliary_solver']['inner_krylov_solve'] is False
    transcript=Path(str(STEM)+'.txt').read_text()
    assert 'Python: irksome.RanaLD' in transcript and 'matrix ordering: rcm' in transcript and 'package used to perform factorization: petsc' in transcript
    assert 'RANA CONSTRUCTION CHECKPOINT PASSED.' in transcript
    return dict(status='passed',independent_operator_solves=3,physical_steps_per_realization=1,solve_summary=summary,
        field_comparisons=fields,matrix_checks=rana['matrix_checks'],auxiliary_action_check=probe,
        rana_operator_event_counts=events,auxiliary_factor_nonzeros=rana['auxiliary_solver']['factor_nonzeros'],extrema=extrema,
        performance_measurement=False,trajectory_verified=False,
        audit_limit='Saved arithmetic and source/raw/installed-interface hashes rechecked without numerical imports. Full CSR, inverse-action and actual-field operations were checked at researcher runtime; these are not rerun by this auditor.')

if __name__=='__main__':print(json.dumps(audit(),indent=2,allow_nan=False))

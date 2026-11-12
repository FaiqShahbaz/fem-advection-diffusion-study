#!/usr/bin/env python3
"""Audit saved researcher-run ASM evidence using only the Python standard library."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
STEM=ROOT/'results/raw/asm-construction-20261019T111303Z-86965'
METRICS=('l2_error','h1_seminorm_error')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def audit():
    sys.path.insert(0,str(ROOT/'environment'))
    import check_asm_construction as runner
    runner.plan()  # Read-only hashes/configuration; no numerical imports or execution.
    manifest=json.loads(Path(str(STEM)+'-manifest.json').read_text())
    raw=ROOT/manifest['measurement_records']['serial']['path']
    data=json.loads(raw.read_text());config=data['config'];checks=config['checks']
    assert manifest['checkpoint_exit_status']==0 and manifest['working_tree_status_before_run']==''
    assert manifest['performance_measurement'] is False and manifest['completed_steps']==['serial']
    assert manifest['inherited_options_or_library_settings']=={}
    assert manifest['selected_environment']['OMP_NUM_THREADS']==manifest['selected_environment']['OPENBLAS_NUM_THREADS']=='1'
    for key in ('accuracy_cost_review','verified_environment','config'):
        assert digest(ROOT/manifest[key])==manifest[key+'_sha256'],key
    for path,expected in manifest['script_sha256'].items():
        assert digest(ROOT/path)==expected,path
        historical=subprocess.check_output(['git','show',manifest['code_commit']+':'+path],cwd=ROOT)
        assert hashlib.sha256(historical).hexdigest()==expected,path
    assert digest(raw)==manifest['measurement_records']['serial']['sha256']
    assert config==json.loads(runner.CONFIG.read_text())
    assert runner.verify_complete(data,config)==manifest['study_validation']
    extrema=dict(max_true_residual_over_nominal_target=0.,max_field_difference=0.,max_quadrature_change=0.)
    summary={}
    def residual_check(residual,target):
        assert all(math.isfinite(v) and v>=0 for v in residual.values()) and residual['rhs_norm']>0
        assert math.isclose(residual['relative_norm'],residual['norm']/residual['rhs_norm'],rel_tol=1e-14)
        assert residual['norm']<=checks['residual_tolerance_factor']*target
        extrema['max_true_residual_over_nominal_target']=max(extrema['max_true_residual_over_nominal_target'],residual['norm']/target)
    for label,row in data['solves'].items():
        params=row['solver_parameters'];actual=row['actual_solver']
        expected=dict(config['asm_solver_parameters'] if label=='manual_asm' else config['solver_parameters'])
        if label=='manual_jacobi_reference':
            for key in ('ksp_rtol','ksp_atol'):expected[key]*=.1
        assert all(params[k]==v for k,v in expected.items())
        assert actual['ksp_type']=='gmres' and actual['pc_type']==expected['pc_type'] and actual['side']=='right' and actual['norm']=='unpreconditioned'
        assert actual['rtol']==params['ksp_rtol'] and actual['atol']==params['ksp_atol'] and actual['max_it']==1000 and actual['requested_restart']==50
        if label=='irksome_jacobi':
            step=row['step_records'][0];history=row['last_step_reported_residual_history']
            assert step['step']==1 and step['t_start']==0 and step['t_end']==.1
            assert step['stage_times']==[.1/3,.1]
            assert step['update_identity_l2']<=1e-12 and step['stiff_accuracy_l2']<=1e-12
            assert row['solver_stats']==[1,1,row['total_iterations']]
            assert row['total_iterations']==row['maximum_step_iterations']==step['iterations']
            assert all(row['checks'].values()) and all(s['source_residual_l2']<=1e-10 for s in row['source_checks'])
            for values in (row['metrics'],row['control_metrics']):
                assert abs(values['domain_area']-1)<=1e-12 and values['boundary_trace_l2']<=1e-12
                assert math.isclose(values['exact_l2_norm'],math.exp(-.1)/2,rel_tol=1e-10,abs_tol=1e-12)
                assert math.isclose(values['exact_h1_seminorm'],math.pi*math.exp(-.1)/math.sqrt(2),rel_tol=1e-10,abs_tol=1e-12)
            for metric in METRICS:
                change=abs(row['metrics'][metric]-row['control_metrics'][metric])
                assert change<=1e-13+1e-6*row['control_metrics'][metric]
            assert all(v<=1e-12 for v in row['final_slope_boundary_trace_l2'])
        else:
            step=row;history=row['reported_residual_history']
            assert row['source_times']==[.1/3,.1] and all(v<=1e-12 for v in row['stage_boundary_trace_l2'])
        residual=step['true_block_residual'];target=max(params['ksp_atol'],params['ksp_rtol']*residual['rhs_norm'])
        assert target==step['true_residual_target'] and step['converged_reason']>0 and 0<step['iterations']<=1000
        residual_check(residual,target)
        assert history[0][0]==0 and history[-1][0]==step['iterations']
        assert all(math.isfinite(v) and v>=0 for _,v in history)
        summary[label]=dict(iterations=step['iterations'],converged_reason=step['converged_reason'],true_block_residual=residual,nominal_target=target)
    fields={}
    for label,item in data['comparisons'].items():
        measured=item['difference_measurement'];assert measured['status']=='passed'
        fields[label]={}
        for metric,row in item['norms'].items():
            value=measured['metrics'][metric];primary=measured['primary_metrics'][metric]
            allowed=checks['pc_field_comparison_atol']+checks['pc_field_comparison_rtol']*row['reference_field_norm']
            assert row['allowed_difference']==allowed and row['actual_field_difference']==value and value<=allowed
            q=measured['quadrature_comparison'][metric]
            assert q['absolute_difference']==abs(primary-value) and q['allowed_difference']==1e-13+1e-6*value
            assert q['absolute_difference']<=q['allowed_difference']
            extrema['max_field_difference']=max(extrema['max_field_difference'],value)
            extrema['max_quadrature_change']=max(extrema['max_quadrature_change'],abs(primary-value))
            fields[label][metric]=dict(difference=value,allowed=allowed)
    asm=data['solves']['manual_asm'];domains=asm['domains'];local=asm['local_solvers'];work=asm['local_work']
    assert all(v==config['asm_solver_parameters'][key] for key,v in asm['options_observed_in_scope'].items())
    assert sorted(i for d in domains for i in d['core_indices'])==list(range(2178))
    for i,(domain,solver,counts) in enumerate(zip(domains,local,work)):
        core,outer=domain['core_indices'],domain['overlap_indices']
        assert domain['index']==solver['index']==counts['index']==i
        assert domain['core_size']==len(core)==len(set(core)) and domain['overlap_size']==len(outer)==len(set(outer))
        assert core and set(core)<=set(outer) and all(0<=v<2178 for v in outer) and domain['one_layer_check']
        assert solver['matrix_size']==[len(outer),len(outer)] and solver['factor_nonzeros']>0
        assert counts['outer_solve']['calls']==asm['outer_preconditioner_applications_from_local_calls']
        assert counts['outer_solve']['calls']==counts['outer_solve']['reported_iterations']>0
        assert counts['action_check']==dict(calls=1,reported_iterations=1)
    action=asm['additive_action_check']
    assert action['reference_action_norm']>0 and action['allowed_defect']==1e-12+1e-11*action['reference_action_norm']
    assert action['relative_defect']==action['defect_norm']/action['reference_action_norm']
    assert action['actual_global_PC_applications']==1 and action['independent_direct_local_PC_applications']==4
    transcript=Path(str(STEM)+'.txt').read_text()
    assert '0 levels of fill' in transcript and 'matrix ordering: rcm' in transcript and 'restriction/interpolation type - BASIC' in transcript
    return dict(status='passed',independent_operator_solves=3,physical_steps_per_realization=1,
        solve_summary=summary,field_comparisons=fields,domain_sizes=[dict(core=d['core_size'],overlap=d['overlap_size']) for d in domains],
        outer_local_calls_per_domain=[w['outer_solve']['calls'] for w in work],
        outer_local_calls_total=sum(w['outer_solve']['calls'] for w in work),
        additive_action_check=action,extrema=extrema,performance_measurement=False,
        audit_limit='Raw arithmetic and index consistency rechecked; matrix adjacency and numerical PC/field operations were checked at researcher runtime, not rerun by this auditor.')

if __name__=='__main__':
    print(json.dumps(audit(),indent=2,allow_nan=False))

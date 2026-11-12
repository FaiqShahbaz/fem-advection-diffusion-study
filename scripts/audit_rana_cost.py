#!/usr/bin/env python3
"""Audit saved matched residual-form Jacobi/RanaLD timing without numerical execution."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
STEM=ROOT/'results/raw/rana-cost-20261021T114632Z-21462'
METRICS=('l2_error','h1_seminorm_error')

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def audit():
    sys.path.insert(0,str(ROOT/'environment'))
    import check_rana_cost as runner
    runner.plan()
    m=json.loads(Path(str(STEM)+'-manifest.json').read_text());raw=ROOT/m['measurement_records']['serial']['path']
    d=json.loads(raw.read_text());config=d['config']
    assert m['checkpoint_exit_status']==0 and m['working_tree_status_before_run']=='' and m['completed_steps']==['serial']
    assert m['performance_measurement'] is True and m['inherited_options_or_library_settings']=={}
    assert m['selected_environment']['OMP_NUM_THREADS']==m['selected_environment']['OPENBLAS_NUM_THREADS']=='1'
    for p in ('power_state','power_state_after'):assert m[p]['source']=='AC Power' and m[p]['exit_status']==0
    for key in ('rana_trajectory_review','verified_environment','config','installed_source_inspection'):
        assert digest(ROOT/m[key])==m[key+'_sha256'],key
    for path,expected in m['script_sha256'].items():
        assert digest(ROOT/path)==expected,path
        assert hashlib.sha256(subprocess.check_output(['git','show',m['code_commit']+':'+path],cwd=ROOT)).hexdigest()==expected,path
    for path,expected in m['inspected_installed_source_sha256'].items():assert digest(ROOT/path)==expected,path
    assert digest(raw)==m['measurement_records']['serial']['sha256'] and config==json.loads(runner.CONFIG.read_text())
    assert runner.verify_complete(d,config)==m['study_validation']
    extrema=dict(max_true_residual_over_nominal_target=0.,max_timed_baseline_field_distance=0.,max_reference_field_distance=0.,max_quadrature_change=0.,max_update_or_stiff_accuracy_l2=0.,max_matrix_frobenius_defect=0.)
    def residual(row,params):
        r=row['true_block_residual'];target=max(params['ksp_atol'],params['ksp_rtol']*r['rhs_norm'])
        assert all(math.isfinite(v) and v>=0 for v in r.values()) and r['rhs_norm']>0
        assert target==row['true_residual_target'] and r['norm']<=10*target and math.isclose(r['relative_norm'],r['norm']/r['rhs_norm'],rel_tol=1e-14)
        extrema['max_true_residual_over_nominal_target']=max(extrema['max_true_residual_over_nominal_target'],r['norm']/target)
        for key in ('update_identity_l2','stiff_accuracy_l2'):
            assert 0<=row[key]<=1e-12
            extrema['max_update_or_stiff_accuracy_l2']=max(extrema['max_update_or_stiff_accuracy_l2'],row[key])
        assert all(v<=1e-12 for v in row['stage_boundary_trace_l2'])
    def difference(measured):
        assert measured['status']=='passed'
        for metric in METRICS:
            value=measured['metrics'][metric];primary=measured['primary_metrics'][metric];q=measured['quadrature_comparison'][metric]
            assert math.isfinite(value) and value>=0 and q['absolute_difference']==abs(primary-value)
            assert q['allowed_difference']==1e-13+1e-6*value and q['absolute_difference']<=q['allowed_difference']
            extrema['max_quadrature_change']=max(extrema['max_quadrature_change'],abs(primary-value))
    def matrix_checks(rows):
        for row in rows.values():
            assert row['status']=='passed' and 0<=row['frobenius_defect']<=row['allowed_defect']==1e-11+1e-12*row['reference_frobenius_norm']
            extrema['max_matrix_frobenius_defect']=max(extrema['max_matrix_frobenius_defect'],row['frobenius_defect'])
    previous_review=json.loads((ROOT/m['rana_trajectory_review']).read_text())
    previous=json.loads((ROOT/previous_review['measurement_records']['serial']['path']).read_text())['trajectories']
    steps=0
    for label,row in d['trajectories'].items():
        params=row['solver_parameters'];role=row['role'];timed=role in ('warmup','measured')
        assert row['total_iterations']==sum(s['iterations'] for s in row['step_records']) and row['maximum_step_iterations']==max(s['iterations'] for s in row['step_records'])
        for i,step in enumerate(row['step_records']):
            assert step['step']==i+1 and abs(step['t_start']-i*.1)<=1e-14 and abs(step['t_end']-(i+1)*.1)<=1e-14
            assert len(step['stage_times'])==2 and all(abs(a-b)<=1e-14 for a,b in zip(step['stage_times'],[(i+1/3)*.1,(i+1)*.1]))
            assert step['converged_reason']>0 and 0<step['iterations']<=1000
            if not timed:
                residual(step,params)
                assert all(s['source_residual_l2']<=1e-10 for s in step['source_checks'])
                if role=='verified':
                    matrix_checks(step['matrix_checks'])
                    assert step['matrix_reuse_tokens']==row['step_records'][0]['matrix_reuse_tokens']
                    for item in step['reference_comparisons'].values():
                        difference(item['difference_measurement'])
                        for metric,r in item['norms'].items():
                            value=item['difference_measurement']['metrics'][metric]
                            assert value==r['actual_field_difference'] and r['allowed_difference']==1e-11+1e-8*r['reference_field_norm'] and value<=r['allowed_difference']
                            extrema['max_reference_field_distance']=max(extrema['max_reference_field_distance'],value)
        for values in (row['metrics'],row['control_metrics']):
            assert all(math.isfinite(v) and v>=0 for v in values.values()) and abs(values['domain_area']-1)<=1e-12 and values['boundary_trace_l2']<=1e-12
            assert math.isclose(values['exact_l2_norm'],math.exp(-1)/2,rel_tol=1e-10,abs_tol=1e-12)
            assert math.isclose(values['exact_h1_seminorm'],math.pi*math.exp(-1)/math.sqrt(2),rel_tol=1e-10,abs_tol=1e-12)
        for metric in METRICS:
            q=row['quadrature_comparison'][metric];change=abs(row['metrics'][metric]-row['control_metrics'][metric])
            assert q['absolute_difference']==change and q['allowed_difference']==1e-13+1e-6*row['control_metrics'][metric] and change<=q['allowed_difference']
            extrema['max_quadrature_change']=max(extrema['max_quadrature_change'],change)
        if role=='verified':
            case=next(c for c in config['cases'] if c['case_id']==row['case_id']);old=previous[case['accepted_role']]
            assert row['operator_event_counts']==old['operator_event_counts'] and [s['iterations'] for s in row['step_records']]==[s['iterations'] for s in old['step_records']]
            assert all(math.isclose(row['control_metrics'][metric],old['control_metrics'][metric],rel_tol=1e-9,abs_tol=1e-12) for metric in METRICS)
        if timed:
            baseline=d['trajectories'][row['case_id']+'_verified'];timing=row['timing'];events=timing['petsc_event_deltas']
            assert all(math.isfinite(v) and v>=0 for k,v in timing.items() if k.endswith('_seconds'))
            assert math.isclose(timing['trajectory_wall_seconds'],timing['construction_seconds']+sum(s['wall_seconds'] for s in row['step_records'])+timing['loop_bookkeeping_seconds'],rel_tol=1e-12,abs_tol=1e-12)
            assert all(events[e]['count']==baseline['operator_event_counts'][e] for e in events)
            assert row['timed_python_monitors'] is False and row['timed_local_callbacks'] is False
            residual(row['final_equation_check'],params);matrix_checks(row['matrix_checks']);difference(row['difference_measurement'])
            for metric,r in row['equivalence'].items():
                value=row['difference_measurement']['metrics'][metric]
                assert r['actual_field_difference']==value and r['allowed_difference']==1e-12+1e-10*r['baseline_field_norm'] and value<=r['allowed_difference']
                extrema['max_timed_baseline_field_distance']=max(extrema['max_timed_baseline_field_distance'],value)
            if row['case_id']=='RanaLD':
                assert events['MatSolve']['count']==events['PCApply']['count']==80 and events['MatLUFactorSym']['count']==events['MatLUFactorNum']['count']==1
                probe=row['auxiliary_action_check'];diag=row['diagnostic_event_deltas']
                assert probe['relative_residual']==probe['residual_norm']/probe['rhs_norm'] and probe['allowed_residual']==1e-11+1e-10*probe['rhs_norm'] and probe['residual_norm']<=probe['allowed_residual']
                assert diag['MatSolve']['count']==diag['PCApply']['count']==1 and all(diag[e]['count']==0 for e in ('KSPSolve','KSPSetUp','PCSetUp','MatLUFactorSym','MatLUFactorNum'))
        steps+=row['completed_steps']
    assert steps==d['completed_steps']==110 and len(d['trajectories'])==11
    difference(d['verified_pair_difference'])
    summary={}
    for case in ('Jacobi','RanaLD'):
        rows=[d['trajectories'][case+'_measured'+str(i)] for i in (1,2,3)];wall=d['timing_summary'][case]['trajectory_wall_seconds']
        values=[r['timing']['trajectory_wall_seconds'] for r in rows]
        assert wall==dict(values=values,minimum=min(values),median=statistics.median(values),maximum=max(values))
        summary[case]=dict(**wall,relative_range=(max(values)-min(values))/statistics.median(values),total_iterations=rows[0]['total_iterations'],
            physical_event_counts={e:r['count'] for e,r in rows[0]['timing']['petsc_event_deltas'].items()})
    ratio=summary['Jacobi']['median']/summary['RanaLD']['median'];assert ratio==d['jacobi_over_rana_median_core_ratio']
    return dict(status='passed',trajectories=11,steps=110,power='AC confirmed at both boundaries',case_summary=summary,
        jacobi_over_rana_median_core_ratio=ratio,extrema=extrema,performance_measurement=True,
        ranking_scope='One fixed residual-form Irksome Radau P2/n16 kappa=1 dt=0.1 T=1 trajectory on one laptop',
        limits=['Three measured repeats per PC; all samples retained; ranges are not confidence intervals.',
            'Fresh construction plus ten advances with warm caches; lazy assembly/factorization included.',
            'Auxiliary work is actual MatSolve/factor event counts; the diagnostic probe is excluded.',
            'Do not pool absolute values with older manual ASM or different Irksome drivers.',
            'Saved arithmetic and hashes audited without numerical imports; matrix/field/probe operations remain researcher runtime evidence.'])

if __name__=='__main__':print(json.dumps(audit(),indent=2,allow_nan=False))

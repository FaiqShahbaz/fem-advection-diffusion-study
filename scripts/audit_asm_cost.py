#!/usr/bin/env python3
"""Audit saved matched manual Jacobi/ASM timings without numerical imports."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
STEM=ROOT/'results/raw/asm-cost-20261021T102000Z-16397'
METRICS=('l2_error','h1_seminorm_error')

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def audit():
    sys.path.insert(0,str(ROOT/'environment'))
    import check_asm_cost as runner
    runner.plan()
    m=json.loads(Path(str(STEM)+'-manifest.json').read_text())
    raw=ROOT/m['measurement_records']['serial']['path'];d=json.loads(raw.read_text());config=d['config']
    assert m['checkpoint_exit_status']==0 and m['working_tree_status_before_run']=='' and m['completed_steps']==['serial']
    assert m['performance_measurement'] and m['inherited_options_or_library_settings']=={}
    assert m['selected_environment']['OPENBLAS_NUM_THREADS']==m['selected_environment']['OMP_NUM_THREADS']=='1'
    for key in ('power_state','power_state_after'):assert m[key]['source']=='AC Power' and m[key]['exit_status']==0
    for key in ('asm_trajectory_review','verified_environment','config'):
        assert digest(ROOT/m[key])==m[key+'_sha256'],key
    for path,expected in m['script_sha256'].items():
        assert digest(ROOT/path)==expected,path
        assert hashlib.sha256(subprocess.check_output(['git','show',m['code_commit']+':'+path],cwd=ROOT)).hexdigest()==expected,path
    assert digest(raw)==m['measurement_records']['serial']['sha256'] and config==json.loads(runner.CONFIG.read_text())
    assert runner.verify_complete(d,config)==m['study_validation']
    extrema=dict(max_true_residual_over_nominal_target=0.,max_update_or_stiff_accuracy_l2=0.,
        max_timed_baseline_field_distance=0.,max_continuum_quadrature_change=0.)
    def residual_check(r,target):
        assert all(math.isfinite(v) and v>=0 for v in r.values()) and r['rhs_norm']>0
        assert math.isclose(r['relative_norm'],r['norm']/r['rhs_norm'],rel_tol=1e-14)
        assert r['norm']<=10*target
        extrema['max_true_residual_over_nominal_target']=max(extrema['max_true_residual_over_nominal_target'],r['norm']/target)
    def identities(row):
        for key in ('update_identity_l2','stiff_accuracy_l2'):
            assert math.isfinite(row[key]) and 0<=row[key]<=1e-12
            extrema['max_update_or_stiff_accuracy_l2']=max(extrema['max_update_or_stiff_accuracy_l2'],row[key])
    def measured_check(row):
        assert row['status']=='passed'
        for metric in METRICS:
            a,b=row['primary_metrics'][metric],row['metrics'][metric];q=row['quadrature_comparison'][metric]
            assert math.isfinite(a) and math.isfinite(b) and a>=0 and b>=0
            assert q['absolute_difference']==abs(a-b) and q['allowed_difference']==1e-13+1e-6*b
            assert q['status']=='passed' and q['absolute_difference']<=q['allowed_difference']
    measured_check(d['verified_pair_difference'])
    for metric,row in d['verified_pair_equivalence'].items():
        assert row['actual_field_difference']==d['verified_pair_difference']['metrics'][metric]
        assert row['allowed_difference']==1e-12+1e-10*row['baseline_field_norm']
        assert row['actual_field_difference']<=row['allowed_difference']
    summaries={};steps=0
    for label,row in d['trajectories'].items():
        params=row['solver_parameters'];actual=row['actual_solver'];timed=row['role'] in ('warmup','measured')
        assert actual['ksp_type']=='gmres' and actual['pc_type']==params['pc_type'] and actual['side']=='right' and actual['norm']=='unpreconditioned'
        assert actual['rtol']==1e-11 and actual['atol']==1e-13 and actual['max_it']==1000 and actual['requested_restart']==50
        assert row['total_iterations']==sum(s['iterations'] for s in row['step_records'])
        assert row['maximum_step_iterations']==max(s['iterations'] for s in row['step_records'])
        for i,step in enumerate(row['step_records']):
            assert step['step']==i+1 and abs(step['t_start']-.1*i)<1e-14 and abs(step['t_end']-.1*(i+1))<1e-14
            assert len(step['stage_times'])==2 and all(abs(a-b)<1e-14 for a,b in zip(step['stage_times'],[(i+1/3)*.1,(i+1)*.1]))
            assert step['converged_reason']>0 and 0<step['iterations']<=1000
            if timed:assert math.isfinite(step['wall_seconds']) and step['wall_seconds']>=0
            else:
                r=step['true_block_residual'];target=max(params['ksp_atol'],params['ksp_rtol']*r['rhs_norm'])
                assert target==step['true_residual_target'];residual_check(r,target);identities(step)
                assert step['event_counts']['PCSetUp']==0 and step['event_counts']['KSPSetUp']==0
                assert all(v<=1e-12 for v in step['stage_boundary_trace_l2'])
                assert [s['time'] for s in step['source_checks']]==step['stage_times'] and all(s['source_residual_l2']<=1e-10 for s in step['source_checks'])
        for values in (row['metrics'],row['control_metrics']):
            assert all(math.isfinite(v) and v>=0 for v in values.values())
            assert abs(values['domain_area']-1)<=1e-12 and values['boundary_trace_l2']<=1e-12
            assert math.isclose(values['exact_l2_norm'],math.exp(-1)/2,rel_tol=1e-10,abs_tol=1e-12)
            assert math.isclose(values['exact_h1_seminorm'],math.pi*math.exp(-1)/math.sqrt(2),rel_tol=1e-10,abs_tol=1e-12)
        for metric in METRICS:
            change=abs(row['metrics'][metric]-row['control_metrics'][metric]);q=row['quadrature_comparison'][metric]
            assert q['absolute_difference']==change and q['allowed_difference']==1e-13+1e-6*row['control_metrics'][metric]
            assert change<=q['allowed_difference']
            extrema['max_continuum_quadrature_change']=max(extrema['max_continuum_quadrature_change'],change)
        if timed:
            baseline=d['trajectories'][row['case_id']+'_verified'];final=row['final_equation_check']
            assert [s['iterations'] for s in row['step_records']]==[s['iterations'] for s in baseline['step_records']]
            r=final['true_block_residual'];target=max(params['ksp_atol'],params['ksp_rtol']*r['rhs_norm'])
            assert target==final['true_residual_target'];residual_check(r,target);identities(final)
            assert all(abs(a-b)<1e-14 for a,b in zip(final['stage_times'],[(9+1/3)*.1,1.]))
            assert all(v<=1e-12 for v in final['stage_boundary_trace_l2'])
            assert all(s['source_residual_l2']<=1e-10 for s in row['source_checks'])
            measured_check(row['difference_measurement'])
            for metric,eq in row['equivalence'].items():
                value=row['difference_measurement']['metrics'][metric]
                assert value==eq['actual_field_difference'] and eq['allowed_difference']==1e-12+1e-10*eq['baseline_field_norm']
                assert value<=eq['allowed_difference']
                extrema['max_timed_baseline_field_distance']=max(extrema['max_timed_baseline_field_distance'],value)
            timing=row['timing'];events=timing['petsc_event_deltas']
            assert all(math.isfinite(v) and v>=0 for event in events.values() for v in event.values())
            assert events['KSPSolve']['count']==10 and events['KSPSetUp']['count']==events['PCSetUp']['count']==1
            assert all(events[e]['count']==baseline['operator_event_counts'][e] for e in events)
            assert all(v==params[k] for k,v in row['options_observed_in_scope'].items())
            assert row['timed_python_monitors'] is False and row['timed_local_callbacks'] is False
            if row['case_id']=='ASM':
                assert row['domains']==baseline['domains'] and row['local_solvers']==baseline['local_solvers']
                applications=int(events['PCApply']['count']);derived=row['local_work_derived']
                assert all(w['outer_solve']['calls']==applications for w in baseline['local_work'])
                assert derived['applications_per_domain']==applications and derived['total_local_applications']==4*applications and derived['domains']==4
                probe=row['diagnostic_additive_action'];assert probe['allowed_defect']==1e-12+1e-11*probe['reference_action_norm']
                assert probe['status']=='passed' and probe['defect_norm']<=probe['allowed_defect']
                assert probe['actual_global_PC_applications']==1 and probe['independent_direct_local_PC_applications']==4
        steps+=row['completed_steps']
    assert steps==d['completed_steps']==100 and len(d['trajectories'])==10
    for case,summary in d['timing_summary'].items():
        wall=summary['trajectory_wall_seconds']
        summaries[case]=dict(median=wall['median'],minimum=wall['minimum'],maximum=wall['maximum'],values=wall['values'],
            relative_range=(wall['maximum']-wall['minimum'])/wall['median'],
            total_iterations=d['trajectories'][case+'_verified']['total_iterations'],
            pc_applications=d['trajectories'][case+'_verified']['operator_event_counts']['PCApply'])
    ratio=summaries['Jacobi']['median']/summaries['ASM']['median'];assert ratio==d['jacobi_over_asm_median_core_ratio']
    return dict(status='passed',trajectories=10,steps=100,power='AC confirmed at both boundaries',
        case_summary=summaries,jacobi_over_asm_median_core_ratio=ratio,extrema=extrema,
        performance_measurement=True,ranking_scope='One fixed manual-driver P2/n16 kappa=1 dt=0.1 Radau trajectory on one laptop',
        limits=['Three measured repeats per PC; retain the slower third ASM sample; no cause inferred or sample discarded.',
            'Timed local work is explicitly derived from calibrated top-level PCApply counts and fixed additive composition.',
            'Warm construction/advance core wall boundary; do not pool absolute values with prior Irksome-driver measurements.',
            'Saved arithmetic and hashes audited without numerical operations; field/graph/PC operations remain researcher runtime evidence.'])

if __name__=='__main__':print(json.dumps(audit(),indent=2,allow_nan=False))

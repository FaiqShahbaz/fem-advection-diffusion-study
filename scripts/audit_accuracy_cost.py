#!/usr/bin/env python3
"""Audit saved targeted accuracy/cost observations without numerical imports or solves."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STEM = ROOT/'results/raw/accuracy-cost-20261019T102618Z-70377'
METRICS = ('l2_error','h1_seminorm_error')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit():
    sys.path.insert(0,str(ROOT/'environment'))
    import check_accuracy_cost as runner
    runner.plan()
    manifest = json.loads(Path(str(STEM)+'-manifest.json').read_text())
    raw = ROOT/manifest['measurement_records']['serial']['path']
    data = json.loads(raw.read_text());config = data['config']
    assert manifest['checkpoint_exit_status']==0 and manifest['working_tree_status_before_run']==''
    for key in ('power_state','power_state_after'):
        assert manifest[key]['source']=='AC Power' and manifest[key]['exit_status']==0
    for path,expected in manifest['script_sha256'].items():
        assert digest(ROOT/path)==expected,path
        assert hashlib.sha256(subprocess.check_output(['git','show',manifest['code_commit']+':'+path],cwd=ROOT)).hexdigest()==expected,path
    for key in ('config','cost_pilot_review','verified_environment'):
        assert digest(ROOT/manifest[key])==manifest[key+'_sha256'],key
    assert digest(raw)==manifest['measurement_records']['serial']['sha256']
    assert config==json.loads(runner.CONFIG.read_text())
    assert runner.verify_complete(data,config)==manifest['study_validation']
    assert data['total_dofs']==1089 and data['expected_free_dofs']==961 and data['expected_cells']==512
    assert all(data['initial_baseline_comparison'].values())
    extrema = dict(max_true_residual_over_nominal_target=0.0,max_update_or_stiff_accuracy_l2=0.0,
        max_timed_baseline_field_distance=0.0,max_continuum_norm_quadrature_change=0.0)
    def residual_check(residual,target):
        assert all(math.isfinite(v) and v>=0 for v in residual.values())
        assert math.isclose(residual['relative_norm'],residual['norm']/residual['rhs_norm'],rel_tol=1e-14)
        assert residual['norm']<=10*target
        extrema['max_true_residual_over_nominal_target']=max(extrema['max_true_residual_over_nominal_target'],residual['norm']/target)
    def identities(record):
        for key in ('update_identity_l2','stiff_accuracy_l2'):
            assert math.isfinite(record[key]) and record[key]<=1e-12
            extrema['max_update_or_stiff_accuracy_l2']=max(extrema['max_update_or_stiff_accuracy_l2'],record[key])
    for label,row in data['trajectories'].items():
        method,dt=row['method'],row['dt'];parameters=row.get('solver_parameters',config['solver_parameters'])
        actual=row['actual_solver']
        assert actual['ksp_type']=='gmres' and actual['pc_type']=='jacobi' and actual['side']=='right' and actual['norm']=='unpreconditioned'
        assert actual['rtol']==parameters['ksp_rtol'] and actual['atol']==parameters['ksp_atol'] and actual['max_it']==1000 and actual['requested_restart']==50
        assert row['total_iterations']==sum(step['iterations'] for step in row['step_records'])
        assert row['maximum_step_iterations']==max(step['iterations'] for step in row['step_records'])
        timed=row['role'] in ('warmup','measured')
        if timed:
            baseline=data['trajectories'][row['case_id']+'_verified']
            assert [r['iterations'] for r in row['step_records']]==[r['iterations'] for r in baseline['step_records']]
        for index,step in enumerate(row['step_records']):
            assert step['step']==index+1 and math.isclose(step['t_start'],index*dt,rel_tol=0,abs_tol=1e-14)
            assert math.isclose(step['t_end'],(index+1)*dt,rel_tol=0,abs_tol=1e-14)
            assert step['converged_reason']>0 and 0<step['iterations']<=1000
            assert math.isfinite(step['reported_residual_norm']) and step['reported_residual_norm']>=0
            times=step['source_times'] if timed else step.get('stage_times',[step.get('source_time')])
            expected=[(index+1)*dt] if method=='manual_backward_euler' else [(index+1/3)*dt,(index+1)*dt]
            assert len(times)==len(expected) and all(math.isclose(a,b,rel_tol=0,abs_tol=1e-14) for a,b in zip(times,expected)),(label,index)
            if timed:
                assert math.isfinite(step['wall_seconds']) and step['wall_seconds']>=0
            else:
                residual=step['true_BE_residual' if method=='manual_backward_euler' else 'true_block_residual']
                target=max(parameters['ksp_atol'],parameters['ksp_rtol']*residual['rhs_norm'])
                assert target==step['true_residual_target'];residual_check(residual,target)
                if method=='RadauIIA':identities(step)
        if method=='RadauIIA':
            assert row['solver_stats'][0]==row['completed_steps'] and row['solver_stats'][2]==row['total_iterations']
        if timed:
            final=row['final_equation_check'];residual=final['true_residual']
            target=max(parameters['ksp_atol'],parameters['ksp_rtol']*residual['rhs_norm'])
            assert target==final['nominal_target'];residual_check(residual,target)
            count=row['completed_steps']
            expected=[1.0] if method=='manual_backward_euler' else [(count-1+1/3)*dt,1.0]
            assert len(final['source_times'])==len(expected) and all(math.isclose(a,b,rel_tol=0,abs_tol=1e-14) for a,b in zip(final['source_times'],expected))
            if method=='RadauIIA':
                identities(final);assert all(v<=1e-12 for v in final['slope_boundary_trace_l2'])
            for metric in METRICS:
                eq=row['equivalence'][metric];difference=row['difference_measurement']['metrics'][metric]
                assert eq['actual_field_difference']==difference and eq['allowed_difference']==1e-12+1e-10*eq['baseline_field_norm']
                assert difference<=eq['allowed_difference']
                extrema['max_timed_baseline_field_distance']=max(extrema['max_timed_baseline_field_distance'],difference)
                assert row['control_metrics'][metric]==baseline['control_metrics'][metric]
                transfer=row['time_error_transfer']['norms'][metric]
                curve=next(c for c in data['curve'] if c['case_id']==row['case_id'])
                error=curve['time_errors'][metric]
                assert transfer['verified_baseline_time_error']==error and transfer['measured_timed_baseline_distance']==difference
                assert transfer['allowed_transfer_distance']==1e-12+0.01*error
                assert transfer['lower']==max(0,error-difference) and transfer['upper']==error+difference and difference<=transfer['allowed_transfer_distance']
            timing=row['timing'];steps=row['step_records'];events=timing['petsc_event_deltas']
            assert timing['first_step_seconds']==steps[0]['wall_seconds']
            assert timing['remaining_steps_seconds']==sum(s['wall_seconds'] for s in steps[1:])
            assert events['KSPSolve']['count']==count and events['KSPSetUp']['count']==count and events['PCSetUp']['count']==1
            assert all(math.isfinite(v) and v>=0 for event in events.values() for v in event.values())
            assert row['system_dofs']==(1089 if method=='manual_backward_euler' else 2178)
        else:
            assert all(s['source_residual_l2']<=1e-10 for s in row['source_checks'])
        control=row['control_metrics']
        assert abs(control['domain_area']-1)<=1e-12 and control['boundary_trace_l2']<=1e-12
        assert math.isclose(control['exact_l2_norm'],math.exp(-1)/2,rel_tol=1e-10,abs_tol=1e-12)
        assert math.isclose(control['exact_h1_seminorm'],math.pi*math.exp(-1)/math.sqrt(2),rel_tol=1e-10,abs_tol=1e-12)
        for metric in METRICS:
            change=abs(row['metrics'][metric]-control[metric]);assert change<=1e-13+1e-6*control[metric]
            extrema['max_continuum_norm_quadrature_change']=max(extrema['max_continuum_norm_quadrature_change'],change)
    def difference_check(item):
        assert item['status']=='passed'
        for metric in METRICS:
            q10,q14=item['primary_metrics'][metric],item['metrics'][metric]
            assert math.isfinite(q14) and q14>=0 and abs(q10-q14)<=1e-13+1e-6*q14
    for item in [data['reference_gap'],data['reference_solver_difference']]+[c['difference_measurement'] for c in data['curve']]:difference_check(item)
    fractions={}
    for metric in METRICS:
        reference=data['reference_resolution']['norms'][metric]
        combined=data['reference_gap']['metrics'][metric]+data['reference_solver_difference']['metrics'][metric]
        minimum=min(c['time_errors'][metric] for c in data['curve'])
        assert reference['combined_sensitivity']==combined and reference['allowed_combined_sensitivity']==1e-12+0.01*minimum
        assert combined<=reference['allowed_combined_sensitivity'];fractions[metric]=combined/minimum
    for method,control in data['curve_solver_controls'].items():
        difference_check(control['measured_difference'])
        finest=min([c for c in data['curve'] if c['method']==method],key=lambda c:c['dt'])
        for metric in METRICS:
            row=control['norms'][metric];difference=control['measured_difference']['metrics'][metric]
            assert row['actual_field_difference']==difference and row['temporal_error_scale']==finest['time_errors'][metric]
            assert row['allowed_difference']==1e-12+0.01*finest['time_errors'][metric] and difference<=row['allowed_difference']
    for rate in data['rates']:
        coarse=next(c for c in data['curve'] if c['method']==rate['method'] and c['dt']==rate['coarse_dt'])
        fine=next(c for c in data['curve'] if c['method']==rate['method'] and c['dt']==rate['fine_dt'])
        value=math.log(coarse['time_errors'][rate['metric']]/fine['time_errors'][rate['metric']])/math.log(coarse['dt']/fine['dt'])
        assert value==rate['observed_order'] and abs(value-config['expected_temporal_orders'][rate['method']])<=0.4
    assert Path(str(STEM)+'.txt').read_text().count('restart=50')==25
    summary={c['case_id']:dict(time_errors=c['time_errors'],wall=c['timing_summary']['trajectory_wall_seconds'],
        total_iterations=data['trajectories'][c['trajectory']]['total_iterations']) for c in data['curve']}
    ratios={}
    for target in config['target_time_errors_l2']:
        chosen={r['method']:r['selected_case'] for r in data['candidate_selections'] if r['target_l2_time_error']==target}
        be=summary[chosen['manual_backward_euler']]['wall']['median'];radau=summary[chosen['RadauIIA']]['wall']['median']
        ratios[str(target)]=dict(BE_case=chosen['manual_backward_euler'],Radau_case=chosen['RadauIIA'],BE_over_Radau_median=be/radau)
    loaded=[n for n in sys.modules if n.split('.')[0] in ('firedrake','petsc4py','irksome','mpi4py')];assert not loaded
    return dict(status='passed',trajectories=25,steps=15770,AC_confirmed_before_after=True,extrema=extrema,
        reference_sensitivity_fraction=fractions,candidate_summary=summary,target_median_ratios=ratios,numerical_modules_loaded=loaded)


if __name__=='__main__':
    print(json.dumps(audit(),indent=2))

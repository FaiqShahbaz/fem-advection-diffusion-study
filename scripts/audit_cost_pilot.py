#!/usr/bin/env python3
"""Audit saved cost-pilot evidence using only the standard library; run no FEM code."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STEM = ROOT/'results/raw/cost-pilot-20261018T120233Z-81068'
METRICS = ('l2_error', 'h1_seminorm_error')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit():
    sys.path.insert(0, str(ROOT/'environment'))
    import check_cost_pilot as runner
    runner.plan()  # Guard the preceding accepted temporal/environment chain.
    manifest_path = Path(str(STEM)+'-manifest.json')
    transcript_path = Path(str(STEM)+'.txt')
    manifest = json.loads(manifest_path.read_text())
    raw_path = ROOT/manifest['measurement_records']['serial']['path']
    data = json.loads(raw_path.read_text())
    assert manifest['checkpoint_exit_status'] == 0 and manifest['working_tree_status_before_run'] == ''
    for name, expected in manifest['script_sha256'].items():
        assert digest(ROOT/name) == expected, name
        historical = subprocess.check_output(['git','show',manifest['code_commit']+':'+name], cwd=ROOT)
        assert hashlib.sha256(historical).hexdigest() == expected, name
    for key in ('config','temporal_review','verified_environment'):
        assert digest(ROOT/manifest[key]) == manifest[key+'_sha256'], key
    assert digest(raw_path) == manifest['measurement_records']['serial']['sha256']
    assert runner.verify_complete(data, data['config']) == manifest['study_validation']
    assert data['config'] == json.loads(runner.CONFIG.read_text())
    assert data['total_dofs'] == 1089 and data['expected_free_dofs'] == 961 and data['expected_cells'] == 512
    extrema = dict(max_true_residual_over_target=0.0, max_update_or_stiff_defect=0.0,
                   max_field_equivalence_difference=0.0)
    def residual_check(residual, target):
        assert all(math.isfinite(value) and value >= 0 for value in residual.values())
        assert residual['norm'] <= 10*target
        assert math.isclose(residual['relative_norm'], residual['norm']/residual['rhs_norm'], rel_tol=1e-14)
        extrema['max_true_residual_over_target'] = max(extrema['max_true_residual_over_target'], residual['norm']/target)
    for label, row in data['trajectories'].items():
        method, dt = row['method'], row['dt']
        par, actual = row.get('solver_parameters', data['config']['solver_parameters']), row['actual_solver']
        assert method in ('manual_backward_euler','RadauIIA') and dt == 0.05
        assert actual['ksp_type'] == 'gmres' and actual['pc_type'] == 'jacobi'
        assert actual['side'] == 'right' and actual['norm'] == 'unpreconditioned'
        assert actual['rtol'] == 1e-11 and actual['atol'] == 1e-13 and actual['max_it'] == 1000 and actual['requested_restart'] == 50
        assert row['total_iterations'] == sum(step['iterations'] for step in row['step_records'])
        assert row['maximum_step_iterations'] == max(step['iterations'] for step in row['step_records'])
        baseline = data['trajectories'][method+'_verified']
        assert [s['iterations'] for s in row['step_records']] == [s['iterations'] for s in baseline['step_records']]
        for index, step in enumerate(row['step_records']):
            assert step['step'] == index+1 and math.isclose(step['t_start'], index*dt, rel_tol=0, abs_tol=1e-14)
            assert math.isclose(step['t_end'], (index+1)*dt, rel_tol=0, abs_tol=1e-14) and step['converged_reason'] > 0
            if row['role'] != 'verified_baseline':
                source = step['source_times']
                assert math.isfinite(step['wall_seconds']) and step['wall_seconds'] >= 0
            else:
                source = step.get('stage_times', [step.get('source_time')])
            expected = [(index+1)*dt] if method == 'manual_backward_euler' else [(index+1/3)*dt, (index+1)*dt]
            assert len(source) == len(expected) and all(math.isclose(a,b,rel_tol=0,abs_tol=1e-14) for a,b in zip(source,expected)), (label,index)
            if row['role'] == 'verified_baseline':
                residual = step['true_BE_residual' if method == 'manual_backward_euler' else 'true_block_residual']
                target = max(par['ksp_atol'], par['ksp_rtol']*residual['rhs_norm'])
                assert target == step['true_residual_target']
                residual_check(residual, target)
                if method == 'RadauIIA':
                    for key in ('update_identity_l2','stiff_accuracy_l2'):
                        assert step[key] <= 1e-12
                        extrema['max_update_or_stiff_defect'] = max(extrema['max_update_or_stiff_defect'], step[key])
        if row['role'] != 'verified_baseline':
            final = row['final_equation_check']
            residual = final['true_residual']
            target = max(1e-13, 1e-11*residual['rhs_norm'])
            assert target == final['nominal_target']
            residual_check(residual, target)
            for metric, eq in row['equivalence'].items():
                assert eq['allowed_difference'] == 1e-12+1e-10*eq['baseline_field_norm']
                assert eq['actual_field_difference'] == row['difference_measurement']['metrics'][metric]
                assert eq['actual_field_difference'] <= eq['allowed_difference']
                extrema['max_field_equivalence_difference'] = max(extrema['max_field_equivalence_difference'],eq['actual_field_difference'])
            for metric in METRICS:
                assert row['control_metrics'][metric] == baseline['control_metrics'][metric]
                diff = row['difference_measurement']
                assert abs(diff['primary_metrics'][metric]-diff['metrics'][metric]) <= 1e-13+1e-6*diff['metrics'][metric]
            timing, steps = row['timing'], row['step_records']
            assert timing['first_step_seconds'] == steps[0]['wall_seconds']
            assert timing['remaining_steps_seconds'] == sum(s['wall_seconds'] for s in steps[1:])
            events = timing['petsc_event_deltas']
            assert events['KSPSolve']['count'] == 20 and events['KSPSetUp']['count'] == 20 and events['PCSetUp']['count'] == 1
            assert all(v >= 0 and math.isfinite(v) for event in events.values() for v in event.values())
            assert row['system_dofs'] == (1089 if method == 'manual_backward_euler' else 2178)
            if method == 'RadauIIA':
                for key in ('update_identity_l2','stiff_accuracy_l2'):
                    assert final[key] <= 1e-12
                    extrema['max_update_or_stiff_defect'] = max(extrema['max_update_or_stiff_defect'],final[key])
                assert all(x <= 1e-12 for x in final['slope_boundary_trace_l2'])
        else:
            assert all(s['source_residual_l2'] <= 1e-10 for s in row['source_checks'])
        for metric in METRICS:
            primary, control = row['metrics'][metric], row['control_metrics'][metric]
            assert abs(primary-control) <= 1e-13+1e-6*control
        metrics = row['control_metrics']
        assert metrics['boundary_trace_l2'] <= 1e-12
        assert math.isclose(metrics['domain_area'],1.0,rel_tol=0,abs_tol=1e-12)
        assert math.isclose(metrics['exact_l2_norm'],math.exp(-1)/2,rel_tol=1e-10,abs_tol=1e-12)
        assert math.isclose(metrics['exact_h1_seminorm'],math.pi*math.exp(-1)/math.sqrt(2),rel_tol=1e-10,abs_tol=1e-12)
    assert transcript_path.read_text().count('restart=50') == 10
    summary = {}
    for method, item in data['timing_summary'].items():
        wall = item['trajectory_wall_seconds']
        summary[method] = dict(wall_median=wall['median'], wall_min=wall['minimum'], wall_max=wall['maximum'],
            range_over_median=(wall['maximum']-wall['minimum'])/wall['median'])
    loaded = [n for n in sys.modules if n.split('.')[0] in ('firedrake','petsc4py','irksome','mpi4py')]
    assert not loaded
    return dict(audit='passed', trajectories=10, steps=200, extrema=extrema, timing_summary=summary,
                power_source='Battery Power' if "'Battery Power'" in manifest['power_state']['output'] else 'other',
                numerical_modules_loaded=loaded)


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))

#!/usr/bin/env python3
"""Researcher-run matched warm core timing after accepted selected full trajectories."""
import argparse
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

from check_selected_trajectories import plan as previous_environment_plan
from check_accuracy_cost import read_power_state
from configure_petsc import ROOT,digest,run_logged,validate_tools

CONFIG = ROOT/'configs/phase3_selected_cost.json'
SCRIPT = ROOT/'src/phase3_selected_cost.py'
SNAPSHOT = ROOT/'environment/verified_environment.json'


def plan():
    _,_,installation,petsc,venv,env=previous_environment_plan()
    review_path=ROOT/'results/processed/selected_trajectories_review.json'
    review=json.loads(review_path.read_text())
    pairs=[(review[key],review[key+'_sha256']) for key in
           ('trajectory_manifest','trajectory_transcript','config','solver_sensitivity_review','verified_environment','installed_source_inspection')]
    pairs.extend((r['path'],r['sha256']) for r in review['measurement_records'].values())
    pairs.extend(review['run_script_sha256'].items())
    pairs.append((review['audit_source']['path'],review['audit_source']['sha256']))
    for path,expected in pairs:
        if digest(ROOT/path)!=expected:raise RuntimeError(f'Accepted trajectory evidence/source changed: {path}')
    if review['status']!='accepted selected full trajectories and fixed-factor reuse' or not review['selected_trajectories_verified']:
        raise RuntimeError('Accepted selected full-trajectory verification is required.')
    sys.path.insert(0,str(ROOT/'src'))
    from phase3_selected_cost import validate_config
    validate_config(json.loads(CONFIG.read_text()))
    if review['independent_audit']['completed_candidate_trajectories']!=8 or review['independent_audit']['new_capped_candidates']!=0:
        raise RuntimeError('All eight selected full candidates must be accepted before timing.')
    return review_path,review,installation,petsc,venv,env


def commands(venv,stem):
    return [('serial',[str(venv/'bin/python'),str(SCRIPT),'--config',str(CONFIG),
                       '--output',str(stem)+'-serial.json'])]


def verify_complete(record,config):
    from check_selected_trajectories import verify_complete as verify_checked
    full_config=json.loads((ROOT/'configs/phase3_selected_trajectories.json').read_text())
    checked_validation=verify_checked(record['checked_study'],full_config)
    if (record['status']!='passed' or not record['completed_study'] or record['config']!=config or record['ranks']!=1
            or record['performance_measurement'] is not True or record['completed_trajectories']!=43 or record['completed_steps']!=430
            or record['predeclared_exclusions']!=full_config['planned_skips']):raise RuntimeError('Selected matched timing study did not pass.')
    expected={case['case_id']+suffix for case in config['cases'] for suffix in ('_warmup0','_measured1','_measured2','_measured3')}
    order=[c['case_id']+'_warmup0' for c in config['cases']]
    order += [case+'_measured'+str(i) for i,ids in enumerate(config['measured_case_order'],1) for case in ids]
    if set(record['trajectories'])!=expected or record['execution_order']!=order:raise RuntimeError('Missing/reordered samples.')
    for case in config['cases']:
        baseline=record['checked_study']['points'][case['point_id']]['candidates'][case['solver']]
        if baseline['status']!='passed' or not all(baseline['accepted_metric_reproduction'].values()):raise RuntimeError('Fresh accepted-path reproduction failed.')
        for suffix in ('_warmup0','_measured1','_measured2','_measured3'):
            row=record['trajectories'][case['case_id']+suffix];params=row['solver_parameters'];actual=row['actual_solver']
            expected_params=config['rana_solver_parameters'] if case['solver']=='RanaLD' else config['asm_solver_parameters'] if case['solver']=='ASM' else {**config['solver_parameters'],'mat_type':'aij','snes_type':'ksponly'}
            if (row['status']!='passed' or row['completed_steps']!=10 or len(row['step_records'])!=10 or row['final_time']!=1.
                    or not all(row['checks'].values()) or params!=expected_params or row['case_id']!=case['case_id'] or row['solver']!=case['solver']
                    or row['total_iterations']!=sum(s['iterations'] for s in row['step_records']) or row['weak_form_argument_count']!=1
                    or row['stage_total_dofs']!=baseline['stage_total_dofs'] or not row['constant_jacobian']
                    or row['timed_python_monitors'] or row['timed_local_callbacks'] or row['solver_stats']!=[10,10,row['total_iterations']]):raise RuntimeError('Incomplete timed trajectory.')
            if (actual['ksp_type']!='gmres' or actual['pc_type']!=params['pc_type'] or actual['snes_type']!='ksponly' or actual['side']!='right'
                    or actual['norm']!='unpreconditioned' or actual['rtol']!=params['ksp_rtol'] or actual['atol']!=params['ksp_atol'] or actual['max_it']!=1000):raise RuntimeError('Timed solver parameters changed.')
            if [s['iterations'] for s in row['step_records']]!=[s['iterations'] for s in baseline['step_records']]:raise RuntimeError('Timed iteration sequence changed.')
            for i,step in enumerate(row['step_records']):
                if (step['step']!=i+1 or step['converged_reason']<=0 or not 0<step['iterations']<=1000 or abs(step['t_start']-i*.1)>1e-14
                        or abs(step['t_end']-(i+1)*.1)>1e-14 or any(abs(a-b)>1e-14 for a,b in zip(step['stage_times'],[(i+1/3)*.1,(i+1)*.1]))):raise RuntimeError('Timed stage clock or convergence failed.')
            timing=row['timing'];events=timing['petsc_event_deltas']
            if not all(math.isfinite(v) and v>=0 for k,v in timing.items() if k.endswith('_seconds')):raise RuntimeError('Invalid timing partition.')
            partition=sum(timing[k] for k in ('construction_seconds','first_step_seconds','remaining_steps_seconds','loop_bookkeeping_seconds'))
            if not math.isclose(partition,timing['trajectory_wall_seconds'],rel_tol=1e-12,abs_tol=1e-12):raise RuntimeError('Timing partition mismatch.')
            if any(events[e]['count']!=count for e,count in baseline['operator_event_counts'].items()):raise RuntimeError('Timed setup/application work changed.')
            final=row['final_equation_check'];r=final['true_block_residual'];target=max(params['ksp_atol'],params['ksp_rtol']*r['rhs_norm'])
            if (not all(math.isfinite(v) and v>=0 for v in r.values()) or r['rhs_norm']<=0 or final['true_residual_target']!=target or r['norm']>10*target
                    or final['update_identity_l2']>1e-12 or final['stiff_accuracy_l2']>1e-12 or any(v>1e-12 for v in final['stage_boundary_trace_l2'])
                    or not all(s['source_residual_l2']<=1e-10 for s in row['source_checks'])):raise RuntimeError('Timed final physical equation failed.')
            for check in row['matrix_checks'].values():
                allowed=config['checks']['auxiliary_matrix_atol']+config['checks']['auxiliary_matrix_rtol']*check['reference_frobenius_norm']
                if check['status']!='passed' or check['allowed_defect']!=allowed or not 0<=check['frobenius_defect']<=allowed:raise RuntimeError('Timed coefficient check failed.')
            if row['difference_measurement']['status']!='passed' or set(row['equivalence'])!={'l2_error','h1_seminorm_error'}:raise RuntimeError('Timed field quadrature comparison failed.')
            for check in row['equivalence'].values():
                allowed=config['checks']['equivalence_atol']+config['checks']['equivalence_rtol']*check['baseline_field_norm']
                if check['allowed_difference']!=allowed or not math.isfinite(check['actual_field_difference']) or not 0<=check['actual_field_difference']<=allowed:raise RuntimeError('Timed actual-field equivalence failed.')
            if case['solver']=='RanaLD':
                if (row['auxiliary_action_check']['status']!='passed' or row['auxiliary_solver']!=baseline['auxiliary_solver']
                        or row['diagnostic_event_deltas']['MatSolve']['count']!=1):raise RuntimeError('Timed Rana structure/action failed.')
            elif case['solver']=='ASM':
                if (row['domains']!=baseline['domains'] or row['local_solvers']!=baseline['local_solvers'] or row['additive_action_check']['status']!='passed'
                        or row['local_work_derived']['total']!=4*events['PCApply']['count'] or events['MatSolve']['count']!=row['local_work_derived']['total']
                        or row['diagnostic_event_deltas']['MatSolve']['count']!=8):raise RuntimeError('Timed ASM structure/action failed.')
        for key in ('construction_seconds','first_step_seconds','remaining_steps_seconds','trajectory_wall_seconds'):
            values=[record['trajectories'][case['case_id']+'_measured'+str(i)]['timing'][key] for i in (1,2,3)]
            s=record['timing_summary'][case['case_id']][key]
            if s!=dict(values=values,minimum=min(values),median=sorted(values)[1],maximum=max(values)):raise RuntimeError('Timing summary does not retain all samples.')
    return dict(status='passed',completed_trajectories=43,completed_steps=430,checked_trajectories=11,warmups=8,measured_trajectories=24,
        checked_validation=checked_validation,predeclared_exclusions=full_config['planned_skips'],performance_measurement=True,
        interpretation='Matched residual-form warm core costs within each selected point; all samples retained; mentor review pending')


def run(review_path,review,installation,petsc,venv,env):
    selected_tools = validate_tools(petsc)
    snapshot = json.loads(SNAPSHOT.read_text())
    inventory = subprocess.check_output([str(venv/'bin/python'),'-m','pip','--no-cache-dir','freeze','--all'],env=env,text=True)
    if inventory!=(ROOT/snapshot['resolved_python_inventory']).read_text():
        raise RuntimeError('Installed inventory differs from the tested snapshot.')
    for key in ('FIREDRAKE_CACHE_DIR','PYOP2_CACHE_DIR','FIREDRAKE_TSFC_KERNEL_CACHE_DIR','TMPDIR'):
        Path(env[key]).mkdir(parents=True,exist_ok=True)
    config = json.loads(CONFIG.read_text())
    power_state=read_power_state()
    if power_state['source']!=config['required_power_source'] or power_state['exit_status']!=0:
        raise RuntimeError('Connect the charger and confirm AC Power before this measurement run; current power source: '+str(power_state['source']))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    stem = ROOT/'results/raw'/f'selected-cost-{stamp}-{os.getpid()}'
    transcript = stem.with_suffix('.txt')
    manifest_path = Path(str(stem)+'-manifest.json')
    steps = commands(venv,stem)
    sources=tuple(dict.fromkeys([SCRIPT,*[ROOT/path for path in review['run_script_sha256']],Path(__file__).resolve()]))
    state = {'started_utc':datetime.now(timezone.utc).isoformat(),'status':'started',
        'measurement_kind':'Matched residual-form warm core timing for eight accepted candidates at three selected points; retained finer-Jacobi exclusion',
        'selected_trajectories_review':str(review_path.relative_to(ROOT)),'selected_trajectories_review_sha256':digest(review_path),
        'verified_environment':str(SNAPSHOT.relative_to(ROOT)),'verified_environment_sha256':digest(SNAPSHOT),
        'config':str(CONFIG.relative_to(ROOT)),'config_sha256':digest(CONFIG),
        'script_sha256':{str(path.relative_to(ROOT)):digest(path) for path in sources},
        'installed_source_inspection':review['installed_source_inspection'],
        'installed_source_inspection_sha256':review['installed_source_inspection_sha256'],
        'inspected_installed_source_sha256':review['inspected_installed_source_sha256'],
        'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'working_tree_status_before_run':subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
        'selected_tools':selected_tools,
        'selected_environment':{key:env[key] for key in ('PETSC_DIR','PETSC_ARCH','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS',
            'FIREDRAKE_CACHE_DIR','PYOP2_CACHE_DIR','FIREDRAKE_TSFC_KERNEL_CACHE_DIR','XDG_CACHE_HOME','TMPDIR',
            'PYTHONNOUSERSITE','PYTHONDONTWRITEBYTECODE')},
        'inherited_options_or_library_settings':{key:env[key] for key in
            ('PETSC_OPTIONS','DYLD_LIBRARY_PATH','DYLD_FALLBACK_LIBRARY_PATH','LD_LIBRARY_PATH') if key in env},
        'commands':{name:argv for name,argv in steps},'completed_steps':[],
        'power_state':power_state,'selected_cost_review_pending':True,'performance_measurement':True,'accuracy_cost_ranking_established':False}
    status = 1
    manifest_path.write_text(json.dumps(state,indent=2)+'\n')
    with transcript.open('w') as log:
        try:
            for name,argv in steps:
                run_logged(argv,ROOT,env,log)
                state['completed_steps'].append(name)
            state['power_state_after']=read_power_state()
            if state['power_state_after']['source']!=config['required_power_source'] or state['power_state_after']['exit_status']!=0:
                raise RuntimeError('AC Power was not confirmed after the run; preserve records without accepting an AC comparison.')
            state['study_validation'] = verify_complete(json.loads(Path(str(stem)+'-serial.json').read_text()),config)
            state['status'] = 'Selected matched core timing checks passed; mentor review pending'
            status = 0
            message = 'SELECTED COST CHECKPOINT PASSED. Review within-point matched timings and all repeats before further work.'
            print(message,flush=True);log.write(message+'\n')
        except (RuntimeError,OSError,subprocess.SubprocessError,KeyError,ValueError,KeyboardInterrupt) as error:
            state['status'] = 'failed or interrupted; review required'
            state['error'] = str(error) or 'Interrupted by researcher'
            message = f"STOP: {state['error']}"
            print(message,flush=True);log.write(message+'\n')
        finally:
            state['measurement_records'] = {}
            path = Path(str(stem)+'-serial.json')
            if path.is_file():
                state['measurement_records']['serial'] = {'path':str(path.relative_to(ROOT)),'sha256':digest(path)}
            state['finished_utc'] = datetime.now(timezone.utc).isoformat()
            state['checkpoint_exit_status'] = status
            manifest_path.write_text(json.dumps(state,indent=2)+'\n')
    print(f'Saved transcript: {transcript}')
    print(f'Saved manifest: {manifest_path}')
    for record in state['measurement_records'].values():
        print(f"Saved measurement record: {ROOT/record['path']}")
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--show-commands',action='store_true',help='Display inputs/command without numerical imports or experiments.')
    args = parser.parse_args()
    parameters = plan()
    if args.show_commands:
        print(CONFIG.read_text())
        stem = ROOT/'results/raw/selected-cost-<UTC timestamp>-<process id>'
        for name,argv in commands(parameters[4],stem):
            print(f'{name}: {shlex.join(argv)}')
        print('Display only: no numerical imports, assembly, solves, time stepping, or MPI launch run.')
        return 0
    return run(*parameters)


if __name__=='__main__':
    try:
        sys.exit(main())
    except (RuntimeError,OSError,subprocess.SubprocessError,KeyError,ValueError) as error:
        print(f'STOP before selected matched cost comparison: {error}',file=sys.stderr)
        sys.exit(1)

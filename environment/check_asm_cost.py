#!/usr/bin/env python3
"""Researcher-run matched manual Radau Jacobi/ASM repeated timing after accepted trajectory verification."""
import argparse
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

from check_asm_trajectory import plan as previous_environment_plan
from check_accuracy_cost import read_power_state
from configure_petsc import ROOT,digest,run_logged,validate_tools

CONFIG = ROOT/'configs/phase3_asm_cost.json'
SCRIPT = ROOT/'src/phase3_asm_cost.py'
SNAPSHOT = ROOT/'environment/verified_environment.json'


def plan():
    _,_,installation,petsc,venv,env=previous_environment_plan()
    review_path=ROOT/'results/processed/asm_trajectory_review.json'
    review=json.loads(review_path.read_text())
    pairs=[(review[key],review[key+'_sha256']) for key in
           ('trajectory_manifest','trajectory_transcript','config','asm_construction_review','verified_environment')]
    pairs.extend((r['path'],r['sha256']) for r in review['measurement_records'].values())
    pairs.extend(review['run_script_sha256'].items())
    pairs.append((review['audit_source']['path'],review['audit_source']['sha256']))
    for path,expected in pairs:
        if digest(ROOT/path)!=expected:raise RuntimeError(f'Accepted trajectory evidence/source changed: {path}')
    if review['status']!='accepted full ASM trajectory agreement and reuse' or not review['asm_trajectory_verified']:
        raise RuntimeError('Accepted full ASM trajectory verification is required.')
    return review_path,review,installation,petsc,venv,env


def commands(venv,stem):
    return [('serial',[str(venv/'bin/python'),str(SCRIPT),'--config',str(CONFIG),
                       '--output',str(stem)+'-serial.json'])]


def verify_complete(record,config):
    if (record['status']!='passed' or not record['completed_study'] or record['config']!=config
            or record['ranks']!=1 or record['performance_measurement'] is not True
            or record['completed_trajectories']!=10 or record['completed_steps']!=100
            or record['total_dofs']!=1089 or record['stage_total_dofs']!=2178):
        raise RuntimeError('The matched timing comparison did not pass.')
    expected={case['case_id']+suffix for case in config['cases'] for suffix in ('_verified','_warmup0','_measured1','_measured2','_measured3')}
    if set(record['trajectories'])!=expected:raise RuntimeError('Missing declared trajectories.')
    order=[c['case_id']+'_warmup0' for c in config['cases']]
    order += [case+'_measured'+str(i) for i,ids in enumerate(config['measured_case_order'],1) for case in ids]
    if record['execution_order']!=order:raise RuntimeError('Warmup/repetition order changed.')
    def equivalence(rows):
        if set(rows)!={'l2_error','h1_seminorm_error'}:raise RuntimeError('Missing field norms.')
        for r in rows.values():
            allowed=config['checks']['equivalence_atol']+config['checks']['equivalence_rtol']*r['baseline_field_norm']
            if not math.isfinite(r['actual_field_difference']) or r['actual_field_difference']<0 or allowed!=r['allowed_difference'] or r['actual_field_difference']>allowed:
                raise RuntimeError('Actual field equivalence failed.')
    if record['verified_pair_difference']['status']!='passed':raise RuntimeError('Checked pair quadrature failed.')
    equivalence(record['verified_pair_equivalence'])
    for label,row in record['trajectories'].items():
        if (row['status']!='passed' or row['completed_steps']!=10 or len(row['step_records'])!=10
                or row['final_time']!=1.0 or not all(row['checks'].values()) or row['matrix_assemblies']!=1
                or row['total_iterations']!=sum(s['iterations'] for s in row['step_records'])):
            raise RuntimeError('Incomplete trajectory.')
        for i,step in enumerate(row['step_records']):
            if (step['step']!=i+1 or abs(step['t_start']-i*.1)>1e-14 or abs(step['t_end']-(i+1)*.1)>1e-14
                    or step['converged_reason']<=0 or not 0<step['iterations']<=1000
                    or len(step['stage_times'])!=2 or any(abs(a-b)>1e-14 for a,b in zip(step['stage_times'],[(i+1/3)*.1,(i+1)*.1]))):
                raise RuntimeError('Clock, stages or convergence failed.')
        params=row['solver_parameters'];actual=row['actual_solver']
        expected_params=config['asm_solver_parameters'] if row['case_id']=='ASM' else config['solver_parameters']
        if (params!=expected_params or actual['ksp_type']!='gmres' or actual['pc_type']!=params['pc_type']
                or actual['side']!='right' or actual['norm']!='unpreconditioned'
                or actual['rtol']!=params['ksp_rtol'] or actual['atol']!=params['ksp_atol'] or actual['max_it']!=1000):
            raise RuntimeError('Actual solver parameters changed.')
        if row['role']=='verified':
            if not all(row['accepted_metric_reproduction'].values()):raise RuntimeError('Accepted baseline reproduction failed.')
            for step in row['step_records']:
                r=step['true_block_residual'];target=max(params['ksp_atol'],params['ksp_rtol']*r['rhs_norm'])
                if not all(math.isfinite(v) for v in r.values()) or target!=step['true_residual_target'] or r['norm']>10*target:
                    raise RuntimeError('Verified true physical residual failed.')
        else:
            if row['timed_python_monitors'] or row['timed_local_callbacks']:raise RuntimeError('Unexpected timed Python solver callbacks.')
            checked=record['trajectories'][row['case_id']+'_verified']
            if [s['iterations'] for s in row['step_records']]!=[s['iterations'] for s in checked['step_records']]:
                raise RuntimeError('Timed work sequence changed.')
            timing=row['timing'];events=timing['petsc_event_deltas']
            if not all(math.isfinite(v) and v>=0 for k,v in timing.items() if k.endswith('_seconds')):
                raise RuntimeError('Invalid timing.')
            partition=sum(timing[k] for k in ('construction_seconds','first_step_seconds','remaining_steps_seconds','loop_bookkeeping_seconds'))
            if (not math.isclose(partition,timing['trajectory_wall_seconds'],rel_tol=1e-12,abs_tol=1e-12)
                    or timing['first_step_seconds']!=row['step_records'][0]['wall_seconds']
                    or timing['remaining_steps_seconds']!=sum(s['wall_seconds'] for s in row['step_records'][1:])):
                raise RuntimeError('Timing partition failed.')
            if not all(events[e]['count']==checked['operator_event_counts'][e] for e in events):
                raise RuntimeError('Timed profiling configuration/work changed.')
            final=row['final_equation_check'];r=final['true_block_residual'];target=max(params['ksp_atol'],params['ksp_rtol']*r['rhs_norm'])
            if (not all(math.isfinite(v) for v in r.values()) or target!=final['true_residual_target'] or r['norm']>10*target
                    or final['update_identity_l2']>1e-12 or final['stiff_accuracy_l2']>1e-12
                    or any(v>1e-12 for v in final['stage_boundary_trace_l2'])
                    or any(abs(a-b)>1e-14 for a,b in zip(final['stage_times'],[(9+1/3)*.1,1.]))):
                raise RuntimeError('Timed final physical equation failed.')
            if row['difference_measurement']['status']!='passed' or not all(s['source_residual_l2']<=1e-10 for s in row['source_checks']):
                raise RuntimeError('Timed field quadrature/source failed.')
            equivalence(row['equivalence'])
            if row['case_id']=='ASM':
                applications=int(events['PCApply']['count']);work=row['local_work_derived']
                if (work['applications_per_domain']!=applications or work['total_local_applications']!=4*applications or work['domains']!=4
                        or row['domains']!=checked['domains'] or row['local_solvers']!=checked['local_solvers']
                        or row['diagnostic_additive_action']['status']!='passed'):
                    raise RuntimeError('ASM construction or declared local-work derivation failed.')
    for case in config['cases']:
        rows=[record['trajectories'][case['case_id']+'_measured'+str(i)] for i in (1,2,3)]
        for key in ('construction_seconds','first_step_seconds','remaining_steps_seconds','trajectory_wall_seconds'):
            values=[r['timing'][key] for r in rows]
            if record['timing_summary'][case['case_id']][key]!=dict(values=values,minimum=min(values),median=sorted(values)[1],maximum=max(values)):
                raise RuntimeError('Timing summary disagrees with repetitions.')
    ratio=record['timing_summary']['Jacobi']['trajectory_wall_seconds']['median']/record['timing_summary']['ASM']['trajectory_wall_seconds']['median']
    if ratio!=record['jacobi_over_asm_median_core_ratio']:raise RuntimeError('Cost-ratio arithmetic failed.')
    return dict(status='passed',trajectories=10,steps=100,performance_measurement=True,
        interpretation='Matched manual Radau Jacobi/ASM warm core comparison on one fixed problem; mentor review pending')


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
    stem = ROOT/'results/raw'/f'asm-cost-{stamp}-{os.getpid()}'
    transcript = stem.with_suffix('.txt')
    manifest_path = Path(str(stem)+'-manifest.json')
    steps = commands(venv,stem)
    sources = (SCRIPT,ROOT/'src/phase3_asm_trajectory.py',ROOT/'environment/check_asm_trajectory.py',ROOT/'src/phase3_asm_construction.py',ROOT/'environment/check_asm_construction.py',ROOT/'src/phase3_accuracy_cost.py',ROOT/'environment/check_accuracy_cost.py',ROOT/'src/phase3_cost_pilot.py',ROOT/'environment/check_cost_pilot.py',ROOT/'src/phase2_temporal.py',ROOT/'environment/check_temporal.py',ROOT/'src/phase0_be_smoke.py',ROOT/'src/phase0_irksome_be.py',ROOT/'src/phase0_radau_smoke.py',
       ROOT/'src/phase1_interpolation.py',ROOT/'src/phase1_spatial.py',Path(__file__).resolve(),
       ROOT/'environment/check_spatial.py',ROOT/'environment/check_interpolation.py',
       ROOT/'environment/check_radau_smoke.py',ROOT/'environment/check_irksome_be.py',ROOT/'environment/check_be_smoke.py',
       ROOT/'environment/check_python_runtime.py',ROOT/'environment/configure_petsc.py')
    state = {'started_utc':datetime.now(timezone.utc).isoformat(),'status':'started',
        'measurement_kind':'Matched manual Radau Jacobi/ASM repeated warm core timing on one fixed problem',
        'asm_trajectory_review':str(review_path.relative_to(ROOT)),'asm_trajectory_review_sha256':digest(review_path),
        'verified_environment':str(SNAPSHOT.relative_to(ROOT)),'verified_environment_sha256':digest(SNAPSHOT),
        'config':str(CONFIG.relative_to(ROOT)),'config_sha256':digest(CONFIG),
        'script_sha256':{str(path.relative_to(ROOT)):digest(path) for path in sources},
        'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'working_tree_status_before_run':subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
        'selected_tools':selected_tools,
        'selected_environment':{key:env[key] for key in ('PETSC_DIR','PETSC_ARCH','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS',
            'FIREDRAKE_CACHE_DIR','PYOP2_CACHE_DIR','FIREDRAKE_TSFC_KERNEL_CACHE_DIR','XDG_CACHE_HOME','TMPDIR',
            'PYTHONNOUSERSITE','PYTHONDONTWRITEBYTECODE')},
        'inherited_options_or_library_settings':{key:env[key] for key in
            ('PETSC_OPTIONS','DYLD_LIBRARY_PATH','DYLD_FALLBACK_LIBRARY_PATH','LD_LIBRARY_PATH') if key in env},
        'commands':{name:argv for name,argv in steps},'completed_steps':[],
        'power_state':power_state,'asm_cost_review_pending':True,'performance_measurement':True,'accuracy_cost_ranking_established':False}
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
            state['status'] = 'Matched ASM/Jacobi timing checks passed; mentor review pending'
            status = 0
            message = 'ASM COST CHECKPOINT PASSED. Review matched-path timings before further parameter changes.'
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
        stem = ROOT/'results/raw/asm-cost-<UTC timestamp>-<process id>'
        for name,argv in commands(parameters[4],stem):
            print(f'{name}: {shlex.join(argv)}')
        print('Display only: no numerical imports, assembly, solves, time stepping, or MPI launch run.')
        return 0
    return run(*parameters)


if __name__=='__main__':
    try:
        sys.exit(main())
    except (RuntimeError,OSError,subprocess.SubprocessError,KeyError,ValueError) as error:
        print(f'STOP before matched ASM cost comparison: {error}',file=sys.stderr)
        sys.exit(1)

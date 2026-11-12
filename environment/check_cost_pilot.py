#!/usr/bin/env python3
"""Researcher-run bounded timing/equivalence pilot against accepted temporal evidence."""
import argparse
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

from check_temporal import plan as previous_environment_plan
from configure_petsc import ROOT,digest,run_logged,validate_tools

CONFIG = ROOT/'configs/phase3_cost_pilot.json'
SCRIPT = ROOT/'src/phase3_cost_pilot.py'
SNAPSHOT = ROOT/'environment/verified_environment.json'


def plan():
    _,_,installation,petsc,venv,env = previous_environment_plan()
    review_path = ROOT/'results/processed/temporal_review.json'
    review = json.loads(review_path.read_text())
    pairs = [(review[key],review[key+'_sha256']) for key in
             ('temporal_manifest','temporal_transcript','config','verified_environment','spatial_review')]
    pairs.extend((record['path'],record['sha256']) for record in review['measurement_records'].values())
    pairs.extend(review['run_script_sha256'].items())
    for path,expected in pairs:
        if digest(ROOT/path)!=expected:
            raise RuntimeError(f'Accepted temporal evidence/source changed: {path}')
    if review['checkpoint_exit_status']!=0 or review['status']!='accepted controlled same-mesh temporal verification' or not review['temporal_order_verified']:
        raise RuntimeError('Accepted controlled temporal verification is required.')
    return review_path,review,installation,petsc,venv,env


def commands(venv,stem):
    return [('serial',[str(venv/'bin/python'),str(SCRIPT),'--config',str(CONFIG),
                       '--output',str(stem)+'-serial.json'])]


def verify_complete(record,config):
    if (record['status']!='passed' or not record['completed_study'] or record['config']!=config
            or record['ranks']!=1 or not record['performance_measurement']):
        raise RuntimeError('The timing/equivalence pilot did not pass.')
    methods = config['time_methods']
    expected = {method+'_verified' for method in methods}
    expected |= {method+'_warmup0' for method in methods}
    expected |= {f'{method}_measured{rep}' for method in methods for rep in (1,2,3)}
    if set(record['trajectories'])!=expected:
        raise RuntimeError('Incomplete fixed pilot schedule.')
    order = [method+'_warmup0' for method in methods]
    order += [f'{method}_measured{rep}' for rep,pair in enumerate(config['measured_order'],1) for method in pair]
    if record['execution_order']!=order or not all(record['initial_baseline_comparison'].values()):
        raise RuntimeError('Execution order or initialization changed.')
    steps = 0
    for label,row in record['trajectories'].items():
        if (row['status']!='passed' or row['dt']!=0.05 or row['completed_steps']!=20
                or len(row['step_records'])!=20 or not all(row['checks'].values())
                or not math.isclose(row['final_time'],1.0,rel_tol=0,abs_tol=1e-14)):
            raise RuntimeError('Incomplete or failed pilot trajectory.')
        if not all(step['converged_reason']>0 for step in row['step_records']):
            raise RuntimeError('A pilot solve did not converge.')
        if row['role']=='verified_baseline':
            if not all(row['accepted_temporal_metric_comparison'].values()):
                raise RuntimeError('Verified replay changed.')
        else:
            timing = row['timing']
            if not all(math.isfinite(v) and v>=0 for key,v in timing.items() if key.endswith('_seconds')):
                raise RuntimeError('Invalid timing.')
            partition = (timing['construction_seconds']+timing['first_step_seconds']
                +timing['remaining_steps_seconds']+timing['loop_bookkeeping_seconds'])
            if not math.isclose(partition,timing['trajectory_wall_seconds'],rel_tol=1e-12,abs_tol=1e-12):
                raise RuntimeError('Wall-clock partition disagrees.')
            events = timing['petsc_event_deltas']
            if events['KSPSolve']['count']!=20 or events['PCSetUp']['count']<1 or events['PCApply']['count']<=0:
                raise RuntimeError('Missing PETSc work evidence.')
            if row['final_equation_check']['status']!='passed' or not all(
                v['actual_field_difference']<=v['allowed_difference'] for v in row['equivalence'].values()):
                raise RuntimeError('Timed loop residual/equivalence failed.')
        steps += row['completed_steps']
    if steps!=200 or record['completed_steps']!=200 or record['completed_trajectories']!=10:
        raise RuntimeError('Expected 10 trajectories and 200 steps.')
    for method in methods:
        summary = record['timing_summary'][method]
        if summary['repetitions']!=3:
            raise RuntimeError('Missing measured repetitions.')
        for key in ('construction_seconds','first_step_seconds','remaining_steps_seconds','trajectory_wall_seconds'):
            values = [record['trajectories'][f'{method}_measured{rep}']['timing'][key] for rep in (1,2,3)]
            if (summary[key]['values']!=values or summary[key]['minimum']!=min(values)
                    or summary[key]['median']!=sorted(values)[1] or summary[key]['maximum']!=max(values)):
                raise RuntimeError('Timing summary disagrees with individual measurements.')
    return {'status':'passed','trajectories':10,'validated_steps':steps,'measured_repetitions_per_method':3,
        'interpretation':'Timing instrumentation and actual-field equivalence pilot; mentor review pending',
        'performance_measurement':True,'accuracy_cost_ranking_established':False}


def run(review_path,review,installation,petsc,venv,env):
    selected_tools = validate_tools(petsc)
    snapshot = json.loads(SNAPSHOT.read_text())
    inventory = subprocess.check_output([str(venv/'bin/python'),'-m','pip','--no-cache-dir','freeze','--all'],env=env,text=True)
    if inventory!=(ROOT/snapshot['resolved_python_inventory']).read_text():
        raise RuntimeError('Installed inventory differs from the tested snapshot.')
    for key in ('FIREDRAKE_CACHE_DIR','PYOP2_CACHE_DIR','FIREDRAKE_TSFC_KERNEL_CACHE_DIR','TMPDIR'):
        Path(env[key]).mkdir(parents=True,exist_ok=True)
    config = json.loads(CONFIG.read_text())
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    stem = ROOT/'results/raw'/f'cost-pilot-{stamp}-{os.getpid()}'
    transcript = stem.with_suffix('.txt')
    manifest_path = Path(str(stem)+'-manifest.json')
    steps = commands(venv,stem)
    sources = (SCRIPT,ROOT/'src/phase2_temporal.py',ROOT/'environment/check_temporal.py',ROOT/'src/phase0_be_smoke.py',ROOT/'src/phase0_irksome_be.py',ROOT/'src/phase0_radau_smoke.py',
       ROOT/'src/phase1_interpolation.py',ROOT/'src/phase1_spatial.py',Path(__file__).resolve(),
       ROOT/'environment/check_spatial.py',ROOT/'environment/check_interpolation.py',
       ROOT/'environment/check_radau_smoke.py',ROOT/'environment/check_irksome_be.py',ROOT/'environment/check_be_smoke.py',
       ROOT/'environment/check_python_runtime.py',ROOT/'environment/configure_petsc.py')
    state = {'started_utc':datetime.now(timezone.utc).isoformat(),'status':'started',
        'measurement_kind':'Timing-instrumentation/equivalence pilot; one fixed serial BE/Radau case',
        'temporal_review':str(review_path.relative_to(ROOT)),'temporal_review_sha256':digest(review_path),
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
        'power_state':None,
        'timing_pilot_review_pending':True,'performance_measurement':True,'accuracy_cost_ranking_established':False}
    try:
        power = subprocess.run(['/usr/bin/pmset','-g','batt'],capture_output=True,text=True,timeout=5,check=False)
        state['power_state'] = {'command':['/usr/bin/pmset','-g','batt'],'exit_status':power.returncode,'output':power.stdout.strip()}
    except (OSError,subprocess.SubprocessError) as error:
        state['power_state'] = {'unavailable':str(error)}
    status = 1
    manifest_path.write_text(json.dumps(state,indent=2)+'\n')
    with transcript.open('w') as log:
        try:
            for name,argv in steps:
                run_logged(argv,ROOT,env,log)
                state['completed_steps'].append(name)
            state['study_validation'] = verify_complete(json.loads(Path(str(stem)+'-serial.json').read_text()),config)
            state['status'] = 'Timing/equivalence pilot passed; mentor review pending'
            status = 0
            message = 'COST PILOT CHECKPOINT PASSED. Review before accuracy-cost comparisons or preconditioner changes.'
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
        stem = ROOT/'results/raw/cost-pilot-<UTC timestamp>-<process id>'
        for name,argv in commands(parameters[4],stem):
            print(f'{name}: {shlex.join(argv)}')
        print('Display only: no numerical imports, assembly, solves, time stepping, or MPI launch run.')
        return 0
    return run(*parameters)


if __name__=='__main__':
    try:
        sys.exit(main())
    except (RuntimeError,OSError,subprocess.SubprocessError,KeyError,ValueError) as error:
        print(f'STOP before timing/equivalence pilot: {error}',file=sys.stderr)
        sys.exit(1)

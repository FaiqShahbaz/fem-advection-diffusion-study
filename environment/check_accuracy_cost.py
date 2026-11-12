#!/usr/bin/env python3
"""Researcher-run targeted temporal accuracy/cost comparison after the accepted timing pilot."""
import argparse
from datetime import datetime,timezone
import json
import math
import os
import re
from pathlib import Path
import shlex
import subprocess
import sys

from check_cost_pilot import plan as previous_environment_plan
from configure_petsc import ROOT,digest,run_logged,validate_tools

CONFIG = ROOT/'configs/phase3_accuracy_cost.json'
SCRIPT = ROOT/'src/phase3_accuracy_cost.py'
SNAPSHOT = ROOT/'environment/verified_environment.json'


def plan():
    _,_,installation,petsc,venv,env = previous_environment_plan()
    review_path = ROOT/'results/processed/cost_pilot_review.json'
    review = json.loads(review_path.read_text())
    pairs = [(review[key],review[key+'_sha256']) for key in
             ('pilot_manifest','pilot_transcript','config','verified_environment','temporal_review')]
    pairs.extend((record['path'],record['sha256']) for record in review['measurement_records'].values())
    pairs.extend(review['run_script_sha256'].items())
    pairs.append((review['audit_source']['path'],review['audit_source']['sha256']))
    for path,expected in pairs:
        if digest(ROOT/path)!=expected:
            raise RuntimeError(f'Accepted pilot evidence/source changed: {path}')
    if review['checkpoint_exit_status']!=0 or review['status']!='accepted timing instrumentation and solution equivalence pilot' or not review['timing_instrumentation_verified'] or not review['solution_equivalence_verified']:
        raise RuntimeError('Accepted timing/equivalence pilot is required.')
    return review_path,review,installation,petsc,venv,env


def commands(venv,stem):
    return [('serial',[str(venv/'bin/python'),str(SCRIPT),'--config',str(CONFIG),
                       '--output',str(stem)+'-serial.json'])]


def extract_power_source(output):
    match = re.search(r"Now drawing from '([^']+)'",output)
    return match.group(1) if match else None


def read_power_state():
    command = ['/usr/bin/pmset','-g','batt']
    power = subprocess.run(command,capture_output=True,text=True,timeout=5,check=False)
    return {'command':command,'exit_status':power.returncode,'source':extract_power_source(power.stdout),
            'header':power.stdout.splitlines()[0] if power.stdout.splitlines() else ''}


def verify_complete(record,config):
    if (record['status']!='passed' or not record['completed_study'] or record['config']!=config
            or record['ranks']!=1 or not record['performance_measurement']):
        raise RuntimeError('The controlled accuracy/cost comparison did not pass.')
    expected = {'reference_coarse':('RadauIIA',config['reference']['coarse_dt']),
                'reference_fine':('RadauIIA',config['reference']['fine_dt']),
                'reference_fine_tighter':('RadauIIA',config['reference']['fine_dt'])}
    for case in config['cases']:
        for suffix in ('_verified','_warmup0','_measured1','_measured2','_measured3'):
            expected[case['case_id']+suffix] = (case['method'],case['dt'])
    for method in config['time_methods']:
        expected[method+'_finest_tighter'] = (method,min(c['dt'] for c in config['cases'] if c['method']==method))
    if set(record['trajectories'])!=set(expected) or len(record['curve'])!=4 or len(record['rates'])!=4:
        raise RuntimeError('Missing declared candidates, references or controls.')
    if [{key:row[key] for key in ('case_id','method','dt')} for row in record['curve']]!=config['cases']:
        raise RuntimeError('Unexpected candidate cut.')
    if not all(record[key]['status']=='passed' for key in ('reference_gap','reference_solver_difference','reference_resolution')):
        raise RuntimeError('Reference controls failed.')
    if (set(record['curve_solver_controls'])!=set(config['time_methods'])
            or not all(row['status']=='passed' for row in record['curve_solver_controls'].values())
            or not all(row['status']=='passed' for row in record['curve']+record['rates'])
            or not all(record['initial_baseline_comparison'].values())):
        raise RuntimeError('Time-error, solver or initialization controls failed.')
    order = [c['case_id']+'_warmup0' for c in config['cases']]
    order += [f'{case_id}_measured{rep}' for rep,ids in enumerate(config['measured_case_order'],1) for case_id in ids]
    if record['execution_order']!=order:
        raise RuntimeError('Measured/warmup order changed.')
    steps = 0
    for label,row in record['trajectories'].items():
        method,dt = expected[label];count = round(config['final_time']/dt)
        if (row['status']!='passed' or row['method']!=method or row['dt']!=dt
                or row['completed_steps']!=count or len(row['step_records'])!=count
                or not all(row['checks'].values())
                or not math.isclose(row['final_time'],1.0,rel_tol=0,abs_tol=1e-14)
                or not all(step['converged_reason']>0 for step in row['step_records'])):
            raise RuntimeError('Incomplete or failed trajectory.')
        if row['role'] in ('measured','warmup'):
            timing = row['timing'];events = timing['petsc_event_deltas']
            if not all(math.isfinite(v) and v>=0 for key,v in timing.items() if key.endswith('_seconds')):
                raise RuntimeError('Invalid timing.')
            partition = timing['construction_seconds']+timing['first_step_seconds']+timing['remaining_steps_seconds']+timing['loop_bookkeeping_seconds']
            if (not math.isclose(partition,timing['trajectory_wall_seconds'],rel_tol=1e-12,abs_tol=1e-12)
                    or events['KSPSolve']['count']!=count or events['PCSetUp']['count']<1 or events['PCApply']['count']<=0):
                raise RuntimeError('Timing partition or profiling evidence failed.')
            if row['final_equation_check']['status']!='passed' or row['time_error_transfer']['status']!='passed':
                raise RuntimeError('Timed final equation or error transfer failed.')
            for norm in row['time_error_transfer']['norms'].values():
                error,difference = norm['verified_baseline_time_error'],norm['measured_timed_baseline_distance']
                if (norm['lower']!=max(0,error-difference) or norm['upper']!=error+difference
                        or difference>norm['allowed_transfer_distance']):
                    raise RuntimeError('Error-transfer interval disagrees.')
        steps += count
    if steps!=15770 or record['completed_steps']!=15770 or record['completed_trajectories']!=25:
        raise RuntimeError('Expected 25 trajectories and 15770 steps.')
    for curve in record['curve']:
        rows = [record['trajectories'][f"{curve['case_id']}_measured{i}"] for i in (1,2,3)]
        for key in ('construction_seconds','first_step_seconds','remaining_steps_seconds','trajectory_wall_seconds'):
            values = [r['timing'][key] for r in rows];summary = curve['timing_summary'][key]
            if summary!=dict(values=values,minimum=min(values),median=sorted(values)[1],maximum=max(values)):
                raise RuntimeError('Timing summary disagrees with repetitions.')
        for metric in ('l2_error','h1_seminorm_error'):
            bounds = curve['timed_error_envelope'][metric]
            if bounds!=dict(lower=min(r['time_error_transfer']['norms'][metric]['lower'] for r in rows),
                            upper=max(r['time_error_transfer']['norms'][metric]['upper'] for r in rows)):
                raise RuntimeError('Timed error envelope disagrees.')
    for selection in record['candidate_selections']:
        candidates = [row for row in record['curve'] if row['method']==selection['method']
            and row['timed_error_envelope']['l2_error']['upper']<=selection['target_l2_time_error']]
        best = min(candidates,key=lambda row:row['timing_summary']['trajectory_wall_seconds']['median']) if candidates else None
        if (selection['eligible_cases']!=[row['case_id'] for row in candidates]
                or selection['selected_case']!=(best['case_id'] if best else None)):
            raise RuntimeError('Target eligibility or candidate selection disagrees.')
    if {(row['method'],row['target_l2_time_error']) for row in record['candidate_selections']}!={(method,target) for method in config['time_methods'] for target in config['target_time_errors_l2']} or len(record['candidate_selections'])!=4:
        raise RuntimeError('Missing declared method/target selections.')
    return {'status':'passed','candidates':4,'trajectories':25,'validated_steps':steps,
            'measured_repetitions_per_candidate':3,'performance_measurement':True,
            'interpretation':'Controlled fixed-mesh time-error/cost candidates; mentor review pending'}


def run(review_path,review,installation,petsc,venv,env):
    selected_tools = validate_tools(petsc)
    config = json.loads(CONFIG.read_text())
    power_state = read_power_state()
    if power_state['exit_status']!=0 or power_state['source']!=config['required_power_source']:
        raise RuntimeError('Connect the charger and confirm AC Power before this measurement run; current power source: '+str(power_state['source']))
    snapshot = json.loads(SNAPSHOT.read_text())
    inventory = subprocess.check_output([str(venv/'bin/python'),'-m','pip','--no-cache-dir','freeze','--all'],env=env,text=True)
    if inventory!=(ROOT/snapshot['resolved_python_inventory']).read_text():
        raise RuntimeError('Installed inventory differs from the tested snapshot.')
    for key in ('FIREDRAKE_CACHE_DIR','PYOP2_CACHE_DIR','FIREDRAKE_TSFC_KERNEL_CACHE_DIR','TMPDIR'):
        Path(env[key]).mkdir(parents=True,exist_ok=True)
    config = json.loads(CONFIG.read_text())
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    stem = ROOT/'results/raw'/f'accuracy-cost-{stamp}-{os.getpid()}'
    transcript = stem.with_suffix('.txt')
    manifest_path = Path(str(stem)+'-manifest.json')
    steps = commands(venv,stem)
    sources = (SCRIPT,ROOT/'src/phase3_cost_pilot.py',ROOT/'environment/check_cost_pilot.py',ROOT/'src/phase2_temporal.py',ROOT/'environment/check_temporal.py',ROOT/'src/phase0_be_smoke.py',ROOT/'src/phase0_irksome_be.py',ROOT/'src/phase0_radau_smoke.py',
       ROOT/'src/phase1_interpolation.py',ROOT/'src/phase1_spatial.py',Path(__file__).resolve(),
       ROOT/'environment/check_spatial.py',ROOT/'environment/check_interpolation.py',
       ROOT/'environment/check_radau_smoke.py',ROOT/'environment/check_irksome_be.py',ROOT/'environment/check_be_smoke.py',
       ROOT/'environment/check_python_runtime.py',ROOT/'environment/configure_petsc.py')
    state = {'started_utc':datetime.now(timezone.utc).isoformat(),'status':'started',
        'measurement_kind':'Four targeted fixed-mesh temporal-accuracy/core-cost candidates; Jacobi baseline',
        'cost_pilot_review':str(review_path.relative_to(ROOT)),'cost_pilot_review_sha256':digest(review_path),
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
        'power_state':power_state,
        'accuracy_cost_review_pending':True,'performance_measurement':True,'accuracy_cost_ranking_established':False}
    status = 1
    manifest_path.write_text(json.dumps(state,indent=2)+'\n')
    with transcript.open('w') as log:
        try:
            for name,argv in steps:
                run_logged(argv,ROOT,env,log)
                state['completed_steps'].append(name)
            state['power_state_after'] = read_power_state()
            if state['power_state_after']['source']!=config['required_power_source'] or state['power_state_after']['exit_status']!=0:
                raise RuntimeError('AC Power was not confirmed after the run; retain measurements for review without pooling them as AC results.')
            state['study_validation'] = verify_complete(json.loads(Path(str(stem)+'-serial.json').read_text()),config)
            state['status'] = 'Targeted temporal accuracy/cost checks passed; mentor review pending'
            status = 0
            message = 'ACCURACY-COST CHECKPOINT PASSED. Review the four candidates before preconditioner changes.'
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
        stem = ROOT/'results/raw/accuracy-cost-<UTC timestamp>-<process id>'
        for name,argv in commands(parameters[4],stem):
            print(f'{name}: {shlex.join(argv)}')
        print('Display only: no numerical imports, assembly, solves, time stepping, or MPI launch run.')
        return 0
    return run(*parameters)


if __name__=='__main__':
    try:
        sys.exit(main())
    except (RuntimeError,OSError,subprocess.SubprocessError,KeyError,ValueError) as error:
        print(f'STOP before targeted accuracy/cost comparison: {error}',file=sys.stderr)
        sys.exit(1)

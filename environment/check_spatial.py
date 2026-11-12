#!/usr/bin/env python3
"""Researcher-run controlled transient spatial verification against accepted evidence."""
import argparse
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

from check_interpolation import plan as previous_environment_plan
from configure_petsc import ROOT,digest,run_logged,validate_tools

CONFIG = ROOT/'configs/phase1_spatial.json'
SCRIPT = ROOT/'src/phase1_spatial.py'
SNAPSHOT = ROOT/'environment/verified_environment.json'


def plan():
    _,_,installation,petsc,venv,env = previous_environment_plan()
    review_path = ROOT/'results/processed/interpolation_review.json'
    review = json.loads(review_path.read_text())
    pairs = [(review[key],review[key+'_sha256']) for key in
             ('interpolation_manifest','interpolation_transcript','config','verified_environment','radau_review')]
    pairs.extend((record['path'],record['sha256']) for record in review['measurement_records'].values())
    pairs.extend(review['run_script_sha256'].items())
    for path,expected in pairs:
        if digest(ROOT/path)!=expected:
            raise RuntimeError(f'Accepted interpolation evidence/source changed: {path}')
    if review['checkpoint_exit_status']!=0 or review['status']!='accepted initial interpolation verification':
        raise RuntimeError('Accepted initial interpolation verification is required.')
    return review_path,review,installation,petsc,venv,env


def commands(venv,stem):
    return [('serial',[str(venv/'bin/python'),str(SCRIPT),'--config',str(CONFIG),
                       '--output',str(stem)+'-serial.json'])]


def verify_complete(record,config):
    if record['status']!='passed' or not record['completed_study'] or record['config']!=config or record['ranks']!=1:
        raise RuntimeError('The controlled serial spatial study did not pass.')
    if len(record['measurements'])!=6 or len(record['rates'])!=8:
        raise RuntimeError('Missing cases or diagnostic rates.')
    expected_cases = {f'P{d}-n{n}' for d in config['element_degrees'] for n in config['mesh_subdivisions']}
    if {row['case_id'] for row in record['measurements']}!=expected_cases:
        raise RuntimeError('Unexpected set of mesh/degree cases.')
    total_steps,trajectories = 0,0
    for row in record['measurements']:
        if row['status']!='passed' or row['time_control']['status']!='passed':
            raise RuntimeError('A spatial case or time control failed.')
        finest = row['element_degree']==2 and row['mesh_subdivisions']==32
        expected = {'primary','half_dt','tighter_solver'} if finest else {'primary','half_dt'}
        if set(row['trajectories'])!=expected:
            raise RuntimeError('Missing baseline/control trajectories.')
        if finest and row['solver_control']['status']!='passed':
            raise RuntimeError('The finest-P2 tolerance sensitivity check failed.')
        for label,trajectory in row['trajectories'].items():
            dt = config['dt'] if label=='primary' else config['control_dt']
            count = round(config['final_time']/dt)
            if trajectory['status']!='passed' or trajectory['completed_steps']!=count or len(trajectory['step_records'])!=count:
                raise RuntimeError('Incomplete or failed trajectory.')
            if not math.isclose(trajectory['final_time'],config['final_time'],rel_tol=0.0,abs_tol=1e-14):
                raise RuntimeError('Incorrect final time.')
            if not all(trajectory['checks'].values()):
                raise RuntimeError('A trajectory validation failed.')
            total_steps += count
            trajectories += 1
    if trajectories!=13 or total_steps!=4000 or not all(rate['status']=='passed' for rate in record['rates']):
        raise RuntimeError('Incomplete controls or failed finite-range rates.')
    return {'status':'passed','cases':6,'trajectories':trajectories,'validated_steps':total_steps,
            'interpretation':'Transient spatial trends over the recorded mesh range, conditional on sensitivity controls; mentor review pending',
            'temporal_order_verified':False,'performance_measurement':False}


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
    stem = ROOT/'results/raw'/f'spatial-{stamp}-{os.getpid()}'
    transcript = stem.with_suffix('.txt')
    manifest_path = Path(str(stem)+'-manifest.json')
    steps = commands(venv,stem)
    sources = (SCRIPT,ROOT/'src/phase0_be_smoke.py',ROOT/'src/phase0_irksome_be.py',ROOT/'src/phase0_radau_smoke.py',
       ROOT/'src/phase1_interpolation.py',Path(__file__).resolve(),ROOT/'environment/check_interpolation.py',
       ROOT/'environment/check_radau_smoke.py',ROOT/'environment/check_irksome_be.py',ROOT/'environment/check_be_smoke.py',
       ROOT/'environment/check_python_runtime.py',ROOT/'environment/configure_petsc.py')
    state = {'started_utc':datetime.now(timezone.utc).isoformat(),'status':'started',
        'measurement_kind':'Controlled transient P1/P2 spatial verification at T=1',
        'interpolation_review':str(review_path.relative_to(ROOT)),'interpolation_review_sha256':digest(review_path),
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
        'temporal_order_verified':False,'performance_measurement':False}
    status = 1
    manifest_path.write_text(json.dumps(state,indent=2)+'\n')
    with transcript.open('w') as log:
        try:
            for name,argv in steps:
                run_logged(argv,ROOT,env,log)
                state['completed_steps'].append(name)
            state['study_validation'] = verify_complete(json.loads(Path(str(stem)+'-serial.json').read_text()),config)
            state['status'] = 'Controlled transient spatial checks passed; mentor review pending'
            status = 0
            message = 'TRANSIENT SPATIAL CHECKPOINT PASSED. Review before temporal-order or cost experiments.'
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
        stem = ROOT/'results/raw/spatial-<UTC timestamp>-<process id>'
        for name,argv in commands(parameters[4],stem):
            print(f'{name}: {shlex.join(argv)}')
        print('Display only: no numerical imports, assembly, solves, time stepping, or MPI launch run.')
        return 0
    return run(*parameters)


if __name__=='__main__':
    try:
        sys.exit(main())
    except (RuntimeError,OSError,subprocess.SubprocessError,KeyError,ValueError) as error:
        print(f'STOP before transient spatial verification: {error}',file=sys.stderr)
        sys.exit(1)

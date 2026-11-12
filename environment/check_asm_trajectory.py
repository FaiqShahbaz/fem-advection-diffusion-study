#!/usr/bin/env python3
"""Researcher-run full-trajectory ASM/local-ILU verification after accepted construction."""
import argparse
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

from check_asm_construction import plan as previous_environment_plan
from configure_petsc import ROOT,digest,run_logged,validate_tools

CONFIG = ROOT/'configs/phase3_asm_trajectory.json'
SCRIPT = ROOT/'src/phase3_asm_trajectory.py'
SNAPSHOT = ROOT/'environment/verified_environment.json'


def plan():
    _,_,installation,petsc,venv,env = previous_environment_plan()
    review_path=ROOT/'results/processed/asm_construction_review.json'
    review=json.loads(review_path.read_text())
    pairs=[(review[key],review[key+'_sha256']) for key in
           ('asm_manifest','asm_transcript','config','accuracy_cost_review','verified_environment')]
    pairs.extend((r['path'],r['sha256']) for r in review['measurement_records'].values())
    pairs.extend(review['run_script_sha256'].items())
    pairs.append((review['audit_source']['path'],review['audit_source']['sha256']))
    for path,expected in pairs:
        if digest(ROOT/path)!=expected:raise RuntimeError(f'Accepted construction evidence/source changed: {path}')
    if review['status']!='accepted standard ASM local ILU one-step construction' or not review['asm_construction_verified']:
        raise RuntimeError('Accepted standard ASM construction is required.')
    return review_path,review,installation,petsc,venv,env


def commands(venv,stem):
    return [('serial',[str(venv/'bin/python'),str(SCRIPT),'--config',str(CONFIG),
                       '--output',str(stem)+'-serial.json'])]


def verify_complete(record,config):
    if (record['status']!='passed' or not record['completed_study'] or record['config']!=config
            or record['ranks']!=1 or record['performance_measurement'] is not False
            or record['completed_trajectories']!=4 or record['completed_steps']!=40
            or record['total_dofs']!=1089 or record['stage_total_dofs']!=2178):
        raise RuntimeError('The full-trajectory verification did not pass.')
    if set(record['trajectories'])!=set(config['trajectory_roles']) or not all(record['baseline_reproduction'].values()):
        raise RuntimeError('Missing trajectory or accepted-baseline reproduction.')
    def field_check(item):
        if item['status']!='passed' or item['difference_measurement']['status']!='passed':
            raise RuntimeError('Field/quadrature comparison failed.')
        for row in item['norms'].values():
            value=row['actual_field_difference'];allowed=config['checks']['pc_field_comparison_atol']+config['checks']['pc_field_comparison_rtol']*row['reference_field_norm']
            if not math.isfinite(value) or value<0 or allowed!=row['allowed_difference'] or value>allowed:
                raise RuntimeError('Field comparison arithmetic failed.')
    expected_comparisons={label+'_vs_irksome' for label in config['trajectory_roles'] if label!='irksome_jacobi'}|{'asm_vs_manual_jacobi'}
    if set(record['comparisons'])!=expected_comparisons:raise RuntimeError('Missing final field comparisons.')
    for item in record['comparisons'].values():field_check(item)
    for label,row in record['trajectories'].items():
        if (row['status']!='passed' or row['completed_steps']!=10 or len(row['step_records'])!=10
                or not all(row['checks'].values()) or row['final_time']!=1.0
                or row['total_iterations']!=sum(s['iterations'] for s in row['step_records'])):
            raise RuntimeError('Incomplete trajectory.')
        params=row['solver_parameters']
        for index,step in enumerate(row['step_records']):
            r=step['true_block_residual'];target=max(params['ksp_atol'],params['ksp_rtol']*r['rhs_norm'])
            if (step['step']!=index+1 or abs(step['t_start']-index*.1)>1e-14
                    or abs(step['t_end']-(index+1)*.1)>1e-14 or step['converged_reason']<=0
                    or not all(math.isfinite(v) and v>=0 for v in r.values()) or target!=step['true_residual_target']
                    or r['norm']>10*target or step['update_identity_l2']>1e-12 or step['stiff_accuracy_l2']>1e-12):
                raise RuntimeError('Clock, residual or identity check failed.')
            if len(step['stage_times'])!=2 or any(abs(a-b)>1e-14 for a,b in zip(step['stage_times'],[(index+1/3)*.1,(index+1)*.1])):
                raise RuntimeError('Stage source times failed.')
            if label!='irksome_jacobi':
                if (index>0 and step['event_counts']['PCSetUp']!=0) or any(v>1e-12 for v in step['stage_boundary_trace_l2']):
                    raise RuntimeError('Reuse or stage boundary check failed.')
                if not all(s['source_residual_l2']<=1e-10 for s in step['source_checks']):
                    raise RuntimeError('Manufactured source check failed.')
                if label!='manual_jacobi_reference':
                    if set(step['reference_comparisons'])!={'update','slope1','slope2'}:raise RuntimeError('Missing step field comparisons.')
                    for item in step['reference_comparisons'].values():field_check(item)
        if label!='irksome_jacobi' and (row['matrix_assemblies']!=1 or row['operator_event_counts']['PCSetUp']<1):
            raise RuntimeError('Matrix/PC setup evidence failed.')
        if label in ('manual_jacobi','manual_asm'):
            for r in row['accepted_time_error_budget_check'].values():
                E,D=r['accepted_baseline_time_error'],r['actual_field_distance']
                if not math.isfinite(D) or D<0 or D>r['allowed_distance'] or r['allowed_distance']!=1e-12+.01*E:
                    raise RuntimeError('Accepted time-error budget check failed.')
    asm=record['trajectories']['manual_asm']
    if len(asm['domains'])!=4 or sorted(i for d in asm['domains'] for i in d['core_indices'])!=list(range(2178)):
        raise RuntimeError('ASM core coverage failed.')
    if not all(d['one_layer_check'] for d in asm['domains']):raise RuntimeError('Graph overlap check failed.')
    if len(asm['local_work'])!=4 or len(asm['local_solvers'])!=4:raise RuntimeError('Missing local work evidence.')
    for i,w in enumerate(asm['local_work']):
        calls=sum(s['local_calls_per_domain'][i] for s in asm['step_records'])
        if w['outer_solve']!=dict(calls=calls,reported_iterations=calls) or calls<=0 or w['action_check']!=dict(calls=1,reported_iterations=1):
            raise RuntimeError('Local action accounting failed.')
    if not all(len(set(s['local_calls_per_domain']))==1 and min(s['local_calls_per_domain'])>0 for s in asm['step_records']):
        raise RuntimeError('Unequal local work.')
    if not all(r['ksp_type']=='preonly' and r['pc_type']=='ilu' and r['factor_solver_type']=='petsc' for r in asm['local_solvers']):
        raise RuntimeError('Local solver/backend changed.')
    action=asm['additive_action_check']
    if action['status']!='passed' or action['defect_norm']>action['allowed_defect']:
        raise RuntimeError('Final additive action check failed.')
    return dict(status='passed',trajectories=4,steps=40,performance_measurement=False,
                interpretation='Full-trajectory field agreement and constant-matrix reuse; mentor review pending')


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
    stem = ROOT/'results/raw'/f'asm-trajectory-{stamp}-{os.getpid()}'
    transcript = stem.with_suffix('.txt')
    manifest_path = Path(str(stem)+'-manifest.json')
    steps = commands(venv,stem)
    sources = (SCRIPT,ROOT/'src/phase3_asm_construction.py',ROOT/'environment/check_asm_construction.py',ROOT/'src/phase3_accuracy_cost.py',ROOT/'environment/check_accuracy_cost.py',ROOT/'src/phase3_cost_pilot.py',ROOT/'environment/check_cost_pilot.py',ROOT/'src/phase2_temporal.py',ROOT/'environment/check_temporal.py',ROOT/'src/phase0_be_smoke.py',ROOT/'src/phase0_irksome_be.py',ROOT/'src/phase0_radau_smoke.py',
       ROOT/'src/phase1_interpolation.py',ROOT/'src/phase1_spatial.py',Path(__file__).resolve(),
       ROOT/'environment/check_spatial.py',ROOT/'environment/check_interpolation.py',
       ROOT/'environment/check_radau_smoke.py',ROOT/'environment/check_irksome_be.py',ROOT/'environment/check_be_smoke.py',
       ROOT/'environment/check_python_runtime.py',ROOT/'environment/configure_petsc.py')
    state = {'started_utc':datetime.now(timezone.utc).isoformat(),'status':'started',
        'measurement_kind':'Full T=1 standard ASM/local ILU trajectory and reuse verification; no timing ranking',
        'asm_construction_review':str(review_path.relative_to(ROOT)),'asm_construction_review_sha256':digest(review_path),
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
        'asm_trajectory_review_pending':True,'performance_measurement':False,'accuracy_cost_ranking_established':False}
    status = 1
    manifest_path.write_text(json.dumps(state,indent=2)+'\n')
    with transcript.open('w') as log:
        try:
            for name,argv in steps:
                run_logged(argv,ROOT,env,log)
                state['completed_steps'].append(name)
            state['study_validation'] = verify_complete(json.loads(Path(str(stem)+'-serial.json').read_text()),config)
            state['status'] = 'Full ASM trajectory checks passed; mentor review pending'
            status = 0
            message = 'ASM TRAJECTORY CHECKPOINT PASSED. Review before repeated timing experiments.'
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
        stem = ROOT/'results/raw/asm-trajectory-<UTC timestamp>-<process id>'
        for name,argv in commands(parameters[4],stem):
            print(f'{name}: {shlex.join(argv)}')
        print('Display only: no numerical imports, assembly, solves, time stepping, or MPI launch run.')
        return 0
    return run(*parameters)


if __name__=='__main__':
    try:
        sys.exit(main())
    except (RuntimeError,OSError,subprocess.SubprocessError,KeyError,ValueError) as error:
        print(f'STOP before ASM trajectory verification: {error}',file=sys.stderr)
        sys.exit(1)

#!/usr/bin/env python3
"""Researcher-run one-step installed RanaLD construction after accepted matched PC timing."""
import argparse
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

from check_asm_cost import plan as previous_environment_plan
from configure_petsc import ROOT,digest,run_logged,validate_tools

CONFIG = ROOT/'configs/phase3_rana_construction.json'
SCRIPT = ROOT/'src/phase3_rana_construction.py'
SNAPSHOT = ROOT/'environment/verified_environment.json'


def plan():
    _,_,installation,petsc,venv,env=previous_environment_plan()
    review_path=ROOT/'results/processed/asm_cost_review.json';review=json.loads(review_path.read_text())
    pairs=[(review[key],review[key+'_sha256']) for key in
        ('cost_manifest','cost_transcript','config','asm_trajectory_review','verified_environment')]
    pairs.extend((r['path'],r['sha256']) for r in review['measurement_records'].values())
    pairs.extend(review['run_script_sha256'].items());pairs.append((review['audit_source']['path'],review['audit_source']['sha256']))
    for path,expected in pairs:
        if digest(ROOT/path)!=expected:raise RuntimeError(f'Accepted matched timing evidence/source changed: {path}')
    if review['status']!='accepted matched manual Radau Jacobi ASM timing' or not review['asm_cost_verified']:
        raise RuntimeError('Accepted matched PC timing review is required.')
    inspection=json.loads((ROOT/'environment/rana_source_inspection.json').read_text())
    for path,expected in inspection['installed_source_sha256'].items():
        if digest(ROOT/path)!=expected:raise RuntimeError(f'Inspected installed Rana/auxiliary-PC source changed: {path}')
    return review_path,review,installation,petsc,venv,env


def commands(venv,stem):
    return [('serial',[str(venv/'bin/python'),str(SCRIPT),'--config',str(CONFIG),
                       '--output',str(stem)+'-serial.json'])]


def verify_complete(record,config):
    if (record['status']!='passed' or not record['completed_study'] or record['config']!=config
            or record['ranks']!=1 or record['performance_measurement'] is not False
            or record['completed_operator_solves']!=3 or record['total_dofs']!=1089
            or record['stage_total_dofs']!=2178 or record['checkpoint_end_time']!=.1
            or set(record['solves'])!={'irksome_jacobi','manual_jacobi_reference','rana_ld'}):
        raise RuntimeError('Rana construction did not complete.')
    baseline=record['solves']['irksome_jacobi'];reference=record['solves']['manual_jacobi_reference'];rana=record['solves']['rana_ld']
    if baseline['status']!='passed' or baseline['completed_steps']!=1 or not all(baseline['checks'].values()):
        raise RuntimeError('Accepted first-step baseline failed.')
    for row in (reference,rana):
        if row['status']!='passed' or row['converged_reason']<=0 or row['stage_total_dofs']!=2178:
            raise RuntimeError('Independent physical stage solve failed.')
        r=row['true_block_residual'];params=row['solver_parameters'];target=max(params['ksp_atol'],params['ksp_rtol']*r['rhs_norm'])
        if (not all(math.isfinite(v) and v>=0 for v in r.values()) or target!=row['true_residual_target'] or r['norm']>10*target
                or any(v>1e-12 for v in row['stage_boundary_trace_l2'])):
            raise RuntimeError('Physical residual/boundary check failed.')
    if rana['weak_form_argument_count']!=1 or rana['weak_form_representation']!='state_function_residual':
        raise RuntimeError('Installed RanaLD requires the checked one-argument residual form.')
    actual=rana['actual_solver'];aux=rana['auxiliary_solver']
    if (rana['solver_parameters']!=config['rana_solver_parameters'] or actual['ksp_type']!='gmres' or actual['pc_type']!='python'
            or actual['python_context']!='irksome.pc.RanaLD' or actual['side']!='right' or actual['norm']!='unpreconditioned'
            or actual['rtol']!=1e-11 or actual['atol']!=1e-13 or actual['max_it']!=1000
            or aux['pc_type']!='lu' or aux['factor_solver_type']!='petsc' or aux['inner_krylov_solve'] is not False):
        raise RuntimeError('Actual Rana/auxiliary solver differs.')
    if rana['completed_steps']!=1 or rana['final_time']!=.1 or not all(rana['checks'].values()) or rana['solver_stats']!=[1,1,rana['iterations']]:
        raise RuntimeError('Rana state/update completion failed.')
    if rana['update_identity_l2']>1e-12 or rana['stiff_accuracy_l2']>1e-12 or not all(s['source_residual_l2']<=1e-10 for s in rana['source_checks']):
        raise RuntimeError('Rana update/source check failed.')
    if set(rana['matrix_checks'])!={'physical_B','auxiliary_P','auxiliary_PC_Pmat'}:raise RuntimeError('Missing matrix construction checks.')
    for row in rana['matrix_checks'].values():
        allowed=config['checks']['auxiliary_matrix_atol']+config['checks']['auxiliary_matrix_rtol']*row['reference_frobenius_norm']
        if row['status']!='passed' or not math.isfinite(row['frobenius_defect']) or row['allowed_defect']!=allowed or row['frobenius_defect']>allowed:
            raise RuntimeError('Independent CSR matrix comparison failed.')
    probe=rana['auxiliary_action_check'];allowed=config['checks']['auxiliary_action_atol']+config['checks']['auxiliary_action_rtol']*probe['rhs_norm']
    if probe['status']!='passed' or probe['allowed_residual']!=allowed or probe['residual_norm']>allowed or probe['diagnostic_outer_PC_applications']!=1:
        raise RuntimeError('Actual Rana auxiliary inverse action failed.')
    names={'reference_update_vs_irksome','rana_update_vs_reference','rana_slope1_vs_reference','rana_slope2_vs_reference'}
    if set(record['comparisons'])!=names:raise RuntimeError('Missing actual field comparisons.')
    for item in record['comparisons'].values():
        if item['status']!='passed' or item['difference_measurement']['status']!='passed':raise RuntimeError('Field/quadrature comparison failed.')
        for r in item['norms'].values():
            allowed=1e-11+1e-8*r['reference_field_norm']
            if r['allowed_difference']!=allowed or not math.isfinite(r['actual_field_difference']) or r['actual_field_difference']>allowed:
                raise RuntimeError('Field difference exceeded its fixed allowance.')
    return dict(status='passed',independent_physical_stage_solves=3,performance_measurement=False,
        interpretation='Installed RanaLD matrix/action construction and first-step field agreement; mentor review pending')


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
    stem = ROOT/'results/raw'/f'rana-construction-{stamp}-{os.getpid()}'
    transcript = stem.with_suffix('.txt')
    manifest_path = Path(str(stem)+'-manifest.json')
    steps = commands(venv,stem)
    sources = (SCRIPT,ROOT/'src/phase3_asm_construction.py',ROOT/'environment/check_asm_construction.py',ROOT/'src/phase3_asm_trajectory.py',ROOT/'environment/check_asm_trajectory.py',ROOT/'src/phase3_asm_cost.py',ROOT/'environment/check_asm_cost.py',ROOT/'src/phase3_accuracy_cost.py',ROOT/'environment/check_accuracy_cost.py',ROOT/'src/phase3_cost_pilot.py',ROOT/'environment/check_cost_pilot.py',ROOT/'src/phase2_temporal.py',ROOT/'environment/check_temporal.py',ROOT/'src/phase0_be_smoke.py',ROOT/'src/phase0_irksome_be.py',ROOT/'src/phase0_radau_smoke.py',
       ROOT/'src/phase1_interpolation.py',ROOT/'src/phase1_spatial.py',Path(__file__).resolve(),
       ROOT/'environment/check_spatial.py',ROOT/'environment/check_interpolation.py',
       ROOT/'environment/check_radau_smoke.py',ROOT/'environment/check_irksome_be.py',ROOT/'environment/check_be_smoke.py',
       ROOT/'environment/check_python_runtime.py',ROOT/'environment/configure_petsc.py')
    state = {'started_utc':datetime.now(timezone.utc).isoformat(),'status':'started',
        'measurement_kind':'One-step installed RanaLD/exact auxiliary LU construction; no timing ranking',
        'asm_cost_review':str(review_path.relative_to(ROOT)),'asm_cost_review_sha256':digest(review_path),
        'verified_environment':str(SNAPSHOT.relative_to(ROOT)),'verified_environment_sha256':digest(SNAPSHOT),
        'config':str(CONFIG.relative_to(ROOT)),'config_sha256':digest(CONFIG),
        'installed_source_inspection':'environment/rana_source_inspection.json',
        'installed_source_inspection_sha256':digest(ROOT/'environment/rana_source_inspection.json'),
        'inspected_installed_source_sha256':json.loads((ROOT/'environment/rana_source_inspection.json').read_text())['installed_source_sha256'],
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
        'rana_construction_review_pending':True,'performance_measurement':False,'accuracy_cost_ranking_established':False}
    status = 1
    manifest_path.write_text(json.dumps(state,indent=2)+'\n')
    with transcript.open('w') as log:
        try:
            for name,argv in steps:
                run_logged(argv,ROOT,env,log)
                state['completed_steps'].append(name)
            state['study_validation'] = verify_complete(json.loads(Path(str(stem)+'-serial.json').read_text()),config)
            state['status'] = 'RanaLD construction checks passed; mentor review pending'
            status = 0
            message = 'RANA CONSTRUCTION CHECKPOINT PASSED. Review before full trajectories or timing.'
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
        stem = ROOT/'results/raw/rana-construction-<UTC timestamp>-<process id>'
        for name,argv in commands(parameters[4],stem):
            print(f'{name}: {shlex.join(argv)}')
        print('Display only: no numerical imports, assembly, solves, time stepping, or MPI launch run.')
        return 0
    return run(*parameters)


if __name__=='__main__':
    try:
        sys.exit(main())
    except (RuntimeError,OSError,subprocess.SubprocessError,KeyError,ValueError) as error:
        print(f'STOP before Rana construction: {error}',file=sys.stderr)
        sys.exit(1)

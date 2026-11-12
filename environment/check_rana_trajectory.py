#!/usr/bin/env python3
"""Researcher-run full RanaLD trajectory verification after accepted first-step construction."""
import argparse
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

from check_rana_construction import plan as previous_environment_plan
from configure_petsc import ROOT,digest,run_logged,validate_tools

CONFIG = ROOT/'configs/phase3_rana_trajectory.json'
SCRIPT = ROOT/'src/phase3_rana_trajectory.py'
SNAPSHOT = ROOT/'environment/verified_environment.json'


def plan():
    _,_,installation,petsc,venv,env=previous_environment_plan()
    review_path=ROOT/'results/processed/rana_construction_review.json';review=json.loads(review_path.read_text())
    pairs=[(review[key],review[key+'_sha256']) for key in
        ('rana_manifest','rana_transcript','config','asm_cost_review','verified_environment','installed_source_inspection','preserved_prior_failure')]
    pairs.extend((r['path'],r['sha256']) for r in review['measurement_records'].values())
    pairs.extend(review['run_script_sha256'].items());pairs.append((review['audit_source']['path'],review['audit_source']['sha256']))
    for path,expected in pairs:
        if digest(ROOT/path)!=expected:raise RuntimeError(f'Accepted Rana construction evidence/source changed: {path}')
    if review['status']!='accepted installed RanaLD one-step construction' or not review['rana_construction_verified']:
        raise RuntimeError('Accepted installed RanaLD construction review is required.')
    inspection=json.loads((ROOT/'environment/rana_trajectory_source_inspection.json').read_text())
    for path,expected in inspection['installed_source_sha256'].items():
        if digest(ROOT/path)!=expected:raise RuntimeError(f'Inspected trajectory backend/interface source changed: {path}')
    return review_path,review,installation,petsc,venv,env


def commands(venv,stem):
    return [('serial',[str(venv/'bin/python'),str(SCRIPT),'--config',str(CONFIG),
                       '--output',str(stem)+'-serial.json'])]


def verify_complete(record,config):
    if (record['status']!='passed' or not record['completed_study'] or record['config']!=config
            or record['ranks']!=1 or record['performance_measurement'] is not False
            or record['completed_trajectories']!=4 or record['completed_steps']!=40
            or record['total_dofs']!=1089 or record['stage_total_dofs']!=2178
            or set(record['trajectories'])!=set(config['trajectory_roles'])
            or not all(record['baseline_reproduction'].values()) or not record['baseline_iteration_sequence_matches']):
        raise RuntimeError('Full Rana trajectory/baseline reproduction did not complete.')
    def field_check(item):
        if item['status']!='passed' or item['difference_measurement']['status']!='passed':raise RuntimeError('Field/quadrature comparison failed.')
        for row in item['norms'].values():
            value=row['actual_field_difference'];allowed=config['checks']['pc_field_comparison_atol']+config['checks']['pc_field_comparison_rtol']*row['reference_field_norm']
            if not math.isfinite(value) or value<0 or row['allowed_difference']!=allowed or value>allowed:raise RuntimeError('Actual field difference failed.')
    expected={label+'_vs_original_irksome' for label in ('manual_jacobi_reference','residual_jacobi','rana_ld')}|{'rana_vs_residual_jacobi'}
    if set(record['comparisons'])!=expected:raise RuntimeError('Missing final actual-field comparisons.')
    for item in record['comparisons'].values():field_check(item)
    for label,row in record['trajectories'].items():
        if (row['status']!='passed' or row['completed_steps']!=10 or len(row['step_records'])!=10 or row['final_time']!=1.
                or not all(row['checks'].values()) or row['total_iterations']!=sum(s['iterations'] for s in row['step_records'])):
            raise RuntimeError('Incomplete trajectory.')
        params=row['solver_parameters'];expected=dict(config['rana_solver_parameters'] if label=='rana_ld' else config['solver_parameters'])
        if label=='manual_jacobi_reference':
            for key in ('ksp_rtol','ksp_atol'):expected[key]*=config['manual_reference_tolerance_factor']
        if label=='residual_jacobi':expected.update(mat_type='aij',snes_type='ksponly')
        if any(params[k]!=v for k,v in expected.items()):raise RuntimeError('Declared solver changed.')
        if label in ('residual_jacobi','rana_ld'):
            if (row['weak_form_argument_count']!=1 or row['weak_form_representation']!='state_function_residual'
                    or not row['constant_jacobian'] or row['solver_stats']!=[10,10,row['total_iterations']]):
                raise RuntimeError('Residual representation or SNES/KSP work counters failed.')
            actual=row['actual_solver']
            if (actual['ksp_type']!='gmres' or actual['pc_type']!=params['pc_type'] or actual['snes_type']!='ksponly'
                    or actual['side']!='right' or actual['norm']!='unpreconditioned' or actual['rtol']!=params['ksp_rtol']
                    or actual['atol']!=params['ksp_atol'] or actual['max_it']!=1000):raise RuntimeError('Actual residual-form solver changed.')
        for index,step in enumerate(row['step_records']):
            r=step['true_block_residual'];target=max(params['ksp_atol'],params['ksp_rtol']*r['rhs_norm'])
            if (step['step']!=index+1 or abs(step['t_start']-index*.1)>1e-14 or abs(step['t_end']-(index+1)*.1)>1e-14
                    or step['converged_reason']<=0 or not all(math.isfinite(v) and v>=0 for v in r.values())
                    or target!=step['true_residual_target'] or r['norm']>10*target or step['update_identity_l2']>1e-12 or step['stiff_accuracy_l2']>1e-12
                    or len(step['stage_times'])!=2 or any(abs(a-b)>1e-14 for a,b in zip(step['stage_times'],[(index+1/3)*.1,(index+1)*.1]))):
                raise RuntimeError('Clock, physical residual or update check failed.')
            if label!='irksome_jacobi':
                if any(v>1e-12 for v in step['stage_boundary_trace_l2']) or not all(s['source_residual_l2']<=1e-10 for s in step['source_checks']):
                    raise RuntimeError('Stage boundary or source check failed.')
                if index>0 and step['event_counts']['PCSetUp']!=0:raise RuntimeError('Unexpected PC rebuild after first step.')
            if label in ('residual_jacobi','rana_ld'):
                if set(step['reference_comparisons'])!={'update','slope1','slope2'}:raise RuntimeError('Missing per-step field comparisons.')
                for item in step['reference_comparisons'].values():field_check(item)
                keys={'physical_B','auxiliary_P','auxiliary_PC_Pmat'} if label=='rana_ld' else {'physical_B'}
                if set(step['matrix_checks'])!=keys:raise RuntimeError('Missing independent matrix checks.')
                for matrix in step['matrix_checks'].values():
                    allowed=config['checks']['auxiliary_matrix_atol']+config['checks']['auxiliary_matrix_rtol']*matrix['reference_frobenius_norm']
                    if matrix['status']!='passed' or not math.isfinite(matrix['frobenius_defect']) or matrix['allowed_defect']!=allowed or not 0<=matrix['frobenius_defect']<=allowed:
                        raise RuntimeError('Physical/auxiliary CSR check failed.')
                if not step['reuse_matches_first_step'] or step['matrix_reuse_tokens']!=row['step_records'][0]['matrix_reuse_tokens']:
                    raise RuntimeError('Matrix/factor identity or state changed.')
                if step['event_counts']['KSPSolve']!=1:raise RuntimeError('Expected one physical KSP solve per advance.')
                if index>0 and any(step['event_counts'][event]!=0 for event in ('MatLUFactorSym','MatLUFactorNum')):
                    raise RuntimeError('Auxiliary LU refactorized after first step.')
                if label=='rana_ld' and (step['event_counts']['MatSolve']!=step['event_counts']['PCApply'] or step['event_counts']['MatSolve']<=0):
                    raise RuntimeError('Auxiliary LU action accounting failed.')
        if label in ('residual_jacobi','rana_ld'):
            counts=row['operator_event_counts']
            if counts['KSPSolve']!=10 or counts['PCSetUp']<1:raise RuntimeError('Missing solve/setup evidence.')
            for event,value in counts.items():
                if value!=row['construction_event_counts'][event]+sum(s['event_counts'][event] for s in row['step_records']):
                    raise RuntimeError('Event-count decomposition failed.')
            for budget in row['accepted_time_error_budget_check'].values():
                if budget['allowed_distance']!=1e-12+.01*budget['accepted_baseline_time_error'] or not 0<=budget['actual_field_distance']<=budget['allowed_distance']:
                    raise RuntimeError('Inherited temporal-error budget check failed.')
    rana=record['trajectories']['rana_ld'];aux=rana['auxiliary_solver'];probe=rana['auxiliary_action_check']
    if (rana['actual_solver']['python_context']!='irksome.pc.RanaLD' or aux['pc_type']!='lu' or aux['factor_solver_type']!='petsc'
            or aux['inner_krylov_solve'] is not False or not aux['factor_nonzeros']>0):raise RuntimeError('Actual installed Rana auxiliary action changed.')
    if (any(rana['operator_event_counts'][event]<1 for event in ('MatLUFactorSym','MatLUFactorNum'))
            or rana['diagnostic_event_counts']['MatSolve']!=1
            or any(rana['diagnostic_event_counts'][event]!=0 for event in ('KSPSolve','PCSetUp','MatLUFactorSym','MatLUFactorNum'))):
        raise RuntimeError('Factorization/probe event evidence failed.')
    allowed=config['checks']['auxiliary_action_atol']+config['checks']['auxiliary_action_rtol']*probe['rhs_norm']
    if (probe['status']!='passed' or probe['allowed_residual']!=allowed or not math.isfinite(probe['residual_norm'])
            or not 0<=probe['residual_norm']<=allowed or probe['diagnostic_outer_PC_applications']!=1):raise RuntimeError('Final actual auxiliary inverse action failed.')
    return dict(status='passed',trajectories=4,steps=40,performance_measurement=False,
        interpretation='Full RanaLD trajectory, matched residual-form agreement and fixed-matrix/factor reuse; mentor review pending')


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
    stem = ROOT/'results/raw'/f'rana-trajectory-{stamp}-{os.getpid()}'
    transcript = stem.with_suffix('.txt')
    manifest_path = Path(str(stem)+'-manifest.json')
    steps = commands(venv,stem)
    sources = (SCRIPT,ROOT/'src/phase3_rana_construction.py',ROOT/'environment/check_rana_construction.py',ROOT/'src/phase3_asm_construction.py',ROOT/'environment/check_asm_construction.py',ROOT/'src/phase3_asm_trajectory.py',ROOT/'environment/check_asm_trajectory.py',ROOT/'src/phase3_asm_cost.py',ROOT/'environment/check_asm_cost.py',ROOT/'src/phase3_accuracy_cost.py',ROOT/'environment/check_accuracy_cost.py',ROOT/'src/phase3_cost_pilot.py',ROOT/'environment/check_cost_pilot.py',ROOT/'src/phase2_temporal.py',ROOT/'environment/check_temporal.py',ROOT/'src/phase0_be_smoke.py',ROOT/'src/phase0_irksome_be.py',ROOT/'src/phase0_radau_smoke.py',
       ROOT/'src/phase1_interpolation.py',ROOT/'src/phase1_spatial.py',Path(__file__).resolve(),
       ROOT/'environment/check_spatial.py',ROOT/'environment/check_interpolation.py',
       ROOT/'environment/check_radau_smoke.py',ROOT/'environment/check_irksome_be.py',ROOT/'environment/check_be_smoke.py',
       ROOT/'environment/check_python_runtime.py',ROOT/'environment/configure_petsc.py')
    state = {'started_utc':datetime.now(timezone.utc).isoformat(),'status':'started',
        'measurement_kind':'Full T=1 installed RanaLD/exact auxiliary LU trajectory, matched residual path and reuse; no timing ranking',
        'rana_construction_review':str(review_path.relative_to(ROOT)),'rana_construction_review_sha256':digest(review_path),
        'verified_environment':str(SNAPSHOT.relative_to(ROOT)),'verified_environment_sha256':digest(SNAPSHOT),
        'config':str(CONFIG.relative_to(ROOT)),'config_sha256':digest(CONFIG),
        'installed_source_inspection':'environment/rana_trajectory_source_inspection.json',
        'installed_source_inspection_sha256':digest(ROOT/'environment/rana_trajectory_source_inspection.json'),
        'inspected_installed_source_sha256':json.loads((ROOT/'environment/rana_trajectory_source_inspection.json').read_text())['installed_source_sha256'],
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
        'rana_trajectory_review_pending':True,'performance_measurement':False,'accuracy_cost_ranking_established':False}
    status = 1
    manifest_path.write_text(json.dumps(state,indent=2)+'\n')
    with transcript.open('w') as log:
        try:
            for name,argv in steps:
                run_logged(argv,ROOT,env,log)
                state['completed_steps'].append(name)
            state['study_validation'] = verify_complete(json.loads(Path(str(stem)+'-serial.json').read_text()),config)
            state['status'] = 'Full RanaLD trajectory checks passed; mentor review pending'
            status = 0
            message = 'RANA TRAJECTORY CHECKPOINT PASSED. Review before repeated timing experiments.'
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
        stem = ROOT/'results/raw/rana-trajectory-<UTC timestamp>-<process id>'
        for name,argv in commands(parameters[4],stem):
            print(f'{name}: {shlex.join(argv)}')
        print('Display only: no numerical imports, assembly, solves, time stepping, or MPI launch run.')
        return 0
    return run(*parameters)


if __name__=='__main__':
    try:
        sys.exit(main())
    except (RuntimeError,OSError,subprocess.SubprocessError,KeyError,ValueError) as error:
        print(f'STOP before Rana trajectory verification: {error}',file=sys.stderr)
        sys.exit(1)

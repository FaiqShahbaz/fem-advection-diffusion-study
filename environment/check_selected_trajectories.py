#!/usr/bin/env python3
"""Researcher-run selected full trajectories after accepted first-step sensitivity outcomes."""
import argparse
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

from check_solver_sensitivity import plan as previous_environment_plan
from configure_petsc import ROOT,digest,run_logged,validate_tools

CONFIG = ROOT/'configs/phase3_selected_trajectories.json'
SCRIPT = ROOT/'src/phase3_selected_trajectories.py'
SNAPSHOT = ROOT/'environment/verified_environment.json'


def plan():
    _,_,installation,petsc,venv,env=previous_environment_plan()
    review_path=ROOT/'results/processed/solver_sensitivity_review.json';review=json.loads(review_path.read_text())
    pairs=[(review[key],review[key+'_sha256']) for key in
        ('pilot_manifest','pilot_transcript','config','rana_cost_review','verified_environment','installed_source_inspection')]
    pairs.extend((r['path'],r['sha256']) for r in review['measurement_records'].values())
    pairs.extend(review['run_script_sha256'].items());pairs.append((review['audit_source']['path'],review['audit_source']['sha256']))
    for path,expected in pairs:
        if digest(ROOT/path)!=expected:raise RuntimeError(f'Accepted sensitivity evidence/source changed: {path}')
    if review['status']!='accepted selected first-step solver sensitivity with preserved iteration limit' or not review['solver_sensitivity_verified'] or review['accepted_candidate_solutions']!=8 or review['preserved_capped_candidates']!=1:
        raise RuntimeError('Accepted first-step sensitivity outcomes are required.')
    pilot=json.loads((ROOT/review['measurement_records']['serial']['path']).read_text())
    excluded=pilot['points']['mesh_refined']['candidates']['Jacobi']
    if excluded['status']!='iteration_limit' or excluded['iterations']!=1000 or excluded['converged_reason']!=-3 or excluded['completed_steps']!=0:
        raise RuntimeError('The predeclared skip must retain the accepted finer-mesh Jacobi limit.')
    sys.path.insert(0,str(ROOT/'src'))
    from phase3_selected_trajectories import validate_config
    validate_config(json.loads(CONFIG.read_text()))
    inspection=json.loads((ROOT/'environment/solver_sensitivity_source_inspection.json').read_text())
    for path,expected in inspection['installed_source_sha256'].items():
        if digest(ROOT/path)!=expected:raise RuntimeError(f'Inspected trajectory backend/interface source changed: {path}')
    return review_path,review,installation,petsc,venv,env


def commands(venv,stem):
    return [('serial',[str(venv/'bin/python'),str(SCRIPT),'--config',str(CONFIG),
                       '--output',str(stem)+'-serial.json'])]


def verify_complete(record,config):
    if (record['status']!='collected' or not record['completed_study'] or record['config']!=config or record['ranks']!=1
            or record['performance_measurement'] is not False or record['reference_trajectories']!=3 or record['reference_completed_steps']!=30
            or record['candidate_attempts']!=8 or record['predeclared_skipped_candidates']!=1
            or set(record['points'])!={p['point_id'] for p in config['points']}):raise RuntimeError('Selected trajectory collection did not complete.')
    passed=capped=skipped=completed=0
    def matrix(row):
        allowed=config['checks']['auxiliary_matrix_atol']+config['checks']['auxiliary_matrix_rtol']*row['reference_frobenius_norm']
        if row['status']!='passed' or row['allowed_defect']!=allowed or not math.isfinite(row['frobenius_defect']) or not 0<=row['frobenius_defect']<=allowed:
            raise RuntimeError('Independent full matrix check failed.')
    def steps(trajectory,rtol,atol):
        if len(trajectory['step_records'])!=trajectory['completed_steps']:raise RuntimeError('Step count mismatch.')
        first=None
        for index,row in enumerate(trajectory['step_records']):
            r=row['true_block_residual'];target=max(atol,rtol*r['rhs_norm'])
            if (row['step']!=index+1 or row['t_start']!=index*config['dt'] or row['t_end']!=(index+1)*config['dt'] or row['converged_reason']<=0
                    or not all(math.isfinite(v) and v>=0 for v in r.values()) or r['rhs_norm']<=0 or row['true_residual_target']!=target or r['norm']>10*target
                    or row['update_identity_l2']>1e-12 or row['stiff_accuracy_l2']>1e-12 or any(v>1e-12 for v in row['stage_boundary_trace_l2'])
                    or not all(s['source_residual_l2']<=1e-10 for s in row['source_checks'])):raise RuntimeError('Physical per-step verification failed.')
            if first is None:first=row['matrix_reuse_tokens']
            if (row['matrix_reuse_tokens']!=first or not row['reuse_matches_first_step']
                    or index>0 and any(row['event_counts'][e]!=0 for e in ('PCSetUp','MatLUFactorSym','MatLUFactorNum'))):raise RuntimeError('Fixed factors were not reused.')
            for value in row.get('matrix_checks',{}).values():matrix(value)
            for item in row.get('reference_comparisons',{}).values():
                if item['status']!='passed' or item['difference_measurement']['status']!='passed':raise RuntimeError('Actual field quadrature comparison failed.')
                for field in item['norms'].values():
                    allowed=config['checks']['pc_field_comparison_atol']+config['checks']['pc_field_comparison_rtol']*field['reference_field_norm']
                    if field['allowed_difference']!=allowed or not math.isfinite(field['actual_field_difference']) or not 0<=field['actual_field_difference']<=allowed:
                        raise RuntimeError('Per-step actual field agreement failed.')
        if trajectory['status']=='passed' and (trajectory['completed_steps']!=10 or trajectory['final_time']!=1. or not all(trajectory['checks'].values())):
            raise RuntimeError('Full trajectory did not verify.')
    for point in config['points']:
        row=record['points'][point['point_id']];n=point['mesh_subdivisions'];meta=row['metadata']
        if (row['status']!='collected' or meta['stage_total_dofs']!=2*(2*n+1)**2 or meta['total_dofs']!=(2*n+1)**2
                or meta['free_dofs']!=(2*n-1)**2 or meta['diffusion']!=point['diffusion'] or meta['checkpoint_end_time']!=1.
                or set(row['candidates'])!=set(config['solver_cases'])):raise RuntimeError('Point metadata changed.')
        reference=row['reference']
        if (reference['status']!='passed' or reference['actual_solver']!=dict(ksp_type='preonly',pc_type='lu',factor_solver_type='petsc')
                or reference['solver_parameters']!=config['reference_solver_parameters']):raise RuntimeError('Independent reference changed.')
        steps(reference,config['reference_residual_rtol'],config['reference_residual_atol'])
        for case,candidate in row['candidates'].items():
            if candidate['status']=='not_attempted':
                if (point['point_id']!='mesh_refined' or case!='Jacobi' or candidate['completed_steps']!=0 or candidate['accepted_final_solution'] is not False
                        or candidate['pilot_iterations']!=1000 or candidate['pilot_converged_reason']!=-3):raise RuntimeError('Unplanned candidate exclusion.')
                skipped+=1;continue
            expected=config['rana_solver_parameters'] if case=='RanaLD' else config['asm_solver_parameters'] if case=='ASM' else {**config['solver_parameters'],'mat_type':'aij','snes_type':'ksponly'}
            actual=candidate['actual_solver']
            if (candidate['solver_parameters']!=expected or candidate['weak_form_argument_count']!=1 or actual['ksp_type']!='gmres'
                    or actual['pc_type']!=expected['pc_type'] or actual['snes_type']!='ksponly' or actual['side']!='right' or actual['norm']!='unpreconditioned'
                    or actual['rtol']!=expected['ksp_rtol'] or actual['atol']!=expected['ksp_atol'] or actual['max_it']!=1000):raise RuntimeError('Actual candidate solver changed.')
            steps(candidate,expected['ksp_rtol'],expected['ksp_atol']);completed+=candidate['completed_steps']
            for step in candidate['step_records']:
                if set(step['reference_comparisons'])!={'update','slope1','slope2'}:raise RuntimeError('Missing actual field comparison.')
                if case=='RanaLD' and set(step['matrix_checks'])!={'physical_B','auxiliary_P','auxiliary_PC_Pmat'}:raise RuntimeError('Missing auxiliary coefficient check.')
                if case=='RanaLD' and step['event_counts']['MatSolve']!=step['event_counts']['PCApply']:raise RuntimeError('Rana work mismatch.')
                if case=='ASM' and step['event_counts']['MatSolve']!=4*step['event_counts']['PCApply']:raise RuntimeError('ASM work mismatch.')
            if candidate['status']=='iteration_limit':
                failure=candidate['failed_step'];matrix(candidate['failed_physical_matrix_check'])
                if (failure['converged_reason']!=-3 or failure['iterations']!=1000 or candidate['completed_steps']>=10
                        or candidate['step_committed'] is not False or candidate['failed_step_state_change_l2']>1e-12
                        or 'final_time' in candidate or 'metrics' in candidate):raise RuntimeError('Invalid or accepted capped trajectory.')
                capped+=1;continue
            if candidate['status']!='passed' or candidate['solver_stats']!=[10,10,candidate['total_iterations']]:raise RuntimeError('Unexpected trajectory outcome.')
            if case=='RanaLD':
                probe=candidate['auxiliary_action_check'];allowed=config['checks']['auxiliary_action_atol']+config['checks']['auxiliary_action_rtol']*probe['rhs_norm']
                if probe['status']!='passed' or probe['allowed_residual']!=allowed or not 0<=probe['residual_norm']<=allowed:raise RuntimeError('Auxiliary probe failed.')
            elif case=='ASM':
                if (len(candidate['domains'])!=4 or sorted(i for d in candidate['domains'] for i in d['core_indices'])!=list(range(meta['stage_total_dofs']))
                        or not all(d['one_layer_check'] for d in candidate['domains']) or candidate['additive_action_check']['status']!='passed'):
                    raise RuntimeError('ASM domain/action check failed.')
            passed+=1
    if (passed+capped!=8 or skipped!=1 or record['completed_candidate_trajectories']!=passed or record['capped_candidate_trajectories']!=capped
            or record['candidate_completed_steps']!=completed):raise RuntimeError('Outcome count mismatch.')
    return dict(status='collected',reference_trajectories=3,reference_completed_steps=30,candidate_attempts=8,completed_candidate_trajectories=passed,
        capped_candidate_trajectories=capped,predeclared_skipped_candidates=skipped,candidate_completed_steps=completed,
        performance_measurement=False,interpretation='Selected full trajectories; recorded pilot exclusion and any new caps retained; mentor review pending')


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
    stem = ROOT/'results/raw'/f'selected-trajectories-{stamp}-{os.getpid()}'
    transcript = stem.with_suffix('.txt')
    manifest_path = Path(str(stem)+'-manifest.json')
    steps = commands(venv,stem)
    sources=tuple(dict.fromkeys([SCRIPT,*[ROOT/path for path in review['run_script_sha256']],Path(__file__).resolve()]))
    state = {'started_utc':datetime.now(timezone.utc).isoformat(),'status':'started',
        'measurement_kind':'Selected full T=1 trajectories and factor reuse; predeclared pilot exclusion retained; no timing',
        'solver_sensitivity_review':str(review_path.relative_to(ROOT)),'solver_sensitivity_review_sha256':digest(review_path),
        'verified_environment':str(SNAPSHOT.relative_to(ROOT)),'verified_environment_sha256':digest(SNAPSHOT),
        'config':str(CONFIG.relative_to(ROOT)),'config_sha256':digest(CONFIG),
        'installed_source_inspection':'environment/solver_sensitivity_source_inspection.json',
        'installed_source_inspection_sha256':digest(ROOT/'environment/solver_sensitivity_source_inspection.json'),
        'inspected_installed_source_sha256':json.loads((ROOT/'environment/solver_sensitivity_source_inspection.json').read_text())['installed_source_sha256'],
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
        'selected_trajectories_review_pending':True,'performance_measurement':False,'accuracy_cost_ranking_established':False}
    status = 1
    manifest_path.write_text(json.dumps(state,indent=2)+'\n')
    with transcript.open('w') as log:
        try:
            for name,argv in steps:
                run_logged(argv,ROOT,env,log)
                state['completed_steps'].append(name)
            state['study_validation'] = verify_complete(json.loads(Path(str(stem)+'-serial.json').read_text()),config)
            state['status'] = 'Selected trajectories collected; outcomes and reuse await mentor review'
            status = 0
            message = 'SELECTED TRAJECTORY RECORDS SAVED. Review full-trajectory outcomes and reuse before timing.'
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
        stem = ROOT/'results/raw/selected-trajectories-<UTC timestamp>-<process id>'
        for name,argv in commands(parameters[4],stem):
            print(f'{name}: {shlex.join(argv)}')
        print('Display only: no numerical imports, assembly, solves, time stepping, or MPI launch run.')
        return 0
    return run(*parameters)


if __name__=='__main__':
    try:
        sys.exit(main())
    except (RuntimeError,OSError,subprocess.SubprocessError,KeyError,ValueError) as error:
        print(f'STOP before selected trajectories: {error}',file=sys.stderr)
        sys.exit(1)

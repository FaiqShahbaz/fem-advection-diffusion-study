#!/usr/bin/env python3
"""Researcher-run three-point first-step solver sensitivity collection after accepted matched Rana timing."""
import argparse
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

from check_rana_cost import plan as previous_environment_plan
from configure_petsc import ROOT,digest,run_logged,validate_tools

CONFIG = ROOT/'configs/phase3_solver_sensitivity.json'
SCRIPT = ROOT/'src/phase3_solver_sensitivity.py'
SNAPSHOT = ROOT/'environment/verified_environment.json'


def plan():
    _,_,installation,petsc,venv,env=previous_environment_plan()
    review_path=ROOT/'results/processed/rana_cost_review.json';review=json.loads(review_path.read_text())
    pairs=[(review[key],review[key+'_sha256']) for key in
        ('cost_manifest','cost_transcript','config','rana_trajectory_review','verified_environment','installed_source_inspection')]
    pairs.extend((r['path'],r['sha256']) for r in review['measurement_records'].values())
    pairs.extend(review['run_script_sha256'].items());pairs.append((review['audit_source']['path'],review['audit_source']['sha256']))
    for path,expected in pairs:
        if digest(ROOT/path)!=expected:raise RuntimeError(f'Accepted Rana timing evidence/source changed: {path}')
    if review['status']!='accepted matched residual Radau Jacobi RanaLD timing' or not review['rana_cost_verified']:
        raise RuntimeError('Accepted matched Rana timing review is required.')
    inspection=json.loads((ROOT/'environment/solver_sensitivity_source_inspection.json').read_text())
    for path,expected in inspection['installed_source_sha256'].items():
        if digest(ROOT/path)!=expected:raise RuntimeError(f'Inspected trajectory backend/interface source changed: {path}')
    return review_path,review,installation,petsc,venv,env


def commands(venv,stem):
    return [('serial',[str(venv/'bin/python'),str(SCRIPT),'--config',str(CONFIG),
                       '--output',str(stem)+'-serial.json'])]


def verify_complete(record,config):
    if (record['status']!='collected' or not record['completed_study'] or record['config']!=config or record['ranks']!=1
            or record['performance_measurement'] is not False or record['reference_solves']!=3 or record['candidate_attempts']!=9
            or set(record['points'])!={p['point_id'] for p in config['points']}):raise RuntimeError('The bounded sensitivity collection did not complete.')
    converged=capped=0
    def residual(row,rtol,atol):
        r=row['true_block_residual'];target=max(atol,rtol*r['rhs_norm'])
        if (not all(math.isfinite(v) and v>=0 for v in r.values()) or r['rhs_norm']<=0 or row['true_residual_target']!=target or r['norm']>10*target
                or row['update_identity_l2']>1e-12 or row['stiff_accuracy_l2']>1e-12 or any(v>1e-12 for v in row['stage_boundary_trace_l2'])):
            raise RuntimeError('Physical equation/update/boundary check failed.')
    def matrix(row):
        allowed=config['checks']['auxiliary_matrix_atol']+config['checks']['auxiliary_matrix_rtol']*row['reference_frobenius_norm']
        if row['status']!='passed' or row['allowed_defect']!=allowed or not math.isfinite(row['frobenius_defect']) or not 0<=row['frobenius_defect']<=allowed:
            raise RuntimeError('Independent full coefficient check failed.')
    for point in config['points']:
        row=record['points'][point['point_id']];meta=row['metadata'];n=point['mesh_subdivisions']
        if (row['status']!='collected' or meta['stage_total_dofs']!=2*(2*n+1)**2 or meta['total_dofs']!=(2*n+1)**2 or meta['free_dofs']!=(2*n-1)**2
                or meta['diffusion']!=point['diffusion'] or set(row['candidates'])!=set(config['solver_cases'])
                or not all(s['source_residual_l2']<=1e-10 for s in row['source_checks'])):raise RuntimeError('Point metadata or MMS forcing check failed.')
        reference=row['reference']
        if reference['status']!='passed' or not all(reference['checks'].values()) or reference['actual_solver']!=dict(ksp_type='preonly',pc_type='lu',factor_solver_type='petsc'):
            raise RuntimeError('Independent physical reference failed.')
        residual(reference,config['reference_residual_rtol'],config['reference_residual_atol'])
        for case,candidate in row['candidates'].items():
            if candidate['weak_form_argument_count']!=1:raise RuntimeError('Residual form interface changed.')
            matrix(candidate['physical_matrix_check']);actual=candidate['actual_solver'];params=candidate['solver_parameters']
            expected=config['rana_solver_parameters'] if case=='RanaLD' else config['asm_solver_parameters'] if case=='ASM' else {**config['solver_parameters'],'mat_type':'aij','snes_type':'ksponly'}
            if (params!=expected or actual['ksp_type']!='gmres' or actual['pc_type']!=params['pc_type'] or actual['snes_type']!='ksponly' or actual['side']!='right' or actual['norm']!='unpreconditioned'
                    or actual['rtol']!=params['ksp_rtol'] or actual['atol']!=params['ksp_atol'] or actual['max_it']!=1000):raise RuntimeError('Actual candidate parameters changed.')
            if candidate['status']=='iteration_limit':
                if (candidate['converged_reason']!=-3 or candidate['iterations']!=1000 or candidate['step_committed'] is not False
                        or candidate['completed_steps']!=0 or candidate['initial_state_change_l2']>1e-12
                        or 'reference_comparisons' in candidate):raise RuntimeError('Invalid or accepted capped candidate.')
                capped+=1;continue
            if (candidate['status']!='passed' or candidate['converged_reason']<=0 or candidate['completed_steps']!=1 or candidate['step_committed'] is not True
                    or not all(candidate['checks'].values()) or set(candidate['reference_comparisons'])!={'update','slope1','slope2'}):raise RuntimeError('Converged candidate did not verify.')
            residual(candidate,params['ksp_rtol'],params['ksp_atol'])
            for item in candidate['reference_comparisons'].values():
                if item['status']!='passed' or item['difference_measurement']['status']!='passed':raise RuntimeError('Actual field comparison failed.')
                for field in item['norms'].values():
                    allowed=config['checks']['pc_field_comparison_atol']+config['checks']['pc_field_comparison_rtol']*field['reference_field_norm']
                    if field['allowed_difference']!=allowed or not math.isfinite(field['actual_field_difference']) or not 0<=field['actual_field_difference']<=allowed:
                        raise RuntimeError('Field comparison allowance failed.')
            if case=='RanaLD':
                for value in candidate['auxiliary_matrix_checks'].values():matrix(value)
                probe=candidate['auxiliary_action_check'];allowed=config['checks']['auxiliary_action_atol']+config['checks']['auxiliary_action_rtol']*probe['rhs_norm']
                if probe['status']!='passed' or probe['allowed_residual']!=allowed or not 0<=probe['residual_norm']<=allowed:raise RuntimeError('Auxiliary inverse action failed.')
            elif case=='ASM':
                if (len(candidate['domains'])!=4 or sorted(i for d in candidate['domains'] for i in d['core_indices'])!=list(range(meta['stage_total_dofs']))
                        or not all(d['one_layer_check'] for d in candidate['domains']) or candidate['additive_action_check']['status']!='passed'):
                    raise RuntimeError('ASM domain/action checks failed.')
            converged+=1
    if record['converged_candidates']!=converged or record['capped_candidates']!=capped or converged+capped!=9:raise RuntimeError('Outcome count mismatch.')
    return dict(status='collected',reference_solves=3,candidate_attempts=9,converged_candidates=converged,capped_candidates=capped,
        performance_measurement=False,interpretation='Three selected first-step points; only converged and independently checked candidates are accepted solutions; mentor review pending')


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
    stem = ROOT/'results/raw'/f'solver-sensitivity-{stamp}-{os.getpid()}'
    transcript = stem.with_suffix('.txt')
    manifest_path = Path(str(stem)+'-manifest.json')
    steps = commands(venv,stem)
    sources=tuple(dict.fromkeys([SCRIPT,*[ROOT/path for path in review['run_script_sha256']],Path(__file__).resolve()]))
    state = {'started_utc':datetime.now(timezone.utc).isoformat(),'status':'started',
        'measurement_kind':'Three selected first-step mesh/diffusion solver sensitivity points; no timing or full-trajectory ranking',
        'rana_cost_review':str(review_path.relative_to(ROOT)),'rana_cost_review_sha256':digest(review_path),
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
        'solver_sensitivity_review_pending':True,'performance_measurement':False,'accuracy_cost_ranking_established':False}
    status = 1
    manifest_path.write_text(json.dumps(state,indent=2)+'\n')
    with transcript.open('w') as log:
        try:
            for name,argv in steps:
                run_logged(argv,ROOT,env,log)
                state['completed_steps'].append(name)
            state['study_validation'] = verify_complete(json.loads(Path(str(stem)+'-serial.json').read_text()),config)
            state['status'] = 'Solver sensitivity pilot collected; convergence outcomes await mentor review'
            status = 0
            message = 'SOLVER SENSITIVITY PILOT RECORDS SAVED. Review convergence outcomes before full trajectories or timing.'
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
        stem = ROOT/'results/raw/solver-sensitivity-<UTC timestamp>-<process id>'
        for name,argv in commands(parameters[4],stem):
            print(f'{name}: {shlex.join(argv)}')
        print('Display only: no numerical imports, assembly, solves, time stepping, or MPI launch run.')
        return 0
    return run(*parameters)


if __name__=='__main__':
    try:
        sys.exit(main())
    except (RuntimeError,OSError,subprocess.SubprocessError,KeyError,ValueError) as error:
        print(f'STOP before solver sensitivity pilot: {error}',file=sys.stderr)
        sys.exit(1)

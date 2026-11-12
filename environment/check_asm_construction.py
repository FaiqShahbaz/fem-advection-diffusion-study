#!/usr/bin/env python3
"""Researcher-run one-step ASM/local-ILU construction after the accepted cost baseline."""
import argparse
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

from check_accuracy_cost import plan as previous_environment_plan
from configure_petsc import ROOT,digest,run_logged,validate_tools

CONFIG = ROOT/'configs/phase3_asm_construction.json'
SCRIPT = ROOT/'src/phase3_asm_construction.py'
SNAPSHOT = ROOT/'environment/verified_environment.json'


def plan():
    _,_,installation,petsc,venv,env = previous_environment_plan()
    review_path = ROOT/'results/processed/accuracy_cost_review.json'
    review = json.loads(review_path.read_text())
    pairs = [(review[key],review[key+'_sha256']) for key in
             ('accuracy_cost_manifest','accuracy_cost_transcript','config','verified_environment','cost_pilot_review')]
    pairs.extend((record['path'],record['sha256']) for record in review['measurement_records'].values())
    pairs.extend(review['run_script_sha256'].items())
    pairs.append((review['audit_source']['path'],review['audit_source']['sha256']))
    for path,expected in pairs:
        if digest(ROOT/path)!=expected:
            raise RuntimeError(f'Accepted accuracy/cost evidence/source changed: {path}')
    if review['checkpoint_exit_status']!=0 or review['status']!='accepted targeted fixed-mesh temporal accuracy cost baseline' or not review['accuracy_cost_baseline_verified']:
        raise RuntimeError('Accepted targeted accuracy/cost baseline is required.')
    conf = (ROOT/'environment/build/petsc/arch-firedrake-default/include/petscconf.h').read_text()
    if '#define PETSC_HAVE_PTSCOTCH 1' not in conf:
        raise RuntimeError('The tested PETSc build must already contain PT-Scotch; do not change the environment.')
    return review_path,review,installation,petsc,venv,env


def commands(venv,stem):
    return [('serial',[str(venv/'bin/python'),str(SCRIPT),'--config',str(CONFIG),
                       '--output',str(stem)+'-serial.json'])]


def verify_complete(record,config):
    if (record['status']!='passed' or not record['completed_study'] or record['config']!=config
            or record['ranks']!=1 or record['performance_measurement'] is not False
            or record['completed_operator_solves']!=3 or record['total_dofs']!=1089
            or record['stage_total_dofs']!=2178 or record['checkpoint_end_time']!=0.1):
        raise RuntimeError('The one-step construction study did not pass.')
    expected={'irksome_jacobi','manual_jacobi_reference','manual_asm'}
    if set(record['solves'])!=expected:
        raise RuntimeError('Missing independent stage solves.')
    baseline=record['solves']['irksome_jacobi']
    if (baseline['status']!='passed' or baseline['completed_steps']!=1
            or not all(baseline['checks'].values()) or baseline['final_time']!=0.1):
        raise RuntimeError('Irksome one-step baseline failed.')
    for label in ('manual_jacobi_reference','manual_asm'):
        row=record['solves'][label]
        if row['status']!='passed' or row['completed_operator_solves']!=1 or row['stage_total_dofs']!=2178 or row['converged_reason']<=0:
            raise RuntimeError('Independent coupled solve failed.')
        r=row['true_block_residual'];par=row['solver_parameters']
        target=max(par['ksp_atol'],par['ksp_rtol']*r['rhs_norm'])
        if (not all(math.isfinite(v) for v in r.values()) or row['true_residual_target']!=target
                or r['norm']>config['checks']['residual_tolerance_factor']*target
                or not all(v<=config['checks']['stage_boundary_trace_l2_max'] for v in row['stage_boundary_trace_l2'])):
            raise RuntimeError('Physical stage residual/boundary check failed.')
    names={'manual_reference_update_vs_irksome','asm_update_vs_manual_reference',
           'asm_slope1_vs_manual_reference','asm_slope2_vs_manual_reference'}
    if set(record['comparisons'])!=names or not all(r['status']=='passed' for r in record['comparisons'].values()):
        raise RuntimeError('Field comparisons failed.')
    for comparison in record['comparisons'].values():
        if comparison['difference_measurement']['status']!='passed' or not all(
            row['actual_field_difference']<=row['allowed_difference'] for row in comparison['norms'].values()):
            raise RuntimeError('Field comparison arithmetic/quadrature failed.')
    asm=record['solves']['manual_asm'];domains=asm['domains'];work=asm['local_work']
    if len(domains)!=4 or len(work)!=4 or len(asm['local_solvers'])!=4:
        raise RuntimeError('Expected four local subdomains and solvers.')
    core=[i for domain in domains for i in domain['core_indices']]
    if sorted(core)!=list(range(2178)) or not all(d['one_layer_check'] and set(d['core_indices'])<=set(d['overlap_indices']) for d in domains):
        raise RuntimeError('Domain coverage or one-layer checks failed.')
    if not all(w['outer_solve']['calls']>0 and w['outer_solve']['reported_iterations']==w['outer_solve']['calls']
               and w['action_check']==dict(calls=1,reported_iterations=1) for w in work):
        raise RuntimeError('Local PREONLY work accounting failed.')
    if len({w['outer_solve']['calls'] for w in work})!=1 or not all(
        r['ksp_type']=='preonly' and r['pc_type']=='ilu' and r['factor_solver_type']=='petsc' for r in asm['local_solvers']):
        raise RuntimeError('Unexpected local solver action/backend.')
    action=asm['additive_action_check']
    if action['status']!='passed' or action['defect_norm']>action['allowed_defect']:
        raise RuntimeError('Independent additive action check failed.')
    return {'status':'passed','independent_operator_solves':3,'subdomains':4,'performance_measurement':False,
            'interpretation':'One-step standard ASM/local ILU construction and field equivalence; mentor review pending'}


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
    stem = ROOT/'results/raw'/f'asm-construction-{stamp}-{os.getpid()}'
    transcript = stem.with_suffix('.txt')
    manifest_path = Path(str(stem)+'-manifest.json')
    steps = commands(venv,stem)
    sources = (SCRIPT,ROOT/'src/phase3_accuracy_cost.py',ROOT/'environment/check_accuracy_cost.py',ROOT/'src/phase3_cost_pilot.py',ROOT/'environment/check_cost_pilot.py',ROOT/'src/phase2_temporal.py',ROOT/'environment/check_temporal.py',ROOT/'src/phase0_be_smoke.py',ROOT/'src/phase0_irksome_be.py',ROOT/'src/phase0_radau_smoke.py',
       ROOT/'src/phase1_interpolation.py',ROOT/'src/phase1_spatial.py',Path(__file__).resolve(),
       ROOT/'environment/check_spatial.py',ROOT/'environment/check_interpolation.py',
       ROOT/'environment/check_radau_smoke.py',ROOT/'environment/check_irksome_be.py',ROOT/'environment/check_be_smoke.py',
       ROOT/'environment/check_python_runtime.py',ROOT/'environment/configure_petsc.py')
    state = {'started_utc':datetime.now(timezone.utc).isoformat(),'status':'started',
        'measurement_kind':'One-step standard ASM(4 blocks, overlap 1)/local ILU(0) construction; no timing ranking',
        'accuracy_cost_review':str(review_path.relative_to(ROOT)),'accuracy_cost_review_sha256':digest(review_path),
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
        'asm_construction_review_pending':True,'performance_measurement':False,'accuracy_cost_ranking_established':False}
    status = 1
    manifest_path.write_text(json.dumps(state,indent=2)+'\n')
    with transcript.open('w') as log:
        try:
            for name,argv in steps:
                run_logged(argv,ROOT,env,log)
                state['completed_steps'].append(name)
            state['study_validation'] = verify_complete(json.loads(Path(str(stem)+'-serial.json').read_text()),config)
            state['status'] = 'Standard ASM/local ILU construction checks passed; mentor review pending'
            status = 0
            message = 'ASM CONSTRUCTION CHECKPOINT PASSED. Review before integrating ASM into timed trajectories.'
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
        stem = ROOT/'results/raw/asm-construction-<UTC timestamp>-<process id>'
        for name,argv in commands(parameters[4],stem):
            print(f'{name}: {shlex.join(argv)}')
        print('Display only: no numerical imports, assembly, solves, time stepping, or MPI launch run.')
        return 0
    return run(*parameters)


if __name__=='__main__':
    try:
        sys.exit(main())
    except (RuntimeError,OSError,subprocess.SubprocessError,KeyError,ValueError) as error:
        print(f'STOP before ASM construction: {error}',file=sys.stderr)
        sys.exit(1)

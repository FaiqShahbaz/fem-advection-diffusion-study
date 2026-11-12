#!/usr/bin/env python3
"""Audit saved three-point first-step solver sensitivity and capped-solve evidence."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
STEM=ROOT/'results/raw/solver-sensitivity-20261021T191746Z-23418'
METRICS=('l2_error','h1_seminorm_error')

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def audit():
    sys.path.insert(0,str(ROOT/'environment'))
    import check_solver_sensitivity as runner
    runner.plan()
    m=json.loads(Path(str(STEM)+'-manifest.json').read_text());raw=ROOT/m['measurement_records']['serial']['path']
    d=json.loads(raw.read_text());cfg=d['config']
    assert m['checkpoint_exit_status']==0 and m['working_tree_status_before_run']=='' and m['completed_steps']==['serial']
    assert m['performance_measurement'] is False and m['accuracy_cost_ranking_established'] is False and m['inherited_options_or_library_settings']=={}
    assert m['selected_environment']['OMP_NUM_THREADS']==m['selected_environment']['OPENBLAS_NUM_THREADS']=='1'
    for key in ('rana_cost_review','verified_environment','config','installed_source_inspection'):assert digest(ROOT/m[key])==m[key+'_sha256'],key
    for path,expected in m['script_sha256'].items():
        assert digest(ROOT/path)==expected,path
        assert hashlib.sha256(subprocess.check_output(['git','show',m['code_commit']+':'+path],cwd=ROOT)).hexdigest()==expected,path
    for path,expected in m['inspected_installed_source_sha256'].items():assert digest(ROOT/path)==expected,path
    assert digest(raw)==m['measurement_records']['serial']['sha256'] and cfg==json.loads(runner.CONFIG.read_text())
    assert runner.verify_complete(d,cfg)==m['study_validation']
    extrema=dict(max_accepted_true_residual_over_nominal_target=0.,max_field_difference=0.,max_quadrature_change=0.,max_matrix_frobenius_defect=0.)
    def residual(row,rtol,atol):
        r=row['true_block_residual'];target=max(atol,rtol*r['rhs_norm'])
        assert all(math.isfinite(v) and v>=0 for v in r.values()) and r['rhs_norm']>0 and target==row['true_residual_target']
        assert math.isclose(r['relative_norm'],r['norm']/r['rhs_norm'],rel_tol=1e-14) and r['norm']<=10*target
        extrema['max_accepted_true_residual_over_nominal_target']=max(extrema['max_accepted_true_residual_over_nominal_target'],r['norm']/target)
        assert 0<=row['update_identity_l2']<=1e-12 and 0<=row['stiff_accuracy_l2']<=1e-12 and all(0<=v<=1e-12 for v in row['stage_boundary_trace_l2'])
        assert row['stage_times']==[.1/3,.1]
    def norms(row):
        assert all(row['checks'].values())
        for values in (row['metrics'],row['control_metrics']):
            assert all(math.isfinite(v) and v>=0 for v in values.values()) and abs(values['domain_area']-1)<=1e-12 and values['boundary_trace_l2']<=1e-12
            assert math.isclose(values['exact_l2_norm'],math.exp(-.1)/2,rel_tol=1e-10,abs_tol=1e-12)
            assert math.isclose(values['exact_h1_seminorm'],math.pi*math.exp(-.1)/math.sqrt(2),rel_tol=1e-10,abs_tol=1e-12)
        for metric in METRICS:
            q=row['quadrature_comparison'][metric];change=abs(row['metrics'][metric]-row['control_metrics'][metric])
            assert q['absolute_difference']==change and q['allowed_difference']==1e-13+1e-6*row['control_metrics'][metric] and change<=q['allowed_difference']
            extrema['max_quadrature_change']=max(extrema['max_quadrature_change'],change)
    def matrix(row):
        assert row['status']=='passed' and 0<=row['frobenius_defect']<=row['allowed_defect']==1e-11+1e-12*row['reference_frobenius_norm']
        extrema['max_matrix_frobenius_defect']=max(extrema['max_matrix_frobenius_defect'],row['frobenius_defect'])
    summary={};caps=[]
    for point in cfg['points']:
        name=point['point_id'];row=d['points'][name];meta=row['metadata'];n=point['mesh_subdivisions'];kappa=point['diffusion'];h=math.sqrt(2)/n
        assert meta['total_dofs']==(2*n+1)**2 and meta['free_dofs']==(2*n-1)**2 and meta['stage_total_dofs']==2*(2*n+1)**2 and meta['cells']==2*n*n
        assert meta['h_max']==h and meta['diffusion']==kappa and math.isclose(meta['cell_peclet'],1/(n*kappa),rel_tol=1e-14)
        assert math.isclose(meta['advective_step_ratio'],n*.1,rel_tol=1e-14) and math.isclose(meta['diffusive_step_ratio'],kappa*.1*n*n/2,rel_tol=1e-14)
        assert all(s['source_residual_l2']<=1e-10 for s in row['source_checks'])
        ref=row['reference'];residual(ref,cfg['reference_residual_rtol'],cfg['reference_residual_atol']);norms(ref)
        cases={}
        for case,r in row['candidates'].items():
            matrix(r['physical_matrix_check']);history=r['reported_residual_history']
            assert history[0][0]==0 and history[-1][0]==r['iterations'] and all(math.isfinite(v) and v>=0 for _,v in history)
            events=r['operator_event_counts'];assert events['KSPSolve']==events['KSPSetUp']==1
            basic=dict(status=r['status'],iterations=r['iterations'],converged_reason=r['converged_reason'],physical_event_counts=events)
            if r['status']=='iteration_limit':
                failed=r['failed_step'];correction=failed['petsc_correction_residual']
                assert name=='mesh_refined' and case=='Jacobi' and r['iterations']==1000 and r['converged_reason']==-3
                assert r['completed_steps']==0 and r['step_committed'] is False and failed['step_committed'] is False and r['initial_state_change_l2']==0.
                assert 'metrics' not in r and 'reference_comparisons' not in r and 'true_block_residual' not in r
                assert failed['reported_residual_norm']==history[-1][1] and 'DIVERGED_ITS' in failed['error']
                assert math.isclose(correction['relative_norm'],correction['norm']/correction['rhs_norm'],rel_tol=1e-14)
                assert correction['norm']>correction['nominal_target'] and correction['nominal_target']==max(1e-13,1e-11*correction['rhs_norm'])
                assert all(v==0 for v in r['failure_diagnostic_event_counts'].values())
                caps.append(dict(point_id=name,solver=case,iterations=1000,reason=-3,reported_residual_norm=history[-1][1],correction_residual=correction,
                    accepted_physical_solution=False,physical_update_change_l2=0.))
                basic['accepted_physical_solution']=False;cases[case]=basic;continue
            residual(r,cfg['solver_parameters']['ksp_rtol'],cfg['solver_parameters']['ksp_atol']);norms(r)
            for item in r['reference_comparisons'].values():
                measured=item['difference_measurement'];assert item['status']==measured['status']=='passed'
                for metric,f in item['norms'].items():
                    value=measured['metrics'][metric];q=measured['quadrature_comparison'][metric]
                    assert f['actual_field_difference']==value and f['allowed_difference']==1e-11+1e-8*f['reference_field_norm'] and value<=f['allowed_difference']
                    assert q['absolute_difference']==abs(value-measured['primary_metrics'][metric]) and q['allowed_difference']==1e-13+1e-6*value and q['absolute_difference']<=q['allowed_difference']
                    extrema['max_field_difference']=max(extrema['max_field_difference'],value)
            if case=='ASM':
                domains=r['domains'];assert sorted(i for v in domains for i in v['core_indices'])==list(range(meta['stage_total_dofs']))
                for domain,local in zip(domains,r['local_solvers']):
                    assert domain['one_layer_check'] and len(domain['core_indices'])==domain['core_size'] and len(domain['overlap_indices'])==domain['overlap_size']
                    assert set(domain['core_indices'])<=set(domain['overlap_indices']) and local['matrix_size']==[domain['overlap_size']]*2
                    assert local['ksp_type']=='preonly' and local['pc_type']=='ilu' and local['factor_solver_type']=='petsc' and local['factor_nonzeros']>0
                assert r['local_work_derived']['per_domain']==events['PCApply'] and r['local_work_derived']['total']==4*events['PCApply']==events['MatSolve']
                probe=r['additive_action_check'];assert probe['defect_norm']<=probe['allowed_defect']==1e-12+1e-11*probe['reference_action_norm']
                assert probe['actual_global_PC_applications']==1 and probe['independent_direct_local_PC_applications']==4 and r['diagnostic_event_counts']['MatSolve']==8
            elif case=='RanaLD':
                for value in r['auxiliary_matrix_checks'].values():matrix(value)
                probe=r['auxiliary_action_check'];assert probe['relative_residual']==probe['residual_norm']/probe['rhs_norm'] and probe['residual_norm']<=probe['allowed_residual']==1e-11+1e-10*probe['rhs_norm']
                assert events['MatSolve']==events['PCApply'] and events['MatLUFactorSym']==events['MatLUFactorNum']==1
                assert r['diagnostic_event_counts']['MatSolve']==1 and r['auxiliary_solver']['pc_type']=='lu' and r['auxiliary_solver']['factor_solver_type']=='petsc'
                basic['auxiliary_factor_nonzeros']=r['auxiliary_solver']['factor_nonzeros']
            basic.update(accepted_physical_solution=True,true_block_residual=r['true_block_residual'],update_reference_distance={metric:r['reference_comparisons']['update']['norms'][metric]['actual_field_difference'] for metric in METRICS})
            cases[case]=basic
        summary[name]=dict(metadata=meta,reference_residual=ref['true_block_residual'],reference_factor_nonzeros=ref['factor_nonzeros'],candidates=cases)
    assert len(caps)==d['capped_candidates']==1 and d['converged_candidates']==8 and d['candidate_attempts']==9 and d['reference_solves']==3
    growth=summary['mesh_refined']['candidates']['RanaLD']['auxiliary_factor_nonzeros']/summary['baseline']['candidates']['RanaLD']['auxiliary_factor_nonzeros']
    return dict(status='passed',reference_solves=3,candidate_attempts=9,accepted_candidate_solutions=8,preserved_iteration_limit_outcomes=caps,
        point_summary=summary,rana_auxiliary_factor_nnz_mesh_ratio=growth,extrema=extrema,performance_measurement=False,
        limits=['Only the first interval [0,0.1]; no full T=1 or elapsed-cost conclusion.',
            'One finer mesh and one lower diffusion; no mesh-independent iteration bound or broad robustness proof.',
            'Rana applies full auxiliary sparse LU; constant iteration count does not imply constant storage, time or scalable subsolves.',
            'Saved arithmetic and hashes rechecked without numerical imports; actual matrix/graph/field/probe operations remain researcher runtime evidence.'])

if __name__=='__main__':print(json.dumps(audit(),indent=2,allow_nan=False))

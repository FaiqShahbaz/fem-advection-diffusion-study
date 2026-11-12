#!/usr/bin/env python3
"""Audit saved selected full trajectories, actual-field checks and fixed-factor reuse without FEM imports."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
STEM=ROOT/'results/raw/selected-trajectories-20261021T193925Z-25133'
METRICS=('l2_error','h1_seminorm_error')
EVENTS=('KSPSolve','KSPSetUp','PCSetUp','PCApply','MatSolve','MatLUFactorSym','MatLUFactorNum')

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def audit():
    sys.path.insert(0,str(ROOT/'environment'))
    import check_selected_trajectories as runner
    runner.plan()
    m=json.loads(Path(str(STEM)+'-manifest.json').read_text());raw=ROOT/m['measurement_records']['serial']['path'];d=json.loads(raw.read_text());cfg=d['config']
    assert m['checkpoint_exit_status']==0 and m['working_tree_status_before_run']=='' and m['completed_steps']==['serial']
    assert m['performance_measurement'] is False and m['accuracy_cost_ranking_established'] is False and m['inherited_options_or_library_settings']=={}
    assert m['selected_environment']['OMP_NUM_THREADS']==m['selected_environment']['OPENBLAS_NUM_THREADS']=='1'
    for key in ('solver_sensitivity_review','verified_environment','config','installed_source_inspection'):assert digest(ROOT/m[key])==m[key+'_sha256'],key
    for path,expected in m['script_sha256'].items():
        assert digest(ROOT/path)==expected,path
        assert hashlib.sha256(subprocess.check_output(['git','show',m['code_commit']+':'+path],cwd=ROOT)).hexdigest()==expected,path
    for path,expected in m['inspected_installed_source_sha256'].items():assert digest(ROOT/path)==expected,path
    assert digest(raw)==m['measurement_records']['serial']['sha256'] and cfg==json.loads(runner.CONFIG.read_text())
    assert runner.verify_complete(d,cfg)==m['study_validation']
    extrema=dict(max_true_residual_over_nominal_target=0.,max_update_defect=0.,max_field_difference=0.,max_field_quadrature_change=0.,max_matrix_frobenius_defect=0.)
    def final_norms(r):
        assert r['status']=='passed' and r['final_time']==1. and all(r['checks'].values())
        for values in (r['metrics'],r['control_metrics']):
            assert all(math.isfinite(v) and v>=0 for v in values.values()) and abs(values['domain_area']-1)<=1e-12 and values['boundary_trace_l2']<=1e-12
            assert math.isclose(values['exact_l2_norm'],math.exp(-1)/2,rel_tol=1e-10,abs_tol=1e-12)
            assert math.isclose(values['exact_h1_seminorm'],math.pi*math.exp(-1)/math.sqrt(2),rel_tol=1e-10,abs_tol=1e-12)
        for metric in METRICS:
            q=r['quadrature_comparison'][metric];change=abs(r['metrics'][metric]-r['control_metrics'][metric])
            assert q['absolute_difference']==change and q['allowed_difference']==1e-13+1e-6*r['control_metrics'][metric] and change<=q['allowed_difference']
    def steps(r,reference=False):
        final_norms(r);first=r['step_records'][0]['matrix_reuse_tokens'];summed=dict(r['construction_event_counts'])
        for index,s in enumerate(r['step_records']):
            residual=s['true_block_residual'];rtol=cfg['reference_residual_rtol'] if reference else r['solver_parameters']['ksp_rtol'];atol=cfg['reference_residual_atol'] if reference else r['solver_parameters']['ksp_atol']
            target=max(atol,rtol*residual['rhs_norm']);assert target==s['true_residual_target'] and residual['rhs_norm']>0
            assert math.isclose(residual['relative_norm'],residual['norm']/residual['rhs_norm'],rel_tol=1e-14) and 0<=residual['norm']<=10*target
            extrema['max_true_residual_over_nominal_target']=max(extrema['max_true_residual_over_nominal_target'],residual['norm']/target)
            assert s['matrix_reuse_tokens']==first and s['reuse_matches_first_step']
            assert s['stage_times']==[index*.1+.1/3,index*.1+.1]
            assert 0<=s['update_identity_l2']<=1e-12 and 0<=s['stiff_accuracy_l2']<=1e-12 and all(0<=v<=1e-12 for v in s['stage_boundary_trace_l2'])
            extrema['max_update_defect']=max(extrema['max_update_defect'],s['update_identity_l2'],s['stiff_accuracy_l2'])
            assert all(0<=v['source_residual_l2']<=1e-10 for v in s['source_checks'])
            if index:assert all(s['event_counts'][e]==0 for e in ('PCSetUp','MatLUFactorSym','MatLUFactorNum'))
            for e in EVENTS:summed[e]+=s['event_counts'][e]
            for value in s.get('matrix_checks',{}).values():
                assert value['status']=='passed' and 0<=value['frobenius_defect']<=value['allowed_defect']==1e-11+1e-12*value['reference_frobenius_norm']
                extrema['max_matrix_frobenius_defect']=max(extrema['max_matrix_frobenius_defect'],value['frobenius_defect'])
            for item in s.get('reference_comparisons',{}).values():
                measured=item['difference_measurement'];assert item['status']==measured['status']=='passed'
                for metric,f in item['norms'].items():
                    value=measured['metrics'][metric];q=measured['quadrature_comparison'][metric]
                    assert f['actual_field_difference']==value and f['allowed_difference']==1e-11+1e-8*f['reference_field_norm'] and 0<=value<=f['allowed_difference']
                    assert q['absolute_difference']==abs(value-measured['primary_metrics'][metric]) and q['allowed_difference']==1e-13+1e-6*value and q['absolute_difference']<=q['allowed_difference']
                    extrema['max_field_difference']=max(extrema['max_field_difference'],value);extrema['max_field_quadrature_change']=max(extrema['max_field_quadrature_change'],q['absolute_difference'])
        assert summed==r['operator_event_counts'] and r['completed_steps']==len(r['step_records'])==10
    summary={};skips=[]
    for point in cfg['points']:
        name=point['point_id'];p=d['points'][name];n=point['mesh_subdivisions'];meta=p['metadata'];kappa=point['diffusion']
        assert meta['cells']==2*n*n and meta['total_dofs']==(2*n+1)**2 and meta['free_dofs']==(2*n-1)**2 and meta['stage_total_dofs']==2*(2*n+1)**2
        assert meta['diffusion']==kappa and meta['h_max']==math.sqrt(2)/n and math.isclose(meta['cell_peclet'],1/(n*kappa),rel_tol=1e-14)
        assert math.isclose(meta['advective_step_ratio'],n*.1,rel_tol=1e-14) and math.isclose(meta['diffusive_step_ratio'],kappa*.1*n*n/2,rel_tol=1e-14)
        ref=p['reference'];steps(ref,True);assert ref['matrix_assemblies']==1
        assert ref['operator_event_counts']['KSPSolve']==ref['operator_event_counts']['MatSolve']==10 and ref['operator_event_counts']['MatLUFactorSym']==ref['operator_event_counts']['MatLUFactorNum']==1
        cases={}
        for case,r in p['candidates'].items():
            if r['status']=='not_attempted':
                assert name=='mesh_refined' and case=='Jacobi' and r['accepted_final_solution'] is False and r['completed_steps']==0 and 'metrics' not in r
                assert digest(ROOT/r['evidence']['path'])==r['evidence']['sha256'] and r['pilot_iterations']==1000 and r['pilot_converged_reason']==-3
                cases[case]=dict(status='not_attempted',reason=r['reason'],pilot_iterations=1000,pilot_reason=-3,accepted_final_solution=False);skips.append(dict(point_id=name,solver=case,**cases[case]));continue
            steps(r);events=r['operator_event_counts'];assert events['KSPSolve']==events['KSPSetUp']==10
            assert r['total_iterations']==sum(s['iterations'] for s in r['step_records']) and r['solver_stats']==[10,10,r['total_iterations']]
            for s in r['step_records']:
                h=s['reported_residual_history'];assert h[0][0]==0 and h[-1][0]==s['iterations'] and all(math.isfinite(v) and v>=0 for _,v in h)
            basic=dict(status='passed',completed_steps=10,total_iterations=r['total_iterations'],step_iterations=[s['iterations'] for s in r['step_records']],physical_event_counts=events,
                final_mms_errors={metric:r['control_metrics'][metric] for metric in METRICS},final_reference_distances={metric:r['step_records'][-1]['reference_comparisons']['update']['norms'][metric]['actual_field_difference'] for metric in METRICS},
                reuse_verified=True)
            if case=='ASM':
                domains=r['domains'];assert sorted(i for v in domains for i in v['core_indices'])==list(range(meta['stage_total_dofs']))
                for domain,local in zip(domains,r['local_solvers']):
                    assert domain['one_layer_check'] and len(domain['core_indices'])==domain['core_size'] and len(domain['overlap_indices'])==domain['overlap_size']
                    assert set(domain['core_indices'])<=set(domain['overlap_indices']) and local['matrix_size']==[domain['overlap_size']]*2
                    assert local['ksp_type']=='preonly' and local['pc_type']=='ilu' and local['factor_solver_type']=='petsc'
                assert events['MatSolve']==4*events['PCApply'] and events['MatLUFactorSym']==0 and events['MatLUFactorNum']==4
                for s in r['step_records']:assert s['local_work_derived']['total']==4*s['local_work_derived']['per_domain']==s['event_counts']['MatSolve']
                probe=r['additive_action_check'];assert 0<=probe['defect_norm']<=probe['allowed_defect']==1e-12+1e-11*probe['reference_action_norm']
                assert r['diagnostic_event_counts']==dict(KSPSolve=4,KSPSetUp=0,PCSetUp=0,PCApply=5,MatSolve=8,MatLUFactorSym=0,MatLUFactorNum=0)
                basic['local_factor_nonzeros']=[s['factor_nonzeros'] for s in r['local_solvers']]
            elif case=='RanaLD':
                assert events['MatSolve']==events['PCApply'] and events['MatLUFactorSym']==events['MatLUFactorNum']==1
                probe=r['auxiliary_action_check'];assert 0<=probe['residual_norm']<=probe['allowed_residual']==1e-11+1e-10*probe['rhs_norm']
                assert math.isclose(probe['relative_residual'],probe['residual_norm']/probe['rhs_norm'],rel_tol=1e-14)
                assert r['diagnostic_event_counts']==dict(KSPSolve=0,KSPSetUp=0,PCSetUp=0,PCApply=1,MatSolve=1,MatLUFactorSym=0,MatLUFactorNum=0)
                basic['auxiliary_factor_nonzeros']=r['auxiliary_solver']['factor_nonzeros']
            cases[case]=basic
        summary[name]=dict(metadata=meta,reference_factor_nonzeros=ref['factor_nonzeros'],reference_mms_errors={metric:ref['control_metrics'][metric] for metric in METRICS},candidates=cases)
    assert d['completed_candidate_trajectories']==8 and d['capped_candidate_trajectories']==0 and d['candidate_completed_steps']==80 and len(skips)==1
    return dict(status='passed',reference_trajectories=3,completed_candidate_trajectories=8,completed_steps=110,new_capped_candidates=0,
        predeclared_exclusions=skips,point_summary=summary,extrema=extrema,performance_measurement=False,
        limits=['Three selected points and dt=0.1, T=1; no broad mesh/diffusion robustness proof.',
            'The earlier finer-mesh Jacobi cap remains a negative result; no full trajectory or cost accepted for that case.',
            'Rana full auxiliary sparse LU factor storage grows with mesh; low iterations do not establish scalable subsolves.',
            'MMS errors combine spatial, temporal and algebraic effects here; no new order or accuracy/cost claim.',
            'Saved hashes/arithmetic checked without numerical imports; field/matrix/PC operations are researcher runtime evidence.'])

if __name__=='__main__':print(json.dumps(audit(),indent=2,allow_nan=False))

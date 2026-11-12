#!/usr/bin/env python3
"""Preserve the first Rana setup failure using historical sources and raw bytes."""
import hashlib
import json
import math
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
STEM=ROOT/'results/raw/rana-construction-20261021T104551Z-17748'

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def audit():
    m=json.loads(Path(str(STEM)+'-manifest.json').read_text())
    raw=ROOT/m['measurement_records']['serial']['path'];d=json.loads(raw.read_text())
    assert m['checkpoint_exit_status']==1 and m['working_tree_status_before_run']=='' and m['completed_steps']==[]
    assert d['status']=='failed' and d['completed_study'] is False and d['performance_measurement'] is False
    assert digest(raw)==m['measurement_records']['serial']['sha256']
    for key in ('asm_cost_review','verified_environment','config','installed_source_inspection'):
        assert digest(ROOT/m[key])==m[key+'_sha256'],key
    for path,expected in m['script_sha256'].items():
        historical=subprocess.check_output(['git','show',m['code_commit']+':'+path],cwd=ROOT)
        assert hashlib.sha256(historical).hexdigest()==expected,path
        # These two unaccepted drivers are repaired after failure; original
        # bytes remain recoverable at the manifest's historical commit.
        if path not in ('src/phase3_rana_construction.py','environment/check_rana_construction.py'):
            assert digest(ROOT/path)==expected,path
    for path,expected in m['inspected_installed_source_sha256'].items():assert digest(ROOT/path)==expected,path
    baseline=d['solves']['irksome_jacobi'];reference=d['solves']['manual_jacobi_reference'];rana=d['solves']['rana_ld']
    assert baseline['status']=='passed' and baseline['completed_steps']==1 and baseline['total_iterations']==734
    assert all(baseline['checks'].values()) and baseline['solver_stats']==[1,1,734]
    assert reference['status']=='passed' and reference['completed_operator_solves']==1 and reference['iterations']==800
    completed={}
    for name,row,r in [('irksome_jacobi',baseline,baseline['step_records'][0]),('manual_jacobi_reference',reference,reference)]:
        residual=r['true_block_residual'];p=row['solver_parameters'];target=max(p['ksp_atol'],p['ksp_rtol']*residual['rhs_norm'])
        assert all(math.isfinite(v) and v>=0 for v in residual.values()) and residual['rhs_norm']>0
        assert target==r['true_residual_target'] and residual['norm']<=10*target and r['converged_reason']>0
        assert math.isclose(residual['relative_norm'],residual['norm']/residual['rhs_norm'],rel_tol=1e-14)
        completed[name]=dict(iterations=r['iterations'],true_block_residual=residual,nominal_target=target)
    failed=rana['failed_step']
    assert d['active_solve']=='rana_ld' and rana['status']=='failed' and rana['completed_steps']==0
    assert failed['iterations']==failed['converged_reason']==0 and failed['step_committed'] is False
    assert failed['reported_residual_history']==[] and 'petsc_correction_residual' not in failed
    text=Path(str(STEM)+'.txt').read_text()
    assert 'v0, = F.arguments()' in text and 'ValueError: too many values to unpack (expected 1)' in text
    assert 'PCSetUp()' in text and failed['diagnostic_error'].startswith('error code 91')
    assert not d['comparisons'] and 'matrix_checks' not in rana and 'auxiliary_action_check' not in rana
    return dict(status='preserved failed RanaLD setup; no Rana numerical result accepted',
        historical_source_commit=m['code_commit'],completed_physical_reference_solves=completed,
        failure_phase='Python PC setup before first GMRES iteration',root_cause='Two-argument trial/test weak form passed to installed RanaLD one-test-argument residual interface',
        rana_completed_steps=0,rana_outer_iterations=0,failed_iterate_accepted=False,
        secondary_diagnostic='KSPBuildResidual failed because setup had not initialized the Krylov basis; no physical residual was obtained',
        numerical_environment_modified=False,configuration_changed=False,
        audit_limit='Historical source/raw hashes and saved partial records audited without numerical imports or rerunning operations')

if __name__=='__main__':print(json.dumps(audit(),indent=2,allow_nan=False))

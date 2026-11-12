#!/usr/bin/env python3
"""Check the report's saved evidence, exact tableau algebra and artifact hashes; never run FEM."""
import argparse
from fractions import Fraction
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys

from prepare_report import ROOT,PROJECT,collect,FIGURES,digest

class NoNumericalImports(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'firedrake','irksome','petsc4py','mpi4py','numpy','scipy','h5py'}:
            raise RuntimeError('Numerical imports are forbidden in this saved-evidence audit: '+fullname)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--inputs-only',action='store_true',help='Check evidence before rebuilding an existing report artifact.');args=parser.parse_args()
    sys.meta_path.insert(0,NoNumericalImports())
    from audit_selected_cost import audit
    review_path=ROOT/'results/processed/selected_cost_review.json';review=json.loads(review_path.read_text())
    assert digest(ROOT/review['audit_source']['path'])==review['audit_source']['sha256']
    result=audit();assert result==review['independent_audit']
    for key in ('cost_manifest','cost_transcript','config','selected_trajectories_review','verified_environment','installed_source_inspection'):
        assert digest(ROOT/review[key])==review[key+'_sha256'],key
    text,ledger=collect();assert (PROJECT/'evidence/data.tex').read_text()==text
    assert json.loads((PROJECT/'evidence/claims.json').read_text())==ledger
    for name,path in FIGURES.items():assert digest(PROJECT/'images'/name)==digest(ROOT/path),name
    cfg=json.loads((ROOT/review['config']).read_text());tableau=cfg['radau_formulation']['tableau_rational']
    A=[[Fraction(v) for v in row] for row in tableau['A']];b=[Fraction(v) for v in tableau['b']];c=[Fraction(v) for v in tableau['c']]
    assert c==[sum(row) for row in A] and b==A[-1] and sum(b)==1
    assert sum(bi*ci for bi,ci in zip(b,c))==Fraction(1,2) and sum(bi*ci**2 for bi,ci in zip(b,c))==Fraction(1,3)
    assert sum(b[i]*sum(A[i][j]*c[j] for j in range(2)) for i in range(2))==Fraction(1,6)
    factors=cfg['rana_tableau_rational'];mat=lambda key:[[Fraction(v) for v in row] for row in factors[key]]
    multiply=lambda X,Y:[[sum(X[i][k]*Y[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
    assert multiply(multiply(mat('L'),mat('D')),mat('U'))==A and multiply(mat('L'),mat('D'))==mat('Atilde')
    # Scalar stability numerator/denominator from the actual 2x2 tableau.
    trace=A[0][0]+A[1][1];det=A[0][0]*A[1][1]-A[0][1]*A[1][0]
    assert trace==Fraction(2,3) and det==Fraction(1,6)
    assert -trace+sum(b)==Fraction(1,3)
    assert det+b[0]*(A[0][1]-A[1][1])+b[1]*(A[1][0]-A[0][0])==0
    artifact=None
    if not args.inputs_only:
        build=json.loads((PROJECT/'evidence/report-build.json').read_text());artifact=build['pdf']
        assert 8<=artifact['pages']<=12 and digest(ROOT/artifact['path'])==artifact['sha256']
        for path,expected in build['input_sha256'].items():assert digest(ROOT/path)==expected,path
        assert build['numerical_experiment_run'] is False
    print(json.dumps(dict(status='passed',saved_numerical_trajectories_audited=result['completed_trajectories'],saved_steps_audited=result['completed_steps'],
        report_claims=len(ledger['claims']),report_figures=len(FIGURES),exact_Radau_order_three_and_LDU_identities=True,
        report_artifact=artifact,numerical_modules_imported=False,numerical_experiments_run=False),indent=2))

if __name__=='__main__':main()

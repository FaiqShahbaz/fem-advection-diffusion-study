#!/usr/bin/env python3
"""Prepare report tables, claim ledger and figure copies from accepted evidence; no numerical imports."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT/'report/technical_report'
REVIEWS=('spatial','temporal','accuracy_cost','solver_sensitivity','selected_trajectories','selected_cost')
FIGURES={
 'spatial.png':'results/figures/spatial-20261008T133314Z-38837.png',
 'temporal.png':'results/figures/temporal-20261018T113908Z-75912.png',
 'accuracy_cost.png':'results/figures/accuracy-cost-20261019T102618Z-70377.png',
 'solver_work.png':'results/figures/selected-trajectories-20261021T193925Z-25133.png',
 'selected_cost.png':'results/figures/selected-cost-20261022T095152Z-28440.png'}

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def sci(value):
    a,b=f'{value:.3e}'.split('e');return '$'+a+r'\times10^{'+str(int(b))+'}$'

def collect():
    reviews={name:json.loads((ROOT/f'results/processed/{name}_review.json').read_text()) for name in REVIEWS}
    assert reviews['selected_cost']['selected_cost_verified'] and reviews['temporal']['temporal_order_verified']
    ledger=dict(status='completed serial study; report revised 12 November 2026',generated_from_accepted_evidence_only=True,new_numerical_experiment_run=False,
        evidence_reviews={name:dict(path=f'results/processed/{name}_review.json',sha256=digest(ROOT/f'results/processed/{name}_review.json')) for name in REVIEWS},
        verified_environment=dict(path='environment/verified_environment.json',sha256=digest(ROOT/'environment/verified_environment.json')),
        figures={name:dict(path=path,sha256=digest(ROOT/path)) for name,path in FIGURES.items()},claims={})
    commands=[]
    def macro(name,value,review,pointer):
        commands.append("\\newcommand{\\"+name+'}{'+str(value)+'}')
        ledger['claims'][name]=dict(rendered_value=value,review=review,source_selector=pointer)
    spatial=reviews['spatial'];rows=[]
    for degree in (1,2):
        rates=[next(r['observed_order'] for r in spatial['rates'] if r['element_degree']==degree and r['metric']==metric and r['fine_case']==f'P{degree}-n32') for metric in ('l2_error','h1_seminorm_error')]
        rows.append(f'P{degree} & {degree+1} / {degree} & {rates[0]:.6f} & {rates[1]:.6f} '+r'\\')
    macro('SpatialRateRows','\n'.join(rows),'spatial','/rates (finest n16-to-n32 pairs)')
    temporal=reviews['temporal'];rows=[]
    for method,label in [('manual_backward_euler','Backward Euler'),('RadauIIA','Radau IIA(2)')]:
        rates=[next(r['observed_order'] for r in temporal['rates'] if r['method']==method and r['metric']==metric and r['fine_dt']==.0125) for metric in ('l2_error','h1_seminorm_error')]
        rows.append(f'{label} & {rates[0]:.6f} & {rates[1]:.6f} '+r'\\')
    macro('TemporalRateRows','\n'.join(rows),'temporal','/rates (finest dt pair)')
    accuracy=reviews['accuracy_cost']['independent_audit'];config=json.loads((ROOT/reviews['accuracy_cost']['config']).read_text());rows=[]
    for case in config['cases']:
        name=case['case_id'];s=accuracy['candidate_summary'][name]
        label=name.replace('_',r'\_');rows.append(f"{label} & {case['dt']:g} & {sci(s['time_errors']['l2_error'])} & {s['wall']['median']:.6f} "+r'\\')
    macro('AccuracyCostRows','\n'.join(rows),'accuracy_cost','/independent_audit/candidate_summary; config/cases')
    ratio=accuracy['target_median_ratios']['3e-06']['BE_over_Radau_median'];macro('TightAccuracyRatio',f'{ratio:.3f}','accuracy_cost','/independent_audit/target_median_ratios/3e-06/BE_over_Radau_median')
    audit=reviews['selected_cost']['independent_audit'];summary=audit['case_summary'];rows=[]
    baseline=audit['checked_study']['point_summary']['baseline']['candidates']
    work_rows=[label+' & '+str(baseline[key]['total_iterations'])+' '+r'\\' for key,label in
               [('Jacobi','Jacobi'),('ASM','ASM/ILU(0)'),('RanaLD','RanaLD/auxiliary LU')]]
    macro('BaselineIterationRows','\n'.join(work_rows),'selected_cost',
          '/independent_audit/checked_study/point_summary/baseline/candidates/*/total_iterations')
    macro('BaselineJacobiRanaRatio',f"{audit['within_point_ratios']['baseline']['jacobi_over_rana_median_core_ratio']:.3f}",
          'selected_cost','/independent_audit/within_point_ratios/baseline/jacobi_over_rana_median_core_ratio')
    for point,label in [('baseline','16, 1'),('mesh_refined','32, 1'),('lower_diffusion','16, 0.1')]:
        for solver in ('Jacobi','ASM','RanaLD'):
            key=point+'_'+solver
            if key not in summary:
                rows.append(label+' & Jacobi & excluded & --- & --- '+r'\\');continue
            s=summary[key];v=s['core_seconds'];rows.append(f"{label} & {solver} & {v['median']:.6f} & {v['minimum']:.6f}--{v['maximum']:.6f} & {s['relative_min_max_range_percent']:.2f}\\% "+r'\\')
    macro('SelectedCostRows','\n'.join(rows),'selected_cost','/independent_audit/case_summary')
    for point,prefix in [('baseline','Baseline'),('mesh_refined','Refined'),('lower_diffusion','Lower')]:
        value=audit['within_point_ratios'][point]['asm_over_rana_median_core_ratio'];macro(prefix+'AsmRanaRatio',f'{value:.3f}','selected_cost',f'/independent_audit/within_point_ratios/{point}/asm_over_rana_median_core_ratio')
    return '\n'.join(commands)+'\n',ledger

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true');args=parser.parse_args()
    text,ledger=collect();evidence=PROJECT/'evidence'
    if args.check:
        assert (evidence/'data.tex').read_text()==text and json.loads((evidence/'claims.json').read_text())==ledger
        for name,path in FIGURES.items():assert digest(PROJECT/'images'/name)==digest(ROOT/path),name
        print('Report tables, claim ledger and figure copies match accepted evidence; no numerical imports.');return
    evidence.mkdir(parents=True,exist_ok=True);(PROJECT/'images').mkdir(exist_ok=True)
    (evidence/'data.tex').write_text(text);(evidence/'claims.json').write_text(json.dumps(ledger,indent=2,allow_nan=False)+'\n')
    for name,path in FIGURES.items():shutil.copyfile(ROOT/path,PROJECT/'images'/name)
    print('Prepared report tables, claim ledger and figure copies from accepted saved evidence only.')

if __name__=='__main__':main()

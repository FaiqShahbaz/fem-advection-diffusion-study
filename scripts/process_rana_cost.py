#!/usr/bin/env python3
"""Regenerate accepted matched residual-form Jacobi/RanaLD tables and figure."""
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
REVIEW=ROOT/'results/processed/rana_cost_review.json'

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    review=json.loads(REVIEW.read_text());assert review['rana_cost_verified']
    record=review['measurement_records']['serial'];raw=ROOT/record['path'];assert digest(raw)==record['sha256']
    data=json.loads(raw.read_text());assert data['status']=='passed'
    inventory=ROOT/'environment/requirements-postprocessing-resolved.txt'
    assert subprocess.check_output([sys.executable,'-m','pip','--no-cache-dir','freeze','--all'],text=True)==inventory.read_text()
    stem=raw.name.removesuffix('-serial.json');table_stem=ROOT/'results/processed'/stem;fig_stem=ROOT/'results/figures'/stem
    rows=[dict(trajectory=name,case_id=row['case_id'],total_iterations=row['total_iterations'],timing=row['timing'],
        equivalence=row['equivalence'],auxiliary_solver=row.get('auxiliary_solver')) for name,row in data['trajectories'].items() if row['role']=='measured']
    table=dict(raw_source=record,accepted_review=str(REVIEW.relative_to(ROOT)),rows=rows,summary=data['timing_summary'],
        jacobi_over_rana_median_core_ratio=data['jacobi_over_rana_median_core_ratio'],power='AC confirmed before/after',
        timing_scope=review['timing_scope'],timing_exclusions=review['timing_exclusions'],ranking_scope=review['ranking_scope'])
    table_json,table_md=Path(str(table_stem)+'-table.json'),Path(str(table_stem)+'-table.md')
    table_json.write_text(json.dumps(table,indent=2,allow_nan=False)+'\n')
    lines=['# Matched residual-form Radau Jacobi/RanaLD timing observations','',
        'CG P2/n16, beta=(1,1), kappa=1, dt=0.1, T=1; one process/thread; AC confirmed before and after. Warm code/cache, fresh solver per trajectory with fixed within-trajectory reuse.','',
        '| PC / repeat | Construction (s) | First advance (s) | Later 9 advances (s) | Core wall (s) |',
        '|---|---:|---:|---:|---:|']
    for row in rows:
        t=row['timing'];lines.append(f"| {row['trajectory']} | {t['construction_seconds']:.6f} | {t['first_step_seconds']:.6f} | {t['remaining_steps_seconds']:.6f} | {t['trajectory_wall_seconds']:.6f} |")
    lines+=['','| PC | Median core (s) | Min–max (s) | Relative range | GMRES iterations | PC applications | Auxiliary LU solves | LU symbolic/numeric |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for case in ('Jacobi','RanaLD'):
        selected=next(r for r in rows if r['case_id']==case);t=data['timing_summary'][case]['trajectory_wall_seconds'];e=selected['timing']['petsc_event_deltas']
        spread=100*(t['maximum']-t['minimum'])/t['median']
        lines.append(f"| {case} | {t['median']:.6f} | {t['minimum']:.6f}–{t['maximum']:.6f} | {spread:.2f}% | {selected['total_iterations']} | {e['PCApply']['count']} | {e['MatSolve']['count']} | {e['MatLUFactorSym']['count']}/{e['MatLUFactorNum']['count']} |")
    lines+=['',f"Ratio of Jacobi/RanaLD median core costs: {data['jacobi_over_rana_median_core_ratio']:.6f}. RanaLD/auxiliary LU had lower measured core cost on this fixed problem, including fresh setup/factorization. No universal ranking is established.",
        'All three measured samples per case are retained. Warmups are stored separately and excluded from medians. These are observed ranges rather than confidence intervals.',
        'All eleven trajectories/110 steps passed. Each warmup/measured actual-field distance from its fresh same-PC checked field was recorded as zero in L2/H1. The distinct verified Jacobi/Rana fields agreed within the declared field budget.',
        'Each timed path recorded ten KSPSetUp events; Rana nevertheless recorded one symbolic/numeric LU factorization pair. KSP setup and factor rebuilding are different event categories.',
        'Rana physical work included 80 actual auxiliary MatSolve applications with 167432 factor nonzeros. Its final diagnostic probe added one MatSolve outside the timer/event delta and rebuilt no factors.',
        'PETSc event times are inclusive and nested; do not sum them into another total. Near-zero Jacobi PC setup times cannot support a setup speed ratio.',
        review['timing_scope']+'. Excludes: '+review['timing_exclusions']+'.',
        'Keep these residual-form Irksome costs distinct from the accepted manual ASM and older method-driver costs. Fixed full-auxiliary sparse LU is not a scalable multigrid stage-subsolve implementation.','',
        f'[Raw observations](../raw/{raw.name}); [accepted review](rana_cost_review.json).','']
    table_md.write_text('\n'.join(lines))
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'environment/build/plot-cache'));os.environ.setdefault('MPLBACKEND','Agg')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.hashsalt':stem,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(8.7,4.3),layout='constrained')
    for index,case in enumerate(('Jacobi','RanaLD')):
        wall=data['timing_summary'][case]['trajectory_wall_seconds'];values=wall['values'];color=('#1f5a99','#aa4f13')[index]
        ax.plot([min(values),max(values)],[index,index],color=color,lw=2)
        ax.scatter(values,[index-.07,index,index+.07],s=40,color=color,zorder=3)
        ax.scatter([wall['median']],[index],s=270,marker='|',color='black',zorder=4)
        ax.annotate(f"median {wall['median']:.6f} s",(max(values),index),xytext=(12,0),textcoords='offset points',va='center')
    ax.set_yticks([0,1],['Jacobi','RanaLD / auxiliary LU']);ax.set_ylim(-.4,1.4)
    ax.set_xlim(0,1.33*max(data['timing_summary'][case]['trajectory_wall_seconds']['maximum'] for case in ('Jacobi','RanaLD')))
    ax.set_xlabel('Fresh construction + 10 advances: core wall time (s)');ax.grid(axis='x',alpha=.2)
    ax.set_title('Matched Radau PC timing on AC power — all three repeats')
    fig.supxlabel('CG P2/n16 • κ = 1 • Δt = 0.1 • T = 1 • one process/thread\nPoints: measured repeats; black tick: median. Warm residual-form Irksome driver; fixed-case comparison.',fontsize=9)
    png,svg=Path(str(fig_stem)+'.png'),Path(str(fig_stem)+'.svg')
    fig.savefig(png,dpi=200,metadata={'Software':f'Matplotlib {matplotlib.__version__}'});fig.savefig(svg,metadata={'Date':None,'Creator':f'Matplotlib {matplotlib.__version__}'})
    plt.close(fig)
    provenance=dict(raw_source=record,review=dict(path=str(REVIEW.relative_to(ROOT)),sha256=digest(REVIEW)),
        processor=dict(path=str(Path(__file__).resolve().relative_to(ROOT)),sha256=digest(Path(__file__))),
        plotting_inventory=dict(path=str(inventory.relative_to(ROOT)),sha256=digest(inventory)),python=sys.executable,python_version=platform.python_version(),
        matplotlib_version=matplotlib.__version__,regeneration_command=shlex.join([sys.executable,str(Path(__file__).resolve())]),
        output_sha256={str(p.relative_to(ROOT)):digest(p) for p in (table_json,table_md,png,svg)},processing_only=True,new_fem_experiment_run=False)
    Path(str(table_stem)+'-processing.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('Regenerated accepted matched Rana cost table and figure:',fig_stem)

if __name__=='__main__':main()

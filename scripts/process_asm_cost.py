#!/usr/bin/env python3
"""Regenerate accepted matched Jacobi/ASM timing tables and scientific figure."""
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
REVIEW=ROOT/'results/processed/asm_cost_review.json'

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    review=json.loads(REVIEW.read_text());assert review['asm_cost_verified']
    record=review['measurement_records']['serial'];raw=ROOT/record['path'];assert digest(raw)==record['sha256']
    data=json.loads(raw.read_text());assert data['status']=='passed'
    inventory_path=ROOT/'environment/requirements-postprocessing-resolved.txt'
    assert subprocess.check_output([sys.executable,'-m','pip','--no-cache-dir','freeze','--all'],text=True)==inventory_path.read_text()
    stem=raw.name.removesuffix('-serial.json');table_stem=ROOT/'results/processed'/stem;fig_stem=ROOT/'results/figures'/stem
    rows=[dict(trajectory=name,case_id=row['case_id'],total_iterations=row['total_iterations'],timing=row['timing'],
        equivalence=row['equivalence'],local_work_derived=row.get('local_work_derived')) for name,row in data['trajectories'].items() if row['role']=='measured']
    table=dict(raw_source=record,accepted_review=str(REVIEW.relative_to(ROOT)),rows=rows,
        summary=data['timing_summary'],jacobi_over_asm_median_core_ratio=data['jacobi_over_asm_median_core_ratio'],
        power='AC confirmed before/after',timing_scope=review['timing_scope'],timing_exclusions=review['timing_exclusions'],ranking_scope=review['ranking_scope'])
    table_json,table_md=Path(str(table_stem)+'-table.json'),Path(str(table_stem)+'-table.md')
    table_json.write_text(json.dumps(table,indent=2,allow_nan=False)+'\n')
    lines=['# Matched manual Radau Jacobi/ASM timing observations','',
        'CG P2/n16, beta=(1,1), kappa=1, dt=0.1, T=1; one process/thread; AC confirmed before and after. Warm code/cache, fresh solver/operator per trajectory with fixed within-trajectory reuse.','',
        '| PC / repeat | Construction (s) | First step (s) | Later 9 steps (s) | Core wall (s) |',
        '|---|---:|---:|---:|---:|']
    for row in rows:
        t=row['timing'];lines.append(f"| {row['trajectory']} | {t['construction_seconds']:.6f} | {t['first_step_seconds']:.6f} | {t['remaining_steps_seconds']:.6f} | {t['trajectory_wall_seconds']:.6f} |")
    lines+=['','| PC | Median core (s) | Min–max (s) | GMRES iterations | Top-level PC applications | Derived local ILU applications |',
        '|---|---:|---:|---:|---:|---:|']
    for case in ('Jacobi','ASM'):
        selected=next(r for r in rows if r['case_id']==case);t=data['timing_summary'][case]['trajectory_wall_seconds'];events=selected['timing']['petsc_event_deltas']
        local=str(selected['local_work_derived']['total_local_applications']) if case=='ASM' else 'n/a'
        lines.append(f"| {case} | {t['median']:.6f} | {t['minimum']:.6f}–{t['maximum']:.6f} | {selected['total_iterations']} | {events['PCApply']['count']} | {local} |")
    lines+=['',f"Ratio of Jacobi/ASM median core costs: {data['jacobi_over_asm_median_core_ratio']:.6f}. ASM had lower measured core cost on this fixed problem, including setup. This is not a universal preconditioner ranking.",
        'Retain all repeats: ASM measured3 was 0.109308 s, compared with 0.087728 and 0.087675 s. Its slower sample has no established cause. The ASM min–max range is 24.66% of its median; the Jacobi range is 1.17%. These are observed ranges, not confidence intervals.',
        'All ten trajectories/100 steps passed. All eight timed/warmup actual-field distances from their same-PC checked baseline were recorded as zero in L2/H1. Each timed trajectory recorded ten KSPSolve, one KSPSetUp and one top-level PCSetUp event.',
        'ASM local work is explicitly derived from 620 top-level PCApply applications times four fixed PREONLY/ILU domains, calibrated by fresh untimed callbacks: 2480 local applications. The separate diagnostic PC action and its four independent local-PC actions are excluded from timed work.',
        'PETSc event times are inclusive; do not add nested times. Near-zero Jacobi setup event times have insufficient resolution for a setup speed ratio.',
        review['timing_scope']+'. Excludes: '+review['timing_exclusions']+'.',
        'Keep these manual-driver costs distinct from earlier Irksome-driver measurements. This comparison verifies actual solution agreement; it does not freshly evaluate temporal-reference errors or demonstrate broader mesh/diffusion robustness.','',
        f'[Raw observations](../raw/{raw.name}); [accepted review](asm_cost_review.json).','']
    table_md.write_text('\n'.join(lines))
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'environment/build/plot-cache'));os.environ.setdefault('MPLBACKEND','Agg')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.hashsalt':stem,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(8.5,4.2),layout='constrained')
    for index,case in enumerate(('Jacobi','ASM')):
        wall=data['timing_summary'][case]['trajectory_wall_seconds'];values=wall['values'];color=('#1f5a99','#aa4f13')[index]
        ax.plot([min(values),max(values)],[index,index],color=color,lw=2)
        ax.scatter(values,[index-.07,index,index+.07],s=40,color=color,zorder=3)
        ax.scatter([wall['median']],[index],s=270,marker='|',color='black',zorder=4)
        ax.annotate(f"median {wall['median']:.6f} s",(max(values),index),xytext=(12,0),textcoords='offset points',va='center')
    ax.set_yticks([0,1],['Jacobi','ASM / local ILU(0)']);ax.set_ylim(-.4,1.4);ax.set_xlim(0,.31)
    ax.set_xlabel('Fresh construction + 10 advances: core wall time (s)');ax.grid(axis='x',alpha=.2)
    ax.set_title('Matched Radau PC timing on AC power — all three repeats')
    fig.supxlabel('CG P2/n16 • κ = 1 • Δt = 0.1 • T = 1 • one process/thread\nPoints: all measured repeats; black tick: median. Warm manual driver; fixed-problem comparison.',fontsize=9)
    png,svg=Path(str(fig_stem)+'.png'),Path(str(fig_stem)+'.svg')
    fig.savefig(png,dpi=200,metadata={'Software':f'Matplotlib {matplotlib.__version__}'});fig.savefig(svg,metadata={'Date':None,'Creator':f'Matplotlib {matplotlib.__version__}'})
    plt.close(fig)
    provenance=dict(raw_source=record,review=dict(path=str(REVIEW.relative_to(ROOT)),sha256=digest(REVIEW)),
        processor=dict(path=str(Path(__file__).resolve().relative_to(ROOT)),sha256=digest(Path(__file__))),
        plotting_inventory=dict(path=str(inventory_path.relative_to(ROOT)),sha256=digest(inventory_path)),
        python=sys.executable,python_version=platform.python_version(),matplotlib_version=matplotlib.__version__,
        regeneration_command=shlex.join([sys.executable,str(Path(__file__).resolve())]),
        output_sha256={str(p.relative_to(ROOT)):digest(p) for p in (table_json,table_md,png,svg)},processing_only=True,new_fem_experiment_run=False)
    Path(str(table_stem)+'-processing.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('Regenerated accepted matched PC table and figure:',fig_stem)

if __name__=='__main__':main()

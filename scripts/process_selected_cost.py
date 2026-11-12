#!/usr/bin/env python3
"""Regenerate matched selected-point cost tables/figure from accepted saved measurements only."""
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
REVIEW=ROOT/'results/processed/selected_cost_review.json'

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    r=json.loads(REVIEW.read_text());assert r['selected_cost_verified'];source=r['measurement_records']['serial'];raw=ROOT/source['path']
    assert digest(raw)==source['sha256'];d=json.loads(raw.read_text());inventory=ROOT/'environment/requirements-postprocessing-resolved.txt'
    assert subprocess.check_output([sys.executable,'-m','pip','--no-cache-dir','freeze','--all'],text=True)==inventory.read_text()
    stem=raw.name.removesuffix('-serial.json');table_stem=ROOT/'results/processed'/stem;fig_stem=ROOT/'results/figures'/stem
    summary=r['independent_audit']['case_summary'];ratios=r['independent_audit']['within_point_ratios']
    table=dict(raw_source=source,accepted_review=str(REVIEW.relative_to(ROOT)),cases=summary,within_point_ratios=ratios,
        timing_partitions=d['timing_summary'],limits=r['independent_audit']['limits'])
    tj,tm=Path(str(table_stem)+'-table.json'),Path(str(table_stem)+'-table.md');tj.write_text(json.dumps(table,indent=2,allow_nan=False)+'\n')
    lines=['# Matched selected-point warm core cost','',
        'CG P2, beta=(1,1), dt=0.1, T=1; one process/thread, AC before/after. Fresh construction/setup plus ten advances; warm caches. All 43 trajectories/430 steps passed.','',
        '| Point | Solver | Repeat 1 (s) | Repeat 2 (s) | Repeat 3 (s) | Median (s) | Relative min-max range |',
        '|---|---|---:|---:|---:|---:|---:|']
    for point in d['config']['points']:
        for solver in ('Jacobi','ASM','RanaLD'):
            key=point['point_id']+'_'+solver
            if key not in summary:lines.append(f"| {point['point_id']} | {solver} | not attempted: pilot cap | — | — | — | — |");continue
            s=summary[key];v=s['core_seconds'];values=' | '.join(f'{x:.9f}' for x in v['values'])
            lines.append(f"| {s['point_id']} | {solver} | {values} | {v['median']:.9f} | {s['relative_min_max_range_percent']:.2f}% |")
    lines+=['','| Point | Jacobi/ASM median ratio | Jacobi/Rana median ratio | ASM/Rana median ratio |','|---|---:|---:|---:|']
    for point,values in ratios.items():
        a=values.get('jacobi_over_asm_median_core_ratio');b=values.get('jacobi_over_rana_median_core_ratio');c=values['asm_over_rana_median_core_ratio']
        lines.append(f"| {point} | {a:.6f}"+f" | {b:.6f} | {c:.6f} |" if a is not None else f"| {point} | — | — | {c:.6f} |")
    lines+=['','Every repeat is retained, including the finer-mesh Rana sample 0.154305083 s. Its observed range is 9.50% of its median; no cause is inferred. Three samples do not define a confidence interval.',
        'All 32 warmup/measured actual final fields matched fresh same-PC checked fields at recorded precision. Physical equations, full coefficients, PC structure, complete work/reuse sequences and quadrature checks passed.',
        'Physical local/auxiliary applications are included; final additive/inverse probes and independent verification are outside the timer/event delta. Rana uses full auxiliary sparse LU. No scalable-subsolve or parallel result follows.',
        'Compare these matched costs within each point. Older different-driver absolute costs are separate. The finer-Jacobi pilot cap supplies no accepted full trajectory or elapsed cost.','',
        f'[Raw record](../raw/{raw.name}); [accepted review](selected_cost_review.json).','']
    tm.write_text('\n'.join(lines))
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'environment/build/plot-cache'));os.environ.setdefault('MPLBACKEND','Agg')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.hashsalt':stem,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(11.5,4.2),layout='constrained',sharey=True)
    colors={'Jacobi':'#1f5a99','ASM':'#aa4f13','RanaLD':'#2d7748'}
    for ax,point in zip(axes,d['config']['points']):
        for i,solver in enumerate(('Jacobi','ASM','RanaLD')):
            key=point['point_id']+'_'+solver
            if key not in summary:
                ax.text(i,.78,'Not attempted\npilot cap',transform=ax.get_xaxis_transform(),ha='center',color=colors[solver],fontsize=9);continue
            s=summary[key]['core_seconds'];values=s['values'];median=s['median'];color=colors[solver]
            ax.vlines(i,min(values),max(values),color=color,lw=1.5)
            ax.scatter([i-.085,i,i+.085],values,s=38,color=color,zorder=3)
            ax.hlines(median,i-.23,i+.23,color=color,lw=2.3)
            ax.annotate(f'{median:.4f}',(i,max(values)),xytext=(0,10),textcoords='offset points',ha='center',fontsize=9)
        ax.set_xlim(-.5,2.5);ax.set_ylim(.03,.95);ax.set_yscale('log');ax.set_xticks(range(3),('Jacobi','ASM','RanaLD'))
        ax.set_title(f"n={point['mesh_subdivisions']}, κ={point['diffusion']:g}");ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Warm core wall time (s, log scale)')
    fig.suptitle('Matched residual-form solver cost at three selected points')
    fig.supxlabel('Dots: all three repeats • horizontal lines: medians • labels: median seconds\nCG P2 • Δt=0.1 • T=1 • fresh construction/setup + ten advances • one process/thread • AC power',fontsize=9)
    png,svg=Path(str(fig_stem)+'.png'),Path(str(fig_stem)+'.svg')
    fig.savefig(png,dpi=200,metadata={'Software':f'Matplotlib {matplotlib.__version__}'});fig.savefig(svg,metadata={'Date':None,'Creator':f'Matplotlib {matplotlib.__version__}'});plt.close(fig)
    provenance=dict(raw_source=source,review=dict(path=str(REVIEW.relative_to(ROOT)),sha256=digest(REVIEW)),processor=dict(path=str(Path(__file__).resolve().relative_to(ROOT)),sha256=digest(Path(__file__))),
        plotting_inventory=dict(path=str(inventory.relative_to(ROOT)),sha256=digest(inventory)),python=sys.executable,python_version=platform.python_version(),matplotlib_version=matplotlib.__version__,
        regeneration_command=shlex.join([sys.executable,str(Path(__file__).resolve())]),output_sha256={str(p.relative_to(ROOT)):digest(p) for p in (tj,tm,png,svg)},processing_only=True,new_fem_experiment_run=False)
    Path(str(table_stem)+'-processing.json').write_text(json.dumps(provenance,indent=2)+'\n');print('Regenerated accepted selected-cost table and figure:',fig_stem)

if __name__=='__main__':main()

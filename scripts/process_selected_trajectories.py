#!/usr/bin/env python3
"""Regenerate accepted selected full-trajectory tables and scientific figure from saved records."""
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
REVIEW=ROOT/'results/processed/selected_trajectories_review.json'

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    r=json.loads(REVIEW.read_text());assert r['selected_trajectories_verified'];source=r['measurement_records']['serial'];raw=ROOT/source['path']
    assert digest(raw)==source['sha256'];d=json.loads(raw.read_text());inventory=ROOT/'environment/requirements-postprocessing-resolved.txt'
    assert subprocess.check_output([sys.executable,'-m','pip','--no-cache-dir','freeze','--all'],text=True)==inventory.read_text()
    stem=raw.name.removesuffix('-serial.json');table_stem=ROOT/'results/processed'/stem;fig_stem=ROOT/'results/figures'/stem
    summary=r['independent_audit']['point_summary'];table=dict(raw_source=source,accepted_review=str(REVIEW.relative_to(ROOT)),points=summary,
        predeclared_exclusions=r['independent_audit']['predeclared_exclusions'],limits=r['independent_audit']['limits'])
    tj,tm=Path(str(table_stem)+'-table.json'),Path(str(table_stem)+'-table.md');tj.write_text(json.dumps(table,indent=2,allow_nan=False)+'\n')
    lines=['# Selected full-trajectory solver work and reuse','',
        'CG P2, beta=(1,1), dt=0.1, T=1, ten steps per completed trajectory; one process/thread. Eight candidate plus three physical-LU reference trajectories passed all 110 steps. No elapsed-cost measurement.','',
        '| Point | n | kappa | Solver | Total outer iterations | Physical PCApply | Physical MatSolve | Rana factor entries |',
        '|---|---:|---:|---|---:|---:|---:|---:|']
    for point in d['config']['points']:
        name=point['point_id'];p=summary[name]
        for case in ('Jacobi','ASM','RanaLD'):
            row=p['candidates'][case]
            if row['status']=='not_attempted':lines.append(f"| {name} | {point['mesh_subdivisions']} | {point['diffusion']:g} | {case} | not attempted: pilot cap | — | — | — |");continue
            e=row['physical_event_counts'];nnz=str(int(row['auxiliary_factor_nonzeros'])) if case=='RanaLD' else '—'
            lines.append(f"| {name} | {point['mesh_subdivisions']} | {point['diffusion']:g} | {case} | {row['total_iterations']} | {e['PCApply']} | {e['MatSolve']} | {nnz} |")
    lines+=['','No new cap occurred. Finer-mesh Jacobi remains explicitly excluded because its first step failed at the unchanged 1000-iteration limit; no full trajectory is claimed.',
        'Physical and factor object IDs/states stayed unchanged; no PC/factor rebuild occurred after step one. Each Rana trajectory used one auxiliary LU factorization; each ASM trajectory used four local ILU numeric setups. KSPSetUp counts do not count factorizations.',
        'ASM MatSolve counts are actual local ILU actions; fixed composition corroborates four per outer PCApply. The final additive probe adds eight separate diagnostic MatSolve actions. Each final Rana probe adds one separate auxiliary action.',
        'Rana factor entries grow 7.136844 times on refinement. They are a storage proxy, not measured bytes or elapsed time. No scalable-subsolve or elapsed-cost ranking follows.',
        'Per-step actual update/slopes agree with independent numerical references. Final MMS errors combine spatial, temporal and algebraic effects; no new convergence order is inferred.','',
        f'[Raw record](../raw/{raw.name}); [accepted review](selected_trajectories_review.json).','']
    tm.write_text('\n'.join(lines))
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'environment/build/plot-cache'));os.environ.setdefault('MPLBACKEND','Agg')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.hashsalt':stem,'axes.spines.top':False,'axes.spines.right':False})
    points=d['config']['points'];labels=[f"n={p['mesh_subdivisions']}, κ={p['diffusion']:g}" for p in points]
    fig,(ax,bx)=plt.subplots(1,2,figsize=(10.7,4.6),layout='constrained',gridspec_kw={'width_ratios':[1.35,1]})
    for offset,case,color,marker in zip((-.22,0,.22),('Jacobi','ASM','RanaLD'),('#1f5a99','#aa4f13','#2d7748'),('o','s','^')):
        for i,point in enumerate(points):
            row=summary[point['point_id']]['candidates'][case];x=i+offset
            if row['status']=='not_attempted':
                ax.text(x,.91,'Jacobi: not attempted\npilot cap retained',transform=ax.get_xaxis_transform(),ha='center',fontsize=8,color=color);continue
            y=row['total_iterations'];ax.scatter([x],[y],marker=marker,color=color,s=60,label=case if i==0 else None,zorder=3)
            ax.annotate(str(y),(x,y),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8)
    ax.set_yscale('log');ax.set_ylim(40,18000);ax.set_xticks(range(3),labels);ax.set_ylabel('Total outer GMRES iterations (log scale)')
    ax.set_title('Ten verified steps per candidate');ax.grid(axis='y',alpha=.2);ax.legend(loc='upper right',fontsize=8)
    values=[summary[p['point_id']]['candidates']['RanaLD']['auxiliary_factor_nonzeros']/1e6 for p in points]
    bx.bar(range(3),values,color='#2d7748',width=.6)
    for i,value in enumerate(values):bx.text(i,value+.035,f'{value:.3f}M',ha='center',fontsize=9)
    bx.set_ylim(0,max(values)*1.23);bx.set_xticks(range(3),labels);bx.set_ylabel('Rana auxiliary LU factor entries (millions)')
    bx.set_title('One fixed auxiliary factor per trajectory');bx.grid(axis='y',alpha=.2)
    fig.suptitle('Selected full trajectories — eight verified candidates, factor reuse checked')
    fig.supxlabel('CG P2 • β=(1,1) • Δt=0.1 • T=1 • one process/thread\nFiner-mesh Jacobi has no accepted full solution. Full auxiliary LU; entries do not measure bytes or elapsed time.',fontsize=9)
    png,svg=Path(str(fig_stem)+'.png'),Path(str(fig_stem)+'.svg')
    fig.savefig(png,dpi=200,metadata={'Software':f'Matplotlib {matplotlib.__version__}'});fig.savefig(svg,metadata={'Date':None,'Creator':f'Matplotlib {matplotlib.__version__}'});plt.close(fig)
    provenance=dict(raw_source=source,review=dict(path=str(REVIEW.relative_to(ROOT)),sha256=digest(REVIEW)),processor=dict(path=str(Path(__file__).resolve().relative_to(ROOT)),sha256=digest(Path(__file__))),
        plotting_inventory=dict(path=str(inventory.relative_to(ROOT)),sha256=digest(inventory)),python=sys.executable,python_version=platform.python_version(),matplotlib_version=matplotlib.__version__,
        regeneration_command=shlex.join([sys.executable,str(Path(__file__).resolve())]),output_sha256={str(p.relative_to(ROOT)):digest(p) for p in (tj,tm,png,svg)},processing_only=True,new_fem_experiment_run=False)
    Path(str(table_stem)+'-processing.json').write_text(json.dumps(provenance,indent=2)+'\n');print('Regenerated accepted selected full-trajectory table and figure:',fig_stem)

if __name__=='__main__':main()

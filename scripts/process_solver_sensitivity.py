#!/usr/bin/env python3
"""Regenerate accepted first-step solver sensitivity table and scientific figure."""
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
REVIEW=ROOT/'results/processed/solver_sensitivity_review.json'

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    r=json.loads(REVIEW.read_text());assert r['solver_sensitivity_verified']
    source=r['measurement_records']['serial'];raw=ROOT/source['path'];assert digest(raw)==source['sha256'];d=json.loads(raw.read_text())
    inventory=ROOT/'environment/requirements-postprocessing-resolved.txt'
    assert subprocess.check_output([sys.executable,'-m','pip','--no-cache-dir','freeze','--all'],text=True)==inventory.read_text()
    stem=raw.name.removesuffix('-serial.json');table_stem=ROOT/'results/processed'/stem;fig_stem=ROOT/'results/figures'/stem
    summary=r['independent_audit']['point_summary'];table=dict(raw_source=source,accepted_review=str(REVIEW.relative_to(ROOT)),points=summary,
        preserved_caps=r['independent_audit']['preserved_iteration_limit_outcomes'],limits=r['independent_audit']['limits'])
    tj,tm=Path(str(table_stem)+'-table.json'),Path(str(table_stem)+'-table.md');tj.write_text(json.dumps(table,indent=2,allow_nan=False)+'\n')
    lines=['# Selected first-step solver sensitivity','',
        'CG P2, beta=(1,1), dt=0.1; only [0,0.1], one process/thread, no elapsed-cost measurement. Full research T=1 is not completed by this pilot.','',
        '| Point | n | kappa | Coupled size | Jacobi | ASM / ILU(0) | RanaLD / auxiliary LU | Rana factor nonzeros |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for point in d['config']['points']:
        name=point['point_id'];p=summary[name];c=p['candidates']
        counts=[str(c[k]['iterations']) if c[k]['status']=='passed' else f"{c[k]['iterations']} (cap; not converged)" for k in ('Jacobi','ASM','RanaLD')]
        lines.append(f"| {name} | {point['mesh_subdivisions']} | {point['diffusion']:g} | {p['metadata']['stage_total_dofs']} | {' | '.join(counts)} | {int(c['RanaLD']['auxiliary_factor_nonzeros'])} |")
    lines+=['','Eight candidate solutions and all three independent physical-LU references passed. Finer-mesh Jacobi reached 1000 iterations with reason -3; no update or physical slope field was accepted. Its recomputed KSP correction relative residual was 4.107492169e−6; that is not the independent physical residual of a completed step.',
        f"Rana auxiliary factor nonzeros grew by {r['independent_audit']['rana_auxiliary_factor_nnz_mesh_ratio']:.6f} from n16 to n32, while its first-step count stayed seven. Factor entries are a storage proxy, not measured bytes or elapsed time.",
        'The LU reference uses PREONLY: displayed default KSP tolerances are not its stopping rule. Its physical residual is checked separately against max(1e−14,1e−12 norm(g)) with the fixed allowance.',
        'ASM physical MatSolve counts corroborate four local actions per outer PCApply. Its additive probe includes one global action and four independent direct local actions; those eight diagnostic MatSolve operations are separate from physical work.',
        'A PETSc MatLUFactorNum event can include the local ILU numeric factor work here; actual recorded PC types identify ILU(0) versus full auxiliary LU.',
        'No mesh-independent bound, full-time accuracy, elapsed-cost ranking or scalable-subsolve result is established.','',f'[Raw record](../raw/{raw.name}); [accepted review](solver_sensitivity_review.json).','']
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
            row=summary[point['point_id']]['candidates'][case];x=i+offset;y=row['iterations'];capped=row['status']!='passed'
            ax.scatter([x],[y],marker='x' if capped else marker,color=color,s=60,label=case if i==0 else None,zorder=3)
            ax.annotate('1000: cap' if capped else str(y),(x,y),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8)
    ax.set_yscale('log');ax.set_ylim(4,2000);ax.set_xticks(range(3),labels);ax.set_ylabel('Outer GMRES iterations (log scale)')
    ax.set_title('First-step outcomes');ax.grid(axis='y',alpha=.2);ax.legend(loc='upper right',fontsize=8)
    values=[summary[p['point_id']]['candidates']['RanaLD']['auxiliary_factor_nonzeros']/1e6 for p in points]
    bx.bar(range(3),values,color='#2d7748',width=.6)
    for i,value in enumerate(values):bx.text(i,value+.035,f'{value:.3f}M',ha='center',fontsize=9)
    bx.set_ylim(0,max(values)*1.23);bx.set_xticks(range(3),labels);bx.set_ylabel('Rana auxiliary LU factor entries (millions)')
    bx.set_title('Factor storage grows with mesh');bx.grid(axis='y',alpha=.2)
    fig.suptitle('Selected solver sensitivity — eight verified solves, one preserved cap')
    fig.supxlabel('CG P2 • β=(1,1) • Δt=0.1 • only [0,0.1] • one process/thread\nCross: cap reached, no solution accepted. Full auxiliary sparse LU; entries do not measure bytes or elapsed time.',fontsize=9)
    png,svg=Path(str(fig_stem)+'.png'),Path(str(fig_stem)+'.svg')
    fig.savefig(png,dpi=200,metadata={'Software':f'Matplotlib {matplotlib.__version__}'});fig.savefig(svg,metadata={'Date':None,'Creator':f'Matplotlib {matplotlib.__version__}'});plt.close(fig)
    provenance=dict(raw_source=source,review=dict(path=str(REVIEW.relative_to(ROOT)),sha256=digest(REVIEW)),processor=dict(path=str(Path(__file__).resolve().relative_to(ROOT)),sha256=digest(Path(__file__))),
        plotting_inventory=dict(path=str(inventory.relative_to(ROOT)),sha256=digest(inventory)),python=sys.executable,python_version=platform.python_version(),matplotlib_version=matplotlib.__version__,
        regeneration_command=shlex.join([sys.executable,str(Path(__file__).resolve())]),output_sha256={str(p.relative_to(ROOT)):digest(p) for p in (tj,tm,png,svg)},processing_only=True,new_fem_experiment_run=False)
    Path(str(table_stem)+'-processing.json').write_text(json.dumps(provenance,indent=2)+'\n');print('Regenerated accepted first-step sensitivity table and figure:',fig_stem)

if __name__=='__main__':main()

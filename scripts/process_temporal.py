#!/usr/bin/env python3
"""Regenerate tables and scientific figures from accepted researcher measurements."""
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / 'results/processed/temporal_review.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    review = json.loads(REVIEW.read_text())
    assert review['checkpoint_exit_status'] == 0
    record = review['measurement_records']['serial']
    raw = ROOT / record['path']
    assert digest(raw) == record['sha256'], 'Accepted raw measurements changed'
    data = json.loads(raw.read_text())
    assert data['status'] == 'passed' and data['ranks'] == 1
    stem = raw.name.removesuffix('-serial.json')
    table_stem = ROOT / 'results/processed' / stem
    fig_stem = ROOT / 'results/figures' / stem
    fig_stem.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in data['curve']:
        trajectory = data['trajectories'][row['trajectory']]
        derived = {key:row[key] for key in ('method','dt','time_errors','trajectory')}
        derived.update(steps=trajectory['completed_steps'],total_iterations=trajectory['total_iterations'])
        for metric in ('l2_error','h1_seminorm_error'):
            rate = next((r['observed_order'] for r in data['rates'] if r['method']==row['method'] and r['fine_dt']==row['dt'] and r['metric']==metric),None)
            derived[metric+'_observed_order'] = rate
        rows.append(derived)
    table = {'raw_source':record,'accepted_review':str(REVIEW.relative_to(ROOT)),
       'quantity':'Actual final-field difference from fine same-mesh Radau reference at T=1; CG P2/n16; q=14',
       'reference':data['config']['reference'],'rows':rows,'rates':data['rates'],
       'temporal_checkpoint_accepted':True,'performance_measurement':False}
    table_json = Path(str(table_stem)+'-table.json')
    table_json.write_text(json.dumps(table,indent=2,allow_nan=False)+'\n')
    table_md = Path(str(table_stem)+'-table.md')
    lines = ['# Measured same-mesh temporal errors','',
       'Researcher data on CG P2/n16, kappa=1, T=1. Errors compare actual final fields with Radau dt=1/1280 on the same space, using quadrature degree 14.','',
       '| Method | dt | Steps | L2 time error | L2 rate | H1 seminorm time error | H1 rate |',
       '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        rate = lambda metric:'—' if row[metric+'_observed_order'] is None else f"{row[metric+'_observed_order']:.6f}"
        name = 'Backward Euler' if row['method']=='manual_backward_euler' else 'Radau IIA (2 stages)'
        lines.append(f"| {name} | {row['dt']} | {row['steps']} | {row['time_errors']['l2_error']:.9e} | {rate('l2_error')} | {row['time_errors']['h1_seminorm_error']:.9e} | {rate('h1_seminorm_error')} |")
    lines += ['', 'Rates use log(E_coarse/E_fine)/log(dt_coarse/dt_fine). The expected orders are one for BE and three for two-stage Radau.','',
       f"[Raw measurements](../raw/{raw.name}); [accepted review](temporal_review.json).",'',
       'Reference-refinement plus tolerance sensitivity is about 0.177% of the smallest measured time error in each norm; the declared controls passed.',
       'These are temporal trends for this fixed Galerkin system and step range. Continuum MMS errors contain a spatial floor; execution cost was not benchmarked.','']
    table_md.write_text('\n'.join(lines))

    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'environment/build/plot-cache'))
    os.environ.setdefault('MPLBACKEND','Agg')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.hashsalt':stem,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(1,2,figsize=(10.6,4.9),layout='constrained')
    for ax,metric,title,ylabel in zip(axes,('l2_error','h1_seminorm_error'),
       ('Function time error','Gradient time error'),
       (r'$\|u_{h,\Delta t}(1)-u_{h,\mathrm{ref}}(1)\|_{L^2}$',r'$|u_{h,\Delta t}(1)-u_{h,\mathrm{ref}}(1)|_{H^1}$')):
        for method,name,color,marker in (('manual_backward_euler','BE','#1f5a99','o'),('RadauIIA','Radau IIA','#aa4f13','s')):
            group = [row for row in rows if row['method']==method]
            rate = math.log(group[-2]['time_errors'][metric]/group[-1]['time_errors'][metric])/math.log(group[-2]['dt']/group[-1]['dt'])
            ax.loglog([row['dt'] for row in group],[row['time_errors'][metric] for row in group],
                      color=color,marker=marker,lw=2,ms=7,label=f'{name}: final rate {rate:.3f}')
        steps = sorted(data['config']['time_step_sizes'])
        ax.xaxis.set_major_locator(FixedLocator(steps))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value,pos:f'{value:g}'))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(steps[0]*.85,steps[-1]*1.15)
        ax.set_xlabel(r'Time step $\Delta t$')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(which='major',alpha=.22)
        ax.legend(loc='best',frameon=False,fontsize=10)
    fig.suptitle(r'Same-mesh temporal verification at $T=1$, $\kappa=1$',fontsize=15)
    fig.supxlabel('CG P2, n = 16   •   Reference: Radau IIA, Δt = 1/1280   •   Quadrature degree 14',fontsize=10)
    png, svg = Path(str(fig_stem)+'.png'),Path(str(fig_stem)+'.svg')
    fig.savefig(png,dpi=200,metadata={'Software':f'Matplotlib {matplotlib.__version__}'})
    fig.savefig(svg,metadata={'Date':None,'Creator':f'Matplotlib {matplotlib.__version__}'})
    plt.close(fig)
    inventory = subprocess.check_output([sys.executable,'-m','pip','--no-cache-dir','freeze','--all'],text=True)
    inventory_path = ROOT/'environment/requirements-postprocessing-resolved.txt'
    assert inventory_path.read_text() == inventory, 'Plotting environment differs from the recorded inventory'
    command = [sys.executable,str(Path(__file__).resolve())]
    provenance = {'raw_source':record,'review':{'path':str(REVIEW.relative_to(ROOT)),'sha256':digest(REVIEW)},
       'processor':{'path':str(Path(__file__).resolve().relative_to(ROOT)),'sha256':digest(Path(__file__))},
       'python':sys.executable,'python_version':platform.python_version(),
       'matplotlib_version':matplotlib.__version__,
       'plotting_inventory':{'path':str(inventory_path.relative_to(ROOT)),'sha256':digest(inventory_path)},
       'regeneration_argv':command,'regeneration_command':shlex.join(command),
       'output_sha256':{str(p.relative_to(ROOT)):digest(p) for p in (table_json,table_md,png,svg)},
       'processing_only':True,'new_fem_experiment_run':False}
    Path(str(table_stem)+'-processing.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('Regenerated temporal table and figure from accepted researcher measurements:',fig_stem)


if __name__ == '__main__':
    main()

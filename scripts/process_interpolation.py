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
REVIEW = ROOT / 'results/processed/interpolation_review.json'


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
    for row in data['measurements']:
        derived = {key: row[key] for key in ('case_id','element_degree','mesh_subdivisions','h','total_dofs')}
        derived.update(row['metrics'])
        for metric in ('l2_error','h1_seminorm_error'):
            rates = [rate for rate in data['rates'] if rate['fine_case'] == row['case_id'] and rate['metric'] == metric]
            derived[metric+'_observed_order'] = rates[0]['observed_order'] if rates else None
        rows.append(derived)
    table = {'raw_source':record, 'accepted_review':str(REVIEW.relative_to(ROOT)),
             'quantity':'Nodal interpolation error at t=0; primary quadrature degree 10',
             'rows':rows, 'rates':data['rates'], 'transient_convergence_claim':False}
    table_json = Path(str(table_stem)+'-table.json')
    table_json.write_text(json.dumps(table,indent=2,allow_nan=False)+'\n')
    table_md = Path(str(table_stem)+'-table.md')
    lines = ['# Measured initial interpolation errors', '',
       'Researcher data at t=0; primary quadrature degree 10. h is the maximum triangle diameter.', '',
       '| Space | n | Total DOFs | h | L2 error | L2 rate | H1 seminorm error | H1 rate |',
       '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        rate = lambda metric: '—' if row[metric+'_observed_order'] is None else f"{row[metric+'_observed_order']:.6f}"
        lines.append(f"| P{row['element_degree']} | {row['mesh_subdivisions']} | {row['total_dofs']} | {row['h']:.9f} | {row['l2_error']:.9e} | {rate('l2_error')} | {row['h1_seminorm_error']:.9e} | {rate('h1_seminorm_error')} |")
    lines += ['', 'Rates use adjacent meshes and log(E_coarse/E_fine)/log(h_coarse/h_fine).', '',
       f"[Raw norms and rates](../raw/{raw.name}); [accepted review](interpolation_review.json).", '',
       'Expected interpolation orders: P1 L2/H1 seminorm = 2/1; P2 = 3/2.',
       'This dataset does not establish transient Galerkin convergence or temporal order.', '']
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
    for ax, metric, title, ylabel in zip(axes,('l2_error','h1_seminorm_error'),
       ('Function error','Gradient error'),(r'$\|I_h g-g\|_{L^2}$',r'$|I_h g-g|_{H^1}$')):
        for degree,color,marker in ((1,'#1f5a99','o'),(2,'#aa4f13','s')):
            group = [row for row in rows if row['element_degree'] == degree]
            # Rates are derived again for the displayed legend, not invented guide lines.
            rate = math.log(group[-2][metric]/group[-1][metric])/math.log(group[-2]['h']/group[-1]['h'])
            ax.loglog([row['h'] for row in group],[row[metric] for row in group],
                      color=color,marker=marker,lw=2,ms=7,label=f'P{degree}: final rate {rate:.3f}')
        hs = [math.sqrt(2)/n for n in (32,16,8)]
        ax.xaxis.set_major_locator(FixedLocator(hs))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value,pos:f'{value:.3f}'))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(hs[0]*.85,hs[-1]*1.15)
        ax.set_xlabel(r'Maximum triangle diameter $h$')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(which='major',alpha=.22)
        ax.legend(loc='best',frameon=False,fontsize=10)
    fig.suptitle(r'Initial interpolation of $g=\sin(\pi x)\sin(\pi y)$',fontsize=15)
    fig.supxlabel('Measured at t = 0   •   Uniform triangular meshes, n = 8, 16, 32   •   Quadrature degree 10',fontsize=10)
    png, svg = Path(str(fig_stem)+'.png'),Path(str(fig_stem)+'.svg')
    fig.savefig(png,dpi=200,metadata={'Software':f'Matplotlib {matplotlib.__version__}'})
    fig.savefig(svg,metadata={'Date':None,'Creator':f'Matplotlib {matplotlib.__version__}'})
    plt.close(fig)
    inventory = subprocess.check_output([sys.executable,'-m','pip','--no-cache-dir','freeze','--all'],text=True)
    inventory_path = ROOT/'environment/requirements-postprocessing-resolved.txt'
    inventory_path.write_text(inventory)
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
    print('Regenerated table and figure from accepted researcher measurements:',fig_stem)


if __name__ == '__main__':
    main()

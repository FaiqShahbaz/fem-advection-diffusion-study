#!/usr/bin/env python3
"""Regenerate the pilot timing table/figure from accepted raw measurements."""
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT/'results/processed/cost_pilot_review.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    review = json.loads(REVIEW.read_text())
    assert review['timing_instrumentation_verified'] and review['solution_equivalence_verified']
    record = review['measurement_records']['serial']
    raw = ROOT/record['path']
    assert digest(raw) == record['sha256']
    data = json.loads(raw.read_text())
    stem = raw.name.removesuffix('-serial.json')
    keys = ('method','repetition','dt','completed_steps','total_iterations','maximum_step_iterations','timing','equivalence','final_equation_check')
    rows = [dict(trajectory=label, **{key:row[key] for key in keys}) for label,row in data['trajectories'].items() if row['role'] == 'measured']
    table_stem, fig_stem = ROOT/'results/processed'/stem, ROOT/'results/figures'/stem
    table = dict(raw_source=record, accepted_review=str(REVIEW.relative_to(ROOT)),
        power_source=review['power_source'], timing_scope=review['timing_scope'],
        rows=rows, summary=data['timing_summary'], accuracy_cost_ranking_established=False)
    table_json, table_md = Path(str(table_stem)+'-table.json'), Path(str(table_stem)+'-table.md')
    table_json.write_text(json.dumps(table,indent=2,allow_nan=False)+'\n')
    lines = ['# Measured timing/equivalence pilot', '',
        'CG P2/n16, kappa=1, dt=0.05, T=1; one serial process and one BLAS/OpenMP thread.',
        '**Recorded power: Battery Power.** Accept these as preliminary instrumentation observations, not formal accuracy–cost comparisons.', '',
        '| Method | Repeat | Construction (s) | First step (s) | Other 19 steps (s) | Total core (s) |',
        '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        t = row['timing']
        name = 'BE' if row['method']=='manual_backward_euler' else 'Radau IIA'
        lines.append(f"| {name} | {row['repetition']} | {t['construction_seconds']:.6f} | {t['first_step_seconds']:.6f} | {t['remaining_steps_seconds']:.6f} | {t['trajectory_wall_seconds']:.6f} |")
    lines += ['', '| Method | Median core (s) | Min–max (s) | GMRES iterations / trajectory | PC setups | PC applications |',
        '|---|---:|---:|---:|---:|---:|']
    for method in data['config']['time_methods']:
        selected = [row for row in rows if row['method']==method]
        t = data['timing_summary'][method]['trajectory_wall_seconds']
        row = selected[0]; e = row['timing']['petsc_event_deltas']
        name = 'BE' if method=='manual_backward_euler' else 'Radau IIA'
        lines.append(f"| {name} | {t['median']:.6f} | {t['minimum']:.6f}–{t['maximum']:.6f} | {row['total_iterations']} | {e['PCSetUp']['count']} | {e['PCApply']['count']} |")
    lines += ['', 'All eight timed/warmup trajectories have recorded final-field differences of zero against their method’s checked baseline in both norms. All convergence, final-equation, boundary and quadrature controls passed.',
        'Each trajectory recorded 20 KSPSolve and 20 KSPSetUp events but only one PCSetUp. Calls to KSPSetUp are not repeated Jacobi construction.',
        'PETSc event times are nested/inclusive; do not add them. Zero/about-one-microsecond setup times are not resolved enough for a setup speed claim.',
        'Warm code/cache, fresh operators/solvers per trajectory. Core wall includes construction and advances; excludes imports, shared mesh, independent verification, printing, I/O and final cleanup.',
        'The methods use the same dt but different coupled-system sizes and accuracy. No equal-accuracy ranking follows from this table.', '',
        f'[Raw measurements](../raw/{raw.name}); [accepted review](cost_pilot_review.json).', '']
    table_md.write_text('\n'.join(lines))
    inventory = subprocess.check_output([sys.executable,'-m','pip','--no-cache-dir','freeze','--all'],text=True)
    inventory_path = ROOT/'environment/requirements-postprocessing-resolved.txt'
    assert inventory == inventory_path.read_text()
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'environment/build/plot-cache'))
    os.environ.setdefault('MPLBACKEND','Agg')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.hashsalt':stem,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig, ax = plt.subplots(figsize=(8.2,4.0),layout='constrained')
    for index,method in enumerate(data['config']['time_methods']):
        group = [row for row in rows if row['method']==method]
        values = [row['timing']['trajectory_wall_seconds'] for row in group]
        median = data['timing_summary'][method]['trajectory_wall_seconds']['median']
        color = '#1f5a99' if index==0 else '#aa4f13'
        ax.plot([min(values),max(values)],[index,index],color=color,lw=2)
        ax.scatter(values,[index-.07,index,index+.07],color=color,s=38,zorder=3)
        ax.scatter([median],[index],marker='|',s=250,color='black',zorder=4)
        ax.annotate(f'median {median:.6f} s',(max(values),index),xytext=(12,0),textcoords='offset points',va='center')
    ax.set_yticks([0,1],['Backward Euler','Radau IIA (2 stages)'])
    ax.set_ylim(-.45,1.45)
    ax.set_xlim(0,max(row['timing']['trajectory_wall_seconds'] for row in rows)*1.43)
    ax.set_xlabel('Construction + 20 steps: core wall time (s)')
    ax.grid(axis='x',alpha=.2)
    ax.set_title('Timing pilot on battery power — three measured repeats')
    fig.supxlabel('CG P2/n16 • κ = 1 • Δt = 0.05 • T = 1 • warm code/cache\nPoints: measured repeats; black tick: median. No equal-accuracy cost ranking.',fontsize=9)
    png,svg=Path(str(fig_stem)+'.png'),Path(str(fig_stem)+'.svg')
    fig.savefig(png,dpi=200,metadata={'Software':f'Matplotlib {matplotlib.__version__}'})
    fig.savefig(svg,metadata={'Date':None,'Creator':f'Matplotlib {matplotlib.__version__}'})
    plt.close(fig)
    provenance=dict(raw_source=record,review=dict(path=str(REVIEW.relative_to(ROOT)),sha256=digest(REVIEW)),
        processor=dict(path=str(Path(__file__).resolve().relative_to(ROOT)),sha256=digest(Path(__file__))),
        plotting_inventory=dict(path=str(inventory_path.relative_to(ROOT)),sha256=digest(inventory_path)),
        python=sys.executable,python_version=platform.python_version(),matplotlib_version=matplotlib.__version__,
        regeneration_command=shlex.join([sys.executable,str(Path(__file__).resolve())]),
        output_sha256={str(p.relative_to(ROOT)):digest(p) for p in (table_json,table_md,png,svg)},
        processing_only=True,new_fem_experiment_run=False)
    Path(str(table_stem)+'-processing.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('Regenerated accepted pilot table/figure:',fig_stem)


if __name__=='__main__':
    main()

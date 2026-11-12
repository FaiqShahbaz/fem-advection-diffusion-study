#!/usr/bin/env python3
"""Regenerate the accepted targeted temporal accuracy/cost table and scientific plot."""
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
REVIEW=ROOT/'results/processed/accuracy_cost_review.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    review=json.loads(REVIEW.read_text());assert review['accuracy_cost_baseline_verified']
    record=review['measurement_records']['serial'];raw=ROOT/record['path'];assert digest(raw)==record['sha256']
    data=json.loads(raw.read_text());assert data['status']=='passed'
    inventory_path=ROOT/'environment/requirements-postprocessing-resolved.txt'
    inventory=subprocess.check_output([sys.executable,'-m','pip','--no-cache-dir','freeze','--all'],text=True)
    assert inventory==inventory_path.read_text()
    stem=raw.name.removesuffix('-serial.json');table_stem=ROOT/'results/processed'/stem;fig_stem=ROOT/'results/figures'/stem
    rows=[]
    for curve in data['curve']:
        baseline=data['trajectories'][curve['trajectory']]
        rows.append({key:curve[key] for key in ('case_id','method','dt','time_errors','timing_summary','timed_error_envelope','measured_trajectories')}
            |dict(steps=baseline['completed_steps'],total_iterations=baseline['total_iterations'],maximum_step_iterations=baseline['maximum_step_iterations']))
    table=dict(raw_source=record,accepted_review=str(REVIEW.relative_to(ROOT)),power='AC before/after run',
        quantity='Time error about finite same-mesh reference, paired with warm construction/advance core wall time',
        rows=rows,candidate_selections=data['candidate_selections'],ranking_scope=review['ranking_scope'])
    table_json,table_md=Path(str(table_stem)+'-table.json'),Path(str(table_stem)+'-table.md')
    table_json.write_text(json.dumps(table,indent=2,allow_nan=False)+'\n')
    lines=['# Measured targeted temporal accuracy/cost baseline','',
        'CG P2/n16, kappa=1, beta=(1,1), T=1; one process/thread, AC confirmed before and after the run. Three measured warm-cache repeats per candidate.','',
        '| Candidate | dt | Steps | L2 time error | H1 seminorm time error | Median core (s) | Core min–max (s) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        t=row['timing_summary']['trajectory_wall_seconds']
        lines.append(f"| {row['case_id']} | {row['dt']} | {row['steps']} | {row['time_errors']['l2_error']:.9e} | {row['time_errors']['h1_seminorm_error']:.9e} | {t['median']:.6f} | {t['minimum']:.6f}–{t['maximum']:.6f} |")
    lines+=['','| L2 time-error target | Selected BE candidate | BE median (s) | Selected Radau candidate | Radau median (s) | BE/Radau median ratio |',
        '|---:|---|---:|---|---:|---:|']
    for target in data['config']['target_time_errors_l2']:
        selected={r['method']:r['selected_case'] for r in data['candidate_selections'] if r['target_l2_time_error']==target}
        be=next(r for r in rows if r['case_id']==selected['manual_backward_euler']);rad=next(r for r in rows if r['case_id']==selected['RadauIIA'])
        btime=be['timing_summary']['trajectory_wall_seconds']['median'];rtime=rad['timing_summary']['trajectory_wall_seconds']['median']
        lines.append(f"| {target:.1e} | {be['case_id']} | {btime:.6f} | {rad['case_id']} | {rtime:.6f} | {btime/rtime:.6f} |")
    lines+=['','Eligibility uses the upper time-error transfer bound over all three measured repeats. All timed-baseline field distances were recorded as zero, so these bounds coincide with the baseline errors at recorded precision.',
        'At 1e−4, the selected BE candidate has lower median core cost. At 3e−6, the selected Radau candidate has about 9.79 times lower median core cost than the selected BE candidate. These compare the declared candidates only; no optimal dt or universal method ranking is established.',
        'The reference is Radau dt=1/1280 on the same mesh; coarse/fine/tighter reference and finest-candidate tolerance controls passed. Reference sensitivity is about 0.00310% of the smallest measured time error in each norm, not a certified reference-error bound.',
        'The continuum spatial floor is about 2.53e−5. Time-error targets do not imply the same total PDE accuracy.',
        'Core time includes fresh construction and all advances under the accepted timing boundaries. Imports, common mesh, reference/verification, printing, I/O and final cleanup are excluded; profiling and small counters are included. PETSc event times are inclusive and must not be added.',
        'All 25 trajectories/15770 steps passed. Each timed trajectory recorded one PCSetUp; outer iterations per candidate were BE_loose 8221, Radau_coarse 7258, BE_fine 50000 and Radau_fine 9648.',
        'Three repeats provide observed ranges, not a precise confidence interval; laptop power/thermal/background conditions and the implementation scope limit generalization.','',
        f'[Raw observations](../raw/{raw.name}); [accepted review](accuracy_cost_review.json).','']
    table_md.write_text('\n'.join(lines))
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'environment/build/plot-cache'));os.environ.setdefault('MPLBACKEND','Agg')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.hashsalt':stem,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(8.8,5.2),layout='constrained')
    used=set()
    offsets={'BE_loose':(12,-15),'BE_fine':(10,-20),'Radau_coarse':(-10,12),'Radau_fine':(12,-7)}
    for row in rows:
        be=row['method']=='manual_backward_euler';color,marker=('#1f5a99','o') if be else ('#aa4f13','s')
        name='Backward Euler' if be else 'Radau IIA (2 stages)'
        wall=row['timing_summary']['trajectory_wall_seconds'];error=row['time_errors']['l2_error'];median=wall['median']
        ax.scatter(wall['values'],[error]*3,color=color,marker=marker,s=20,alpha=.35,zorder=2)
        ax.errorbar([median],[error],xerr=[[median-wall['minimum']],[wall['maximum']-median]],
            fmt=marker,color=color,ms=7,capsize=4,lw=1.4,label=name if name not in used else None,zorder=3)
        used.add(name)
        ax.annotate(f"{'BE' if be else 'Radau'}, Δt={row['dt']:g}",(median,error),xytext=offsets[row['case_id']],
                    textcoords='offset points',ha='right' if row['case_id']=='Radau_coarse' else 'left',fontsize=10)
    ax.set_xscale('log');ax.set_yscale('log');ax.set_xlim(.12,4.5);ax.set_ylim(1.8e-7,2.5e-4)
    for target,label in ((1e-4,'Loose target: 10⁻⁴'),(3e-6,'Tight target: 3×10⁻⁶')):
        ax.axhline(target,color='#777777',ls='--',lw=.9,alpha=.75,zorder=1)
        ax.text(.98,target,label,transform=ax.get_yaxis_transform(),ha='right',va='bottom',fontsize=9,color='#555555')
    ax.set_xlabel('Warm core wall time: construction + advances (s)')
    ax.set_ylabel(r'$\|u_{h,\Delta t}(1)-u_{h,\mathrm{ref}}(1)\|_{L^2}$')
    ax.set_title('Fixed-mesh time accuracy and measured core cost')
    ax.xaxis.set_major_locator(FixedLocator([.2,.5,1,2,4]));ax.xaxis.set_major_formatter(FuncFormatter(lambda value,pos:f'{value:g}'));ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(which='major',alpha=.18);ax.legend(loc='center left',bbox_to_anchor=(.01,.58),frameon=False,fontsize=10)
    fig.supxlabel('CG P2/n16 • κ = 1 • T = 1 • AC power • one process/thread\nMarkers: medians; bars: min–max of three repeats. Four candidates; finite same-mesh reference.',fontsize=9)
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
    print('Regenerated accepted targeted accuracy/cost table and figure:',fig_stem)


if __name__=='__main__':
    main()

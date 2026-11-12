#!/usr/bin/env python3
"""Typeset the saved-evidence technical report using local TeX; no numerical imports or solves."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from prepare_report import ROOT,PROJECT,collect,digest,FIGURES

def main():
    subprocess.run([sys.executable,str(ROOT/'scripts/check_saved_results.py'),'--inputs-only'],check=True,cwd=ROOT)
    subprocess.run([sys.executable,str(ROOT/'scripts/prepare_report.py'),'--check'],check=True,cwd=ROOT)
    engine=shutil.which('pdflatex');info=shutil.which('pdfinfo')
    if not engine or not info:raise RuntimeError('Report building requires local pdflatex and pdfinfo; numerical dependencies are not modified.')
    build=PROJECT/'.build';build.mkdir(exist_ok=True)
    env=dict(os.environ,SOURCE_DATE_EPOCH=str(int(datetime(2026,11,12,tzinfo=timezone.utc).timestamp())),FORCE_SOURCE_DATE='1')
    argv=[engine,'-interaction=nonstopmode','-halt-on-error','-file-line-error','-output-directory',str(build),'main.tex']
    for iteration in (1,2):
        run=subprocess.run(argv,cwd=PROJECT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (build/f'compile-{iteration}.txt').write_text(run.stdout)
        if run.returncode:raise RuntimeError(f'Report TeX compilation failed; inspect {build}/compile-{iteration}.txt')
    output=PROJECT/'technical_report.pdf';shutil.copyfile(build/'main.pdf',output)
    details=subprocess.check_output([info,str(output)],text=True);pages=int(re.search(r'^Pages:\s+(\d+)',details,re.M).group(1))
    if not 8<=pages<=12:raise RuntimeError(f'Report has {pages} pages; review layout against the 8-12 page target.')
    log=(build/'main.log').read_text(errors='replace')
    if 'Overfull \\hbox' in log or 'Overfull \\vbox' in log or 'undefined references' in log or 'undefined citations' in log:
        raise RuntimeError('Report has an overflow or unresolved reference; inspect the saved TeX log and PDF before delivery.')
    inputs=[PROJECT/'main.tex',PROJECT/'evidence/data.tex',PROJECT/'evidence/claims.json',PROJECT/'evidence/primary_sources.json',ROOT/'scripts/prepare_report.py',ROOT/'scripts/check_saved_results.py',ROOT/'scripts/check_report_evidence.py',Path(__file__).resolve(),*[PROJECT/'images'/name for name in FIGURES]]
    manifest=dict(status='completed serial study; report revised 12 November 2026',operation='Saved-data typesetting only',numerical_experiment_run=False,
        pdf=dict(path=str(output.relative_to(ROOT)),sha256=digest(output),pages=pages),
        engine=engine,engine_version=subprocess.check_output([engine,'--version'],text=True).splitlines()[0],
        commands=[argv,argv],source_date_epoch=env['SOURCE_DATE_EPOCH'],input_sha256={str(p.relative_to(ROOT)):digest(p) for p in inputs},
        limitations=['No numerical solve or experiment rerun.', 'Native TeX is separate from the frozen FEM environment.', 'Cross-platform bitwise PDF reproducibility is not asserted.'])
    (PROJECT/'evidence/report-build.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Report compiled: {output} ({pages} pages; no overflow or unresolved reference reported).')

if __name__=='__main__':main()

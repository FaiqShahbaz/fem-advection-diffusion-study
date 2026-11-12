#!/usr/bin/env python3
"""Check published data and report arithmetic without a numerical installation.

This reads saved records. It does not regenerate fields, measure fresh runtimes,
or verify a PETSc binary. Use check_report_evidence.py on the original native
installation for the detailed environment and historical-source audit.
"""
import argparse
from fractions import Fraction
import importlib.abc
import json
import math
from pathlib import Path
import statistics
import sys

from prepare_report import ROOT, PROJECT, FIGURES, REVIEWS, collect, digest


class SavedDataOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'firedrake', 'irksome', 'petsc4py', 'mpi4py', 'numpy', 'scipy', 'h5py'}:
            raise RuntimeError('Saved-data checking must not import ' + fullname)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def same(a, b, label):
    require(math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15), label)


def read(path):
    return json.loads(path.read_text())


def exact_tableau(config):
    table = config['radau_formulation']['tableau_rational']
    A = [[Fraction(x) for x in row] for row in table['A']]
    b, c = ([Fraction(x) for x in table[k]] for k in ('b', 'c'))
    require(c == [sum(row) for row in A] and b == A[-1], 'Stage times or stiff accuracy changed')
    require(sum(b) == 1 and sum(b[i]*c[i] for i in range(2)) == Fraction(1, 2), 'First/second order conditions failed')
    require(sum(b[i]*c[i]**2 for i in range(2)) == Fraction(1, 3), 'Third order condition failed')
    require(sum(b[i]*sum(A[i][j]*c[j] for j in range(2)) for i in range(2)) == Fraction(1, 6), 'Third order coupling failed')
    factors = config['rana_tableau_rational']
    matrices = {k: [[Fraction(x) for x in row] for row in factors[k]] for k in ('L', 'D', 'U', 'Atilde')}
    def multiply(X, Y):
        return [[sum(X[i][k]*Y[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
    require(multiply(matrices['L'], matrices['D']) == matrices['Atilde'], 'LD approximation changed')
    require(multiply(matrices['Atilde'], matrices['U']) == A, 'LDU factorization changed')


def audit(check_pdf=True):
    sys.meta_path.insert(0, SavedDataOnly())
    verified = set()
    def checksum(path, expected):
        require(isinstance(path, str) and not Path(path).is_absolute(), 'Expected a repository-relative evidence path')
        target = (ROOT / path).resolve()
        require(target.is_relative_to(ROOT) and target.is_file(), 'Missing repository evidence: ' + path)
        require(digest(target) == expected, 'Evidence hash differs: ' + path)
        verified.add(path)

    text, ledger = collect()
    require((PROJECT/'evidence/data.tex').read_text() == text, 'Regenerate report tables')
    require(read(PROJECT/'evidence/claims.json') == ledger, 'Regenerate report evidence index')
    reviews, records = {}, {}
    # Check the report's six evidence sets and their immediate prerequisites.
    # Absolute installed-library paths are deliberately not dereferenced here.
    for name in REVIEWS:
        item = ledger['evidence_reviews'][name]
        checksum(item['path'], item['sha256'])
        review = reviews[name] = read(ROOT/item['path'])
        for key, value in review.items():
            if key.endswith('_sha256') and isinstance(value, str):
                source = review.get(key[:-7])
                if isinstance(source, str) and not Path(source).is_absolute():
                    checksum(source, value)
        for source, expected in review.get('run_script_sha256', {}).items():
            checksum(source, expected)
        if 'audit_source' in review:
            checksum(review['audit_source']['path'], review['audit_source']['sha256'])
        for measured in review['measurement_records'].values():
            checksum(measured['path'], measured['sha256'])
        raw = records[name] = read(ROOT/review['measurement_records']['serial']['path'])
        expected_status = 'collected' if name in {'solver_sensitivity', 'selected_trajectories'} else 'passed'
        require(raw['status'] == expected_status and raw['completed_study'] and raw['ranks'] == 1, 'Incomplete study: ' + name)
        require(raw['config'] == read(ROOT/review['config']), 'Raw/config mismatch: ' + name)
    checksum(ledger['verified_environment']['path'], ledger['verified_environment']['sha256'])
    for name, item in ledger['figures'].items():
        checksum(item['path'], item['sha256'])
        require(digest(PROJECT/'images'/name) == item['sha256'], 'Report figure differs: ' + name)

    spatial = records['spatial']
    by_case = {row['case_id']: row for row in spatial['measurements']}
    for rate in spatial['rates']:
        coarse, fine = (by_case[rate[k]] for k in ('coarse_case', 'fine_case'))
        value = math.log(coarse['metrics'][rate['metric']]/fine['metrics'][rate['metric']])/math.log(coarse['h']/fine['h'])
        same(value, rate['observed_order'], 'Spatial rate arithmetic differs')
    require(spatial['rates'] == reviews['spatial']['rates'], 'Spatial review differs')

    temporal = records['temporal']
    by_step = {(row['method'], row['dt']): row for row in temporal['curve']}
    for rate in temporal['rates']:
        coarse, fine = (by_step[(rate['method'], rate[k])] for k in ('coarse_dt', 'fine_dt'))
        value = math.log(coarse['time_errors'][rate['metric']]/fine['time_errors'][rate['metric']])/math.log(coarse['dt']/fine['dt'])
        same(value, rate['observed_order'], 'Temporal rate arithmetic differs')
    require(temporal['rates'] == reviews['temporal']['rates'], 'Temporal review differs')

    accuracy = records['accuracy_cost']
    summaries = reviews['accuracy_cost']['independent_audit']['candidate_summary']
    by_case = {row['case_id']: row for row in accuracy['curve']}
    for case in accuracy['config']['cases']:
        key = case['case_id']
        values = [accuracy['trajectories'][key+'_measured'+str(i)]['timing']['trajectory_wall_seconds'] for i in (1, 2, 3)]
        require(values == summaries[key]['wall']['values'], 'Accuracy/cost repeats differ')
        same(statistics.median(values), summaries[key]['wall']['median'], 'Accuracy/cost median differs')
        require(by_case[key]['time_errors'] == summaries[key]['time_errors'], 'Accuracy/cost errors differ')
    for selection in accuracy['candidate_selections']:
        eligible = [c['case_id'] for c in accuracy['config']['cases'] if c['method'] == selection['method']
                    and by_case[c['case_id']]['time_errors']['l2_error'] <= selection['target_l2_time_error']]
        require(eligible == selection['eligible_cases'], 'Accuracy target eligibility differs')
        winner = min(eligible, key=lambda k: summaries[k]['wall']['median'])
        require(winner == selection['selected_case'], 'Accuracy target selection differs')
    same(summaries['BE_fine']['wall']['median']/summaries['Radau_coarse']['wall']['median'],
         reviews['accuracy_cost']['independent_audit']['target_median_ratios']['3e-06']['BE_over_Radau_median'],
         'Tight-target runtime ratio differs')

    data, review = records['selected_cost'], reviews['selected_cost']['independent_audit']
    config = data['config']
    exact_tableau(config)
    checked = data['checked_study']
    require(checked['completed_candidate_trajectories'] == 8 and checked['reference_trajectories'] == 3, 'Checked trajectory count differs')
    require(checked['candidate_completed_steps'] == 80 and checked['reference_completed_steps'] == 30, 'Checked step count differs')
    expected_order = [c['case_id']+'_warmup0' for c in config['cases']]
    expected_order += [key+'_measured'+str(i) for i, keys in enumerate(config['measured_case_order'], 1) for key in keys]
    require(data['execution_order'] == expected_order and set(data['trajectories']) == set(expected_order), 'Missing or reordered timing samples')
    case_summary = review['case_summary']
    for case in config['cases']:
        key, point, solver = case['case_id'], case['point_id'], case['solver']
        baseline = checked['points'][point]['candidates'][solver]
        stored = review['checked_study']['point_summary'][point]['candidates'][solver]
        require(baseline['status'] == 'passed' and baseline['completed_steps'] == 10, 'Checked candidate incomplete')
        steps = [step['iterations'] for step in baseline['step_records']]
        require(sum(steps) == baseline['total_iterations'] == stored['total_iterations'] and steps == stored['step_iterations'], 'Checked iteration sum differs')
        params = config['rana_solver_parameters'] if solver == 'RanaLD' else config['asm_solver_parameters'] if solver == 'ASM' else {**config['solver_parameters'], 'mat_type': 'aij', 'snes_type': 'ksponly'}
        for suffix in ('_warmup0', '_measured1', '_measured2', '_measured3'):
            row = data['trajectories'][key+suffix]
            require(row['status'] == 'passed' and row['completed_steps'] == len(row['step_records']) == 10 and all(row['checks'].values()), 'Incomplete measured trajectory')
            require(row['solver_parameters'] == params and row['final_time'] == 1 and row['dt'] == .1, 'Measured parameters differ')
            require(row['total_iterations'] == sum(step['iterations'] for step in row['step_records']) == sum(steps), 'Measured iteration sum differs')
            require([step['iterations'] for step in row['step_records']] == steps, 'Measured work differs from checked work')
            for i, step in enumerate(row['step_records'], 1):
                require(step['step'] == i and step['converged_reason'] > 0 and 0 < step['iterations'] <= 1000, 'Measured solve failed')
                same(step['t_end'], i*.1, 'Step endpoint differs')
            final = row['final_equation_check']
            res = final['true_block_residual']
            target = max(params['ksp_atol'], params['ksp_rtol']*res['rhs_norm'])
            require(0 <= res['norm'] <= config['checks']['residual_tolerance_factor']*target, 'Saved final residual exceeds tolerance')
            for value in row['equivalence'].values():
                require(0 <= value['actual_field_difference'] <= value['allowed_difference'], 'Saved field comparison failed')
            timing = row['timing']
            require(math.isfinite(timing['trajectory_wall_seconds']) and timing['trajectory_wall_seconds'] > 0, 'Invalid measured runtime')
            same(sum(timing[k] for k in ('construction_seconds', 'first_step_seconds', 'remaining_steps_seconds', 'loop_bookkeeping_seconds')),
                 timing['trajectory_wall_seconds'], 'Runtime partition differs')
        values = [data['trajectories'][key+'_measured'+str(i)]['timing']['trajectory_wall_seconds'] for i in (1, 2, 3)]
        summary = case_summary[key]['core_seconds']
        require(values == summary['values'], 'Stored runtime samples differ')
        same(statistics.median(values), summary['median'], 'Selected-cost median differs')
        same(min(values), summary['minimum'], 'Selected-cost minimum differs')
        same(max(values), summary['maximum'], 'Selected-cost maximum differs')
    ratios = {}
    for point in ('baseline', 'mesh_refined', 'lower_diffusion'):
        value = case_summary[point+'_ASM']['core_seconds']['median']/case_summary[point+'_RanaLD']['core_seconds']['median']
        same(value, review['within_point_ratios'][point]['asm_over_rana_median_core_ratio'], 'ASM/Rana ratio differs')
        ratios[point] = value
    require(data['completed_trajectories'] == 11+len(data['trajectories']) == 43 and data['completed_steps'] == 110+sum(r['completed_steps'] for r in data['trajectories'].values()) == 430, 'Study totals differ')
    artifact = None
    if check_pdf:
        build = read(PROJECT/'evidence/report-build.json')
        artifact = build['pdf']
        require(8 <= artifact['pages'] <= 12, 'Unexpected report length')
        checksum(artifact['path'], artifact['sha256'])
        for path, expected in build['input_sha256'].items():
            checksum(path, expected)
    return dict(status='passed',mode='saved data only',repository_files_hash_checked=len(verified),
                selected_cost_trajectories_checked=43,selected_cost_steps_checked=430,
                baseline_outer_iterations={k: v['total_iterations'] for k, v in review['checked_study']['point_summary']['baseline']['candidates'].items()},
                asm_over_rana_median_runtime_ratios=ratios,report_claim_groups=len(ledger['claims']),report_figures=len(FIGURES),
                exact_Radau_order_three_and_LDU_identities=True,report_artifact=artifact,
                numerical_experiments_run=False,native_installation_checked=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs-only', action='store_true', help='Check source data before rebuilding the PDF.')
    args = parser.parse_args()
    print(json.dumps(audit(check_pdf=not args.inputs_only), indent=2))


if __name__ == '__main__':
    main()

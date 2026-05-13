#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Validation harness for null_engine models.

Runs a small batch of null mocks (default 10) for a given model and prints a
sanity report: NaN/Inf count, negative count, mean/min/max Δχ², and the ratio
of the mean to naive_dof. Catches obvious bugs before committing to a full
10k-mock run.

Usage (intended via the `null-engine --validate` CLI):
    from null_engine import validate_model, print_validation_report
    report = validate_model('wdagger', data_dir='./data', n_test=10, workers=2)
    status = print_validation_report(report)
"""

import math
import numpy as np
from multiprocessing import Pool

from null_engine.engine import init_worker, run_one_mock
from null_engine.models import MODELS


def validate_model(model_key, data_dir, n_test=10, workers=2,
                   seed_offset=999000, plugin_path=None):
    """Run n_test null mocks for model_key and return a report dict.
    If plugin_path is given, it is forwarded to init_worker so worker
    subprocesses can register the external plugin before lookup."""
    if model_key not in MODELS:
        raise ValueError(f"unknown model_key: {model_key!r}")
    spec = MODELS[model_key]
    naive_dof = spec['naive_dof']
    model_name = spec['name']

    seeds = [seed_offset + i for i in range(n_test)]

    with Pool(workers, initializer=init_worker,
              initargs=(data_dir, model_key, plugin_path)) as pool:
        dchi2 = pool.map(run_one_mock, seeds)

    arr = np.asarray(dchi2, dtype=float)
    bad_mask = np.isnan(arr) | np.isinf(arr)
    nan_inf_count = int(bad_mask.sum())
    good = arr[~bad_mask]

    if good.size > 0:
        mean = float(good.mean())
        mx = float(good.max())
        mn = float(good.min())
        negative_count = int((good < 0).sum())
        ratio_to_naive = mean / naive_dof if naive_dof else float('nan')
    else:
        mean = float('nan')
        mx = float('nan')
        mn = float('nan')
        negative_count = 0
        ratio_to_naive = float('nan')

    return {
        'model_key': model_key,
        'model_name': model_name,
        'n_test': n_test,
        'naive_dof': naive_dof,
        'dchi2': arr,
        'nan_inf_count': nan_inf_count,
        'negative_count': negative_count,
        'mean': mean,
        'max': mx,
        'min': mn,
        'ratio_to_naive': ratio_to_naive,
    }


def print_validation_report(report):
    """Print a human-readable validation report. Returns status string."""
    n = report['n_test']
    naive_dof = report['naive_dof']
    nan_inf = report['nan_inf_count']
    neg = report['negative_count']
    mean = report['mean']
    ratio = report['ratio_to_naive']

    notes = []
    status = 'PASS'

    # FAIL conditions
    if nan_inf > 0:
        status = 'FAIL'
        notes.append(f"{nan_inf} NaN/Inf returns — broken H_func or w_func")
    if not math.isnan(mean):
        if mean > 3 * naive_dof:
            status = 'FAIL'
            notes.append(f"mean Δχ² ({mean:.2f}) > 3× naive_dof ({naive_dof}) — "
                         "something is very wrong")
        if mean < 0.1 * naive_dof:
            status = 'FAIL'
            notes.append(f"mean Δχ² ({mean:.2f}) < 0.1× naive_dof ({naive_dof}) — "
                         "fits aren't moving")
    else:
        status = 'FAIL'
        notes.append("no finite Δχ² values — all mocks broken")

    # WARN conditions (only if not already FAIL)
    if status != 'FAIL':
        if not math.isnan(ratio) and (ratio < 0.5 or ratio > 2.0):
            status = 'WARN'
            notes.append(f"ratio_to_naive = {ratio:.2f}× — outside [0.5, 2.0] "
                         "(possible redundant params or scanning)")
        if neg > 0.3 * n:
            status = 'WARN'
            notes.append(f"{neg}/{n} negative Δχ² (>30%) — LCDM beating extended too often")

    # bounds-hit rate punted in v1
    notes.append("TODO: bounds-hit rate not computed in v1 "
                 "(run_one_mock returns Δχ² only, not best-fit params)")

    sep = '=' * 60
    dash = '-' * 60
    print(sep)
    print(f"Validation: {report['model_name']} ({report['model_key']})")
    print(f"  Test mocks: {n}")
    print(f"  Naive χ²({naive_dof}) expected mean: {naive_dof}")
    print(dash)
    if math.isnan(mean):
        print(f"  Δχ² mean: NaN (no finite values)")
    else:
        print(f"  Δχ² mean: {mean:.3f} (ratio to naive: {ratio:.2f}×)")
        print(f"  Δχ² range: [{report['min']:.3f}, {report['max']:.3f}]")
    print(f"  NaN/Inf:  {nan_inf}/{n}")
    print(f"  Negative: {neg}/{n}")
    print(dash)
    print(f"  Status: {status}")
    if notes:
        print(f"  Notes:")
        for note in notes:
            print(f"    - {note}")
    print(sep)

    return status

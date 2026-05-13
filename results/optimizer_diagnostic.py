#!/usr/bin/env python3
"""Test whether optimizer settings affect the null distribution."""
import numpy as np
import os, sys, time
from scipy.optimize import minimize
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))
import null_engine as ne

# Test model — change to '4pde' for 4-param calibration
MODEL_KEY = os.environ.get('DIAG_MODEL', 'gauss_eos')
N_MOCKS = 200  # enough to see systematic shifts
DATA_DIR = os.path.expanduser('~/null_engine')

def run_mock_flexible(args):
    """One mock with configurable optimizer settings."""
    seed, n_starts, method, maxiter = args
    np.random.seed(seed)
    model = ne.W['model']

    # Generate LCDM mock
    zg, dm = ne.comoving_distance(ne.H_lcdm, ne.H0T, ne.OMT)
    bao_true = ne.bao_predictions(ne.H_lcdm, zg, dm, ne.H0T, ne.OMT)
    bao_mock = bao_true + ne.BCH @ np.random.randn(13)
    sn_true = ne.sn_predictions(ne.H_lcdm, zg, dm)
    sn_mock = sn_true + ne.W['sch'] @ np.random.randn(ne.W['ns']) + np.random.normal(0, 0.01)
    cmb_true = ne.cmb_priors(ne.H0T, ne.OMT, ne.OBT, ne.H_lcdm)
    cmb_mock = cmb_true + ne.CCH @ np.random.randn(3)

    # Fit LCDM
    best_lcdm = 1e10
    best_lcdm_params = None
    for h in [67, 69]:
        for o in [0.29, 0.32]:
            r = minimize(ne.chi2_lcdm, [h, o, 0.02236],
                        args=(bao_mock, sn_mock, cmb_mock),
                        method='Nelder-Mead',
                        options={'maxiter': 1500, 'xatol': 1e-3, 'fatol': 0.05})
            if r.fun < best_lcdm:
                best_lcdm = r.fun
                best_lcdm_params = r.x

    # Fit extended — use requested number of starts and method
    starts = model['starts'][:n_starts]
    # If we need more starts than available, add random perturbations
    while len(starts) < n_starts:
        base = model['starts'][np.random.randint(len(model['starts']))]
        perturbed = tuple(b * (1 + 0.2 * np.random.randn()) for b in base)
        starts.append(perturbed)

    best_ext = 1e10
    for extra_start in starts:
        try:
            x0 = list(best_lcdm_params) + list(extra_start)
            r = minimize(ne.chi2_extended, x0,
                        args=(bao_mock, sn_mock, cmb_mock),
                        method=method,
                        options={'maxiter': maxiter, 'xatol': 1e-3, 'fatol': 0.05}
                        if method == 'Nelder-Mead' else {'maxiter': maxiter})
            if r.fun < best_ext:
                best_ext = r.fun
        except:
            pass

    return best_lcdm - best_ext


def run_mock_engine(seed):
    """Use the engine's actual run_one_mock (24 starts + boundary retry)."""
    return ne.run_one_mock(seed)


# Configurations to test
configs = [
    ("12starts_NM_2000",  12, 'Nelder-Mead', 2000),  # old default
    ("24starts_NM_2000",  24, 'Nelder-Mead', 2000),  # double starts
    ("12starts_NM_4000",  12, 'Nelder-Mead', 4000),  # double iterations
    ("12starts_Powell",   12, 'Powell',      2000),   # different optimizer
    ("24starts_Powell",   24, 'Powell',      2000),   # Powell + more starts
    ("engine_24+retry",   -1, 'engine',      -1),     # actual engine logic
]

if __name__ == '__main__':
    seeds = list(range(500000, 500000 + N_MOCKS))

    for label, n_starts, method, maxiter in configs:
        print(f"\n{'='*50}")
        print(f"Config: {label}")
        print(f"  starts={n_starts}  method={method}  maxiter={maxiter}")

        t0 = time.time()

        with Pool(6, initializer=ne.init_worker,
                  initargs=(DATA_DIR, MODEL_KEY)) as pool:
            if method == 'engine':
                results = pool.map(run_mock_engine, seeds)
            else:
                args_list = [(s, n_starts, method, maxiter) for s in seeds]
                results = pool.map(run_mock_flexible, args_list)

        a = np.array(results)
        elapsed = time.time() - t0
        print(f"  time: {elapsed:.0f}s ({elapsed/N_MOCKS:.2f}s/mock)")
        print(f"  mean={a.mean():.3f}  median={np.median(a):.3f}  std={a.std():.3f}")
        print(f"  max={a.max():.3f}  p95={np.percentile(a,95):.3f}  p99={np.percentile(a,99):.3f}")
        print(f"  frac<0.5: {np.mean(a<0.5):.1%}")

        np.save(f'diag_{label}.npy', a)

    print("\n\nDone. If means differ significantly, optimizer is a systematic.")

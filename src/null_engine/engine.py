#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Unified null mock engine for look-elsewhere effect (LEE) calibration
across multiple dark energy parameterizations.

Generates ΛCDM null mocks against SN (Pantheon+), BAO (DESI DR1), and CMB
shift-parameter priors, and fits each mock with:
  1. ΛCDM baseline
  2. One extended model with free location parameters
The Δχ² distribution over many mocks gives the empirical null for that model.

Built-in models:
  - yates      Two-Gaussian κ(z) perturbation (Yates, this work)
  - wdagger    w†CDM abrupt transition (Scherer et al. 2025)
  - lambda_s   Λ_sCDM sign-switching (Akarsu et al. 2024)
  - gauss_eos  Gaussian EoS bump (Hussain et al. 2025)
  - cplwb      CPL + quadratic (Mukherjee et al. 2510.03779)
  - gde        Generalized DE (Barboza & Alcaniz)
  - pade       Padé EoS (Rezaei et al.)
  - bellde     Gaussian bump + baseline (Hussain et al. 2505.09913)
  - 4pde       4-parameter EoS (Akarsu et al. 2512.08752)

External plugins (bring your own H(z) or w(z)) — see plugin_loader.py.

Usage:
    python null_engine.py --model wdagger -n 10000 -w 8
    python null_engine.py --model all -n 10000 -w 8
    python null_engine.py --model-file my_model.py -n 10000 -w 8
    python null_engine.py --model-file my_model.py --validate
    python null_engine.py --combine file1.npz file2.npz -o combined.npz

Requirements: numpy, scipy
Data files (in --data-dir): sn_zHD_1701.npy, sn_zHEL_1701.npy, sn_inv_cov_1701.npy
"""

import numpy as np
import os, sys, time, argparse, warnings
from scipy.optimize import minimize
from multiprocessing import Pool, cpu_count
warnings.filterwarnings('ignore')
trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))

# ---------------------------------------------------------------------------
# Physical constants and fiducial cosmology
# ---------------------------------------------------------------------------
C = 299792.458          # km/s
OG = 2.469e-5           # Omega_gamma * h^2 (CMB photon density)
ORH2 = OG * (1 + 3.046 * (7/8) * (4/11)**(4/3))  # Omega_rad * h^2 (photons + 3.046 neutrino species)
RB0 = 3 / (4 * OG)      # baryon-to-photon ratio prefactor for sound speed
RD = 147.09             # sound horizon at drag epoch (Mpc); DESI DR1 fiducial

# LCDM fiducial for mock generation. Pantheon+ + DESI DR1 + Planck-like CMB
# best fit; the null distribution is conditioned on this point in parameter
# space. Vary cautiously — every npz output is tied to these numbers.
H0T, OMT, OBT = 68.36, 0.3019, 0.02260

# BAO data points (DESI DR1 format)
BAO_PTS = [
    (0.295, 7.94167639, 'DV'),
    (0.510, 13.58758434, 'DM'), (0.510, 21.86294686, 'DH'),
    (0.706, 17.35069094, 'DM'), (0.706, 19.45534918, 'DH'),
    (0.934, 21.57563956, 'DM'), (0.934, 17.64149464, 'DH'),
    (1.321, 27.60085612, 'DM'), (1.321, 14.17602155, 'DH'),
    (1.484, 30.51190063, 'DM'), (1.484, 12.81699964, 'DH'),
    (2.330, 8.631545674846294, 'DH'),
    (2.330, 38.988973961958784, 'DM'),
]

BAO_COV = np.array([
    [5.78998687e-03,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,2.83473742e-02,-3.26062007e-02,0,0,0,0,0,0,0,0,0,0],
    [0,-3.26062007e-02,1.83928040e-01,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,3.23752442e-02,-2.37445646e-02,0,0,0,0,0,0,0,0],
    [0,0,0,-2.37445646e-02,1.11469198e-01,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,2.61732816e-02,-1.12938006e-02,0,0,0,0,0,0],
    [0,0,0,0,0,-1.12938006e-02,4.04183878e-02,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,1.05336516e-01,-2.90308418e-02,0,0,0,0],
    [0,0,0,0,0,0,0,-2.90308418e-02,5.04233092e-02,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,5.83020277e-01,-1.95215562e-01,0,0],
    [0,0,0,0,0,0,0,0,0,-1.95215562e-01,2.68336193e-01,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,1.02136194e-02,-2.31395216e-02],
    [0,0,0,0,0,0,0,0,0,0,0,-2.31395216e-02,2.82685779e-01],
])

CMB_IC = np.array([
    [94392.3971, -1360.4913, 1664517.2916],
    [-1360.4913, 161.4349, 3671.6180],
    [1664517.2916, 3671.6180, 79719182.5162],
])

# CMB shift-parameter pivots (R and l_A reference values). These cancel
# identically in the null-mock comparison — the fitter and the mock generator
# both subtract them — but are retained for parity with the real-data form
# of cmb_priors, so the same likelihood code can be used outside this engine.
OR_SHIFT, OL_SHIFT = -0.000569, 0.8592

# Integration grids
ZG = np.linspace(0, 2.5, 300); DZG = ZG[1] - ZG[0]
ZL = np.linspace(0.001, 10, 150)
ZHH = np.geomspace(10, 1200, 400)
ZR = np.geomspace(1, 1e7, 1500)

# Precomputed matrix operations
BI = np.linalg.inv(BAO_COV)
BCH = np.linalg.cholesky(BAO_COV)
CCH = np.linalg.cholesky(np.linalg.inv(CMB_IC))


# ===========================================================================
# MODEL DEFINITIONS
# ===========================================================================
# All built-in H(z) functions and the MODELS registry live in models.py.
# Re-exported here so legacy code reading null_engine.engine.MODELS keeps
# working. External plugins extend MODELS via plugin_loader; see model_spec
# for the canonical spec shape.
# ===========================================================================

from null_engine.models import (
    MODELS, SIG2_YATES, _de_density_from_wz,
    H_lcdm, H_yates, H_wdagger, H_lambda_s, H_4pde,
    H_gauss_eos, H_cplwb, H_gde, H_pade, H_bellde,
)




# ---------------------------------------------------------------------------
# Worker globals (loaded once per process)
# ---------------------------------------------------------------------------
W = {}

def init_worker(data_dir, model_key, plugin_path=None):
    """Called once per worker. Loads SN data and model config.
    If plugin_path is given, registers the external model in this worker's
    MODELS table before lookup (workers spawn fresh and don't inherit
    plugins registered in the parent process)."""
    if plugin_path is not None and model_key not in MODELS:
        from null_engine.plugin_loader import register_external_model
        register_external_model(MODELS, plugin_path)
    sz = np.load(os.path.join(data_dir, 'sn_zHD_1701.npy'))
    sh = np.load(os.path.join(data_dir, 'sn_zHEL_1701.npy'))
    si = np.load(os.path.join(data_dir, 'sn_inv_cov_1701.npy'))
    ns = len(sz)
    ss = float(np.sum(si))
    slc = float(np.log(ss / (2 * np.pi)))
    sf = si.astype(np.float32)

    # SN covariance for mock generation (inverse of inverse)
    ev, U = np.linalg.eigh(si)
    sc = U @ np.diag(1 / np.maximum(ev, 1e-10)) @ U.T
    sc = 0.5 * (sc + sc.T)
    try:
        sch = np.linalg.cholesky(sc)
    except np.linalg.LinAlgError:
        sc += np.eye(ns) * 1e-6
        sch = np.linalg.cholesky(sc)

    W['sz'] = sz; W['sh'] = sh; W['sf'] = sf; W['ss'] = ss
    W['slc'] = slc; W['sch'] = sch; W['ns'] = ns
    W['model'] = MODELS[model_key]


# ---------------------------------------------------------------------------
# Cosmology functions (shared infrastructure)
# ---------------------------------------------------------------------------
def comoving_distance(Hfunc, *args):
    """Compute comoving distance on the integration grid."""
    H = Hfunc(ZG, *args)
    if np.any(np.isnan(H) | np.less_equal(H, 0)):
        return None, None
    inv_H = C / H
    d = np.zeros(len(ZG))
    d[1:] = np.cumsum(0.5 * (inv_H[:-1] + inv_H[1:])) * DZG
    return ZG, d

def cmb_priors(H0, Om, ob, Hfunc, *extra):
    """CMB shift parameter priors."""
    h = H0 / 100; Orr = ORH2 / h**2; omh2 = Om * h**2; Rb = RB0 * ob
    g1 = 0.0783 * ob**(-0.238) / (1 + 39.5 * ob**0.763)
    g2 = 0.56 / (1 + 21.1 * ob**1.81)
    zs = 1048 * (1 + 0.00124 * ob**(-0.738)) * (1 + g1 * omh2**g2)

    Hv = Hfunc(ZL, H0, Om, *extra)
    if np.any(np.isnan(Hv) | np.less_equal(Hv, 0)):
        return None
    dc = trapz(C / Hv, ZL)

    zz = np.append(ZHH[ZHH <= zs], zs)
    dc += trapz(C / (H0 * np.sqrt(Orr*(1+zz)**4 + Om*(1+zz)**3)), zz)

    zrr = np.concatenate([[zs], ZR[ZR >= zs]])
    Hr = H0 * np.sqrt(Orr*(1+zrr)**4 + Om*(1+zrr)**3)
    rs = trapz(C / (Hr * np.sqrt(3*(1 + Rb/(1+zrr)))), zrr)

    if dc <= 0 or rs <= 0:
        return None
    return np.array([np.sqrt(Om) * (H0/C) * dc - OR_SHIFT,
                     np.pi * dc / rs - OL_SHIFT, ob])

def bao_predictions(Hfunc, zg, dm, H0, Om, *extra):
    """BAO distance predictions."""
    m = np.zeros(13)
    for i, (z, v, q) in enumerate(BAO_PTS):
        H = Hfunc(z, H0, Om, *extra)
        DH = C / H
        D = np.interp(z, zg, dm)
        if D <= 0:
            return None
        if q == 'DV':
            m[i] = (z * DH * D**2)**(1/3) / RD
        elif q == 'DM':
            m[i] = D / RD
        elif q == 'DH':
            m[i] = DH / RD
    return m

def sn_predictions(Hfunc, zg, dm, *extra):
    """SN distance modulus predictions."""
    D = np.interp(W['sz'], zg, dm)
    dL = (1 + W['sh']) * D
    if np.any(dL <= 0):
        return None
    return 5 * np.log10(dL) + 25

def chi2_sn(mu_pred, mu_mock):
    """SN chi-squared with analytic marginalisation over M_B."""
    d = (mu_pred - mu_mock).astype(np.float32)
    t = W['sf'] @ d
    return float(d @ t) - float(np.sum(t))**2 / W['ss'] + W['slc']


# ---------------------------------------------------------------------------
# Likelihood functions
# ---------------------------------------------------------------------------
def chi2_lcdm(theta, bao_mock, sn_mock, cmb_mock):
    """LCDM chi-squared."""
    H0, Om, ob = theta
    if not (55 < H0 < 85 and 0.15 < Om < 0.50 and 0.018 < ob < 0.026):
        return 1e10
    zg, dm = comoving_distance(H_lcdm, H0, Om)
    if zg is None:
        return 1e10
    b = bao_predictions(H_lcdm, zg, dm, H0, Om)
    s = sn_predictions(H_lcdm, zg, dm)
    p = cmb_priors(H0, Om, ob, H_lcdm)
    if b is None or s is None or p is None:
        return 1e10
    return (float((bao_mock - b) @ BI @ (bao_mock - b)) +
            chi2_sn(s, sn_mock) +
            float((p - cmb_mock) @ CMB_IC @ (p - cmb_mock)))

def chi2_extended(theta, bao_mock, sn_mock, cmb_mock):
    """Extended model chi-squared. Model determined by worker state."""
    model = W['model']
    H0, Om, ob = theta[:3]
    extra = theta[3:]

    if not (55 < H0 < 85 and 0.15 < Om < 0.50 and 0.018 < ob < 0.026):
        return 1e10
    if not model['bounds'](extra):
        return 1e10

    Hfunc = model['H_func']
    zg, dm = comoving_distance(Hfunc, H0, Om, *extra)
    if zg is None:
        return 1e10
    b = bao_predictions(Hfunc, zg, dm, H0, Om, *extra)
    s = sn_predictions(Hfunc, zg, dm, *extra)
    p = cmb_priors(H0, Om, ob, Hfunc, *extra)
    if b is None or s is None or p is None:
        return 1e10
    return (float((bao_mock - b) @ BI @ (bao_mock - b)) +
            chi2_sn(s, sn_mock) +
            float((p - cmb_mock) @ CMB_IC @ (p - cmb_mock)))


# ---------------------------------------------------------------------------
# Worker function
# ---------------------------------------------------------------------------
def run_one_mock(seed):
    """Generate one LCDM mock, fit both models, return delta-chi2."""
    np.random.seed(seed)
    model = W['model']

    # Generate LCDM mock data
    zg, dm = comoving_distance(H_lcdm, H0T, OMT)
    bao_true = bao_predictions(H_lcdm, zg, dm, H0T, OMT)
    bao_mock = bao_true + BCH @ np.random.randn(13)
    sn_true = sn_predictions(H_lcdm, zg, dm)
    # Two noise contributions: (1) correlated SN noise via the Cholesky factor of
    # the full SN covariance, (2) a scalar coherent offset (~0.01 mag) standing in
    # for a sample-wide M_B nuisance shift the analytic marginalization absorbs.
    sn_mock = sn_true + W['sch'] @ np.random.randn(W['ns']) + np.random.normal(0, 0.01)
    cmb_true = cmb_priors(H0T, OMT, OBT, H_lcdm)
    cmb_mock = cmb_true + CCH @ np.random.randn(3)

    # Fit LCDM (4 starting points)
    best_lcdm = 1e10
    best_lcdm_params = None
    for h in [67, 69]:
        for o in [0.29, 0.32]:
            r = minimize(chi2_lcdm, [h, o, 0.02236],
                        args=(bao_mock, sn_mock, cmb_mock),
                        method='Nelder-Mead',
                        options={'maxiter': 1500, 'xatol': 1e-3, 'fatol': 0.05})
            if r.fun < best_lcdm:
                best_lcdm = r.fun
                best_lcdm_params = r.x

    # Fit extended model — 24 starts: 12 grid + 12 random perturbations
    all_starts = list(model['starts'])
    for base in model['starts']:
        perturbed = tuple(b * (1 + 0.2 * np.random.randn()) for b in base)
        all_starts.append(perturbed)

    best_ext = 1e10
    best_ext_params = None
    for extra_start in all_starts:
        try:
            x0 = list(best_lcdm_params) + list(extra_start)
            r = minimize(chi2_extended, x0,
                        args=(bao_mock, sn_mock, cmb_mock),
                        method='Nelder-Mead',
                        options={'maxiter': 2000, 'xatol': 1e-3, 'fatol': 0.05})
            if r.fun < best_ext:
                best_ext = r.fun
                best_ext_params = r.x
        except Exception:
            pass

    # Boundary retry: if best fit is near a bound, retry with random interior starts
    bound_ranges = model.get('bound_ranges')
    if best_ext_params is not None and bound_ranges is not None:
        extra_params = best_ext_params[3:]
        at_bound = any(
            abs(p - lo) < 0.02 * (hi - lo) or abs(p - hi) < 0.02 * (hi - lo)
            for p, (lo, hi) in zip(extra_params, bound_ranges)
        )
        if at_bound:
            for _ in range(12):
                try:
                    rand_extra = [np.random.uniform(lo + 0.05*(hi-lo), hi - 0.05*(hi-lo))
                                  for lo, hi in bound_ranges]
                    x0 = list(best_lcdm_params) + rand_extra
                    r = minimize(chi2_extended, x0,
                                args=(bao_mock, sn_mock, cmb_mock),
                                method='Nelder-Mead',
                                options={'maxiter': 2000, 'xatol': 1e-3, 'fatol': 0.05})
                    if r.fun < best_ext:
                        best_ext = r.fun
                except Exception:
                    pass

    return best_lcdm - best_ext


# ---------------------------------------------------------------------------
# Combine mode
# ---------------------------------------------------------------------------
def combine_results(files, output):
    all_d = []
    for f in files:
        a = np.load(f)['dchi2_null']
        print(f"  {f}: {len(a)} mocks, mean={a.mean():.2f}, max={a.max():.2f}")
        all_d.extend(a)
    c = np.array(all_d)
    np.savez(output, dchi2_null=c)
    print(f"\nCombined: {len(c)} mocks, mean={c.mean():.3f}, max={c.max():.3f}")
    from scipy import stats
    dofs = range(1, 8)
    ks_stats = {dof: stats.kstest(c, 'chi2', args=(dof,)) for dof in dofs}
    closest_dof = min(ks_stats, key=lambda d: ks_stats[d][0])
    for dof in dofs:
        ks, p = ks_stats[dof]
        marker = ' <-- closest' if dof == closest_dof else ''
        print(f"  chi2({dof}): KS={ks:.4f}, p={p:.2e}{marker}")
    for t in [10, 15, 20, 24]:
        n = np.sum(c > t)
        f = n / len(c)
        sig = stats.norm.isf(f) if f > 0 else float('inf')
        print(f"  > {t}: {n}/{len(c)} = {f:.2e} ({sig:.1f}σ)")
    # False positive rates against naive chi^2(k) where k is rounded mean
    naive_dof = round(c.mean())
    for cl, name in [(0.95, '95%'), (0.99, '99%'), (0.999, '99.9%')]:
        thresh = stats.chi2.ppf(cl, naive_dof)
        fpr = np.mean(c > thresh)
        expected = 1 - cl
        inflation = fpr / expected
        print(f"  FPR at {name} (chi2({naive_dof}) thresh={thresh:.2f}): "
              f"{fpr*100:.2f}% vs {expected*100:.2f}% expected = {inflation:.1f}x")
    print(f"\nSaved to {output}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Null mock engine for look-elsewhere calibration')
    parser.add_argument('--model', type=str, default='yates',
                       choices=list(MODELS.keys()) + ['all'],
                       help='Model to test (or "all" for all models)')
    parser.add_argument('-n', '--num-mocks', type=int, default=10000)
    parser.add_argument('-w', '--workers', type=int, default=0)
    parser.add_argument('--seed-offset', type=int, default=100000)
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Directory containing sn_zHD_1701.npy, sn_zHEL_1701.npy, '
                            'sn_inv_cov_1701.npy (default: ./data, i.e. relative to '
                            'the current working directory).')
    parser.add_argument('-o', '--output', type=str, default='')
    parser.add_argument('--checkpoint-interval', type=int, default=500)
    parser.add_argument('--combine', nargs='+', metavar='FILE')
    parser.add_argument('--model-file', type=str, default=None,
                       help='Path to external plugin .py exposing a MODEL dict '
                            '(or register() returning one). The plugin\'s key '
                            'becomes the --model argument.')
    parser.add_argument('--validate', action='store_true',
                       help='Run a small validation batch (default 10 mocks) '
                            'and print a sanity report instead of the full run.')
    parser.add_argument('--validate-n', type=int, default=10,
                       help='Number of mocks for --validate (default 10).')
    args = parser.parse_args()

    if args.combine:
        combine_results(args.combine, args.output or 'null_combined.npz')
        return

    if args.model_file:
        from null_engine.plugin_loader import register_external_model
        plugin_key = register_external_model(MODELS, args.model_file)
        print(f"Registered external model: {plugin_key} (from {args.model_file})")
        args.model = plugin_key

    if args.validate:
        from null_engine.validator import validate_model, print_validation_report
        models_to_validate = list(MODELS.keys()) if args.model == 'all' else [args.model]
        plugin_path = args.model_file
        n_fail = 0
        for mk in models_to_validate:
            report = validate_model(mk, data_dir=args.data_dir,
                                    n_test=args.validate_n,
                                    workers=max(2, min(args.workers or 4, 4)),
                                    plugin_path=plugin_path)
            status = print_validation_report(report)
            if status == 'FAIL':
                n_fail += 1
        sys.exit(1 if n_fail > 0 else 0)

    models_to_run = list(MODELS.keys()) if args.model == 'all' else [args.model]

    if args.workers <= 0:
        args.workers = max(1, cpu_count() - 2)

    # Verify data files
    for fn in ['sn_zHD_1701.npy', 'sn_zHEL_1701.npy', 'sn_inv_cov_1701.npy']:
        if not os.path.exists(os.path.join(args.data_dir, fn)):
            print(f"ERROR: {fn} not found in {args.data_dir}/")
            sys.exit(1)

    for model_key in models_to_run:
        model = MODELS[model_key]
        output = args.output or f'null_{model_key}_N{args.num_mocks}.npz'
        ckpt = output.replace('.npz', '_checkpoint.npz')

        print(f"\n{'='*60}")
        print(f"Model: {model['name']} ({model_key})")
        print(f"  Reference: {model['ref']}")
        print(f"  Extra params: {model['n_extra']}")
        print(f"  Naive dof: {model['naive_dof']}")
        print(f"  Starting points: {len(model['starts'])}")
        print(f"  Mocks: {args.num_mocks}")
        print(f"  Workers: {args.workers}")
        print(f"  Output: {output}")
        print(f"{'='*60}\n")

        # Resume from checkpoint
        completed = []
        if os.path.exists(ckpt):
            completed = list(np.load(ckpt)['dchi2_null'])
            print(f"Resuming from checkpoint: {len(completed)} done")

        start_idx = len(completed)
        n_rem = args.num_mocks - start_idx
        if n_rem <= 0:
            print("All mocks done!")
            continue

        seeds = list(range(args.seed_offset + start_idx,
                          args.seed_offset + args.num_mocks))

        t0 = time.time()
        results = list(completed)
        chunk = args.checkpoint_interval

        with Pool(args.workers, initializer=init_worker,
                  initargs=(args.data_dir, model_key, args.model_file)) as pool:
            for cs in range(0, n_rem, chunk):
                ce = min(cs + chunk, n_rem)
                batch = pool.map(run_one_mock, seeds[cs:ce])
                results.extend(batch)
                n_done = len(results)

                arr = np.array(results)
                el = time.time() - t0
                rate = (n_done - start_idx) / el if el > 0 else 0
                eta = (args.num_mocks - n_done) / rate if rate > 0 else 0

                print(f"  [{n_done:7d}/{args.num_mocks}] "
                      f"mean={arr.mean():.2f} max={arr.max():.2f} "
                      f"rate={rate:.1f}/s ETA={eta/60:.0f}min")

                np.savez(ckpt, dchi2_null=np.array(results))

        final = np.array(results)
        np.savez(output, dchi2_null=final)
        if os.path.exists(ckpt):
            os.remove(ckpt)

        el = time.time() - t0
        print(f"\nDONE [{model_key}]: {len(final)} mocks in {el/3600:.1f}h")
        print(f"  mean={final.mean():.3f} max={final.max():.3f}")
        print(f"  naive chi2({model['naive_dof']}) mean={model['naive_dof']:.1f}")
        print(f"  inflation: {(final.mean()/model['naive_dof'] - 1)*100:.0f}%")


if __name__ == '__main__':
    main()

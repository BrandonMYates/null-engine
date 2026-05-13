# SPDX-License-Identifier: MIT
"""Unit tests: fast checks on individual engine components.

No mock generation, no multiprocessing. Each test should run in well
under a second.
"""

import numpy as np
import pytest

from null_engine import MODELS, normalize_model_spec
from null_engine.models import (
    H_lcdm, _de_density_from_wz,
)
from null_engine.model_spec import (
    _bounds_from_ranges, _starts_from_ranges, _h_from_w,
)


# ---------------------------------------------------------------------------
# H_func sanity: each built-in returns finite positive H over a sane z range
# ---------------------------------------------------------------------------

Z_TEST = np.linspace(0.01, 3.0, 50)


@pytest.mark.parametrize('model_key', list(MODELS.keys()))
def test_builtin_H_func_finite_positive(model_key):
    """Every built-in H_func must return finite positive values
    when evaluated at its first starting point (which is by construction
    a sane interior point in parameter space)."""
    spec = MODELS[model_key]
    H_func = spec['H_func']
    starts = spec['starts'][0]
    H = H_func(Z_TEST, 68.36, 0.3019, *starts)
    assert np.all(np.isfinite(H)), f"{model_key}: non-finite H values"
    assert np.all(H > 0), f"{model_key}: non-positive H values"
    # Sanity: H should be increasing monotonically over this z range
    # (no model in the suite is so pathological that this fails at a sane start)
    assert H[-1] > H[0], f"{model_key}: H not increasing with z"


def test_lcdm_at_fiducial():
    """H_lcdm at fiducial cosmology returns the expected value at z=0."""
    H = H_lcdm(np.array([0.0]), 68.36, 0.3019)
    assert np.isclose(H[0], 68.36, rtol=1e-10)


# ---------------------------------------------------------------------------
# w_func -> H_func (via _h_from_w) matches analytic ΛCDM as a special case
# ---------------------------------------------------------------------------

def test_w_func_matches_lcdm_when_w_is_minus_one():
    """w(z) = -1 (cosmological constant) should produce the same H(z)
    as the analytic LCDM expression to high precision."""
    def w_const(z):
        return -1.0 * np.ones_like(z)

    H_via_w = _h_from_w(w_const)
    H_analytic = H_lcdm(Z_TEST, 68.36, 0.3019)
    H_numerical = H_via_w(Z_TEST, 68.36, 0.3019)
    np.testing.assert_allclose(H_numerical, H_analytic, rtol=1e-4)


def test_w_func_cpl_matches_engine_cpl():
    """w(a) = w0 + wa*(1-a) via _h_from_w should match the built-in
    H_cplwb (with wb=0) to high precision."""
    from null_engine.models import H_cplwb

    def w_cpl(z, w0, wa):
        a = 1.0 / (1.0 + z)
        return w0 + wa * (1 - a)

    H_via_w = _h_from_w(w_cpl)
    H_builtin = H_cplwb(Z_TEST, 68.36, 0.3019, -0.95, 0.1, 0.0)
    H_plugin = H_via_w(Z_TEST, 68.36, 0.3019, -0.95, 0.1)
    np.testing.assert_allclose(H_plugin, H_builtin, rtol=1e-3)


# ---------------------------------------------------------------------------
# Spec normalization: the canonical 10-key shape
# ---------------------------------------------------------------------------

CANONICAL_KEYS = {'key', 'name', 'ref', 'H_func', 'n_extra', 'bound_ranges',
                  'bounds', 'bounds_extra', 'starts', 'naive_dof'}


def test_normalize_minimal_H_func_spec():
    """Bare-minimum spec with H_func: normalize fills in everything else."""
    def H_test(z, H0, Om, p):
        return H_lcdm(z, H0, Om) * (1 + 1e-3 * p)

    raw = {
        'key': 'test_min',
        'name': 'minimal test',
        'H_func': H_test,
        'bound_ranges': [(-1.0, 1.0)],
    }
    norm = normalize_model_spec(raw)
    assert set(norm.keys()) == CANONICAL_KEYS
    assert norm['n_extra'] == 1
    assert norm['naive_dof'] == 1  # default == n_extra
    assert norm['ref'] == ''
    assert callable(norm['bounds'])
    assert len(norm['starts']) >= 12


def test_normalize_minimal_w_func_spec():
    """w_func form: normalize wraps it into an H_func."""
    def w_test(z, w0):
        return -1.0 + w0 * np.ones_like(z)

    raw = {
        'key': 'test_w',
        'name': 'minimal w test',
        'w_func': w_test,
        'bound_ranges': [(-0.5, 0.5)],
    }
    norm = normalize_model_spec(raw)
    assert callable(norm['H_func'])
    H = norm['H_func'](Z_TEST, 68.36, 0.3019, 0.0)
    assert np.all(np.isfinite(H))
    assert np.all(H > 0)


def test_normalize_rejects_missing_required():
    """Missing required fields must raise ValueError."""
    with pytest.raises(ValueError):
        normalize_model_spec({'name': 'x', 'H_func': H_lcdm,
                              'bound_ranges': [(0, 1)]})  # no 'key'

    with pytest.raises(ValueError):
        normalize_model_spec({'key': 'x', 'name': 'x',
                              'bound_ranges': [(0, 1)]})  # no H_func or w_func


def test_normalize_rejects_both_H_and_w():
    """Providing both H_func and w_func is ambiguous -> ValueError."""
    with pytest.raises(ValueError):
        normalize_model_spec({
            'key': 'x', 'name': 'x',
            'H_func': H_lcdm, 'w_func': lambda z: -np.ones_like(z),
            'bound_ranges': [(0, 1)],
        })


# ---------------------------------------------------------------------------
# bounds and starts helpers
# ---------------------------------------------------------------------------

def test_bounds_strict_inequality():
    """_bounds_from_ranges uses strict < (matches engine's existing lambdas)."""
    bounds = _bounds_from_ranges([(0.0, 1.0)])
    assert bounds([0.5]) is True
    assert bounds([0.0]) is False  # strict
    assert bounds([1.0]) is False  # strict
    assert bounds([-0.1]) is False
    assert bounds([1.1]) is False


def test_bounds_extra_predicate():
    """bounds_extra layered on top of box check."""
    bounds = _bounds_from_ranges(
        [(0.0, 1.0), (0.0, 1.0)],
        bounds_extra=lambda p: p[0] < p[1],
    )
    assert bounds([0.3, 0.7]) is True
    assert bounds([0.7, 0.3]) is False  # box OK, extra fails


def test_starts_in_interior():
    """_starts_from_ranges samples in the documented [5%, 95%] interior band."""
    ranges = [(-1.0, 1.0), (0.0, 10.0)]
    starts = _starts_from_ranges(ranges, n=12, seed=42)
    assert len(starts) == 12
    for s in starts:
        assert -0.9 < s[0] < 0.9   # 5%-95% of [-1,1] = [-0.9, 0.9]
        assert 0.5 < s[1] < 9.5    # 5%-95% of [0,10]


# ---------------------------------------------------------------------------
# _de_density_from_wz: f=1 when w=-1 (cosmological constant)
# ---------------------------------------------------------------------------

def test_de_density_at_w_minus_one():
    """w(z) = -1 yields rho_DE constant -> f(z) = 1."""
    zz = np.linspace(0, 3, 500)
    w_z = -1.0 * np.ones_like(zz)
    f = _de_density_from_wz(zz, zz, w_z)
    np.testing.assert_allclose(f, 1.0, atol=1e-6)

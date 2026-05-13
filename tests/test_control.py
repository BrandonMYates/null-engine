# SPDX-License-Identifier: MIT
"""Slow integration test: 20 mocks of CPL-wb (the control model).

CPL-wb is a 3-parameter polynomial extension with zero location
parameters, so its empirical mean Δχ² should land close to the naive
χ²(3) = 3.0 expectation. We check that the engine produces finite,
positive-mean Δχ² values in the right ballpark.

This is the only test that exercises the full mock-generation +
dual-fit pipeline. Sized for GitHub Actions ubuntu-latest (4 vCPUs):
20 mocks × ~14s per mock / 4 workers ≈ 70s + setup. Run with
`pytest -m slow` or skip with `pytest -m 'not slow'`.
"""

import numpy as np
import pytest

from null_engine import validate_model


@pytest.mark.slow
@pytest.mark.timeout(1200)
def test_cplwb_20_mocks_in_expected_range(data_dir):
    """20 mocks of CPL-wb against the canonical fiducial. Mean Δχ²
    should land in [1.0, 5.0] — wide enough to handle 20-sample SE
    (~0.55 around the 2.987 canonical mean → ~3.6σ buffer either side)
    while still catching a broken pipeline (mean ≈ 0 = fits not moving,
    mean ≫ 5 = something is very wrong)."""
    report = validate_model(
        model_key='cplwb',
        data_dir=data_dir,
        n_test=20,
        workers=4,
        seed_offset=900000,  # disjoint from production seeds
    )

    dchi2 = report['dchi2']
    assert dchi2.shape == (20,)
    assert np.all(np.isfinite(dchi2)), "non-finite Δχ² values produced"
    assert report['nan_inf_count'] == 0
    assert dchi2.dtype == np.float64

    mean = report['mean']
    assert 1.0 < mean < 5.0, (
        f"CPL-wb 20-mock mean Δχ² = {mean:.3f} outside expected "
        f"[1.0, 5.0]; canonical 10k mean is 2.987"
    )

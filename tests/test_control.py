# SPDX-License-Identifier: MIT
"""Slow integration test: 50 mocks of CPL-wb (the control model).

CPL-wb is a 3-parameter polynomial extension with zero location
parameters, so its empirical mean Δχ² should land very close to the
naive χ²(3) = 3.0 expectation. We check that the engine produces
finite, positive-mean Δχ² values in the right ballpark.

This is the only test that exercises the full mock-generation +
dual-fit pipeline. It takes ~5–10 minutes on a 2-worker pool. Run
with `pytest -m slow` or skip with `pytest -m 'not slow'`.
"""

import numpy as np
import pytest

from null_engine import validate_model


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_cplwb_50_mocks_in_expected_range(data_dir):
    """50 mocks of CPL-wb against the canonical fiducial. Mean Δχ²
    should land in [1.5, 4.5] — wide enough to absorb 50-sample noise
    while still catching a broken pipeline (mean near 0 = fits not
    moving, mean >>4 = something is very wrong). The 10k-mock canonical
    value is 2.987."""
    report = validate_model(
        model_key='cplwb',
        data_dir=data_dir,
        n_test=50,
        workers=2,
        seed_offset=900000,  # disjoint from production seeds
    )

    dchi2 = report['dchi2']
    assert dchi2.shape == (50,)
    assert np.all(np.isfinite(dchi2)), "non-finite Δχ² values produced"
    assert report['nan_inf_count'] == 0
    assert dchi2.dtype == np.float64

    mean = report['mean']
    assert 1.5 < mean < 4.5, (
        f"CPL-wb 50-mock mean Δχ² = {mean:.3f} outside expected "
        f"[1.5, 4.5]; canonical 10k mean is 2.987"
    )

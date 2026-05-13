# SPDX-License-Identifier: MIT
"""Minimal external plugin example for null_engine.

Single-parameter w(z) parameterization: constant shift from w = -1.
Useful as a template for users writing their own plugins, and as a
target for the test suite.

Run via:
    null-engine --model-file examples/test_external.py --validate
"""

import numpy as np


def w_shift(z, w0):
    return -1.0 + w0 * np.ones_like(z)


MODEL = {
    'key': 'test_shift',
    'name': 'Test constant-w shift',
    'ref': 'test_external.py',
    'w_func': w_shift,
    'bound_ranges': [(-0.5, 0.5)],
    'naive_dof': 1,
}

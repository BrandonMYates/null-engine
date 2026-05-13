# SPDX-License-Identifier: MIT
"""Plugin-loader tests: load examples/test_external.py and verify it
registers cleanly through the public API. Fast — no mock generation."""

import numpy as np
import pytest

from null_engine import (
    MODELS, load_model_file, register_external_model,
)


CANONICAL_KEYS = {'key', 'name', 'ref', 'H_func', 'n_extra', 'bound_ranges',
                  'bounds', 'bounds_extra', 'starts', 'naive_dof'}


def test_load_model_file_returns_canonical_spec(example_plugin_path):
    spec = load_model_file(example_plugin_path)
    assert set(spec.keys()) == CANONICAL_KEYS
    assert spec['key'] == 'test_shift'
    assert spec['name'] == 'Test constant-w shift'
    assert spec['n_extra'] == 1
    assert spec['naive_dof'] == 1
    assert callable(spec['H_func'])
    assert callable(spec['bounds'])
    assert len(spec['starts']) >= 12


def test_register_external_model_inserts_under_key(example_plugin_path):
    """Registering inserts under MODEL['key']. Use a private dict so we
    don't pollute the global MODELS for other tests."""
    private_models = {}
    key = register_external_model(private_models, example_plugin_path)
    assert key == 'test_shift'
    assert key in private_models
    assert private_models[key]['name'] == 'Test constant-w shift'


def test_register_rejects_duplicate_key(example_plugin_path):
    """Re-registering the same plugin into a dict that already has the
    key must raise — never silently overwrite."""
    private_models = {}
    register_external_model(private_models, example_plugin_path)
    with pytest.raises(ValueError):
        register_external_model(private_models, example_plugin_path)


def test_plugin_H_func_finite_at_interior_point(example_plugin_path):
    """The wrapped H_func (built from w_func via _h_from_w) returns
    finite positive H over a representative z range."""
    spec = load_model_file(example_plugin_path)
    z = np.linspace(0.01, 3.0, 50)
    H = spec['H_func'](z, 68.36, 0.3019, 0.1)
    assert np.all(np.isfinite(H))
    assert np.all(H > 0)


def test_plugin_bounds_check_works(example_plugin_path):
    spec = load_model_file(example_plugin_path)
    assert spec['bounds']([0.0]) is True       # interior
    assert spec['bounds']([-0.5]) is False     # at lower bound (strict)
    assert spec['bounds']([0.5]) is False      # at upper bound
    assert spec['bounds']([1.0]) is False      # outside

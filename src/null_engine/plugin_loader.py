# SPDX-License-Identifier: MIT
"""Dynamic plugin loader for user-supplied dark-energy models.

A plugin is a plain .py file that either defines a module-level ``MODEL``
dict, or a ``register()`` callable returning one. The dict is passed through
``model_spec.normalize_model_spec`` and inserted into the engine's MODELS
table under its ``key``. From there it runs under ``--model <key>`` like a
built-in.

Two minimal examples:

    # example 1 -- H(z) form, 2 free parameters
    import numpy as np

    def H_mine(z, H0, Om, p1, p2):
        return H0 * np.sqrt(Om*(1+z)**3 + (1-Om) * (1 + p1*z + p2*z**2))

    MODEL = {
        'key': 'mine',
        'name': 'My H(z) parameterization',
        'ref': 'Smith et al. 2026',
        'H_func': H_mine,
        'bound_ranges': [(-0.5, 0.5), (-0.3, 0.3)],
        'naive_dof': 2,
    }

    # example 2 -- w(z) form, 1 free parameter; engine derives H internally
    # via _de_density_from_wz against a z-grid built from the data redshifts.
    import numpy as np

    def w_mine(z, w0):
        return -1 + w0 * np.exp(-z)

    MODEL = {
        'key': 'mine_w',
        'name': 'My w(z) parameterization',
        'w_func': w_mine,
        'bound_ranges': [(-0.5, 0.5)],
    }

Note: the engine fits with multiprocessing.Pool, so each worker re-imports
the plugin file independently. Heavy top-level imports (theano, torch, ...)
will pay their cost N times. Keep plugin files minimal -- numpy and pure
python only.
"""

import importlib.util
import os
import sys
import uuid

from null_engine.model_spec import normalize_model_spec


def load_model_file(path: str) -> dict:
    """Load a plugin .py and return its normalized MODEL dict.

    The file must expose either a ``MODEL`` dict or a ``register()`` function
    returning one. The result is passed through ``normalize_model_spec``.
    """
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise ValueError(f"plugin file not found: {path}")

    base = os.path.splitext(os.path.basename(path))[0]
    # Unique module name so re-loading the same path (or two plugins sharing
    # a basename) never clobbers an existing sys.modules entry.
    mod_name = f"null_engine_plugin_{base}_{uuid.uuid4().hex[:8]}"

    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot build import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(mod_name, None)
        raise

    raw = getattr(module, 'MODEL', None)
    if raw is None:
        reg = getattr(module, 'register', None)
        if callable(reg):
            raw = reg()
        else:
            raise ValueError(
                f"plugin {path} exposes neither MODEL dict nor register() "
                f"function"
            )
    if not isinstance(raw, dict):
        raise ValueError(
            f"plugin {path} produced a {type(raw).__name__}, expected dict"
        )

    return normalize_model_spec(raw)


def register_external_model(MODELS: dict, path: str) -> str:
    """Load a plugin and insert it into ``MODELS`` under its declared key.

    Raises ``ValueError`` if the key already exists -- we never silently
    shadow a built-in. Returns the key on success.
    """
    spec = load_model_file(path)
    key = spec.get('key')
    if not isinstance(key, str) or not key:
        raise ValueError(f"plugin {path} did not provide a string 'key'")
    if key in MODELS:
        raise ValueError(
            f"model key {key!r} already registered; rename the plugin's "
            f"'key' field to avoid shadowing a built-in"
        )
    MODELS[key] = spec
    return key

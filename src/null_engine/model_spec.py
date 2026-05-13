# SPDX-License-Identifier: MIT
"""Model spec normalizer for null_engine.

User-facing entry: ``normalize_model_spec(raw)``. Accepts a small dict with
either ``H_func`` or ``w_func`` plus ``bound_ranges`` and returns the canonical
spec shape expected by null_engine.MODELS. Auto-fills ``bounds`` (box check +
optional ``bounds_extra``), ``starts`` (Sobol or Latin-hypercube sample of
interior), and ``naive_dof``.
"""

import inspect
import numpy as np

from null_engine.models import _de_density_from_wz

try:
    from scipy.stats import qmc as _qmc
    _HAS_SOBOL = True
except Exception:
    _HAS_SOBOL = False


def _bounds_from_ranges(bound_ranges, bounds_extra=None):
    """Closure: True iff every p[i] strictly inside bound_ranges[i] and bounds_extra(p)."""
    ranges = tuple((float(lo), float(hi)) for lo, hi in bound_ranges)
    extra = bounds_extra if bounds_extra is not None else (lambda p: True)

    def _bounds(p):
        if len(p) != len(ranges):
            return False
        for v, (lo, hi) in zip(p, ranges):
            if not (lo < v < hi):
                return False
        return bool(extra(p))

    return _bounds


def _starts_from_ranges(bound_ranges, n=12, seed=0):
    """Sample n interior points in [5%, 95%] of each range. Sobol if available, else LHS."""
    d = len(bound_ranges)
    lo = np.array([r[0] for r in bound_ranges], dtype=float)
    hi = np.array([r[1] for r in bound_ranges], dtype=float)
    span = hi - lo
    a = lo + 0.05 * span
    b = lo + 0.95 * span

    if _HAS_SOBOL:
        sampler = _qmc.Sobol(d=d, scramble=True, seed=seed)
        u = sampler.random(n)
    else:
        rng = np.random.default_rng(seed)
        u = np.empty((n, d))
        for j in range(d):
            cuts = (np.arange(n) + rng.random(n)) / n
            rng.shuffle(cuts)
            u[:, j] = cuts

    pts = a + u * (b - a)
    return [tuple(float(x) for x in row) for row in pts]


def _h_from_w(w_func):
    """Wrap w_func(z, *extra) into H_func(z, H0, Om, *extra) using _de_density_from_wz."""
    def H_func(z, H0, Om, *extra):
        z = np.asarray(z, dtype=float)
        zz = np.linspace(0.0, float(np.max(z)) + 0.1, 500)
        w_z = w_func(zz, *extra)
        f = _de_density_from_wz(z, zz, w_z)
        return H0 * np.sqrt(np.maximum(Om * (1 + z) ** 3 + (1 - Om) * f, 1e-10))

    return H_func


def _accepts_arity(func, required_positional):
    """Check whether func can accept at least `required_positional` positional args.
    Returns True if *args is present (skip check)."""
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return True  # builtins / C funcs — be lenient

    n_pos = 0
    for p in sig.parameters.values():
        if p.kind == inspect.Parameter.VAR_POSITIONAL:
            return True
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                      inspect.Parameter.POSITIONAL_OR_KEYWORD):
            n_pos += 1
    return n_pos >= required_positional


def normalize_model_spec(raw):
    """Normalize a user-facing model dict into the canonical null_engine spec shape."""
    if not isinstance(raw, dict):
        raise ValueError("raw spec must be a dict")

    if 'key' not in raw:
        raise ValueError("missing required field 'key'")
    if 'name' not in raw:
        raise ValueError("missing required field 'name'")
    key = raw['key']
    name = raw['name']

    has_H = 'H_func' in raw and raw['H_func'] is not None
    has_w = 'w_func' in raw and raw['w_func'] is not None
    if not has_H and not has_w:
        raise ValueError(f"model '{key}': must provide one of 'H_func' or 'w_func'")
    if has_H and has_w:
        raise ValueError(f"model '{key}': provide only one of 'H_func' or 'w_func', not both")

    bound_ranges = raw.get('bound_ranges')
    if not bound_ranges:
        raise ValueError(f"model '{key}': 'bound_ranges' is required and must be non-empty")

    normalized_ranges = []
    for i, br in enumerate(bound_ranges):
        if not (hasattr(br, '__len__') and len(br) == 2):
            raise ValueError(f"model '{key}': bound_ranges[{i}] must be a 2-tuple (lo, hi)")
        lo, hi = float(br[0]), float(br[1])
        if not (lo < hi):
            raise ValueError(f"model '{key}': bound_ranges[{i}] must satisfy lo < hi (got {lo}, {hi})")
        normalized_ranges.append((lo, hi))

    n_extra = len(normalized_ranges)
    if 'n_extra' in raw and raw['n_extra'] != n_extra:
        raise ValueError(
            f"model '{key}': n_extra={raw['n_extra']} disagrees with "
            f"len(bound_ranges)={n_extra}"
        )

    if has_H:
        H_func = raw['H_func']
    else:
        H_func = _h_from_w(raw['w_func'])

    # Arity check: H_func needs (z, H0, Om, *extra) — i.e. 3 + n_extra positional.
    target = raw['H_func'] if has_H else raw['w_func']
    required = (3 + n_extra) if has_H else (1 + n_extra)
    if not _accepts_arity(target, required):
        kind = 'H_func' if has_H else 'w_func'
        raise ValueError(
            f"model '{key}': {kind} must accept at least {required} positional args "
            f"(got fewer for n_extra={n_extra})"
        )

    bounds_extra = raw.get('bounds_extra')
    if bounds_extra is None:
        bounds_extra = lambda p: True
    bounds = _bounds_from_ranges(normalized_ranges, bounds_extra)

    starts = raw.get('starts')
    if starts is None or len(starts) < 12:
        generated = _starts_from_ranges(normalized_ranges, n=12, seed=raw.get('seed', 0))
        if starts is None:
            starts = generated
        else:
            starts = list(starts) + generated[len(starts):]
    starts = [tuple(float(x) for x in s) for s in starts]

    naive_dof = raw.get('naive_dof', n_extra)
    ref = raw.get('ref', "")

    return {
        'key': key,
        'name': name,
        'ref': ref,
        'H_func': H_func,
        'n_extra': n_extra,
        'bound_ranges': normalized_ranges,
        'bounds': bounds,
        'bounds_extra': bounds_extra,
        'starts': starts,
        'naive_dof': naive_dof,
    }

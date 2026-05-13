# SPDX-License-Identifier: MIT
"""Built-in dark-energy model definitions for null_engine.

Contains every H(z) used by the canonical null distributions in this
package and the MODELS registry the engine consumes. External plugins
extend this registry via plugin_loader.register_external_model.

Each entry conforms to the canonical model spec:
  - H_func(z, H0, Om, *extra)  callable returning H(z)
  - n_extra                    number of parameters beyond (H0, Om, ob)
  - bounds(extra)              callable returning True if extra is in-bounds
  - bound_ranges               list of (lo, hi) tuples, one per extra param
  - starts                     list of starting-point tuples for extra params
  - name                       human-readable name
  - ref                        paper citation
  - naive_dof                  number of d.o.f. for naive χ² comparison
"""

import numpy as np

SIG2_YATES = 0.02  # 2 * (0.1)^2


def H_lcdm(z, H0, Om):
    """Standard LCDM Hubble function."""
    return H0 * np.sqrt(Om * (1+z)**3 + (1 - Om))


def H_yates(z, H0, Om, A1, zc1, A2, zc2):
    """Two-Gaussian perturbation to dark energy density."""
    k = 1 + A1*np.exp(-(z-zc1)**2 / SIG2_YATES) + \
            A2*np.exp(-(z-zc2)**2 / SIG2_YATES)
    return H0 * np.sqrt(Om*(1+z)**3 + (1-Om)*k**2)


def H_wdagger(z, H0, Om, delta, z_dag):
    """w†CDM: abrupt EoS transition at z_dag.
    w(z) = -1 + delta for z < z_dag, w(z) = -1 - delta for z > z_dag."""
    a = 1.0 / (1.0 + z)
    a_dag = 1.0 / (1.0 + z_dag)
    f = np.where(
        a >= a_dag,
        a**(-3 * delta),
        a_dag**(-3*delta) * (a / a_dag)**(3*delta)
    )
    return H0 * np.sqrt(Om*(1+z)**3 + (1-Om)*f)


def H_lambda_s(z, H0, Om, z_dag):
    """Lambda_s CDM: cosmological constant switches sign at z_dag."""
    sign = np.where(z < z_dag, 1.0, -1.0)
    val = Om*(1+z)**3 + (1-Om)*sign
    val = np.maximum(val, 1e-10)
    return H0 * np.sqrt(val)


def H_4pde(z, H0, Om, w0, wm, at, delta_de):
    """4-parameter DE EoS (Akarsu et al. 2512.08752).
    w(a) = w0 + (wm-w0) * [1-exp(-(a-1)/d)] / [1-exp(-1/d)]
                         * [1+exp(at/d)] / [1+exp(-(a-at)/d)]"""
    zz = np.linspace(0, np.max(z) + 0.1, 500)
    aa = 1.0 / (1.0 + zz)
    e1 = np.clip(-(1 - aa) / delta_de, -50, 50)
    e2 = np.clip(-1.0 / delta_de, -50, 50)
    e3 = np.clip(at / delta_de, -50, 50)
    e4 = np.clip(-(aa - at) / delta_de, -50, 50)
    ratio1 = (1 - np.exp(e1)) / (1 - np.exp(e2))
    ratio2 = (1 + np.exp(e3)) / (1 + np.exp(e4))
    w_z = w0 + (wm - w0) * ratio1 * ratio2
    integrand = (1 + w_z) / (1 + zz)
    log_rho = 3 * np.cumsum(integrand) * (zz[1] - zz[0])
    log_rho_interp = np.interp(z, zz, log_rho)
    f = np.exp(np.clip(log_rho_interp, -50, 50))
    return H0 * np.sqrt(np.maximum(Om*(1+z)**3 + (1-Om)*f, 1e-10))


def H_gauss_eos(z, H0, Om, A_w, zc, sigma_w):
    """Gaussian bump in EoS: w(z) = -1 + A_w * exp(-(z-zc)^2 / (2*sigma_w^2))."""
    zz = np.linspace(0, np.max(z) + 0.1, 500)
    w_z = -1 + A_w * np.exp(-(zz - zc)**2 / (2*sigma_w**2))
    integrand = (1 + w_z) / (1 + zz)
    log_rho = 3 * np.cumsum(integrand) * (zz[1] - zz[0])
    log_rho_interp = np.interp(z, zz, log_rho)
    f = np.exp(log_rho_interp)
    return H0 * np.sqrt(Om*(1+z)**3 + (1-Om)*f)


def _de_density_from_wz(z_eval, zz, w_z):
    """Shared numerical DE density from w(z). Returns f = rho_DE(z)/rho_DE(0).
    Used by the w_func plugin shortcut and by every w(z)-based built-in below."""
    integrand = (1 + w_z) / (1 + zz)
    log_rho = 3 * np.cumsum(integrand) * (zz[1] - zz[0])
    return np.exp(np.clip(np.interp(z_eval, zz, log_rho), -50, 50))


def H_cplwb(z, H0, Om, w0, wa, wb):
    """CPL-wb: w(a) = w0 + wa*(1-a) + wb*(1-a)^2.
    Polynomial extension of CPL. Mukherjee et al. (2510.03779)."""
    zz = np.linspace(0, np.max(z) + 0.1, 500)
    aa = 1.0 / (1.0 + zz)
    w_z = w0 + wa * (1 - aa) + wb * (1 - aa)**2
    f = _de_density_from_wz(z, zz, w_z)
    return H0 * np.sqrt(np.maximum(Om*(1+z)**3 + (1-Om)*f, 1e-10))


def H_gde(z, H0, Om, w0, w1, beta):
    """GDE: w(a) = w0 - w1*(a^beta - 1)/beta. Barboza & Alcaniz."""
    zz = np.linspace(0, np.max(z) + 0.1, 500)
    aa = 1.0 / (1.0 + zz)
    w_z = w0 - w1 * (aa**beta - 1) / beta
    f = _de_density_from_wz(z, zz, w_z)
    return H0 * np.sqrt(np.maximum(Om*(1+z)**3 + (1-Om)*f, 1e-10))


def H_pade(z, H0, Om, w0, w1, w2):
    """Pade: w(a) = (w0 + w1*(1-a)) / (1 + w2*(1-a)). Rezaei et al."""
    zz = np.linspace(0, np.max(z) + 0.1, 500)
    aa = 1.0 / (1.0 + zz)
    denom = 1 + w2 * (1 - aa)
    denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
    w_z = (w0 + w1 * (1 - aa)) / denom
    f = _de_density_from_wz(z, zz, w_z)
    return H0 * np.sqrt(np.maximum(Om*(1+z)**3 + (1-Om)*f, 1e-10))


def H_bellde(z, H0, Om, w0, w1, zt, delta):
    """BellDE: w(z) = w1 + (w0-w1)*exp(-(z-zt)^2/delta^2).
    Gaussian bump with free baseline. Hussain et al. (2505.09913)."""
    zz = np.linspace(0, np.max(z) + 0.1, 500)
    w_z = w1 + (w0 - w1) * np.exp(-(zz - zt)**2 / delta**2)
    f = _de_density_from_wz(z, zz, w_z)
    return H0 * np.sqrt(np.maximum(Om*(1+z)**3 + (1-Om)*f, 1e-10))


# ---------------------------------------------------------------------------
# MODELS registry
# ---------------------------------------------------------------------------
MODELS = {
    'yates': {
        'H_func': H_yates,
        'n_extra': 4,
        'bounds': lambda p: (-0.5<p[0]<0.5 and 0.05<p[1]<1.5 and
                             -0.5<p[2]<0.5 and 0.05<p[3]<1.5 and p[1]<p[3]),
        'bound_ranges': [(-0.5, 0.5), (0.05, 1.5), (-0.5, 0.5), (0.05, 1.5)],
        'starts': [
            (0.1, 0.2, 0.1, 0.7), (0.1, 0.3, 0.1, 0.7),
            (0.1, 0.3, -0.08, 1.0), (0.1, 0.4, 0.1, 0.8),
            (-0.08, 0.3, 0.1, 0.7), (0.1, 0.5, 0.1, 0.9),
            (-0.08, 0.2, -0.08, 0.7), (0.1, 0.3, 0.1, 1.0),
            (0.1, 0.4, -0.08, 0.8), (-0.08, 0.5, 0.1, 1.1),
            (0.1, 0.6, 0.1, 1.1), (0.1, 0.2, -0.08, 0.6),
        ],
        'name': 'Yates two-Gaussian',
        'ref': 'Yates (this work)',
        'naive_dof': 4,
    },
    'wdagger': {
        'H_func': H_wdagger,
        'n_extra': 2,
        'bounds': lambda p: (-0.5<p[0]<0.5 and 0.01<p[1]<3.0),
        'bound_ranges': [(-0.5, 0.5), (0.01, 3.0)],
        'starts': [
            (0.05, 0.3), (0.05, 0.5), (0.05, 0.8),
            (0.05, 1.2), (0.05, 1.8), (0.05, 2.5),
            (-0.05, 0.3), (-0.05, 0.5), (-0.05, 0.8),
            (-0.05, 1.2), (-0.05, 1.8), (-0.05, 2.5),
        ],
        'name': 'w-dagger CDM',
        'ref': 'Scherer et al. (2025)',
        'naive_dof': 2,
    },
    'lambda_s': {
        'H_func': H_lambda_s,
        'n_extra': 1,
        'bounds': lambda p: (0.5<p[0]<5.0),
        'bound_ranges': [(0.5, 5.0)],
        'starts': [
            (0.8,), (1.2,), (1.5,), (1.8,), (2.0,), (2.5,),
            (3.0,), (3.5,), (4.0,), (0.6,), (1.0,), (4.5,),
        ],
        'name': 'Lambda_s CDM',
        'ref': 'Akarsu et al. (2024)',
        'naive_dof': 1,
    },
    'gauss_eos': {
        'H_func': H_gauss_eos,
        'n_extra': 3,
        'bounds': lambda p: (-0.5<p[0]<0.5 and 0.05<p[1]<2.0 and 0.05<p[2]<0.5),
        'bound_ranges': [(-0.5, 0.5), (0.05, 2.0), (0.05, 0.5)],
        'starts': [
            (0.1, 0.3, 0.15), (0.1, 0.5, 0.15), (0.1, 0.8, 0.15),
            (0.1, 1.0, 0.15), (0.1, 1.5, 0.15), (0.1, 0.5, 0.25),
            (-0.1, 0.3, 0.15), (-0.1, 0.5, 0.15), (-0.1, 0.8, 0.15),
            (-0.1, 1.0, 0.15), (-0.1, 1.5, 0.25), (0.1, 0.8, 0.30),
        ],
        'name': 'Gaussian EoS bump',
        'ref': 'Hussain et al. (2025)',
        'naive_dof': 3,
    },
    '4pde': {
        'H_func': H_4pde,
        'n_extra': 4,
        'bounds': lambda p: (-1.5<p[0]<-0.3 and -2.5<p[1]<-0.3 and
                             0.01<p[2]<0.8 and 0.05<p[3]<1.0),
        'bound_ranges': [(-1.5, -0.3), (-2.5, -0.3), (0.01, 0.8), (0.05, 1.0)],
        'starts': [
            (-0.9, -1.8, 0.05, 0.25), (-0.9, -1.5, 0.1, 0.25),
            (-0.9, -1.8, 0.2, 0.15), (-0.9, -1.2, 0.1, 0.4),
            (-1.1, -1.8, 0.05, 0.25), (-1.1, -1.5, 0.1, 0.25),
            (-1.1, -1.8, 0.2, 0.15), (-1.1, -1.2, 0.1, 0.4),
            (-0.8, -2.0, 0.05, 0.3), (-0.8, -1.5, 0.3, 0.2),
            (-1.0, -2.0, 0.1, 0.5), (-1.0, -1.0, 0.15, 0.3),
        ],
        'name': '4-parameter DE EoS (4pDE)',
        'ref': 'Akarsu et al. (2512.08752)',
        'naive_dof': 4,
    },
    'cplwb': {
        'H_func': H_cplwb,
        'n_extra': 3,
        'bounds': lambda p: (-2.0<p[0]<0.0 and -5.0<p[1]<3.0 and -5.0<p[2]<5.0),
        'bound_ranges': [(-2.0, 0.0), (-5.0, 3.0), (-5.0, 5.0)],
        'starts': [
            (-1.0, 0.1, 0.1), (-1.0, -0.5, 0.3), (-1.0, 0.5, -0.3),
            (-0.9, 0.2, 0.2), (-0.9, -0.3, 0.5), (-0.9, 0.3, -0.5),
            (-1.1, 0.1, 0.1), (-1.1, -0.5, 0.3), (-1.1, 0.5, -0.3),
            (-1.0, -1.0, 0.5), (-1.0, 1.0, -0.5), (-0.8, 0.2, 0.2),
        ],
        'name': 'CPL + w_b quadratic',
        'ref': 'Mukherjee et al. (2510.03779)',
        'naive_dof': 3,
    },
    'gde': {
        'H_func': H_gde,
        'n_extra': 3,
        'bounds': lambda p: (-2.0<p[0]<0.0 and -3.0<p[1]<3.0 and -2.5<p[2]<2.5 and abs(p[2])>0.01),
        'bound_ranges': [(-2.0, 0.0), (-3.0, 3.0), (-2.5, 2.5)],
        'starts': [
            (-1.0, 0.1, 1.0), (-1.0, -0.2, 1.0), (-1.0, 0.3, 0.5),
            (-0.9, 0.2, 1.5), (-0.9, -0.3, 0.8), (-0.9, 0.1, 2.0),
            (-1.1, 0.1, 1.0), (-1.1, -0.2, 0.5), (-1.1, 0.3, 1.5),
            (-1.0, 0.5, 0.3), (-1.0, -0.5, 2.0), (-0.8, 0.2, 1.0),
        ],
        'name': 'Generalized DE (GDE)',
        'ref': 'Barboza & Alcaniz',
        'naive_dof': 3,
    },
    'pade': {
        'H_func': H_pade,
        'n_extra': 3,
        'bounds': lambda p: (-2.0<p[0]<0.0 and -6.5<p[1]<2.0 and -3.0<p[2]<6.5),
        'bound_ranges': [(-2.0, 0.0), (-6.5, 2.0), (-3.0, 6.5)],
        'starts': [
            (-1.0, 0.1, 0.1), (-1.0, -0.5, 0.5), (-1.0, 0.5, -0.5),
            (-0.9, 0.2, 0.3), (-0.9, -0.3, 1.0), (-0.9, 0.3, -1.0),
            (-1.1, 0.1, 0.2), (-1.1, -0.5, 0.5), (-1.1, 0.5, -0.5),
            (-1.0, -1.0, 1.0), (-1.0, 1.0, -1.0), (-0.8, 0.2, 0.5),
        ],
        'name': 'Pade EoS',
        'ref': 'Rezaei et al.',
        'naive_dof': 3,
    },
    'bellde': {
        'H_func': H_bellde,
        'n_extra': 4,
        'bounds': lambda p: (-2.5<p[0]<2.0 and -1.3<p[1]<0.0 and
                             0.1<p[2]<3.0 and 0.05<p[3]<2.0),
        'bound_ranges': [(-2.5, 2.0), (-1.3, 0.0), (0.1, 3.0), (0.05, 2.0)],
        'starts': [
            (-1.2, -1.0, 0.5, 0.5), (-1.3, -0.9, 0.5, 0.8),
            (-1.2, -1.0, 1.0, 0.5), (-1.3, -0.9, 1.0, 0.8),
            (-1.5, -1.0, 0.5, 1.0), (-1.5, -0.8, 1.0, 1.0),
            (-0.8, -1.0, 0.5, 0.5), (-0.8, -1.0, 1.0, 0.5),
            (-1.2, -1.1, 0.3, 0.3), (-1.2, -0.7, 1.5, 0.8),
            (-1.0, -1.0, 0.8, 0.5), (-1.4, -0.9, 0.5, 1.2),
        ],
        'name': 'BellDE (Gaussian bump + baseline)',
        'ref': 'Hussain et al. (2505.09913)',
        'naive_dof': 4,
    },
}

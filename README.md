# Null Engine

Empirical null calibration for dark-energy Δχ² tests against Pantheon+
supernovae, DESI DR1 BAO, and CMB shift-parameter priors. Provides
empirical look-elsewhere-effect (LEE) null distributions for nine
built-in dark-energy parameterizations and a plugin interface for
adding your own.

Companion software to the EPJC Letter
**"Parameter counting fails in post-DESI dark-energy inference"**
(Yates 2026, [DOI:10.5281/zenodo.20145881](https://doi.org/10.5281/zenodo.20145881)).

## Installation

```bash
git clone https://github.com/BrandonMYates/null-engine.git
cd null-engine
pip install .
```

For the test suite and the example plot script:

```bash
pip install ".[test,examples]"
```

Requires Python ≥ 3.9, numpy ≥ 1.21, scipy ≥ 1.7. After install, the
`null-engine` console script is on your PATH:

```bash
null-engine --help
```

## Repository layout

```
null-engine/
├── src/null_engine/        # the package
│   ├── engine.py           # mock generator + fitter + main()
│   ├── models.py           # built-in H(z) functions + MODELS registry
│   ├── model_spec.py       # canonical-spec normalizer (auto-bounds, auto-starts, w_func→H_func)
│   ├── plugin_loader.py    # external plugin loader
│   └── validator.py        # --validate sanity harness
├── data/                   # Pantheon+ inputs (CC-BY 4.0)
│   ├── sn_zHD_1701.npy
│   ├── sn_zHEL_1701.npy
│   └── sn_inv_cov_1701.npy
├── results/                # canonical 10k-mock null distributions (CC-BY 4.0)
│   ├── null_<model>_N10000.npz   ×8
│   ├── null_<model>_N10000_12start.npz   ×4 (supplementary)
│   ├── diag_*.npy                ×6 (optimizer diagnostics)
│   └── optimizer_diagnostic.py   (reproduces convergence study)
├── tests/                  # pytest suite
├── examples/
│   ├── test_external.py    # minimal w(z) plugin (template)
│   └── plot_null_overlay.py # reproduces Fig. 1 of the EPJC Letter
├── paper/                  # JOSS paper
└── pyproject.toml
```

## Canonical results (10,000 mocks each, 24-start + boundary retry)

| File (in `results/`) | Model | k | Mean Δχ² | Naive χ²(k) | Max |
|------|-------|---|----------|-------------|-----|
| `null_lambda_s_N10000.npz` | Λ_sCDM | 1 | -0.059 | 1.0 | 13.03 |
| `null_wdagger_N10000.npz` | w†CDM | 2 | 2.211 | 2.0 | 14.88 |
| `null_gde_N10000.npz` | GDE | 3 | 2.174 | 3.0 | 16.54 |
| `null_pade_N10000.npz` | Padé | 3 | 2.365 | 3.0 | 16.52 |
| `null_gauss_eos_N10000.npz` | Gaussian EoS | 3 | 2.656 | 3.0 | 18.41 |
| `null_cplwb_N10000.npz` | CPL-wb | 3 | 2.987 | 3.0 | 17.00 |
| `null_4pde_N10000.npz` | 4pDE | 4 | 2.551 | 4.0 | 16.18 |
| `null_bellde_N10000.npz` | BellDE | 4 | 3.744 | 4.0 | 19.61 |

Each `.npz` contains a single array `dchi2_null` of length 10,000.

## Loading a null distribution

```python
import numpy as np
d = np.load('results/null_wdagger_N10000.npz')['dchi2_null']
print(d.shape, d.mean(), d.max())   # (10000,) 2.211 14.88
```

Compare your observed Δχ² against the empirical distribution rather
than against naive χ²(k):

```python
observed_dchi2 = 14.86           # e.g. Scherer et al. Planck+DESI+Union3
p_value = (d >= observed_dchi2).mean()
from scipy.stats import norm
sigma_LEE = norm.isf(p_value)    # ≈ 3.72
```

## Running the engine

All commands assume you're at the repository root (so `--data-dir`
defaults to `./data` correctly).

### Built-in model

```bash
null-engine --model wdagger -n 10000 -w 8
```

`-n` mock count, `-w` worker processes. Results checkpoint every 500
mocks; re-running resumes.

### Validation (small batch sanity check)

```bash
null-engine --model wdagger --validate --validate-n 10
```

PASS/WARN/FAIL report. Run this before committing to a full 10k run,
especially for custom models.

### Bring your own model

Write a tiny `.py` file with either an `H_func` or a `w_func`. See
`examples/test_external.py` for the simplest case. H(z) form:

```python
import numpy as np

def H_mine(z, H0, Om, p1, p2):
    return H0 * np.sqrt(Om*(1+z)**3 + (1-Om) * (1 + p1*z + p2*z**2))

MODEL = {
    'key': 'mine',
    'name': 'My H(z) parameterization',
    'H_func': H_mine,
    'bound_ranges': [(-0.5, 0.5), (-0.3, 0.3)],
    'naive_dof': 2,
}
```

EoS-only form (engine integrates w(z) → H(z) for you):

```python
def w_mine(z, w0):
    return -1 + w0 * np.exp(-z)

MODEL = {
    'key': 'mine_w',
    'name': 'My w(z) parameterization',
    'w_func': w_mine,
    'bound_ranges': [(-0.5, 0.5)],
}
```

Then:

```bash
null-engine --model-file my_model.py --validate
null-engine --model-file my_model.py -n 10000 -w 8
```

The plugin loader auto-generates starting points (Sobol-sampled) and
box-bounds from `bound_ranges` if you don't supply them. See
`src/null_engine/plugin_loader.py` and `src/null_engine/model_spec.py`
for the full canonical spec.

### Combining results from multiple runs

```bash
null-engine --combine run1.npz run2.npz -o combined.npz
```

Prints KS-statistic fits against χ²(k) for k=1..7, tail counts above
fixed thresholds, and false-positive-rate inflation against naive χ²(k).

### Reproducing Fig. 1 of the EPJC Letter

```bash
python examples/plot_null_overlay.py --results-dir results/ --out null_overlay.png
```

Renders an 8-panel overlay of every model's empirical null vs its
naive χ²(k), colour-coded by character (redundant / scanning / control
/ constrained).

## Tests

```bash
pip install ".[test]"
pytest -m "not slow"          # 25 unit + plugin tests, ~1 second
pytest -m slow --timeout=900  # one integration test, ~5–10 minutes
```

CI runs both jobs across Python 3.10/3.11/3.12 on push and PR
(`.github/workflows/tests.yml`).

## Reproducibility

- **Fiducial cosmology:** H0 = 68.36, Ωm = 0.3019, Ωb·h² = 0.02260.
  Defined at the top of `src/null_engine/engine.py` (`H0T, OMT, OBT`).
  Every output `.npz` in `results/` is conditioned on this point.
- **Seed offset:** mocks 0..9999 use seeds `100000..109999` (the
  default `--seed-offset 100000`). Validation mocks use seeds starting
  at 999000.
- **Optimizer:** Nelder-Mead, 24 starts (12 grid + 12 random ±20%
  perturbations), 2000 iter, `xatol=1e-3 fatol=0.05`. Boundary retry:
  if any best-fit parameter ends within 2% of a bound, 12 additional
  random interior starts are attempted.
- **Hardware/timing:** ~6 hours per model on an 8-core machine.

The optimizer choice was verified empirically: see
`results/optimizer_diagnostic.py` and the `diag_*.npy` outputs.
24-start Nelder-Mead with boundary retry was selected as the most
aggressive configuration that still converges within a reasonable
wall-clock. Powell was rejected (65% stuck rate on 4-parameter
models). ~13% of mocks yield Δχ² < 0.5 across all configurations —
these are genuine null realizations, not optimizer failures.

## Citation

If you use this engine or its outputs, please cite both:

1. **Companion paper:** Yates, B. (2026). *Parameter counting fails in post-DESI dark-energy inference.* Zenodo. https://doi.org/10.5281/zenodo.20145881
2. **Data deposit:** Yates, B. (2026). *Null Engine — LEE Calibration for Dark Energy Δχ² Tests* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.20145151

BibTeX:

```bibtex
@misc{yates2026paramcount,
  author       = {Yates, Brandon},
  title        = {Parameter counting fails in post-{DESI} dark-energy inference},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.20145881},
  url          = {https://doi.org/10.5281/zenodo.20145881}
}

@misc{yates2026nullengine,
  author       = {Yates, Brandon},
  title        = {Null Engine --- {LEE} Calibration for Dark Energy {$\Delta\chi^2$} Tests},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.20145151},
  url          = {https://doi.org/10.5281/zenodo.20145151},
  note         = {Data set}
}
```

When using the Pantheon+ SN data files redistributed in `data/`, please
also cite Scolnic et al. (2022) — the original Pantheon+ release.

## License

- **Code** (everything under `src/`, `tests/`, `examples/`): MIT — see [`LICENSE`](LICENSE)
- **Data** (everything under `data/`, `results/`): CC-BY 4.0 — see [`LICENSE-DATA`](LICENSE-DATA)

## Contact

Brandon Yates
Idaho Falls, ID 83404
bmichaelyates@gmail.com

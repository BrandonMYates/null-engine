---
title: 'Null Engine: Empirical null calibration for dark-energy Δχ² tests'
tags:
  - Python
  - cosmology
  - dark energy
  - statistical inference
  - Wilks theorem
  - look-elsewhere effect
authors:
  - name: Brandon Yates
    orcid: 0009-0007-7401-7883
    affiliation: 1
affiliations:
  - name: Independent Researcher, Idaho Falls, ID, USA
    index: 1
date: 12 May 2026
bibliography: paper.bib
---

# Summary

`Null Engine` is a Python package that calibrates Δχ² tests for dark-energy
model comparison against the post-DESI dataset stack: Pantheon+ Type Ia
supernovae [@scolnic2022], DESI DR1 baryon acoustic oscillations
[@desi_dr1_bao], and a CMB shift-parameter prior. The standard practice in
the field is to convert a Δχ² between a flat-ΛCDM null and an extended
dark-energy model into a significance using Wilks' theorem [@wilks1938],
which asymptotically maps Δχ² onto a χ²(k) distribution with k the number
of extra free parameters. This mapping silently assumes regularity
conditions — interior maxima, identifiability, and a non-degenerate Fisher
information — that fail when the extended model contains location-style
nuisance parameters (e.g. a transition redshift) that the optimizer is free
to scan across the data. In those cases the Δχ² null is not χ²(k), and
naive use of Wilks under- or over-reports the claimed significance.

`Null Engine` replaces the χ²(k) assumption with the empirical Δχ² null
distribution obtained by generating ΛCDM mocks and dual-fitting each one
under both ΛCDM and the extended model. It ships with nine built-in
parameterizations — Λ\_sCDM, w†CDM, Gaussian-EoS, CPL-wb, GDE, Padé, BellDE,
4pDE, and a two-Gaussian Yates control — and a plugin interface
(`--model-file`) that registers an arbitrary user-supplied `H(z)` or `w(z)`
without editing the source.

# Statement of Need

The post-DESI dark-energy literature has produced a rapid succession of
Δχ² claims interpreted through Wilks' theorem, with reported tensions
ranging from roughly 2σ to >5σ [@scherer2025; @hussain2025;
@alam2025; @nair2025; @akarsu2024_lambda_s]. The models in play are
heterogeneous: some have redundant parameters (e.g. GDE reduces to CPL when
β→1, Padé reduces to CPL when w₂→0), some have location parameters that
freely scan over redshift (Gaussian-EoS zc, BellDE zt), and some are clean
polynomial expansions (CPL-wb). Empirically, the Wilks assumption fails in
opposite directions for these families: redundant-parameter models have
effective degrees of freedom *below* k (so χ²(k) over-reports tails),
while location-scanning models have tails *fatter* than χ²(k) (so χ²(k)
under-reports them, the classic look-elsewhere effect [@gross2010]).

Until now, every group performing such a comparison has either rolled its
own null mocks or skipped the calibration entirely. `Null Engine`
consolidates the calibration into one reusable, plugin-extensible tool:
the user supplies `H(z)` or `w(z)`, the engine returns the empirical Δχ²
null distribution against the same Pantheon+ + DESI + CMB stack used in
the published claims. The intended audience is any cosmologist running
frequentist Δχ² model comparison against this dataset stack, including
groups extending the analyses cited above and groups proposing new
parameterizations who want a one-line significance calibration before
publication.

The companion EPJC Letter [@yates2026paper] applies `Null Engine` to the
nine built-in models; the underlying mock chains and best-fit catalogs are
deposited at [@yates2026data].

# Methodology

ΛCDM mocks are generated from a fiducial cosmology (H₀ = 68.36 km/s/Mpc,
Ωₘ = 0.3019, Ωᵦh² = 0.02260) with three independent likelihood blocks
combined as

$$\chi^2 = \chi^2_{\rm SN} + \chi^2_{\rm BAO} + \chi^2_{\rm CMB}.$$

The supernova block uses the full Pantheon+ statistical-plus-systematic
covariance with an analytically marginalized M\_B nuisance term; the BAO
block uses the public DESI DR1 covariance over the released tracer
combinations; the CMB block uses shift-parameter pivots (R, lₐ, Ωᵦh²).
Each mock is fit twice — once under ΛCDM, once under the extended model —
and the dual-fit Δχ² is

$$\Delta\chi^2 = \chi^2_{\Lambda{\rm CDM,best}} - \chi^2_{\rm ext,best}.$$

Optimization uses Nelder-Mead with 24 starts (12 from a coarse grid, 12
from perturbed best-fits) and a boundary-retry pass for parameters that
land on a hard prior edge. Convergence robustness was verified empirically
by comparing 12-start, 24-start, and Powell runs on identical mocks; the
diagnostic harness ships as `optimizer_diagnostic.py`.

The plugin specification accepts either `H_func(z, params)` or
`w_func(z, params)`; in the latter case the dark-energy density is
constructed by numerical integration via `_de_density_from_wz`, so a user
supplying only an equation-of-state need not reproduce background
boilerplate.

The per-model results are framed in the companion paper as a
*scanning-dimension* effect: the discriminant for whether χ²(k) under- or
over-reports Δχ² appears to be the dimensionality of the location-parameter
scan, not the parameter count k. `Null Engine` does not depend on that
interpretation — it returns the empirical null regardless of why χ²(k)
fails — but the framing motivates why a single tool is useful across
otherwise heterogeneous model families.

# Acknowledgements

I thank the authors of [@scherer2025], [@hussain2025], [@alam2025], and
[@nair2025] for the published Δχ² results that motivated this
calibration, and the Pantheon+ and DESI collaborations for the public data
releases that make this kind of independent reanalysis possible.

# References

# SPDX-License-Identifier: MIT
"""Plot empirical Δχ² null distributions against the naive Wilks χ²(k).

This script is the canonical example for the Null Engine package and doubles
as the figure-generator for the companion EPJC Letter
"Parameter counting fails in post-DESI dark-energy inference."

It loads the per-model null arrays (``null_<model>_N10000.npz`` files,
each containing a single ``dchi2_null`` array of length 10,000) and
renders a 2x4 grid of histograms with the naive χ²(k) overlay drawn on top
of each panel. Panels are color-coded by model character (redundant,
scanning-dim, control, constrained) so the central thesis -- that the
dimensionality of the location-parameter scan, not the total parameter
count, drives the deviation from Wilks -- is visible at a glance.

Usage
-----
    python plot_null_overlay.py [--results-dir ../results] [--out null_overlay.png]

Requires: numpy, scipy, matplotlib (the last is an optional package extra:
``pip install .[examples]``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# (filename, display name, naive_dof, character)
MODELS = [
    ("null_lambda_s_N10000.npz",  "Λ_sCDM",            1, "constrained"),
    ("null_wdagger_N10000.npz",   "w†CDM",             2, "scanning"),
    ("null_gde_N10000.npz",       "GDE",               3, "redundant"),
    ("null_pade_N10000.npz",      "Padé",              3, "redundant"),
    ("null_gauss_eos_N10000.npz", "Gaussian EoS",      3, "scanning"),
    ("null_cplwb_N10000.npz",     "CPL-wb (control)",  3, "control"),
    ("null_4pde_N10000.npz",      "4pDE",              4, "redundant"),
    ("null_bellde_N10000.npz",    "BellDE",            4, "scanning"),
]

CHARACTER_COLORS = {
    "redundant":   "steelblue",
    "scanning":    "crimson",
    "control":     "seagreen",
    "constrained": "darkorange",
}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=here.parent / "results",
        help="Directory containing the null_<model>_N10000.npz files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("null_overlay.png"),
        help="Output figure path (PNG/PDF inferred from suffix).",
    )
    return parser.parse_args()


def load_null(path: Path) -> np.ndarray | None:
    if not path.exists():
        print(f"[warn] missing {path.name}, skipping panel.")
        return None
    with np.load(path) as npz:
        return np.asarray(npz["dchi2_null"], dtype=float)


def main() -> None:
    args = parse_args()

    fig, axes = plt.subplots(2, 4, figsize=(14, 7), dpi=150)
    axes_flat = axes.flatten()

    # Track legend handles once -- one shared legend for the whole figure.
    hist_handle = None
    chi2_handle = None

    for ax, (fname, label, naive_dof, character) in zip(axes_flat, MODELS):
        data = load_null(args.results_dir / fname)
        if data is None:
            ax.set_axis_off()
            ax.set_title(f"{label}  (k={naive_dof})\n[file missing]", fontsize=10)
            continue

        color = CHARACTER_COLORS[character]
        mean = float(np.mean(data))

        # Histogram: clip to a sensible upper edge so tails don't dominate.
        upper = max(np.quantile(data, 0.995), stats.chi2.ppf(0.995, naive_dof))
        bins = np.linspace(min(0.0, float(np.min(data))), upper, 50)
        _, _, patches = ax.hist(
            data, bins=bins, density=True,
            color=color, alpha=0.55, edgecolor="white", linewidth=0.3,
            label="null mocks",
        )
        if hist_handle is None:
            hist_handle = patches[0]

        # χ²(k) reference curve.
        x = np.linspace(1e-3, upper, 400)
        line, = ax.plot(
            x, stats.chi2.pdf(x, naive_dof),
            color="black", linewidth=1.6, linestyle="--",
            label=r"$\chi^2(k)$",
        )
        if chi2_handle is None:
            chi2_handle = line

        ax.set_title(f"{label}  (k={naive_dof})", fontsize=11)
        ax.text(
            0.97, 0.93,
            f"mean={mean:.2f}\nnaive={naive_dof:.1f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="0.7", alpha=0.85),
        )
        ax.set_xlabel(r"$\Delta\chi^2$", fontsize=9)
        ax.set_ylabel("density", fontsize=9)
        ax.tick_params(labelsize=8)

    fig.suptitle(
        r"Empirical null $\Delta\chi^2$ distributions vs naive $\chi^2(k)$",
        fontsize=13,
    )

    handles = [h for h in (hist_handle, chi2_handle) if h is not None]
    if handles:
        fig.legend(
            handles=handles,
            labels=["null mocks", r"$\chi^2(k)$"],
            loc="lower center", ncol=2, frameon=False, fontsize=10,
            bbox_to_anchor=(0.5, -0.01),
        )

    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(args.out, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

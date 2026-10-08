"""
sh_pyramid.py — spherical-harmonic decomposition of a scalar field map, drawn as
the "pyramid" of coefficients used in Figure 10 (right column) of the Shimmer
paper: x axis = degree n, y axis = order m, one square per (n, m) coloured by the
fitted coefficient a_{n,m}.

Model (paper, Section S2, Eq. B.1):

    B(r, theta, phi)  ~  sum_{n=0}^{Nmax} sum_{m=-n}^{n}  a_{n,m} (r/R0)^n Y_n^m(theta, phi)

Y_n^m are real, orthonormal spherical harmonics (m > 0: cos, m < 0: sin), R0 is
the radius at which the field map was recorded. The coefficients are found by
least squares on the measured points. n = 0 (the mean field) is fitted but not
drawn, exactly as in the paper.

Self-contained, same isolation rule as the rest of this verifier: it does not
import from the field_harmonics_viewer project.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
from scipy.special import factorial, lpmv


@dataclass
class SHFit:
    n_max: int
    r0_mm: float
    coeffs: np.ndarray       # ((n_max+1)^2,), ordered n = 0..n_max, m = -n..n
    rms_resid: float         # same units as the fitted field
    cond: float

    def grid(self) -> np.ndarray:
        """Coefficients as an (n_max+1, 2*n_max+1) array indexed [n, m + n_max].
        Cells with |m| > n are NaN."""
        g = np.full((self.n_max + 1, 2 * self.n_max + 1), np.nan)
        k = 0
        for n in range(self.n_max + 1):
            for m in range(-n, n + 1):
                g[n, m + self.n_max] = self.coeffs[k]
                k += 1
        return g


def _real_ylm(n: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Real orthonormal Y_n^m. theta = polar angle from +z, phi = azimuth from +x."""
    am = abs(m)
    norm = np.sqrt((2 * n + 1) / (4 * np.pi) * factorial(n - am) / factorial(n + am))
    plm = lpmv(am, n, np.cos(theta))
    if m == 0:
        return norm * plm
    if m > 0:
        return np.sqrt(2) * norm * plm * np.cos(m * phi)
    return np.sqrt(2) * norm * plm * np.sin(am * phi)


def _design_matrix(x_mm, y_mm, z_mm, n_max: int, r0_mm: float) -> np.ndarray:
    r = np.sqrt(x_mm**2 + y_mm**2 + z_mm**2)
    r_safe = np.where(r > 0, r, 1.0)
    theta = np.arccos(np.clip(z_mm / r_safe, -1.0, 1.0))
    phi = np.arctan2(y_mm, x_mm)
    cols = []
    for n in range(n_max + 1):
        radial = (r / r0_mm) ** n
        for m in range(-n, n + 1):
            cols.append(radial * _real_ylm(n, m, theta, phi))
    return np.column_stack(cols)


def fit_sh(x_mm, y_mm, z_mm, values, n_max: int = 10, r0_mm: Optional[float] = None) -> SHFit:
    """Least-squares fit of the (n_max+1)^2 coefficients to the scattered `values`."""
    x_mm, y_mm, z_mm, values = (np.asarray(a, dtype=float) for a in (x_mm, y_mm, z_mm, values))
    if r0_mm is None:
        # median, not mean: robust to the r = 0 reference points a scan may contain
        r0_mm = float(np.median(np.sqrt(x_mm**2 + y_mm**2 + z_mm**2)))
    n_coef = (n_max + 1) ** 2
    if values.size < n_coef:
        raise ValueError(
            f"{values.size} points are not enough for n_max={n_max} ({n_coef} coefficients)."
        )
    A = _design_matrix(x_mm, y_mm, z_mm, n_max, r0_mm)
    scale = np.linalg.norm(A, axis=0)
    scale[scale == 0] = 1.0
    c_scaled, *_ = np.linalg.lstsq(A / scale, values, rcond=None)
    coeffs = c_scaled / scale
    rms = float(np.sqrt(np.mean((A @ coeffs - values) ** 2)))
    return SHFit(n_max, r0_mm, coeffs, rms, float(np.linalg.cond(A / scale)))


def make_sh_figure(
    meas,
    result,
    *,
    n_max: int = 10,
    r0_mm: Optional[float] = None,
    has_shim: bool = True,
    shim_field_mode: str = "along_B0",
    color_limit: Optional[float] = None,
    share_color_scale: bool = False,
    suptitle: Optional[str] = None,
):
    """High-level entry point: SH pyramids for the measured / shim / shimmed field.

    `meas` is a measurement_io.Measurement and `result` a field_analysis.FieldResult.
    With has_shim=False only the measured pyramid is drawn.

    shim_field_mode:
      "along_B0"  - signed shim-field component along the local measured field direction
                    (needs a measurement with x/y/z components). measured + shim = shimmed,
                    coefficient by coefficient.
      "magnitude" - |B_shim|: always positive, does NOT add to the measured coefficients.

    Returns (figure, {name: SHFit}).
    """
    from field_analysis import ppm

    fields = {"measured": result.baseline_mag_T * 1e3}
    titles = {"measured": f"Measured field\n{ppm(result.baseline_mag_T):,.0f} ppm"}

    if has_shim:
        if shim_field_mode == "along_B0":
            if not meas.has_vector:
                raise ValueError(
                    'shim_field_mode="along_B0" needs a measurement with x/y/z components.'
                )
            shim_mT, shim_label = result.shim_along_b0_T() * 1e3, "along B0"
        elif shim_field_mode == "magnitude":
            shim_mT, shim_label = result.shim_mag_T * 1e3, "|B|"
        else:
            raise ValueError(
                f'shim_field_mode must be "along_B0" or "magnitude", got {shim_field_mode!r}'
            )
        fields["shim"] = shim_mT
        fields["shimmed"] = result.total_mag_T * 1e3
        titles["shim"] = f"Shim magnets' field\n{shim_label}"
        titles["shimmed"] = f"Shimmed field\n{ppm(result.total_mag_T):,.0f} ppm"

    fits = {
        name: fit_sh(meas.x_mm, meas.y_mm, meas.z_mm, values, n_max=n_max, r0_mm=r0_mm)
        for name, values in fields.items()
    }
    fig = plot_sh_pyramids(fits, titles=titles, color_limit=color_limit,
                           share_color_scale=share_color_scale, suptitle=suptitle)
    return fig, fits


def plot_sh_pyramids(
    fits: Dict[str, SHFit],
    titles: Optional[Dict[str, str]] = None,
    color_limit: Optional[float] = None,
    unit: str = "mT",
    cmap: str = "bwr",
    share_color_scale: bool = True,
    suptitle: Optional[str] = None,
):
    """One pyramid per entry of `fits`, side by side.

    color_limit: symmetric colour range +-color_limit (the paper uses 0.1). None =
                 automatic, from the largest |a_{n,m}| with n >= 1.
    share_color_scale: one common range and one colorbar for all panels (needed to
                 compare measured vs shim vs shimmed). False = each panel gets its own.
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle

    names = list(fits)
    titles = titles or {}

    def auto_limit(f: SHFit) -> float:
        g = f.grid()[1:]                     # drop n = 0
        return float(np.nanmax(np.abs(g))) or 1.0

    if color_limit is not None:
        limits = {k: float(color_limit) for k in names}
    elif share_color_scale:
        limits = {k: max(auto_limit(f) for f in fits.values()) for k in names}
    else:
        limits = {k: auto_limit(f) for k, f in fits.items()}

    n_max = max(f.n_max for f in fits.values())
    fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names) + (1.2 if share_color_scale else 0), 6.6),
                             squeeze=False)
    axes = axes[0]

    for ax, name in zip(axes, names):
        fit = fits[name]
        lim = limits[name]
        norm = Normalize(-lim, lim)
        colormap = plt.get_cmap(cmap)
        grid = fit.grid()
        for n in range(1, fit.n_max + 1):
            for m in range(-n, n + 1):
                v = grid[n, m + fit.n_max]
                ax.add_patch(Rectangle((n - 0.4, m - 0.4), 0.8, 0.8,
                                       facecolor=colormap(norm(v)), edgecolor="black", linewidth=1.2))
        ax.set_xlim(0.4, n_max + 0.6)
        ax.set_ylim(-n_max - 0.6, n_max + 0.6)
        ax.set_aspect("equal")
        ax.set_xticks(range(1, n_max + 1))
        ax.set_yticks(range(-n_max, n_max + 1))
        ax.set_xlabel("n")
        ax.set_ylabel("m")
        ax.set_title(titles.get(name, name), fontsize=10)
        if not share_color_scale:
            sm = cm.ScalarMappable(norm=norm, cmap=colormap)
            fig.colorbar(sm, ax=ax, shrink=0.8, label=f"$a_{{n,m}}$ ({unit})")

    if share_color_scale:
        lim = limits[names[0]]
        sm = cm.ScalarMappable(norm=Normalize(-lim, lim), cmap=plt.get_cmap(cmap))
        fig.colorbar(sm, ax=list(axes), shrink=0.8, label=f"$a_{{n,m}}$ ({unit})")
    if suptitle:
        fig.suptitle(suptitle, fontsize=9)
    if not share_color_scale:
        # leave room at the top for the suptitle so it does not touch the panel titles
        fig.tight_layout(rect=(0, 0, 1, 0.93 if suptitle else 1))
    return fig

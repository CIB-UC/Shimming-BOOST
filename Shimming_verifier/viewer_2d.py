"""
viewer_2d.py — static 2D comparison of the measured field with and without the
shim magnets: XY / XZ / YZ projections of the point cloud, coloured by |B|.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from field_analysis import FieldResult, ppm
from measurement_io import Measurement


def plot_comparison(meas: Measurement, result: FieldResult):
    """2x3 figure: top row = without shim, bottom row = with shim (same colour scale).
    Returns the matplotlib Figure; saving/showing is left to the caller."""
    b0_mT = result.baseline_mag_T * 1e3
    b1_mT = result.total_mag_T * 1e3
    x_mm, y_mm, z_mm = meas.x_mm, meas.y_mm, meas.z_mm

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    planes = [("XY", x_mm, y_mm), ("XZ", x_mm, z_mm), ("YZ", y_mm, z_mm)]
    vmin = min(b0_mT.min(), b1_mT.min())
    vmax = max(b0_mT.max(), b1_mT.max())

    sc = None
    for row, (b_vals, title_prefix) in enumerate([(b0_mT, "Without shim"), (b1_mT, "With shim")]):
        for col, (name, h, v) in enumerate(planes):
            ax = axes[row, col]
            sc = ax.scatter(h, v, c=b_vals, cmap="plasma", s=14, vmin=vmin, vmax=vmax)
            ax.set_title(f"{title_prefix} — {name}")
            ax.set_xlabel(f"{name[0]} (mm)")
            ax.set_ylabel(f"{name[1]} (mm)")
            ax.set_aspect("equal")
    fig.colorbar(sc, ax=axes, label="|B| (mT)", shrink=0.8)

    fig.suptitle(
        f"No shim: {ppm(result.baseline_mag_T):,.0f} ppm   |   "
        f"With shim: {ppm(result.total_mag_T):,.0f} ppm"
    )
    return fig

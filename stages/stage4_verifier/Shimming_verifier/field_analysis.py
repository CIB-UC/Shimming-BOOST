"""
field_analysis.py — compute the theoretical field at the measurement points
(with and without shim magnets) and the resulting homogeneity (ppm).

Self-contained: uses only magpylib's own field solver (magnetostatics of
ideal cuboid magnets) and plain arithmetic. No assumption is imported from
any other codebase.

ppm definition used here (the standard MRI-shimming definition, stated
explicitly): for a scalar field magnitude array B (Tesla),

    ppm = (max(B) - min(B)) / mean(B) * 1e6

applied over whichever set of points is passed in (typically: every point of
the loaded measurement).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from measurement_io import Measurement


@dataclass
class FieldResult:
    b_baseline_T: np.ndarray     # (N,3) measured/baseline field vector at each point (T)
    b_shim_T: np.ndarray         # (N,3) shim-magnets-only field vector at each point (T)
    b_total_T: np.ndarray        # (N,3) baseline + shim (T)
    positions_m: np.ndarray      # (N,3)

    @property
    def baseline_mag_T(self):
        return np.linalg.norm(self.b_baseline_T, axis=-1)

    @property
    def shim_mag_T(self):
        return np.linalg.norm(self.b_shim_T, axis=-1)

    @property
    def total_mag_T(self):
        return np.linalg.norm(self.b_total_T, axis=-1)

    def shim_along_b0_T(self) -> np.ndarray:
        """Signed component of the shim field along the local measured field direction (T).

        This is the part of the shim field that actually changes B0, so
        |B_measured| + shim_along_b0 ~ |B_shimmed| (the shim field is small next to B0).
        Unlike shim_mag_T it is signed, and it is what adds to the measured field."""
        direction = self.b_baseline_T / np.linalg.norm(self.b_baseline_T, axis=-1, keepdims=True)
        return np.sum(self.b_shim_T * direction, axis=-1)


def ppm(b_values_T: np.ndarray) -> float:
    """(max - min) / mean * 1e6, over a 1D array of field magnitudes (or any
    consistent scalar field component) in Tesla."""
    b = np.asarray(b_values_T, dtype=float)
    m = np.mean(b)
    if m == 0 or not np.isfinite(m):
        return float("nan")
    return float((np.max(b) - np.min(b)) / m * 1e6)


def compute_fields(meas: Measurement, magnet_collection) -> FieldResult:
    """
    Evaluate:
      - the measured baseline field AT the measurement points (taken directly
        from the measurement itself — this is data, not a magpylib
        computation, since we have no model for whatever produced the
        baseline field, e.g. the Halbach array);
      - the shim magnets' theoretical field at those same points, via
        magpylib;
      - their sum (the theoretical "shimmed" field).

    If the measurement only recorded a scalar |B| (no vector components), the
    baseline vector is reconstructed as (|B|, 0, 0) — i.e. treated as if it
    points along a single axis — WITH A LOUD CAVEAT, since a scalar-only
    measurement genuinely cannot be vector-added to the shim field without
    an assumed direction. Prefer a measurement with recorded vector
    components (the "controlled" format) whenever possible.
    """
    positions_m = np.column_stack([meas.x_m, meas.y_m, meas.z_m])

    if meas.has_vector:
        b_baseline = np.column_stack([meas.bx_T, meas.by_T, meas.bz_T])
    else:
        print(
            "WARNING: this measurement has no recorded field VECTOR, only |B|. "
            "Treating the baseline as pointing along +X for vector addition with "
            "the shim field — this is an assumption, not measured data. Prefer a "
            "'controlled'-format scan (with x/y/z axis (gauss) columns) for a "
            "trustworthy shimmed-field calculation."
        )
        b_baseline = np.column_stack([meas.b_T, np.zeros_like(meas.b_T), np.zeros_like(meas.b_T)])

    if len(magnet_collection.sources_all) == 0:
        b_shim = np.zeros_like(b_baseline)
    else:
        b_shim = magnet_collection.getB(positions_m)
        b_shim = np.atleast_2d(b_shim)
        if b_shim.shape != b_baseline.shape:
            b_shim = b_shim.reshape(b_baseline.shape)

    b_total = b_baseline + b_shim

    return FieldResult(
        b_baseline_T=b_baseline, b_shim_T=b_shim, b_total_T=b_total, positions_m=positions_m,
    )


def summarize(result: FieldResult) -> str:
    b0 = result.baseline_mag_T
    b1 = result.total_mag_T
    ppm0 = ppm(b0)
    ppm1 = ppm(b1)
    lines = [
        "Homogeneity (ppm = (max-min)/mean * 1e6, over |B| at all measured points):",
        f"  WITHOUT shim magnets : ppm = {ppm0:,.1f}   "
        f"(|B| range [{b0.min()*1e3:.4f}, {b0.max()*1e3:.4f}] mT, mean {b0.mean()*1e3:.4f} mT)",
        f"  WITH shim magnets    : ppm = {ppm1:,.1f}   "
        f"(|B| range [{b1.min()*1e3:.4f}, {b1.max()*1e3:.4f}] mT, mean {b1.mean()*1e3:.4f} mT)",
    ]
    if np.isfinite(ppm0) and np.isfinite(ppm1) and ppm0 > 0:
        improvement = (ppm0 - ppm1) / ppm0 * 100
        if abs(improvement) < 1e-9:
            verdict = "no change"
        elif improvement > 0:
            verdict = "improved"
        else:
            verdict = "worsened"
        lines.append(f"  change               : {improvement:+.1f} % ({verdict})")
    return "\n".join(lines)

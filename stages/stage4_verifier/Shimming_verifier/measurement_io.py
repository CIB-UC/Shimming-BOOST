"""
measurement_io.py — read a measured magnetic-field scan CSV into simple arrays,
and turn it into magpylib Sensor objects.

Fully self-contained: no assumptions borrowed from any other codebase, only
what's documented in the CSV itself. Two export formats are auto-detected:

  "simple"      — metadata line 1, header line 2, columns: X, Y, Z, Gauss[, T].
                  One scalar field value (magnitude, or a single component —
                  whichever the "Gauss" column holds) per point.

  "controlled"  — a metadata block, then a header containing radio/theta/phi
                  and "x axis (gauss)", "y axis (gauss)", "z axis (gauss)",
                  "Magnitude (gauss)" columns (vector field, repeated samples
                  per point averaged).

Units handled here: gauss → tesla (1 G = 1e-4 T) for field values, and
millimetres → metres for coordinates (magpylib is strict-SI, metres only).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
import pandas as pd

GAUSS_TO_T = 1e-4
MM_TO_M = 1e-3


@dataclass
class Measurement:
    x_m: np.ndarray            # (N,) point coordinates, METRES (magpylib convention)
    y_m: np.ndarray
    z_m: np.ndarray
    b_T: np.ndarray            # (N,) |B| or scalar field value, TESLA
    bx_T: Optional[np.ndarray] = None   # (N,) field vector, TESLA — only when the
    by_T: Optional[np.ndarray] = None   # source recorded components.
    bz_T: Optional[np.ndarray] = None
    source_path: str = ""
    fmt: str = ""

    @property
    def n_points(self) -> int:
        return self.x_m.size

    @property
    def x_mm(self):
        return self.x_m / MM_TO_M

    @property
    def y_mm(self):
        return self.y_m / MM_TO_M

    @property
    def z_mm(self):
        return self.z_m / MM_TO_M

    @property
    def b_mT(self):
        return self.b_T * 1000.0

    @property
    def has_vector(self) -> bool:
        return self.bx_T is not None

    def summary(self) -> str:
        r_mm = np.sqrt(self.x_mm**2 + self.y_mm**2 + self.z_mm**2)
        lines = [
            f"Measurement ({self.fmt})  <-  {self.source_path}",
            f"  {self.n_points} points",
            f"  radius : min={r_mm.min():.2f}  max={r_mm.max():.2f}  mean={r_mm.mean():.2f} mm",
            f"  |B|    : min={self.b_mT.min():.4f}  max={self.b_mT.max():.4f}  mean={self.b_mT.mean():.4f} mT",
        ]
        return "\n".join(lines)


def _find_header_row(path: str, max_scan: int = 200) -> int:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= max_scan:
                break
            low = line.lower()
            cells = [c.strip() for c in low.split(",")]
            if "x" in cells and ("gauss" in low or "z" in cells):
                return i
    raise ValueError(f"Could not locate a header row in {path} (scanned {max_scan} lines).")


def load_measurement(path: str, coord_unit_mm: float = 1.0) -> Measurement:
    """
    Read a gaussmeter CSV export. Auto-detects the "simple" (X,Y,Z,Gauss) vs.
    "controlled" (radio/theta/phi + vector components, repeated samples) format.

    coord_unit_mm: multiply raw X/Y/Z by this to get millimetres, for scans
                   whose coordinates are not already in mm.
    """
    header_row = _find_header_row(path)
    df = pd.read_csv(path, skiprows=header_row)
    df.columns = [c.strip() for c in df.columns]
    cols_lower = {c.lower(): c for c in df.columns}

    is_controlled = "magnitude (gauss)" in cols_lower or "muestra" in cols_lower
    if is_controlled:
        return _load_controlled(df, cols_lower, path, coord_unit_mm)
    return _load_simple(df, cols_lower, path, coord_unit_mm)


def _load_simple(df, cols_lower, path, coord_unit_mm) -> Measurement:
    def col(name):
        key = name.lower()
        if key not in cols_lower:
            raise ValueError(f"Missing column '{name}'. Found: {list(df.columns)}")
        return cols_lower[key]

    x_mm = df[col("X")].to_numpy(dtype=float) * coord_unit_mm
    y_mm = df[col("Y")].to_numpy(dtype=float) * coord_unit_mm
    z_mm = df[col("Z")].to_numpy(dtype=float) * coord_unit_mm
    gauss_col = cols_lower.get("gauss")
    if gauss_col is None:
        raise ValueError(f"Missing 'Gauss' column. Found: {list(df.columns)}")
    b_T = np.abs(df[gauss_col].to_numpy(dtype=float)) * GAUSS_TO_T

    keep = ~((x_mm == 0.0) & (y_mm == 0.0) & (z_mm == 0.0))  # drop spurious marker rows
    return Measurement(
        x_mm[keep] * MM_TO_M, y_mm[keep] * MM_TO_M, z_mm[keep] * MM_TO_M, b_T[keep],
        source_path=path, fmt="simple",
    )


def _load_controlled(df, cols_lower, path, coord_unit_mm) -> Measurement:
    def col(name, required=True):
        key = name.lower()
        if key not in cols_lower:
            if required:
                raise ValueError(f"Missing column '{name}'. Found: {list(df.columns)}")
            return None
        return cols_lower[key]

    x_c, y_c, z_c = col("X"), col("Y"), col("Z")
    mag_c = col("Magnitude (gauss)")
    bx_c = col("x axis (gauss)", required=False)
    by_c = col("y axis (gauss)", required=False)
    bz_c = col("z axis (gauss)", required=False)

    if all(k in cols_lower for k in ("radio", "theta_deg", "phi_deg")):
        gkeys = [cols_lower["radio"], cols_lower["theta_deg"], cols_lower["phi_deg"]]
    else:
        gkeys = [x_c, y_c, z_c]

    agg = {x_c: "mean", y_c: "mean", z_c: "mean", mag_c: "mean"}
    if bx_c:
        agg.update({bx_c: "mean", by_c: "mean", bz_c: "mean"})
    grouped = df.groupby(gkeys, as_index=False).agg(agg)

    x_mm = grouped[x_c].to_numpy(dtype=float) * coord_unit_mm
    y_mm = grouped[y_c].to_numpy(dtype=float) * coord_unit_mm
    z_mm = grouped[z_c].to_numpy(dtype=float) * coord_unit_mm
    b_T = np.abs(grouped[mag_c].to_numpy(dtype=float)) * GAUSS_TO_T

    bx = by = bz = None
    if bx_c:
        bx = grouped[bx_c].to_numpy(dtype=float) * GAUSS_TO_T
        by = grouped[by_c].to_numpy(dtype=float) * GAUSS_TO_T
        bz = grouped[bz_c].to_numpy(dtype=float) * GAUSS_TO_T

    return Measurement(
        x_mm * MM_TO_M, y_mm * MM_TO_M, z_mm * MM_TO_M, b_T, bx, by, bz,
        source_path=path, fmt="controlled",
    )


def shift_measurement(meas: Measurement, dx_mm: float = 0.0, dy_mm: float = 0.0,
                      dz_mm: float = 0.0) -> Measurement:
    """Copy of `meas` with every point's coordinates shifted by (dx, dy, dz) mm.

    Only the coordinates move; each field value stays attached to its point. So this
    models "the probe was really at file position + shift": everything computed
    afterwards (e.g. the shim field) is evaluated at the shifted positions."""
    return replace(
        meas,
        x_m=meas.x_m + dx_mm * MM_TO_M,
        y_m=meas.y_m + dy_mm * MM_TO_M,
        z_m=meas.z_m + dz_mm * MM_TO_M,
    )


def measurement_to_sensors(meas: Measurement):
    """
    Build one magpylib Sensor per measured point, positioned at that point's
    (x,y,z) in metres. Sensors carry no notion of "value" themselves — they are
    just observation points; the measured value lives in `meas.b_T` (aligned
    by index) for comparison against whatever a source collection reports at
    those same sensors via getB().

    Returns a magpylib.Collection of Sensor objects (one per measured point),
    plus the parallel array of positions (m) for convenience.
    """
    import magpylib as magpy

    positions_m = np.column_stack([meas.x_m, meas.y_m, meas.z_m])
    sensors = [magpy.Sensor(position=p) for p in positions_m]
    return magpy.Collection(sensors), positions_m

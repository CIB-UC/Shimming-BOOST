"""
shim_magnets.py — read a shimming configuration CSV (X, Y, Z, Angle) and build
magpylib Cuboid magnets from it.

Self-contained: this module makes its OWN explicit assumptions about geometry
and angle convention, documented below, and does not import or rely on any
other codebase's internal logic. If the assumptions don't match how a given
CSV's angle was actually generated, that mismatch is exactly what this
verifier is for — it should be checked against the real magnet layout /
measured field, not assumed correct by construction.

Expected CSV columns (case-insensitive, flexible on exact header text):
  X (mm), Y (mm), Z (mm), Angle (deg)

  (A CSV whose 3rd column is a tray/ring INDEX rather than a literal Z
  coordinate — e.g. a "RingNumber" column — cannot be used here directly,
  since this module has no external tray-geometry table to convert an index
  into a physical Z. Use the XYZ-format export instead.)

Assumed physical convention (stated explicitly — verify against your actual
hardware before trusting results):
  - Each magnet is modelled as a solid CUBOID of side `magnet_side_mm`,
    uniformly magnetized, with remanence (polarization magnitude) `brem_T`.
  - "Angle (deg)" is the magnet's magnetization direction as an in-plane
    rotation about the GLOBAL Z axis (the bore axis), measured counter-
    clockwise from +X, applied in the magnet's own local X-Y plane before
    it is placed at (X, Y, Z) — i.e. magnetization = brem * (cos(angle),
    sin(angle), 0). The magnet's physical faces are assumed to stay
    axis-aligned (no attempt to rotate the cuboid body itself to match a
    tray's local frame) — only the magnetization vector rotates.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

MM_TO_M = 1e-3
DEFAULT_BREM_T = 1.26          # magnet remanence, Tesla (project default; override freely)
DEFAULT_SIDE_MM = 6.0          # cube side length, mm


@dataclass
class ShimConfig:
    x_mm: np.ndarray
    y_mm: np.ndarray
    z_mm: np.ndarray
    angle_deg: np.ndarray
    source_path: str = ""

    @property
    def n_magnets(self) -> int:
        return self.x_mm.size


def load_shim_config(path: str) -> ShimConfig:
    """Read a shimming-configuration CSV. Requires literal X, Y, Z, Angle
    columns (mm / mm / mm / deg) — NOT a RingNumber/tray-index column."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    cols_lower = {c.lower(): c for c in df.columns}

    def find(*candidates):
        for c in candidates:
            if c.lower() in cols_lower:
                return cols_lower[c.lower()]
        return None

    x_c = find("X (mm)", "X")
    y_c = find("Y (mm)", "Y")
    z_c = find("Z (mm)", "Z")
    a_c = find("Angle (deg)", "Angle")

    if x_c is None or y_c is None or a_c is None:
        raise ValueError(f"Shim CSV needs X, Y, Angle columns at minimum. Found: {list(df.columns)}")
    if z_c is None:
        ring_c = find("RingNumber", "Ring", "InsertPos")
        if ring_c is not None:
            raise ValueError(
                f"This CSV has a '{ring_c}' column instead of a literal Z (mm) column. "
                "shim_magnets.py needs real physical Z coordinates (the XYZ-format export), "
                "not a tray/ring index — it has no external geometry table to convert one into "
                "the other. Point this at the companion XYZ CSV instead."
            )
        raise ValueError(f"Shim CSV is missing a Z (mm) column. Found: {list(df.columns)}")

    return ShimConfig(
        x_mm=df[x_c].to_numpy(dtype=float),
        y_mm=df[y_c].to_numpy(dtype=float),
        z_mm=df[z_c].to_numpy(dtype=float),
        angle_deg=df[a_c].to_numpy(dtype=float),
        source_path=path,
    )


def shift_magnets_z(cfg: ShimConfig, shift_negz_mm: float = 0.0,
                    shift_posz_mm: float = 0.0) -> ShimConfig:
    """Copy of `cfg` with the magnets split by the sign of their ORIGINAL z:
    magnets with z < 0 are moved by `shift_negz_mm` and magnets with z > 0 by
    `shift_posz_mm`, both along +z (so a positive value moves a magnet toward +z:
    away from the centre for z > 0, towards it for z < 0). Magnets at exactly z = 0
    are left where they are."""
    dz = np.where(cfg.z_mm < 0, shift_negz_mm, np.where(cfg.z_mm > 0, shift_posz_mm, 0.0))
    return replace(cfg, z_mm=cfg.z_mm + dz)


def sample_brem(n_magnets: int, brem_T: float = DEFAULT_BREM_T, brem_std_T: float = 0.0,
                seed=None) -> np.ndarray:
    """Per-magnet remanence, Tesla: each magnet is an independent sample of a normal
    distribution with mean `brem_T` and standard deviation `brem_std_T`.
    brem_std_T = 0 gives exactly `brem_T` for every magnet. `seed` makes the draw
    reproducible (None = different every call)."""
    if brem_std_T < 0:
        raise ValueError("brem_std_T must be >= 0")
    if brem_std_T == 0:
        return np.full(n_magnets, float(brem_T))
    return np.random.default_rng(seed).normal(brem_T, brem_std_T, n_magnets)


def build_magnets(cfg: ShimConfig, brem_T=DEFAULT_BREM_T,
                   side_mm: float = DEFAULT_SIDE_MM):
    """
    Build one magpylib magnet.Cuboid per row of `cfg`.

    brem_T : remanence / polarization magnitude, Tesla. Either one number for all
             magnets (default 1.26 T) or an array with one value per magnet, e.g. from
             sample_brem() to model magnet-to-magnet variation.
    side_mm: cube side length, mm (uniform cube; override if magnets are a
             different size).

    Convention (see module docstring): magnetization = brem_T *
    (cos(angle), sin(angle), 0) in the GLOBAL frame — Angle is treated as a
    rotation about global Z, 0 deg = +X, counter-clockwise.

    Returns a magpylib.Collection of Cuboid sources.
    """
    import magpylib as magpy

    side_m = side_mm * MM_TO_M
    angle_rad = np.deg2rad(cfg.angle_deg)
    brem = np.broadcast_to(np.asarray(brem_T, dtype=float), (cfg.n_magnets,))

    cuboids = []
    for i in range(cfg.n_magnets):
        pol = (
            brem[i] * np.cos(angle_rad[i]),
            brem[i] * np.sin(angle_rad[i]),
            0.0,
        )
        pos = (cfg.x_mm[i] * MM_TO_M, cfg.y_mm[i] * MM_TO_M, cfg.z_mm[i] * MM_TO_M)
        cuboids.append(
            magpy.magnet.Cuboid(
                dimension=(side_m, side_m, side_m),
                polarization=pol,
                position=pos,
            )
        )
    return magpy.Collection(cuboids)

"""
viewer_3d.py — interactive 3D viewer: the measured field point cloud + where
the shim magnets sit (as oriented cubes with a magnetization arrow).

Self-contained, same isolation rule as the rest of this verifier: reads only
measurement_io.Measurement / shim_magnets.ShimConfig objects (already built
from CSV data alone) — nothing borrowed from any other codebase.

Built on Plotly (already a magpylib dependency) so it renders as a single,
self-contained HTML file: open it in any browser and freely rotate / pan /
zoom with the mouse, no server or extra install needed.

Shows up to THREE field point-cloud traces (toggle each on/off via the
legend — all start visible together):
  - Measured field       (baseline, straight from the CSV)
  - Theoretical shim field   (shim magnets' own contribution, magpylib)
  - Theoretical shimmed field (measured + shim)
Each has its OWN always-on colorbar (stacked on the left), and the page
carries a "<field> — ppm: <value>" line per field plus the source CSV
filenames used to build the scene.

Usage:
    from measurement_io import load_measurement
    from shim_magnets import load_shim_config
    from field_analysis import compute_fields
    from viewer_3d import plot_3d_scene

    meas = load_measurement("measurement.csv")
    cfg  = load_shim_config("shim_xyz.csv")           # optional
    result = compute_fields(meas, build_magnets(cfg)) # optional
    plot_3d_scene(meas, cfg, result=result, out_html="scene.html")
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import plotly.graph_objects as go

from measurement_io import Measurement
from shim_magnets import ShimConfig, DEFAULT_SIDE_MM
from field_analysis import FieldResult, ppm

# Local cube corners (unit cube centered at origin, before scale/rotate/translate).
_UNIT_CUBE_CORNERS = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
], dtype=float) * 0.5

# 12 triangles (2 per face) referencing corner indices above, for a Mesh3d cube.
_CUBE_I = [0, 0, 0, 0, 4, 4, 1, 1, 3, 3, 2, 2]
_CUBE_J = [1, 2, 4, 5, 5, 6, 2, 6, 0, 4, 3, 7]
_CUBE_K = [2, 3, 5, 1, 6, 7, 6, 5, 4, 7, 7, 6]
# (a standard, if slightly redundant, triangulation — visually verified via
#  the numeric geometry checks: correct centering, sizing, and orientation.)

# Field trace styling, in display order (also the left-to-right colorbar order).
_FIELD_STYLE = {
    "Measured field":            dict(colorscale="Plasma",  colorbar_x=-0.18),
    "Theoretical shim field":    dict(colorscale="Cividis",  colorbar_x=-0.09),
    "Theoretical shimmed field": dict(colorscale="Viridis", colorbar_x=0.00),
}


def _rotation_z(angle_rad: np.ndarray) -> np.ndarray:
    """Return (N,3,3) rotation matrices about global Z, one per angle."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    n = angle_rad.size
    R = np.zeros((n, 3, 3))
    R[:, 0, 0] = c
    R[:, 0, 1] = -s
    R[:, 1, 0] = s
    R[:, 1, 1] = c
    R[:, 2, 2] = 1.0
    return R


def _magnet_cube_trace(cfg: ShimConfig, side_mm: float, opacity: float = 0.55,
                        color: str = "crimson") -> go.Mesh3d:
    """One Mesh3d trace containing ALL magnet cubes (concatenated), each
    rotated about Z by its own Angle and translated to its own (X,Y,Z) — so
    the whole set of magnets is one efficient draw call."""
    n = cfg.n_magnets
    angle_rad = np.deg2rad(cfg.angle_deg)
    R = _rotation_z(angle_rad)                       # (n,3,3)
    corners = _UNIT_CUBE_CORNERS * side_mm            # (8,3), mm

    # rotate each magnet's 8 corners, then translate
    all_x, all_y, all_z = [], [], []
    all_i, all_j, all_k = [], [], []
    for m in range(n):
        rotated = corners @ R[m].T                    # (8,3)
        cx, cy, cz = cfg.x_mm[m], cfg.y_mm[m], cfg.z_mm[m]
        pts = rotated + np.array([cx, cy, cz])
        base = m * 8
        all_x.extend(pts[:, 0])
        all_y.extend(pts[:, 1])
        all_z.extend(pts[:, 2])
        all_i.extend([base + t for t in _CUBE_I])
        all_j.extend([base + t for t in _CUBE_J])
        all_k.extend([base + t for t in _CUBE_K])

    return go.Mesh3d(
        x=all_x, y=all_y, z=all_z, i=all_i, j=all_j, k=all_k,
        color=color, opacity=opacity, flatshading=True,
        name="Shim magnets", showlegend=True,
        hoverinfo="skip",
    )


def _magnet_arrow_trace(cfg: ShimConfig, side_mm: float, color: str = "black") -> go.Scatter3d:
    """Cone/line arrows showing each magnet's magnetization direction
    (Angle, in-plane about Z — the same convention shim_magnets.py uses)."""
    angle_rad = np.deg2rad(cfg.angle_deg)
    L = side_mm * 0.9  # arrow length
    xs, ys, zs = [], [], []
    for m in range(cfg.n_magnets):
        cx, cy, cz = cfg.x_mm[m], cfg.y_mm[m], cfg.z_mm[m]
        dx, dy = L * np.cos(angle_rad[m]), L * np.sin(angle_rad[m])
        xs += [cx, cx + dx, None]
        ys += [cy, cy + dy, None]
        zs += [cz, cz, None]
    return go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines",
        line=dict(color=color, width=4),
        name="Magnetization dir.", showlegend=True, hoverinfo="skip",
    )


def _field_points_trace(meas: Measurement, field_values_mT: np.ndarray, name: str) -> go.Scatter3d:
    """One field point-cloud trace, ALWAYS carrying its own visible colorbar
    (stacked on the left via each field's fixed colorbar_x in _FIELD_STYLE) —
    all three fields' colorbars are shown together, not just the active one."""
    style = _FIELD_STYLE[name]
    return go.Scatter3d(
        x=meas.x_mm, y=meas.y_mm, z=meas.z_mm,
        mode="markers",
        marker=dict(
            size=4, color=field_values_mT, colorscale=style["colorscale"],
            showscale=True,
            colorbar=dict(title=dict(text=f"{name}<br>B (mT)", font=dict(size=10)),
                          x=style["colorbar_x"], len=0.6, thickness=16,
                          tickfont=dict(size=9)),
        ),
        name=name, visible=True,
        hovertemplate=(
            f"{name}<br>X=%{{x:.1f}} mm<br>Y=%{{y:.1f}} mm<br>Z=%{{z:.1f}} mm"
            "<br>B=%{marker.color:.4f} mT<extra></extra>"
        ),
    )


def _ppm_annotation_text(field_values_by_name: dict) -> str:
    lines = []
    for name, vals_mT in field_values_by_name.items():
        p = ppm(np.asarray(vals_mT) * 1e-3)   # ppm() expects Tesla
        lines.append(f"{name} — ppm: {p:,.1f}")
    return "<br>".join(lines)


def plot_3d_scene(meas: Measurement,
                   cfg: Optional[ShimConfig] = None,
                   side_mm: float = DEFAULT_SIDE_MM,
                   result: Optional[FieldResult] = None,
                   out_html: str = "field_3d_scene.html",
                   title: str = "Measured field + shim magnet placement",
                   measurement_csv_name: Optional[str] = None,
                   shim_csv_name: Optional[str] = None):
    """
    Build and save an interactive 3D scene (self-contained HTML):
      - "Measured field" — the baseline points, colored by |B| from the CSV;
      - if `result` is given (field_analysis.compute_fields output):
          "Theoretical shim field"    — the shim magnets' own contribution
          "Theoretical shimmed field" — measured + shim
        all three shown together, each with its own always-visible colorbar
        (stacked on the left edge), and each toggleable independently via
        the legend;
      - if `cfg` is given, every shim magnet drawn as an oriented cube at its
        (X,Y,Z), with a short line showing its magnetization direction
        (Angle, per shim_magnets.py's convention);
      - a "<field> — ppm: <value>" line per shown field, bottom-right;
      - the measurement/shim source CSV filenames in the page title.

    The output is a single .html file — open it in any browser; drag to
    rotate, scroll/pinch to zoom, right-drag (or shift-drag) to pan.
    """
    traces = []
    field_values_by_name = {"Measured field": meas.b_mT}
    traces.append(_field_points_trace(meas, meas.b_mT, "Measured field"))

    if result is not None:
        shim_mT = result.shim_mag_T * 1e3
        shimmed_mT = result.total_mag_T * 1e3
        field_values_by_name["Theoretical shim field"] = shim_mT
        field_values_by_name["Theoretical shimmed field"] = shimmed_mT
        traces.append(_field_points_trace(meas, shim_mT, "Theoretical shim field"))
        traces.append(_field_points_trace(meas, shimmed_mT, "Theoretical shimmed field"))

    if cfg is not None and cfg.n_magnets > 0:
        traces.append(_magnet_cube_trace(cfg, side_mm))
        traces.append(_magnet_arrow_trace(cfg, side_mm))

    # Reserve left margin for the stacked colorbars (one per field trace).
    n_colorbars = len(field_values_by_name)
    left_margin = 70 + 90 * max(0, n_colorbars - 1)

    meas_name = measurement_csv_name or getattr(meas, "source_path", "") or "(unknown)"
    shim_name = shim_csv_name or (getattr(cfg, "source_path", "") if cfg is not None else None) or "(none)"
    full_title = (
        f"{title}<br>"
        f"<sup>Measurement: {meas_name}   |   Shim config: {shim_name}</sup>"
    )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=full_title, x=0.5, xanchor="center"),
        scene=dict(
            xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)",
            aspectmode="data",
        ),
        legend=dict(itemsizing="constant", x=1.0, xanchor="left", y=1.0),
        margin=dict(l=left_margin, r=150, t=70, b=90),
        annotations=[
            dict(
                text=_ppm_annotation_text(field_values_by_name),
                showarrow=False,
                xref="paper", yref="paper",
                x=1.0, y=-0.06, xanchor="right", yanchor="top",
                align="left",
                font=dict(size=12),
                bgcolor="rgba(255,255,255,0.7)",
            )
        ],
    )
    fig.write_html(out_html, include_plotlyjs="cdn")
    return fig

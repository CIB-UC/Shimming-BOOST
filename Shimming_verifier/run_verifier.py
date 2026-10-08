"""
run_verifier.py — command-line entry point of the magpylib shimming verifier.

`main.py` is the interactive entry point (edit its CONFIGURATION block, press Run in
VSCode). This file is the NON-interactive one: it takes the same knobs as command-line
arguments, so another program (the Julia pipeline's Stage 4, a shell script, CI) can
drive the verifier without editing any file. It does not duplicate any logic — it
loads main.py, overwrites the CONFIGURATION variables from the arguments, and calls
main.main(). The verifier stays fully self-contained: this file knows nothing about
the Julia pipeline.

The three things you must decide:
    --measurement  the measured field CSV (which field to compare against)
    --shim         the shim-magnet CSV: X (mm), Y (mm), Z (mm), Angle (deg)
                   (omit = baseline only, no shim magnets)
    what to run    --comparison-2d / --scene-3d / --sh-pyramid  (each has a --no- form)
plus --out-dir, where every generated file is written.

Example:
    python run_verifier.py --measurement scan.csv --shim shim_xyz.csv --out-dir out ^
        --no-scene-3d --brem-t 1.26 --side-mm 6

Frame of the shim CSV: by default the CSV is used exactly as written (--shim-frame lab).
Some producers (the Julia BOOST pipeline) write it in a frame where the main field B0
points along +y, whatever way the scanner was oriented; --shim-frame optimizer rotates
positions and angles back into the measurement's frame first, using B0's direction from
--b0-direction (or, for "auto", from the mean field vector of the measurement).

Misalignment-study knobs (--brem-std-t, --meas-shift, --z-magnets-shift) default to 0
here, NOT to whatever is currently written in main.py, so a scripted run is always the
ideal setup unless you explicitly ask otherwise.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# rotation phi (deg, CCW) that takes the measurement (lab) frame to the frame where B0 = +y
_B0_TO_PHI_DEG = {"+y": 0.0, "+x": 90.0, "-x": -90.0, "-y": 180.0}


def _parse(argv=None) -> argparse.Namespace:
    B = argparse.BooleanOptionalAction
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--measurement", required=True, help="measured-field CSV (path)")
    p.add_argument("--shim", default=None, help="shim CSV with X, Y, Z (mm) and Angle (deg); omit for baseline only")
    p.add_argument("--out-dir", required=True, help="folder for every generated file (created if missing)")

    g = p.add_argument_group("what to run")
    g.add_argument("--comparison-2d", action=B, default=True, help="2D projections, with vs without shim")
    g.add_argument("--scene-3d", action=B, default=True, help="interactive 3D scene (.html)")
    g.add_argument("--sh-pyramid", action=B, default=True, help="spherical-harmonic pyramids")
    g.add_argument("--show", action=B, default=False, help="open the matplotlib windows at the end")

    g = p.add_argument_group("shim CSV frame")
    g.add_argument("--shim-frame", choices=("lab", "optimizer"), default="lab",
                   help="lab: use the shim CSV as written. optimizer: it is in the frame where B0 = +y; "
                        "rotate it into the measurement's frame first")
    g.add_argument("--b0-direction", choices=("auto", "+x", "-x", "+y", "-y"), default="auto",
                   help="direction of B0 in the measurement's frame (only used with --shim-frame optimizer)")

    g = p.add_argument_group("physics")
    g.add_argument("--brem-t", type=float, default=1.26, help="mean remanence, Tesla")
    g.add_argument("--side-mm", type=float, default=6.0, help="magnet cube side, mm")
    g.add_argument("--coord-unit-mm", type=float, default=1.0, help="measurement X/Y/Z unit, in mm")

    g = p.add_argument_group("spherical harmonics")
    g.add_argument("--sh-max-degree", type=int, default=None, help="max SH degree (main.py default if omitted)")
    g.add_argument("--r0-mm", type=float, default=None, help="SH reference radius; omit = median measured radius")
    g.add_argument("--shim-field-mode", choices=("along_B0", "magnitude"), default=None)
    g.add_argument("--color-limit-mt", type=float, default=None, help="pyramid colour limit (mT); 0 = automatic")
    g.add_argument("--share-color-scale", action=B, default=None)

    g = p.add_argument_group("misalignment study (all default to 0 = ideal)")
    g.add_argument("--brem-std-t", type=float, default=0.0, help="magnet-to-magnet remanence std, Tesla")
    g.add_argument("--brem-seed", type=int, default=0)
    g.add_argument("--meas-shift", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z"),
                   help="shift every measured point by (x, y, z) mm")
    g.add_argument("--z-magnets-shift", type=float, nargs=2, default=(0.0, 0.0), metavar=("NEGZ", "POSZ"),
                   help="z shift (mm) of magnets with z<0 and z>0")
    return p.parse_args(argv)


def detect_b0_direction(measurement_path: str, coord_unit_mm: float = 1.0) -> str:
    """'+x'/'-x'/'+y'/'-y': dominant in-plane axis of the measurement's mean field vector
    (a Halbach field lies in the x-y plane, so z is ignored). Needs a scan with recorded
    components; without them there is nothing to detect and '+y' is assumed."""
    from measurement_io import load_measurement
    m = load_measurement(measurement_path, coord_unit_mm=coord_unit_mm)
    if not m.has_vector:
        print("run_verifier: WARNING: the measurement has no field components, so B0's direction "
              "cannot be detected; assuming +y (shim CSV used as-is).")
        return "+y"
    bx, by = float(np.mean(m.bx_T)), float(np.mean(m.by_T))
    return ("+x" if bx >= 0 else "-x") if abs(bx) >= abs(by) else ("+y" if by >= 0 else "-y")


def shim_to_lab_frame(shim_path: str, direction: str, out_csv: Path) -> Path:
    """Write a copy of the shim CSV rotated from the B0 = +y frame back into the
    measurement's frame (exact inverse of a lab -> optimizer rotation by phi):
        (x, y) -> (x cos phi + y sin phi, -x sin phi + y cos phi),  angle -> angle - phi"""
    from shim_magnets import load_shim_config
    cfg = load_shim_config(shim_path)
    phi = np.deg2rad(_B0_TO_PHI_DEG[direction])
    c, s = np.cos(phi), np.sin(phi)
    lab = replace(cfg,
                  x_mm=cfg.x_mm * c + cfg.y_mm * s,
                  y_mm=-cfg.x_mm * s + cfg.y_mm * c,
                  angle_deg=np.mod(cfg.angle_deg - _B0_TO_PHI_DEG[direction], 360.0))
    pd.DataFrame({"X (mm)": lab.x_mm, "Y (mm)": lab.y_mm, "Z (mm)": lab.z_mm,
                  "Angle (deg)": lab.angle_deg}).to_csv(out_csv, index=False)
    return out_csv


def _load_main():
    spec = importlib.util.spec_from_file_location("verifier_main", HERE / "main.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)      # runs only main.py's configuration + imports
    return mod


def _check_inputs(a: argparse.Namespace) -> None:
    for label, path in (("--measurement", a.measurement), ("--shim", a.shim)):
        if path and not Path(path).is_file():
            sys.exit(f"run_verifier: {label} file not found: {path}")


def main(argv=None) -> None:
    a = _parse(argv)
    _check_inputs(a)

    # Windows pipes default to the ANSI code page; the log contains non-ASCII symbols.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    if not a.show:                     # headless: never try to open a window
        import matplotlib
        matplotlib.use("Agg")

    v = _load_main()

    shim_path = a.shim
    if a.shim and a.shim_frame == "optimizer":
        direction = a.b0_direction
        if direction == "auto":
            direction = detect_b0_direction(a.measurement, a.coord_unit_mm)
        Path(a.out_dir).mkdir(parents=True, exist_ok=True)
        shim_path = str(shim_to_lab_frame(a.shim, direction,
                                          Path(a.out_dir) / f"{Path(a.shim).stem}_labframe.csv"))
        print(f"run_verifier: B0 in the measurement frame = {direction} -> "
              + ("shim CSV already in that frame (no rotation)" if direction == "+y"
                 else f"shim CSV rotated into it: {shim_path}"))
        print()

    # absolute paths make main._resolve ignore its folder arguments
    v.MEASUREMENT_CSV = str(Path(a.measurement).resolve())
    v.SHIM_CSV = str(Path(shim_path).resolve()) if shim_path else None
    v.OUTPUT_DIR = str(Path(a.out_dir).resolve())

    v.BREM_T, v.SIDE_MM, v.COORD_UNIT_MM = a.brem_t, a.side_mm, a.coord_unit_mm
    v.BREM_STD_T, v.BREM_SEED = a.brem_std_t, a.brem_seed
    v.X_MEASUREMENT_SHIFT_MM, v.Y_MEASUREMENT_SHIFT_MM, v.Z_MEASUREMENT_SHIFT_MM = a.meas_shift
    v.Z_MAGNETS_SHIFT_NEGZ_MM, v.Z_MAGNETS_SHIFT_POSZ_MM = a.z_magnets_shift

    v.MAKE_COMPARISON_2D = a.comparison_2d
    v.MAKE_SCENE_3D = a.scene_3d
    v.MAKE_SH_PYRAMID = a.sh_pyramid
    v.SHOW_FIGURES = a.show

    if a.sh_max_degree is not None:
        v.SH_MAX_DEGREE = a.sh_max_degree
    if a.r0_mm is not None:
        v.R0_MM = a.r0_mm
    if a.shim_field_mode is not None:
        v.SHIM_FIELD_MODE = a.shim_field_mode
    if a.color_limit_mt is not None:
        v.COLOR_LIMIT_MT = a.color_limit_mt or None
    if a.share_color_scale is not None:
        v.SHARE_COLOR_SCALE = a.share_color_scale

    Path(v.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    v.main()


if __name__ == "__main__":
    main()

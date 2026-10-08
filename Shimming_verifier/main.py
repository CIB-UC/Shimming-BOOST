import sys
from pathlib import Path
import numpy as np
"""
main.py — the single entry point of the magpylib shimming verifier.

Run it from VSCode (Run Python File, or the "Run Cell" buttons on the `# %%`
markers): edit the variables in the CONFIGURATION block and run. No command line
arguments. Everything else lives in the other modules, which this file just calls:

    measurement_io.py   load_measurement       measured field CSV -> Measurement
    shim_magnets.py     load_shim_config,      shim CSV -> magpylib magnets
                        build_magnets
    field_analysis.py   compute_fields,        measured / shim / shimmed field, ppm
                        summarize
    viewer_2d.py        plot_comparison        2D projections, with vs without shim
    viewer_3d.py        plot_3d_scene          interactive 3D scene (.html)
    sh_pyramid.py       make_sh_figure         spherical-harmonic pyramids (Fig. 10)

Folder layout (next to this script):
    Field_measurements/   measured-field CSVs
    Shimming_magnets/     shim-configuration CSVs
    output/               every generated file is saved here (created if missing)
"""

# %% ── CONFIGURATION ── edit these ─────────────────────────────────────────────

# Folders, relative to this script (absolute paths also work)
MEASUREMENT_DIR = "Field_measurements"
SHIM_DIR = "Shimming_magnets"
OUTPUT_DIR = "output/NOcentershift_withmagnetshift_nostd"

# File names inside those folders (an absolute path here overrides the folder)
MEASUREMENT_CSV = "Try2_UCScanner_NoShim_Sphere200mmDiam_20DegC_08082026.csv"
SHIM_CSV = "InsertSearch_Test_UcNoShim_48inserts_14092026_shim_xyz.csv"   # None = baseline only

# Shim magnets
BREM_T = 1.26            # mean remanence, Tesla
SIDE_MM = 6.0            # cube side, mm
COORD_UNIT_MM = 1.0      # multiply the measurement X/Y/Z by this to get mm

# Magnet-to-magnet variation: every shim magnet gets its own remanence, drawn from a
# normal distribution with mean BREM_T and this standard deviation (Tesla).
# 0 = all magnets exactly BREM_T.
BREM_STD_T = 0 #0.011502929
BREM_SEED = 0            # makes the random draw repeatable; None = new draw every run

# Misalignment study (all in mm, 0 = no shift). Both shifts are applied to the ideal
# setup, and the ppm of the unshifted reference is printed next to the shifted one.
#   measurement shift : moves the coordinates of EVERY measured point. The measured field
#                       values stay attached to their points, so this models "the probe
#                       really was at (file position + shift)".
#   magnet z shifts   : magnets with z < 0 move by Z_MAGNETS_SHIFT_NEGZ_MM and magnets
#                       with z > 0 by Z_MAGNETS_SHIFT_POSZ_MM (both along +z, decided by
#                       the magnet's original z).
X_MEASUREMENT_SHIFT_MM = 0 #np.random.uniform(0, 5)
Y_MEASUREMENT_SHIFT_MM = 0#np.random.uniform(0, 5)
Z_MEASUREMENT_SHIFT_MM = 0 #np.random.uniform(0, 5)
Z_MAGNETS_SHIFT_NEGZ_MM =  0.0 #-3.0
Z_MAGNETS_SHIFT_POSZ_MM = 0.0

# What to produce. Each output has an on/off switch and a file name (saved in OUTPUT_DIR).
MAKE_COMPARISON_2D = True
COMPARISON_2D_PNG = "comparison_2d.png"

MAKE_SCENE_3D = True
SCENE_3D_HTML = "scene_3d.html"

MAKE_SH_PYRAMID = True
SH_PYRAMID_PNG = "sh_pyramid.png"

SHOW_FIGURES = True      # open the matplotlib windows at the end (the PNGs are saved either way)

# Spherical-harmonic pyramid
SH_MAX_DEGREE = 7        # the paper shows n = 1..10
R0_MM = None             # reference radius; None = median radius of the measured points

# What "shim field" means in the pyramid:
#   "along_B0"  : component of B_shim along the local measured field direction, i.e. the
#                 part that actually changes B0 (needs a measurement with x/y/z components).
#                 Use this one: then  measured + shim = shimmed  coefficient by coefficient.
#   "magnitude" : |B_shim|. Always positive, does NOT add to the measured coefficients.
SHIM_FIELD_MODE = "along_B0"

# Colour scale of the pyramid coefficients, in mT (symmetric: +-limit).
#   SHARE_COLOR_SCALE = False : every panel gets its own colour bar
#   SHARE_COLOR_SCALE = True  : one common colour bar for the three panels
#   COLOR_LIMIT_MT = None     : automatic, from the largest coefficient in the panel
#                               (or in all panels, if shared)
#   COLOR_LIMIT_MT = 0.1      : fixed limit, applied to every panel (the paper uses 0.1)
COLOR_LIMIT_MT = 0.1
SHARE_COLOR_SCALE = True

# %% ── CODE ── nothing to edit below ──────────────────────────────────────────

# Folder of this script. `__file__` exists for "Run Python File"; VSCode's interactive
# window (Run Cell) has no `__file__` but sets `__vsc_ipynb_file__` instead.
try:
    _HERE = Path(__file__).resolve().parent
except NameError:
    try:
        _HERE = Path(__vsc_ipynb_file__).resolve().parent   # noqa: F821
    except NameError:
        _HERE = Path.cwd()
if str(_HERE) not in sys.path:          # so the sibling modules import from any cwd
    sys.path.insert(0, str(_HERE))

from field_analysis import compute_fields, ppm, summarize
from measurement_io import load_measurement, shift_measurement
from shim_magnets import build_magnets, load_shim_config, sample_brem, shift_magnets_z


def _resolve(name: str, folder: str = "") -> Path:
    """`name` inside `folder`, where `folder` is relative to this script.
    Absolute paths are returned unchanged."""
    p = Path(name)
    if p.is_absolute():
        return p
    f = Path(folder)
    return (f if f.is_absolute() else _HERE / f) / p


def _out(name: str) -> Path:
    path = _resolve(name, OUTPUT_DIR)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def main():
    # ── load + compute ───────────────────────────────────────────────────────
    meas_ideal = load_measurement(str(_resolve(MEASUREMENT_CSV, MEASUREMENT_DIR)),
                                  coord_unit_mm=COORD_UNIT_MM)
    print(meas_ideal.summary())
    print()

    meas_shift = (X_MEASUREMENT_SHIFT_MM, Y_MEASUREMENT_SHIFT_MM, Z_MEASUREMENT_SHIFT_MM)
    meas = shift_measurement(meas_ideal, *meas_shift)
    if any(meas_shift):
        print(f"Measurement points shifted by (x, y, z) = {meas_shift} mm")

    cfg_ideal = cfg = None
    if SHIM_CSV:
        cfg_ideal = load_shim_config(str(_resolve(SHIM_CSV, SHIM_DIR)))
        print(f"Loaded {cfg_ideal.n_magnets} shim magnets from {SHIM_CSV}")

        cfg = shift_magnets_z(cfg_ideal, Z_MAGNETS_SHIFT_NEGZ_MM, Z_MAGNETS_SHIFT_POSZ_MM)
        if Z_MAGNETS_SHIFT_NEGZ_MM or Z_MAGNETS_SHIFT_POSZ_MM:
            n_neg = int((cfg_ideal.z_mm < 0).sum())
            n_pos = int((cfg_ideal.z_mm > 0).sum())
            print(f"Magnets shifted in z: {n_neg} with z<0 by {Z_MAGNETS_SHIFT_NEGZ_MM} mm, "
                  f"{n_pos} with z>0 by {Z_MAGNETS_SHIFT_POSZ_MM} mm")

        brem = sample_brem(cfg.n_magnets, BREM_T, BREM_STD_T, seed=BREM_SEED)
        print(f"  brem: mean {BREM_T} T, std {BREM_STD_T} T -> sampled mean {brem.mean():.4f}, "
              f"std {brem.std():.4f}, min {brem.min():.4f}, max {brem.max():.4f} T "
              f"(seed {BREM_SEED}); side = {SIDE_MM} mm")
        magnets = build_magnets(cfg, brem_T=brem, side_mm=SIDE_MM)
    else:
        print("No SHIM_CSV given: computing baseline-only (no shim magnets).")
        import magpylib as magpy
        magnets = magpy.Collection()

    print()
    result = compute_fields(meas, magnets)
    print(summarize(result))

    # How much did the perturbations matter? Compare with the ideal setup: no shifts,
    # every magnet exactly BREM_T.
    perturbed = any(meas_shift) or BREM_STD_T > 0 or Z_MAGNETS_SHIFT_NEGZ_MM or Z_MAGNETS_SHIFT_POSZ_MM
    if cfg is not None and perturbed:
        ideal = compute_fields(meas_ideal, build_magnets(cfg_ideal, brem_T=BREM_T, side_mm=SIDE_MM))
        ppm_ideal, ppm_now = ppm(ideal.total_mag_T), ppm(result.total_mag_T)
        print(f"\nShimmed homogeneity, ideal setup     : {ppm_ideal:,.1f} ppm")
        print(f"Shimmed homogeneity, perturbed setup : {ppm_now:,.1f} ppm  "
              f"({ppm_now - ppm_ideal:+,.1f} ppm)")

    made_figures = False

    # ── 2D comparison (with vs without shim) ─────────────────────────────────
    if MAKE_COMPARISON_2D:
        from viewer_2d import plot_comparison
        fig = plot_comparison(meas, result)
        path = _out(COMPARISON_2D_PNG)
        fig.savefig(path, dpi=130, bbox_inches="tight")
        print(f"\nSaved 2D comparison -> {path}")
        made_figures = True

    # ── interactive 3D scene ─────────────────────────────────────────────────
    if MAKE_SCENE_3D:
        from viewer_3d import plot_3d_scene
        path = _out(SCENE_3D_HTML)
        plot_3d_scene(meas, cfg, side_mm=SIDE_MM,      # cfg, meas: as shifted
                      result=result if cfg is not None else None,
                      out_html=str(path),
                      measurement_csv_name=MEASUREMENT_CSV,
                      shim_csv_name=SHIM_CSV)
        print(f"Saved 3D scene -> {path}  (open in any browser)")

    # ── spherical-harmonic pyramids (Figure 10 of the Shimmer paper) ─────────
    if MAKE_SH_PYRAMID:
        from sh_pyramid import make_sh_figure
        header = f"Spherical harmonics — {Path(MEASUREMENT_CSV).name}"
        if SHIM_CSV:
            header += f"\nshim: {Path(SHIM_CSV).name}"
        fig, fits = make_sh_figure(
            meas, result,
            n_max=SH_MAX_DEGREE, r0_mm=R0_MM,
            has_shim=cfg is not None, shim_field_mode=SHIM_FIELD_MODE,
            color_limit=COLOR_LIMIT_MT, share_color_scale=SHARE_COLOR_SCALE,
            suptitle=header,
        )
        print("\nSpherical-harmonic fits:")
        for name, f in fits.items():
            print(f"  {name:9s} n<={f.n_max}: R0 = {f.r0_mm:.1f} mm, "
                  f"residual RMS = {f.rms_resid:.4f} mT, cond = {f.cond:.1e}")
        path = _out(SH_PYRAMID_PNG)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved SH pyramids -> {path}")
        made_figures = True

    if SHOW_FIGURES and made_figures:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()

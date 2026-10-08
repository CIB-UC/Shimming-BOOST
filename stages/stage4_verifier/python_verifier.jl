# python_verifier.jl  —  STAGE 4  (independent Python / magpylib verification)
#
# Run on its own:        julia python_verifier.jl
# Or via the pipeline:   run_pipeline.jl (run_python_verifier = true, or start_stage = 5)
# Or via the GUI:        stage "verify_python"
#
# Hands three decisions to the self-contained Python verifier in Shimming_verifier/:
#   1) which measured field   (verifier_measurement, default: the run's own scan)
#   2) which shim layout      (verifier_use_shim / verifier_shim_csv, default: this
#                              iteration's shim CSV — the Stage-1.5 / insert-search /
#                              OSII seam file)
#   3) what to run            (verifier_make_2d / _3d / _sh)
# and launches  Shimming_verifier/run_verifier.py.  The Python side shares no code with
# Julia: it recomputes the shim field with magpylib, so agreement between its ppm and
# the optimizer's ppm is a genuine cross-check of the operator G, not a tautology.
#
# What this script does itself is only the thin adapter the Python side cannot know:
#   • The seam CSV is  X, Y, RingNumber, Angle.  The verifier needs a literal Z, so
#     RingNumber (the real signed InsertPos) is converted back to axial z with
#     utils/pos_trays.jl's ringpos_from_tray_mm and the current tray-geometry keys.
#     Works for any producer of the seam (gradient/SA export, insert search, OSII).
#     A CSV that already has a Z (mm) column (export_csv.jl's *_shim_xyz.csv) is
#     accepted too; both are treated as living in the OPTIMIZER frame.
#   • Frame. Stage 0 rotates the lab measurement so B0 points along +y before the
#     optimizer sees it, so every shim CSV is in that rotated frame — but the verifier
#     reads the RAW lab-frame measurement. This script only declares the shim CSV's
#     frame (--shim-frame optimizer) and B0's direction (main_field_direction, or
#     "auto" = detected from the scan); run_verifier.py does the rotation. For a scan
#     whose B0 is already +y (the current scanner) it is the identity.
#   • Magnet Br / side and the measurement unit come from the existing config keys
#     (magnet_Br_T, magnet_side_mm, sh_measured_unit_mm), so nothing is entered twice.
#
# All paths + switches come from pipeline_config.jl (§4h).
#   in :  measured_field_path (or verifier_measurement), shim_csv_path (or verifier_shim_csv)
#   out:  optimizer_iter_dir/PythonVerifier/{<ITER>_shim_verifier_input.csv, comparison_2d.png,
#                                            scene_3d.html, sh_pyramid.png}

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

include(joinpath(@__DIR__, "..", "..", "pipeline_config.jl"))
include(joinpath(@__DIR__, "..", "..", "core", "utils", "pos_trays.jl"))       # ringpos_from_tray_mm

# --- resolve the two inputs ----------------------------------------------------
_resolve(name, dir) = isabspath(name) ? name : joinpath(dir, name)

const meas_path = isempty(verifier_measurement) ? measured_field_path :
                                                  _resolve(verifier_measurement, MEASURED_FIELD_DIR)
const shim_in   = verifier_use_shim ?
                  (isempty(verifier_shim_csv) ? shim_csv_path : _resolve(verifier_shim_csv, optimizer_iter_dir)) :
                  nothing

isfile(meas_path) || error("Stage 4: measured field not found:\n    $(meas_path)")
if shim_in !== nothing && !isfile(shim_in)
    error("Stage 4: shim CSV not found:\n    $(shim_in)\n" *
          "Run Stage 1.5 (or an insert search / OSII import) first, point verifier_shim_csv " *
          "at a file, or set verifier_use_shim = false for a baseline-only check.")
end
(verifier_make_2d || verifier_make_3d || verifier_make_sh) ||
    error("Stage 4: nothing to run — enable at least one of verifier_make_2d / _3d / _sh.")

# --- frame: direction of B0 in THIS measurement ---------------------------------
# main_field_direction describes the run's own scan; for any other file, or "auto", the
# Python side detects it from the scan's mean field vector.
const b0_direction = (main_field_direction != "auto" && normpath(meas_path) == normpath(measured_field_path)) ?
                     main_field_direction : "auto"

# --- shim CSV → verifier input (X, Y, Z mm, Angle deg; optimizer frame) ----------
function _col(hdr, prefixes...)
    for p in prefixes, (i, h) in enumerate(hdr)
        startswith(h, p) && return i
    end
    return nothing
end

function write_verifier_shim(in_csv, out_csv)
    lines = filter(!isempty ∘ strip, readlines(in_csv))
    length(lines) >= 2 || error("Stage 4: shim CSV has no magnets: $(in_csv)")
    hdr = lowercase.(strip.(split(lines[1], ',')))
    ix, iy, ia = _col(hdr, "x"), _col(hdr, "y"), _col(hdr, "angle")
    iz    = _col(hdr, "z")
    iring = _col(hdr, "ringnumber", "ring", "insertpos")
    (ix === nothing || iy === nothing || ia === nothing || (iz === nothing && iring === nothing)) &&
        error("Stage 4: can't read $(basename(in_csv)) — need columns X, Y, Angle and either " *
              "Z (mm) or RingNumber. Found: $(hdr)")

    rows = [parse.(Float64, strip.(split(l, ','))) for l in lines[2:end]]
    X = [r[ix] for r in rows]; Y = [r[iy] for r in rows]; A = [r[ia] for r in rows]
    if iz !== nothing
        Z = [r[iz] for r in rows]
        zsrc = "Z (mm) column"
    else
        ring = [r[iring] for r in rows]
        all(r -> r == round(r), ring) ||
            error("Stage 4: RingNumber must be whole tray slots (got e.g. $(first(filter(r -> r != round(r), ring))))")
        Z = ringpos_from_tray_mm(Int.(round.(ring));
                                 tray_slot_spacing_mm = tray_slot_spacing_mm,
                                 front_tray_shift_mm  = front_tray_shift_mm,
                                 back_tray_shift_mm   = back_tray_shift_mm)
        zsrc = "RingNumber → z via the config's tray geometry"
    end

    mkpath(dirname(out_csv))
    open(out_csv, "w") do io
        println(io, "X (mm),Y (mm),Z (mm),Angle (deg)")
        for i in eachindex(X)
            println(io, X[i], ",", Y[i], ",", Z[i], ",", A[i])
        end
    end
    println("Stage 4: shim layout  $(length(X)) magnets, z ∈ [$(minimum(Z)), $(maximum(Z))] mm  ($(zsrc))")
    return out_csv
end

mkpath(verifier_out_dir)
println("Stage 4: measurement  ", meas_path)
shim_arg = shim_in === nothing ? nothing :
           write_verifier_shim(shim_in, joinpath(verifier_out_dir, "$(ITERATION)_shim_verifier_input.csv"))
shim_in === nothing && println("Stage 4: shim layout  none — baseline only")

# --- launch the Python verifier --------------------------------------------------
Sys.which(verifier_python) === nothing &&
    error("Stage 4: Python executable '$(verifier_python)' not found. Set verifier_python in " *
          "config.toml to a full path (a Python with: pip install -r Shimming_verifier/requirements.txt).")

let probe = `$verifier_python -c "import numpy, pandas, scipy, matplotlib, plotly, magpylib"`
    success(pipeline(ignorestatus(probe); stdout = devnull, stderr = devnull)) ||
        error("Stage 4: '$(verifier_python)' is missing packages the verifier needs.\n" *
              "    $(verifier_python) -m pip install -r $(joinpath(VERIFIER_DIR, "requirements.txt"))")
end

flag(name, on) = on ? "--$(name)" : "--no-$(name)"
args = String[verifier_python, joinpath(VERIFIER_DIR, "run_verifier.py"),
              "--measurement", meas_path,
              "--out-dir", verifier_out_dir,
              "--brem-t", string(magnet_Br_T), "--side-mm", string(magnet_side_mm),
              "--coord-unit-mm", string(sh_measured_unit_mm),
              flag("comparison-2d", verifier_make_2d), flag("scene-3d", verifier_make_3d),
              flag("sh-pyramid", verifier_make_sh), flag("show", verifier_show_figures),
              "--sh-max-degree", string(verifier_sh_max_degree)]
# shim_csv_frame = "scan": the CSV is already in the scan's frame -> use as written ("lab");
# "optimizer": it is in the B0 = +y frame and the verifier rotates it into the scan's frame.
shim_arg === nothing || append!(args, ["--shim", shim_arg,
                                       "--shim-frame", shim_csv_frame == "scan" ? "lab" : "optimizer",
                                       "--b0-direction", b0_direction])

cmd = addenv(Cmd(Cmd(args); dir = VERIFIER_DIR), "PYTHONUNBUFFERED" => "1", "PYTHONIOENCODING" => "utf-8")
println("Stage 4: running      ", join(args[1:2], " "), " …\n")
try
    run(pipeline(cmd; stdin = devnull))
catch
    error("Stage 4: the Python verifier failed (see its output above).")
end

# --- confirm the outputs it was asked for ----------------------------------------
expected = String[]
verifier_make_2d && push!(expected, "comparison_2d.png")
verifier_make_3d && push!(expected, "scene_3d.html")
verifier_make_sh && push!(expected, "sh_pyramid.png")
missing_out = [f for f in expected if !isfile(joinpath(verifier_out_dir, f))]
isempty(missing_out) || error("Stage 4: the verifier exited 0 but did not write: $(join(missing_out, ", "))")

println("\nStage 4 done  →  ", verifier_out_dir)
for f in expected
    println("  ", f)
end

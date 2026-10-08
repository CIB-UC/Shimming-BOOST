# osii_to_shim.jl  —  STAGE 1.5 (alternative)  OSII shim-config CSV → BOOST shim CSV
#
# Run on its own:        julia osii_to_shim.jl
# Or via the GUI/backend: stage "osii"  (Stage 1.5 — OSII import)
#
# Converts an OSII passive-shim configuration file (magnet positions + rotations,
# in the OSII coordinate frame, metres) into the shim CSV BOOST Stage 2
# (CSV_to_STL.jl) consumes, with the SAME format export_csv.jl produces:
#       X (mm), Y (mm), RingNumber, Angle (deg)
# so nothing downstream changes, and the iteration name/number (from config.toml)
# flow straight through to the final 3D-printing folder + engraved insert labels.

# ── Frame transform (OSII → BOOST), per the project's axis convention ──────────
#     BOOST X  =  OSII x    ( pos_x )
#     BOOST Y  =  OSII z    ( pos_z )      ← in-plane circle (radius = shim_radius_mm)
#     BOOST Z  = −OSII y    (−pos_y )      ← axial; used only to rank the rings
#   positions: metres → mm   (×1000)
#   Angle_BOOSTs = mod(±rotation_deg + osii_angle_offset_deg, 360)
#     Sign is INVERTED by default (osii_invert_angle = true): OSII's rotation_deg
#     is a twist about OSII +y axis, but BOOST pocket angle (make_square) is a
#     rotation about BOOST +Z = −OSII y — the anti-parallel axial axis — so the
#     same physical twist flips sign. The +offset then places rotation=0 (marker
#     at OSII +z = BOOST +y) at BOOST 90° (BOOST 0° is +x). ⇒ default = mod(90 − rot, 360).
#   RingNumber: the REAL, physical InsertPos (signed tray slot, e.g. -7) that the
#     measured axial position (BOOST Z) falls on, per pos_trays.jl's
#     ringpos_from_tray_mm — NOT a 0-based sequential index. This is what
#     CSV_to_STL.jl / make_label print on the physical part (I / R±NN / T labels),
#     so every Stage-1.5 path (export_csv.jl, insert_search.jl, here) must agree.
#     (The OSII tray/insert/magnet indices are ignored — everything is derived
#      from the positions, exactly as Stage 2 re-derives the tray from X,Y.)
#
# Paths come from pipeline_config.jl:
#   in :  osii_input_path   (OSII_shimming_outputs_toconvert/<osii_input_name>)
#   out:  shim_csv_path     (Optimizer_Output_per_Iteration/<ITER>/<ITER>_shim.csv)

import Pkg
Pkg.activate(@__DIR__)

include("pipeline_config.jl")   # osii_input_path, shim_csv_path, shim_radius_mm, geometry
using CSV, DataFrames, Dates

"""
    osii_to_shim(in_csv, out_csv; angle_offset_deg=osii_angle_offset_deg)

Read an OSII shim-config CSV and write BOOST Stage-2 shim CSV. Returns `out_csv`.
"""
function osii_to_shim(in_csv::AbstractString, out_csv::AbstractString;
                      angle_offset_deg::Real = osii_angle_offset_deg,
                      invert_angle::Bool = osii_invert_angle)
    isfile(in_csv) || error("OSII input not found:\n    $(in_csv)\n" *
                            "Put the OSII CSV in OSII_shimming_outputs_toconvert/ " *
                            "and set osii_input_name in config.toml.")
    src = CSV.read(in_csv, DataFrame)

    for c in ("pos_x_m", "pos_y_m", "pos_z_m", "rotation_deg")
        c in names(src) || error("OSII CSV is missing column '$(c)'.\n" *
                                 "    columns found: $(names(src))")
    end

    # --- frame transform (OSII → BOOSTs) ---------------------------------------
    X   =  1000.0 .* Float64.(src[!, "pos_x_m"])        # BOOST X = OSII x   (mm)
    Y   =  1000.0 .* Float64.(src[!, "pos_z_m"])        # BOOST Y = OSII z   (mm)
    Zax = -1000.0 .* Float64.(src[!, "pos_y_m"])        # BOOST Z = −OSII y  (mm, axial)
    rot = Float64.(src[!, "rotation_deg"])
    rot = invert_angle ? -rot : rot                     # BOOST angle axis (+Z) = −OSII y ⇒ twist flips sign
    Ang = mod.(rot .+ Float64(angle_offset_deg), 360.0)

    # --- rings: recover the REAL, physical InsertPos from the measured axial
    # position, by inverting pos_trays.jl's ringpos_from_tray_mm:
    #     z(+n) =  back_tray_shift_mm  + (n-1)·tray_slot_spacing_mm   (n ≥ 1, +Z "Back")
    #     z(-n) = -(front_tray_shift_mm + (n-1)·tray_slot_spacing_mm) (n ≥ 1, -Z "Front")
    # Rounds to the nearest legal tray slot (measured z is never exact) and warns if
    # any row lands more than half a slot-spacing away — that means the OSII axial
    # data doesn't actually line up with this bore's tray geometry.
    function insertpos_from_z(z::Real)
        if z >= 0
            n = round(Int, (z - back_tray_shift_mm) / tray_slot_spacing_mm) + 1
            resid = z - (back_tray_shift_mm + (n - 1) * tray_slot_spacing_mm)
        else
            n = -(round(Int, (-z - front_tray_shift_mm) / tray_slot_spacing_mm) + 1)
            resid = z - (-(front_tray_shift_mm + (-n - 1) * tray_slot_spacing_mm))
        end
        return n, resid
    end

    zkey        = round.(Zax; digits = 3)                    # collapse FP noise before inverting
    inv         = insertpos_from_z.(zkey)
    Ring        = first.(inv)
    resid       = last.(inv)
    maxresid    = isempty(resid) ? 0.0 : maximum(abs.(resid))
    if maxresid > tray_slot_spacing_mm / 2
        @warn "OSII axial position doesn't line up with the configured tray geometry " *
              "(worst residual $(round(maxresid, digits=2)) mm vs slot spacing " *
              "$(tray_slot_spacing_mm) mm). Check front_tray_shift_mm/back_tray_shift_mm/" *
              "tray_slot_spacing_mm in config.toml against this OSII layout." maxresid = maxresid
    end
    zsort = sort(unique(Zax))                            # for the printed summary only

    # Order rows ring-major (like export_csv's P_cpu ordering), preserving each
    # ring's incoming magnet order so consecutive rows stay within a segment
    # (keeps the intra-segment angle step that CSV_to_STL sanity-checks).
    ord = sortperm(collect(zip(Ring, 1:length(Ring))))

    df = DataFrame("X (mm)"      => X[ord],
                   "Y (mm)"      => Y[ord],
                   "RingNumber"  => Float64.(Ring[ord]),   # real InsertPos (e.g. -7.0), not a sequential index
                   "Angle (deg)" => Ang[ord])

    # --- geometry sanity vs config (same spirit as CSV_to_STL's checks) ------
    meanR = sum(sqrt.(df[!, "X (mm)"] .^ 2 .+ df[!, "Y (mm)"] .^ 2)) / nrow(df)
    if abs(meanR - shim_radius_mm) > 1.0
        @warn "OSII in-plane radius ($(round(meanR, digits=1)) mm) ≠ config shim_radius_mm " *
              "($(shim_radius_mm) mm). Check the axis map / units in the OSII file."
    end

    mkpath(dirname(out_csv))
    CSV.write(out_csv, df)
    println("Wrote ", out_csv, "\n  (", nrow(df), " magnets, ", length(zsort),
            " ring(s) at InsertPos ", join(sort(unique(Ring)), ", "),
            "; angle ", invert_angle ? "−rot" : "+rot", " + ", angle_offset_deg,
            "°, mean radius ", round(meanR, digits = 1), " mm)")
    return out_csv
end

osii_to_shim(osii_input_path, shim_csv_path)

# --- protect this CSV from export_csv.jl -------------------------------------
# The OSII path deliberately skips Stages 0–1, so there is NO optimizer-result
# jld2 describing it. A `run_pipeline.jl` full run, or the GUI "Export + build
# STL" button, calls export_csv.jl — which would either crash on the missing
# result or, worse, silently rebuild the CSV from a STALE result left in this
# iteration folder by an earlier run. Either way the layout you imported is gone.
# This marker makes export_csv refuse with an explanation instead. It records the
# file it describes, so it self-clears the moment anything else rewrites the CSV.
open(joinpath(optimizer_iter_dir, "SPARSE_LAYOUT.marker"), "w") do io
    println(io, "sBOOSTce = osii_to_shim")
    println(io, "written = ", Dates.now())
    println(io, "inserts = (imported layout — not produced by an optimizer)")
    println(io, "magnets = ", countlines(shim_csv_path) - 1)
    println(io, "bytes = ", filesize(shim_csv_path))
end
println("Marked as an imported layout → Stage 2 \"Build STL only\".")

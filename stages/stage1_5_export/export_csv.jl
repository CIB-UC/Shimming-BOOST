# export_csv.jl  —  STAGE 1.5  (the seam: BOOST result → shim CSV)
#
# Run on its own:        julia export_csv.jl
# Or via the pipeline:   run_pipeline.jl launches this as a subprocess
#                        AFTER run_optim.jl has produced the result jld2.
#
# Reads the BOOST optimizer result and the magnet positions, and writes TWO CSVs:
#   1) the insert CSV the geometry stage consumes (unchanged, still the seam):
#       X (mm), Y (mm), RingNumber, Angle (deg)
#   2) a companion "physical" CSV, same rows/order, with the real axial Z (mm)
#      instead of RingNumber — for inspection / external tools that want literal
#      3D coordinates rather than the tray-slot index:
#       X (mm), Y (mm), Z (mm), Angle (deg)
#
# RingNumber is the REAL, physical InsertPos (the signed -25..25 axial tray slot
# from positions_in_tray_new_wished) — NOT a 0-based sequential index. This is what
# CSV_to_STL.jl / make_label print on the physical part (I / R±NN / T labels), and
# what the Stage-3 viewers key off of, so every stage must agree on this one number.
# The companion XYZ CSV is purely a convenience export; nothing downstream reads it.
#
# Angle (deg) in both files is the optimizer's raw solved θ per magnet — the in-plane
# rotation of that magnet's moment (μcosθ, μsinθ) in BOOST's own X–Y frame (0° = +X,
# CCW) — written straight from `bestθ` with NO transform. (Contrast: the OSII import
# path, osii_to_shim.jl, explicitly inverts sign and adds a 90° offset because OSII
# uses a different frame/axis convention; export_csv's angle needs no such correction
# since it's already native to BOOST. CSV_to_STL.jl applies it as-is per magnet,
# independent of which tray/ring, confirming it is not tray-relative.)
#
# All paths come from pipeline_config.jl:
#   in :  optimizer_result_path
#   out:  shim_csv_path, shim_csv_xyz_path

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

include(joinpath(@__DIR__, "..", "..", "pipeline_config.jl"))   # paths (optimizer_result_path, shim_csv_path) + geometry

# --- refuse to clobber a SPARSE per-insert layout -----------------------------
# This script writes EVERY magnet of every ring in positions_in_tray_new_wished.
# stages/stage1_optimize/grad_optim/insert_search.jl instead writes a SPARSE shim CSV (only the inserts it
# chose) and leaves a marker describing it. A `run_pipeline.jl` full run, or the GUI
# "Export + build STL" button, would otherwise silently replace e.g. 6 chosen inserts
# with all 48 of those rings — and you would print the wrong thing with no error.
# The marker is honoured only while it still describes the file actually on disk, so
# it self-clears the moment any other stage legitimately rewrites the CSV.
let marker = joinpath(optimizer_iter_dir, "SPARSE_LAYOUT.marker")
    if isfile(marker) && isfile(shim_csv_path)
        info = Dict{String,String}()
        for ln in readlines(marker)
            parts = split(ln, " = ", limit = 2)
            length(parts) == 2 && (info[strip(parts[1])] = strip(parts[2]))
        end
        if get(info, "bytes", "") == string(filesize(shim_csv_path))
            error("""
            REFUSING to overwrite a sparse per-insert shim CSV.

                $(shim_csv_path)
                written by $(get(info, "source", "?")) on $(get(info, "written", "?"))
                — $(get(info, "inserts", "?")) insert(s), $(get(info, "magnets", "?")) magnets.

            export_csv.jl writes every magnet of every ring in
            positions_in_tray_new_wished, which would replace that layout with a dense
            one and print far more inserts than the search selected.

            -> To build THIS layout:      Stage 2 "Build STL only"  (julia CSV_to_STL.jl)
            -> To go back to a dense run: delete
                   $(marker)
               then re-run Stage 1 ("Grad build+solve") before exporting.
            """)
        end
    end
end

if eval_domain === :shell
    include(joinpath(@__DIR__, "..", "stage1_optimize", "setup_shell.jl"))
else
    include(joinpath(@__DIR__, "..", "stage1_optimize", "setup.jl"))   
end
using CSV, DataFrames, JLD2
include(joinpath(@__DIR__, "..", "..", "core", "utils", "viewer_frame.jl"))   # optimizer <-> scan frame

# --- code from test_mags.jl, path-safe ---
function save_to_csv(Psave, result_file, out_csv, out_csv_xyz)
    @load result_file λ bestθ final_state ppm

    ang_all = Float64.(Array(bestθ))
    st_all  = Float64.(Array(final_state))
    A_all   = Matrix(Psave')       # Nmag x 3  (x, y, z in mm)

    size(A_all, 1) == length(ang_all) == length(st_all) || error(
        "result/geometry mismatch: $(size(A_all,1)) magnet positions but " *
        "$(length(ang_all)) angles and $(length(st_all)) states in $(basename(result_file)).\n" *
        "The result was produced for a different ring set than the current " *
        "positions_in_tray_new_wished — re-run Stage 1, or restore the rings it was solved for.")

    # `final_state` is the optimizer's per-magnet ON/OFF mask. A SPARSE layout marks
    # empty slots with 0 — stages/stage1_optimize/grad_optim/insert_search.jl fills only the inserts it chose,
    # and an SA run can switch magnets off too. Those magnets are not physically there,
    # so writing them would make Stage 2 cut pockets for magnets you never placed.
    # (This filter used to be missing: every magnet was exported regardless of state.)
    keep = findall(>(0.5), st_all)
    isempty(keep) && error("every magnet in $(basename(result_file)) is OFF (final_state all zero).")
    n_off = length(st_all) - length(keep)

    A   = A_all[keep, :]
    ang = ang_all[keep]

    # A is ring-major (setup.jl/setup_shell.jl build P_cpu by looping
    # positions_in_tray_new_wished outer, magnets inner — see positions_from_rings_mm),
    # so distinct-z BLOCKS appear in the same order as positions_in_tray_new_wished.
    # Recover each row's REAL InsertPos by block index into that array, rather than a
    # 0-based counter — RingNumber must be the physical tray slot everywhere downstream.
    z = A[:, 3]
    block = cumsum([0; z[2:end] .!= z[1:end-1]]) .+ 1     # 1-based distinct-z block per row
    n_blocks = maximum(block)
    n_blocks == length(positions_in_tray_new_wished) || error(
        "export_csv.jl: found $(n_blocks) distinct axial ring(s) in the result but " *
        "positions_in_tray_new_wished has $(length(positions_in_tray_new_wished)) — " *
        "geometry/result mismatch. Re-run Stage 1 for the current wished rings before exporting.")
    insert_pos = Float64.(getindex.(Ref(positions_in_tray_new_wished), block))

    # Keep the REAL axial z (mm, as built by positions_from_rings_mm — the literal
    # physical coordinate, before it's swapped for RingNumber below) for the
    # companion XYZ export. This is numerically the same value RingNumber is
    # derived FROM (via ringpos_from_tray_mm), just not yet replaced by the tray-
    # slot index, so the two files describe the identical set of magnets/angles —
    # only the 3rd column's meaning differs (physical Z vs. tray-slot label).
    z_real = copy(z)

    # --- rotate back into the scan's frame (shim_csv_frame = "scan") -----------------------
    # The optimizer works in the frame where B0 -> +y (Stage 0 rotated the scan into it). Here,
    # right after the optimizer, positions AND moment angles are rotated back by the scan's own
    # B0 direction, so every later stage (Stage 2 STLs, viewers, verifier) works in the scan frame.
    # The 12 tray centres map onto each other, so Stage 2 just re-derives the tray from (x,y).
    if shim_csv_frame == "scan"
        sdir = read_field_direction(shell_fieldmap_path, fieldmap_path)
        sdir === nothing && (main_field_direction != "auto" ? (sdir = main_field_direction) :
            error("export_csv.jl: can't tell the scan's B0 direction (no field_direction in the field-map jld2 " *
                  "and main_field_direction = \"auto\"). Re-run Stage 0, or set shim_csv_frame = \"optimizer\"."))
        if sdir != "+y"
            rot = [opt_to_scan_xy(A[i, 1], A[i, 2], sdir) for i in 1:size(A, 1)]
            A[:, 1] = first.(rot); A[:, 2] = last.(rot)
            ang = mod.(ang .+ opt_to_scan_angle_deg(sdir), 360.0)
        end
        println("export: shim CSV rotated back to the SCAN frame (B0 along ", sdir, ")")
    else
        println("export: shim CSV left in the OPTIMIZER frame (B0 -> +y)  [shim_csv_frame = \"optimizer\"]")
    end

    A[:, 3] = insert_pos   # overwrite with RingNumber for the primary (seam) CSV

    data = hcat(A, ang)
    df = DataFrame(data, [:x, :y, :z, :val])

    mkpath(dirname(out_csv))
    CSV.write(out_csv, df, header = ["X (mm)", "Y (mm)", "RingNumber", "Angle (deg)"])
    println("Wrote ", out_csv, "  (", size(df, 1), " magnets, ", n_blocks, " ring(s): ",
            join(positions_in_tray_new_wished, ", "),
            n_off > 0 ? ", $(n_off) empty slot(s) skipped" : "", ")")

    # --- companion CSV: same rows/order, real Z instead of RingNumber ------------
    data_xyz = hcat(A[:, 1], A[:, 2], z_real, ang)
    df_xyz = DataFrame(data_xyz, [:x, :y, :z, :val])
    mkpath(dirname(out_csv_xyz))
    CSV.write(out_csv_xyz, df_xyz, header = ["X (mm)", "Y (mm)", "Z (mm)", "Angle (deg)"])
    println("Wrote ", out_csv_xyz, "  (", size(df_xyz, 1), " magnets, physical X/Y/Z + angle)")

    return out_csv, out_csv_xyz
end

save_to_csv(P_cpu, optimizer_result_path, shim_csv_path, shim_csv_xyz_path)

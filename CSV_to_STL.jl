####### MAGNET INSERT CSV to STL  - JULIA + GMSH ########

# Created by: Low Field Project - Pontificia Universidad Católica de Chile
# Date: December 2025 / January 2026

# This script creates 3D model from a CSV file containing magnet position and rotations, 
# for shimming used in the Low Field MRI project, specifically for OSII 2.1 Halbach Array Configuration
# using trays, not rods. 

####### Packages needed: Gmsh, CSV, Dataframes, Dates ########
####### Files needed in folder: CSV with data, label_generator.jl,  JuliaInsertGenerator.jl ########

# CSV Expected: 
# X (mm), Y(mm), RingNumber, Angle (deg)  (With header)

using Pkg
Pkg.activate(@__DIR__)          # same pinned project as the other stages (has Gmsh)

include(joinpath(@__DIR__, "pipeline_config.jl"))                 # paths + shared geometry
include(joinpath(@__DIR__, "utils/helping_functions_for_JIG.jl")) # defines JIG fns + ensure_pkg

import Gmsh: gmsh


############ Ensuring required packages are installed ############
ensure_pkg("CSV")
ensure_pkg("DataFrames")
ensure_pkg("Dates")

using CSV
using DataFrames
using Dates


############ Pipeline inputs / outputs (from pipeline_config.jl) ############
filename             = shim_csv_path        # Stage 1.5 output (the seam)
create_static_insert = true
Iteration            = iteration_number     # integer engraved on labels


############ Parameters ############

# System Parameters (num_trays, mags_per_segment come from pipeline_config.jl)
angle_between_trays = 2pi / num_trays  # radians
angle_offset_text = 0


# InsertPos -> the printed/foldered P/N sign token, matching make_label's engraved
# N/P glyphs (utils/helping_functions_for_JIG.jl) — e.g. -7 -> "N07", +12 -> "P12".
# Used only for folder/file names on disk; make_label itself takes the raw signed
# Int (ring_num below) and does its own N/P formatting for the physical label.
insertpos_label(pos::Int) = (pos < 0 ? "N" : "P") * lpad(abs(pos), 2, '0')

printingFunction(["Low Field MRI - PUC", "JULIA insert generator"], false)


############ Reading the CSV ############
printingFunction(["Reading Data"], false)
df = CSV.read(filename, DataFrame)
df[!, :Tray] = [assign_tray(r[1], r[2]; num_trays=num_trays) for r in eachrow(df)]

# Geometry is authoritative from pipeline_config.jl (single source of truth).
# theta_deg = inter-magnet angle = arc span / (mags_per_segment - 1).
theta_deg = round(angle_per_segment_deg / (mags_per_segment - 1), digits=3)

# --- sanity check: confirm the CSV actually matches the config geometry ---
csv_radius = round(sum(sqrt.(df[:,1].^2 .+ df[:,2].^2)) / nrow(df), digits=1)
if abs(csv_radius - shim_radius_mm) > 1.0
    @warn "CSV mean radius ($(csv_radius) mm) ≠ config shim_radius_mm ($(shim_radius_mm) mm). Using config value."
end
if nrow(df) >= 2
    t1 = atan(df[1, 2], df[1, 1]) * 180/pi
    t2 = atan(df[2, 2], df[2, 1]) * 180/pi
    csv_theta = round(min(abs(t2 - t1), 360.0 - abs(t2 - t1)), digits=3)
    if abs(csv_theta - theta_deg) > 0.5
        @warn "CSV first-two-row angle step ($(csv_theta)°) ≠ config theta_deg ($(theta_deg)°). Check CSV row ordering."
    end
end

printingFunction(["Shim radius: $(shim_radius_mm) mm  |  theta: $(theta_deg)°"])
ring_amounts = length(unique(df[:, 3])) # Number of rings


############ Creating JuliaInsert ############
printingFunction(["Creating new Magnet Insert for this iteration"], false)
create_magnet_insert_stl(lc, shim_radius_mm, theta_deg, mags_per_segment, mag_cover_thickness, extr_cover_thickness)
if create_static_insert
    printingFunction(["Creating new Static Insert for this iteration"], false)
    if shim_radius_mm < 250
        create_static_insert_stl(lc, shim_radius_mm, theta_deg, removing_depth, mags_per_segment)
    else
        create_static_insert_stl_forJosh(lc, shim_radius_mm, theta_deg, removing_depth, mags_per_segment)
    end
end

# error("Stopping here for testing")
############ Main Loop - Creating Inserts for every Tray and Ring ############

printingFunction(["STARTING LOOP"], false)
for ring_num in sort(unique(Int.(df[:, 3])))
    printingFunction(["Starting with Ring $(insertpos_label(ring_num))  (InsertPos $ring_num)"], false)

    for tray_num in 1:num_trays

        tray_df = filter(r -> Int(r.RingNumber) == ring_num && r.Tray == tray_num, df)
        isempty(tray_df) && continue

        mags_in_tray = nrow(tray_df)
        printingFunction(["Ring $(insertpos_label(ring_num))  Tray $tray_num  ($mags_in_tray magnets)"], true)

        ############ tray center angle (from tray number, not sequential index) ############
        tray_angle_rad = mod((tray_num - 9) * (2pi / num_trays), 2pi)   # e.g. tray 7 → 180°
        insert_rotation = tray_angle_rad - pi/2               # insert starts at +Y (90°)

        ############ load the right insert (radius + magnet count) ############
        gmsh.initialize()
        gmsh.model.add("Base Mesh")
        factory = gmsh.model.occ
        factory.importShapes(joinpath(MAGNET_INSERTS_DIR,
            "MagnetInsert_radius_$(shim_radius_mm)radius_$(mags_per_segment)mags.step"))
        factory.synchronize()

        
        ############ labels ############
        make_label(factory, Iteration, ring_num, tray_num, lc, shim_radius_mm, letter_thickness, label_scale)
        # factory.rotate((3,1), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, insert_rotation)
        factory.synchronize()

        
        vols = gmsh.model.getEntities(3)
        baseVol = vols[1]

        
        ############ rotate insert to tray position ############
        factory.rotate([baseVol], 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, insert_rotation)
        factory.synchronize()


        ############ cut a hole for every magnet in this tray ############
        for row in eachrow(tray_df)
            x_pos     = row[1]
            y_pos     = row[2]
            angle_rad = row[4] * pi / 180
            make_square(factory, (x_pos, y_pos), width_holes,
                        angle_rad - angle_offset_text, lc, total_thickness)
        end
        factory.synchronize()


        ############ boolean cut ############
        allVols     = factory.getEntities(3)
        removingVols = [v for v in allVols if v != baseVol]
        factory.cut([baseVol], removingVols)
        factory.synchronize()

        ############ find the surviving solid — DO NOT assume it kept tag 1 ############
        # OCC's boolean cut does not guarantee the result keeps the input's tag; it can
        # renumber, or (on messy/degenerate geometry — see the BOPAlgo_Alert* warnings
        # this insert may print) even leave more than one solid behind. Re-query instead
        # of hardcoding (3,1), and fail loudly rather than silently rotating/writing the
        # wrong (or a nonexistent) entity.
        postCutVols = gmsh.model.getEntities(3)
        isempty(postCutVols) && error(
            "Ring $(insertpos_label(ring_num)) Tray $tray_num: boolean cut left NO solid " *
            "(likely degenerate/self-intersecting magnet holes — check width_holes, magnet " *
            "spacing, and the BOPAlgo_Alert* warnings above). Skipping is not safe; fix the " *
            "geometry for this tray and re-run.")
        if length(postCutVols) > 1
            @warn "Ring $(insertpos_label(ring_num)) Tray $tray_num: cut left $(length(postCutVols)) " *
                  "solids instead of 1 — keeping the largest by volume, removing the rest" tags = postCutVols
            vols_by_size = sort(postCutVols, by = v -> gmsh.model.occ.getMass(v[1], v[2]), rev = true)
            keepVol = vols_by_size[1]
            factory.remove([v for v in vols_by_size[2:end]], true)
            factory.synchronize()
        else
            keepVol = postCutVols[1]
        end

        ############ rotate back to standard orientation for output ############
        factory.rotate([keepVol], 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -insert_rotation)
        factory.synchronize()


        ### Outputing as .step (to edit) and .stl (to print, without magnet holes) ###
        # Folder/file names use the P/N token (e.g. RingN07) instead of a bare signed
        # integer (Ring-7) — avoids a literal minus sign in a Windows path/filename,
        # and matches the InsertPos label engraved on the part by make_label.
        ring_tag = insertpos_label(ring_num)
        tray_tag = lpad(tray_num, 2, '0')
        outdir =final_iteration_dir
        outdir_stl = joinpath(outdir, "stl_outputs")
        outdir_step = joinpath(outdir, "Step_outputs")
        mkpath(outdir) # creates the folder if it doesn't exist
        mkpath(outdir_stl)  # creates the folder if it doesn't exist
        mkpath(outdir_step)  # creates the folder if it doesn't exist

        gmsh.write(joinpath(outdir_stl, "Ring$(ring_tag)_Tray$(tray_tag).stl"))

        ############ rotate  to magnet orientation for output in step ############
        factory.rotate([keepVol], 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -insert_rotation)
        factory.synchronize()

        gmsh.write(joinpath(outdir_step, "Ring$(ring_tag)_Tray$(tray_tag).step"))
        
        # if !("-nopopup" in ARGS)
        #     gmsh.fltk.run()
        # end

        gmsh.finalize()
        printingFunction(["Finished Insert with data: Ring $(insertpos_label(ring_num)), Tray $tray_num"], true)
    end
    printingFunction(["Finished with Ring $(insertpos_label(ring_num))"], false)
end

# Copy this iteration's static insert into the iteration output folder, so the
# 3D-printing folder is self-contained (everything to print is in one place).
if create_static_insert
    static_base = "StaticInsert_radius_$(shim_radius_mm)radius_$(mags_per_segment)mags"
    for ext in (".stl", ".step")
        src = joinpath(STATIC_INSERTS_DIR, static_base * ext)
        isfile(src) && cp(src, joinpath(final_iteration_dir, static_base * ext); force = true)
    end
    printingFunction(["Static insert copied into: $(final_iteration_dir)"], false)
end

printingFunction(["PROCESS FINISHED!!", "Inserts available in: $(final_iteration_dir)"], false)
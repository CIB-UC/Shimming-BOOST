####### Magnet Array 3D Visualizer - GLMakie ########
# Created by: Low Field Project - Pontificia Universidad Católica de Chile
#
# Loads a CSV (X (mm), Y (mm), RingNumber, Angle (deg)) and renders every
# magnet as a square footprint at its (X, Y) position, rotated by its
# magnetization direction (Angle), with a circle marking the pole side.
#
# Ring Z (axial) positions come from the physical tray geometry in
# Boost_codes/pos_trays.jl (the same ringpos_from_tray_mm the optimization
# pipeline uses). RingNumber IS the real, physical InsertPos (signed tray slot),
# so each ring's z is computed directly from its own RingNumber value.
#
# A measured magnetic field is loaded from pipeline_config.jl's fieldmap_path (By_grid, mT, on
# axes xg/yg/zg in mm). The shim magnets' By contribution is computed on that
# same grid with Boost_codes/imanes.jl (GPU dipole kernel), and the shimmed
# field is the sum. A selector switches the point cloud (and the ppm readout)
# between Measured / Magnet only / Shimmed; a button shows/hides the cloud.
#
# Usage: julia magnet_visualizer.jl   (requires a CUDA GPU for the field calc)



using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))                 # same pinned project as the rest of the pipeline
@isdefined(ITERATION) || include(joinpath(@__DIR__, "..", "..", "pipeline_config.jl"))

using GLMakie
using CSV
using DataFrames
using Statistics
using Dates                                          # timestamped GIF filenames

include(joinpath(@__DIR__, "..", "..", "core", "utils/pos_trays.jl"))   # ringpos_from_tray_mm
include(joinpath(@__DIR__, "..", "..", "core", "utils/viewer_frame.jl")) # optimizer <-> scan frame (display)



# Master switch: when false, skip ALL field work (no jld2 load, no GPU dipole
show_field = true

# FIELD-ONLY mode: with no shim CSV yet (e.g. right after Stage 0), there are no
# magnets to draw — show just the measured field cloud. The field IS the whole view
# then, so force it on.
has_shim = isfile(shim_csv_path)
has_shim || (show_field = true)

if show_field
    using JLD2
    using CUDA
    include(joinpath(@__DIR__, "..", "..", "core", "utils/imanes.jl"))   # dipole field (GPU)
end

############ Config ############
magnet_size_mm   = 6.0                              # side length of each magnet's square footprint
tip_marker_size  = 2.5                              # diameter (mm, data-space) of the pole-side marker

# Shim solution comes straight from the shim CSV (the geometry stage's own output);
# each ring's tray geometry is derived from its RingNumber (real InsertPos) below,
# not from positions_in_tray_new_wished/positions_in_tray_occupied — those describe
# what THIS run of the optimizer was asked to place, not what the CSV on disk holds.

# Measured magnetic field comes from pipeline_config.jl's fieldmap_path
# (By_grid on axes xg,yg,zg). `show_field` is set at the very top of the file.
field_marker_size_mm = 6.0          # data-space diameter of each field sample point
field_colormap       = :plasma      # kept distinct from the rings' :turbo
field_max_points     = 200_000      # subsample the grid above this many points (responsiveness)
default_field        = has_shim ? "Shimmed" : "Measured"   # field-only ⇒ Measured
show_field_initially = true         # start with the field cloud visible (toggle with the button)

# Only show / score field samples within this radius (mm) of the bore centre —
# i.e. BOOST's homogeneity shell. The raw measurement spans ±525 mm in z, so
# WITHOUT this the cloud dwarfs the magnet rings (points far outside the ring).
# Increase it to inspect more of the measured volume. Further restricted below
# by field_valid_radius_mm (read from the jld2) so a :spherical_harmonics field
# map never shows the model's extrapolation past the actual measured shell.
field_clip_radius_mm = Rmax

# Shim magnet strength: pipeline_config.jl's magnet_moment_Am2, derived from
# config.toml's magnet_Br_T (remanence) + magnet_side_mm (cube side length) —
# passed straight to build_dipoles_device as m_mag, so the drawn field uses the
# SAME magnets the optimizer solved for, with no second B1cm-proxy conversion.
shim_moment_Am2      = magnet_moment_Am2
magnet_axis          = :x           # moment base axis for build_dipoles_device (matches the optimizer)

############ Load CSV (only if a shim solution exists) ############
has_shim || println("No shim CSV at $(shim_csv_path)  →  FIELD-ONLY mode (measured field cloud, no magnets).")
if has_shim
    csv_path = shim_csv_path        # Stage 1.5 output (the seam)
    df = CSV.read(csv_path, DataFrame)

    x         = Float32.(df[:, 1])
    y         = Float32.(df[:, 2])
    ring      = Int.(df[:, 3])
    angle_deg = Float32.(df[:, 4])
    angle_rad = angle_deg .* (Float32(pi) / 180)

    ring_lo, ring_hi = extrema(ring)
    n_rings = length(unique(ring))

    println("Loaded $(nrow(df)) magnets across $(n_rings) ring(s): $(sort(unique(ring)))")

    ############ Ring axial (Z) positions from tray geometry ############
    # RingNumber is the REAL, physical InsertPos (signed tray slot) — so each ring's z
    # comes from calling ringpos_from_tray_mm on THAT ring value directly, not from
    # zipping unique_rings against wished_trays positionally (wished_trays' array order
    # need not match unique_rings' ascending sort, and a run can also drop/add rings vs.
    # config, e.g. an OSII import or a sparse insert-search result). Driven by the SAME
    # config keys positions_from_rings_mm uses, so the magnets are drawn exactly where
    # the optimizer placed them. (This used to hardcode half_shift_mm = 0.0, which
    # silently disagreed with the optimizer for any non-zero shift — the viewer drew
    # the right magnets at the wrong z.)
    unique_rings = sort(unique(ring))
    ring_z_mm    = ringpos_from_tray_mm(unique_rings;
                                        tray_slot_spacing_mm = tray_slot_spacing_mm,
                                        front_tray_shift_mm  = front_tray_shift_mm,
                                        back_tray_shift_mm   = back_tray_shift_mm)
    ring_to_z = Dict(r => Float32(z) for (r, z) in zip(unique_rings, ring_z_mm))
    z_all = Float32[ring_to_z[r] for r in ring]   # per-magnet axial position (mm)

    println("Ring -> Z (mm): " * join(["$(r) => $(ring_to_z[r])" for r in unique_rings], ", "))
end



if show_field   # ===== all field work guarded by show_field =====

    ############ Load measured magnetic field ############
    # By_grid: 3D fieldmap in mT on the regular axes xg, yg, zg (mm). Following
    # setup.jl's axis reshapes, By_grid[i,j,k] is the field at (xg[i], yg[j], zg[k]).
    field_path = fieldmap_path
    @load field_path By_grid xg yg zg

    # Sign of the main field as recorded by the scan. Stage 0 stores |B| (positive) in the
    # optimizer frame (B0 -> +y); the resolved scan direction ("+y"/"-y"/...) is saved next to it.
    # A "-" direction means the real field points the opposite way, so the viewer displays the
    # SIGNED component (negative values) instead of |B|, for the measured, shim and shimmed fields.
    # Missing key (older jld2) -> "+y", i.e. unchanged behaviour.
    field_dir = String(let d = "+y"
        jldopen(field_path, "r") do f
            haskey(f, "field_direction") ? (d = f["field_direction"]) :
            haskey(f, "main_field_direction") && (d = f["main_field_direction"])
        end
        d
    end)
    use_scan    = viewer_frame == "scan"
    # Signed display only in the scan frame; in the optimizer frame B0 is +y by construction (|B| > 0).
    field_sign  = (use_scan && startswith(field_dir, "-")) ? -1f0 : 1f0
    field_label = use_scan ? "B (mT), B0 along $(field_dir)" : "By (mT)"
    frame_note  = use_scan ? "scan frame (B0 along $(field_dir))" : "optimizer frame (B0 → +y)"
    # The shim CSV may be in the SCAN frame (shim_csv_frame): bring it into the optimizer frame, where the
    # dipole field is computed (display is rotated back below if viewer_frame = "scan").
    if has_shim && shim_csv_frame == "scan" && field_dir != "+y"
        _r = [scan_to_opt_xy(x[i], y[i], field_dir) for i in eachindex(x)]
        global x = Float32.(first.(_r)); global y = Float32.(last.(_r))
        global angle_deg = Float32.(mod.(angle_deg .+ scan_to_opt_angle_deg(field_dir), 360))
        global angle_rad = angle_deg .* (Float32(pi) / 180)
    end

    xg32, yg32, zg32 = Float32.(xg), Float32.(yg), Float32.(zg)
    nx, ny, nz = size(By_grid)
    @assert (nx, ny, nz) == (length(xg), length(yg), length(zg)) "By_grid dims $(size(By_grid)) don't match axes lengths ($(length(xg)), $(length(yg)), $(length(zg)))"

    # How far the field map can be trusted (written by Stage 0). :reshape saves
    # Inf (the whole grid is literally measured, no interpolation); :spherical_harmonics
    # saves the measured shell's radius (mm) — beyond it the model is
    # extrapolating with NO supporting data. Missing key (older jld2) -> Inf,
    # i.e. no extra clipping beyond field_clip_radius_mm above.
    field_valid_radius_mm = Float32(let r = Inf
        jldopen(field_path, "r") do f
            haskey(f, "field_valid_radius_mm") && (r = f["field_valid_radius_mm"])
        end
        r
    end)

    measured_full = field_sign .* Float32.(By_grid)   # mT (signed: negative when B0 points along -x/-y)
    println("Loaded field grid $(nx)×$(ny)×$(nz); measured By range " *
            "[$(round(minimum(measured_full), digits=3)), $(round(maximum(measured_full), digits=3))] mT")
    isfinite(field_valid_radius_mm) &&
        println("  field map trusted only to r ≤ $(round(field_valid_radius_mm, digits=1)) mm (beyond that is SH extrapolation)")

    ############ Magnet field contribution (GPU dipoles, on the jld2 grid) ############
    # Reuse imanes.jl's building blocks directly on the REAL axes (xg,yg,zg) so the
    # field is evaluated at the exact measured grid points, then summed with it.
    function magnet_By_on_axes(posiciones, θdeg, xgv, ygv, zgv;
                            m_mag::Real, axis::Symbol = :y, to_mT::Bool = true)
        grid = make_grid_gpu_from_axes_mm(xgv, ygv, zgv)
        px, py, pz, mx, my, mz = build_dipoles_device(posiciones, θdeg; axis = axis, m_mag = Float32(m_mag))
        dBy = CUDA.zeros(Float32, Int(grid.nx), Int(grid.ny), Int(grid.nz))
        N = Int(grid.nx) * Int(grid.ny) * Int(grid.nz)
        @cuda threads=256 blocks=cld(N, 256) _by_dipoles_kernel!(
            dBy, grid.X, grid.Y, grid.Z,
            px, py, pz, mx, my, mz,
            μ0_4π_f32, EPS_R2_f32, grid.nx, grid.ny, grid.nz)
        CUDA.synchronize()
        to_mT && (dBy .*= 1_000f0)
        return Array(dBy)
    end

    # Magnets currently displayed: positions in mm (X, Y from CSV; Z from tray geom),
    # angles = the CSV's Angle column (the optimizer's per-magnet θ). Only with a shim.
    if has_shim
        posiciones = [(Float64(x[i]), Float64(y[i]), Float64(z_all[i])) for i in eachindex(x)]
        θdeg       = Float64.(angle_deg)

        magnet_full  = field_sign .* magnet_By_on_axes(posiciones, θdeg, xg, yg, zg;
                                        m_mag = shim_moment_Am2, axis = magnet_axis, to_mT = true)
        shimmed_full = measured_full .+ magnet_full
        println("Magnet By range [$(round(minimum(magnet_full), digits=3)), $(round(maximum(magnet_full), digits=3))] mT")
    end

    # Display frame. The dipole field above is computed in the OPTIMIZER frame (B0 -> +y); for
    # "scan" re-express the field arrays + axes in the scan's own frame (display-only).
    if use_scan && field_dir != "+y"
        measured_full, xg_s, yg_s = grid_to_scan(measured_full, xg, yg, field_dir)
        if has_shim
            magnet_full  = grid_to_scan(magnet_full,  xg, yg, field_dir)[1]
            shimmed_full = grid_to_scan(shimmed_full, xg, yg, field_dir)[1]
        end
        xg32, yg32 = Float32.(xg_s), Float32.(yg_s)
        nx, ny = size(measured_full, 1), size(measured_full, 2)
    end

    ############ Subsample the grid into a point cloud (clipped to the shell) ############
    clip_r = min(field_clip_radius_mm, field_valid_radius_mm)   # homogeneity shell ∧ actual measured extent
    stride = max(1, ceil(Int, (nx * ny * nz / field_max_points)^(1/3)))
    idxs = [(i, j, k) for k in 1:stride:nz for j in 1:stride:ny for i in 1:stride:nx
            if sqrt(xg32[i]^2 + yg32[j]^2 + zg32[k]^2) <= clip_r]
    @assert !isempty(idxs) "No field samples within $(clip_r) mm; increase field_clip_radius_mm or check field_valid_radius_mm."
    field_points = [Point3f(xg32[i], yg32[j], zg32[k]) for (i, j, k) in idxs]
    println("Field cloud: $(length(idxs)) samples within r ≤ $(clip_r) mm of bore centre")

    sample_field(F) = Float32[F[i, j, k] for (i, j, k) in idxs]
    ppm_of(s) = (maximum(s) - minimum(s)) / abs(mean(s)) * 1e6   # over the clipped (shell) samples

    if has_shim
        field_names = ["Measured", "Magnet only", "Shimmed"]
        field_fulls = Dict("Measured" => measured_full, "Magnet only" => magnet_full, "Shimmed" => shimmed_full)
    else
        field_names = ["Measured"]
        field_fulls = Dict("Measured" => measured_full)
    end
    field_samples = Dict(k => sample_field(field_fulls[k]) for k in field_names)
    field_ppm     = Dict(k => ppm_of(field_samples[k])     for k in field_names)
    field_lims    = Dict(k => extrema(field_samples[k])    for k in field_names)

    for k in field_names
        println("$(rpad(k, 12)) ppm = $(round(field_ppm[k], digits=1))")
    end

end   # ===== if show_field =====

############ Square geometry + magnet segments (only with a shim solution) ############
# "front" edge (c1-c2) faces local +X; after rotation it points along the
# magnet's magnetization direction -- that's where the pole-marker circle sits.
# Magnets for display: the CSV is in the optimizer frame; for the "scan" frame rotate positions
# and moment angles back by the scan direction (the dipole field above already used the originals).
if has_shim && viewer_frame == "scan" && !(@isdefined(field_dir)) 
    error("viewer_frame = \"scan\" needs the field map (show_field) to know the scan direction.")
end
if has_shim && viewer_frame == "scan" && field_dir != "+y"
    _rot = [opt_to_scan_xy(x[i], y[i], field_dir) for i in eachindex(x)]
    x = Float32.(first.(_rot)); y = Float32.(last.(_rot))
    angle_rad = angle_rad .+ Float32(deg2rad(opt_to_scan_angle_deg(field_dir)))
end

if has_shim
half_size = magnet_size_mm / 2
local_corners = [
    ( half_size,  half_size),   # c1 - front-left
    ( half_size, -half_size),   # c2 - front-right
    (-half_size, -half_size),   # c3 - back-right
    (-half_size,  half_size),   # c4 - back-left
]

cosA = cos.(angle_rad)
sinA = sin.(angle_rad)

# Accept any Real (half_size is Float64) and coerce to Float32 internally.
rotated_xy(lx::Real, ly::Real) = (Float32(lx) .* cosA .- Float32(ly) .* sinA,
                                  Float32(lx) .* sinA .+ Float32(ly) .* cosA)

corner_x = Vector{Vector{Float32}}(undef, 4)
corner_y = Vector{Vector{Float32}}(undef, 4)
for (k, (lx, ly)) in enumerate(local_corners)
    corner_x[k], corner_y[k] = rotated_xy(Float32(lx), Float32(ly))
end

pole_offset_x, pole_offset_y = rotated_xy(half_size, 0f0)   # midpoint of the front edge

############ Build magnet geometry (static -- Z is fixed) ############
# Each magnet -> 4 line segments (rotated square edges). Manual linesegments!
# instead of arrows!/arrows3d! (broken with Observable+Vector mix in this Makie).
n = length(x)
seg_pts = Vector{Point3f}(undef, 8n)
for i in 1:n
    cx, cy, cz = x[i], y[i], z_all[i]
    for k in 1:4
        k2 = (k % 4) + 1
        base = 8 * (i - 1)
        seg_pts[base + 2k - 1] = Point3f(cx + corner_x[k][i],  cy + corner_y[k][i],  cz)
        seg_pts[base + 2k]     = Point3f(cx + corner_x[k2][i], cy + corner_y[k2][i], cz)
    end
end

pole_points = Point3f.(x .+ pole_offset_x, y .+ pole_offset_y, z_all)

seg_colors = Vector{Int}(undef, 8n)
for i in eachindex(ring)
    base = 8 * (i - 1)
    for k in 1:8
        seg_colors[base + k] = ring[i]
    end
end
end   # if has_shim (magnet geometry)




############ Figure ############

# LScene (not Axis3) for full Camera3D: right-drag pan + accessible near/far.
fig = Figure(size = (1500, 950))
title_label = Label(fig[0, show_field ? (1:2) : (1:1)],
    (has_shim ? "Magnet Array Visualizer" : "Field Map Viewer — measured field, pre-shim") * (@isdefined(frame_note) ? "   —   " * frame_note : ""), fontsize = 20)
ls = LScene(fig[1, 1], show_axis = true)

if has_shim   # draw the magnets only when there's a shim solution
    linesegments!(ls, seg_pts; color = seg_colors, colormap = :turbo, linewidth = 3)
    scatter!(ls, pole_points; color = ring, colormap = :turbo,
        markersize = tip_marker_size, markerspace = :data)
end

if show_field
    # Reactive field point cloud: color + colorrange + ppm follow the selector.
    global name_obs   = Observable(default_field)
    global color_obs  = Observable(field_samples[default_field])
    global crange_obs = Observable(field_lims[default_field])
    global ppm_obs    = Observable(field_ppm[default_field])

    global field_plot = scatter!(ls, field_points; color = color_obs, colormap = field_colormap,
        colorrange = crange_obs, markersize = field_marker_size_mm, markerspace = :data,
        visible = show_field_initially)

    Colorbar(fig[1, 2], limits = crange_obs, colormap = field_colormap, label = field_label, width = 25)
end

############ End labels (fixed Z, at ±200 mm along the bore) ############
front_label_z = -200f0   # −Z end (Gaussmeter side)
back_label_z  =  200f0   # +Z end (Wall side)

text!(ls, Point3f(0, 0, front_label_z); text = "Front - Gaussmeter", fontsize = 20, color = :black,
    markerspace = :data, align = (:center, :center),
    rotation = Makie.qrotation(Vec3f(0, 1, 0), Float32(π)))
text!(ls, Point3f(0, 0, back_label_z); text = "Back - Wall", fontsize = 20, color = :black,
    markerspace = :data, align = (:center, :center))

############ Ring colour legend (colour → printed InsertPos label) ############
# Small floating legend (top-left of the 3D view) mapping each ring's turbo colour to
# its RingNumber. Colours match the magnets exactly: ring value r is normalised over
# [ring_lo, ring_hi] into the :turbo colormap (as the plots do). RingNumber IS the
# physical InsertPos (signed tray slot) now, so there's no separate "tray" to look up —
# the label just shows the same P/N token engraved on the physical part
# (utils/helping_functions_for_JIG.jl's make_label), e.g. -7 -> "N07", +12 -> "P12".
if has_shim
    insertpos_label(pos::Int) = (pos < 0 ? "N" : "P") * lpad(abs(pos), 2, '0')
    _ring_cmap     = cgrad(:turbo)
    ring_colour(r) = ring_hi == ring_lo ? _ring_cmap[0.5] :
                     _ring_cmap[(r - ring_lo) / (ring_hi - ring_lo)]
    legend_swatches = [MarkerElement(color = ring_colour(r), marker = :rect, markersize = 16)
                       for r in unique_rings]
    legend_labels   = ["Ring $(insertpos_label(r))" for r in unique_rings]
    Legend(fig[1, 1], legend_swatches, legend_labels, "Rings";
        tellwidth = false, tellheight = false, halign = :left, valign = :top,
        margin = (10, 10, 10, 10), framevisible = true, labelsize = 14, titlesize = 15)
end

    
# Pin the main 3D view large so neither the colorbar column nor the controls
# row can squeeze it into a strip. (Rings-only mode keeps Makie's defaults.)
if show_field
    colsize!(fig.layout, 1, Relative(0.92))  # 3D view column
    colsize!(fig.layout, 2, Relative(0.08))  # narrow colorbar column
    rowsize!(fig.layout, 1, Relative(0.85))  # main view row stays tall
end

############ Camera clipping fix ############
cc = Makie.cameracontrols(ls.scene)
cc.near[] = 0.01f0
cc.far[]  = 1.0f6

############ Rotating-GIF export (share the 3D view without a live session) ############
# Spins the camera 360° around the vertical (Z, the bore axis) at a fixed
# elevation/distance — derived from wherever you've currently left the camera, so
# rotating/zooming first changes what gets recorded — and writes one frame per
# step via GLMakie's `record`. viewer_gif_frames/viewer_gif_fps come from
# config.toml (pipeline_config.jl); length in seconds = frames / fps.
function save_rotation_gif(path::AbstractString; label::AbstractString = "")
    eye0, look0, up0 = Vec3f(cc.eyeposition[]), Vec3f(cc.lookat[]), Vec3f(cc.upvector[])
    radius_xy = hypot(eye0[1] - look0[1], eye0[2] - look0[2])
    z_off     = eye0[3] - look0[3]
    θ0        = atan(eye0[2] - look0[2], eye0[1] - look0[1])
    old_title = title_label.text[]
    !isempty(label) && (title_label.text[] = old_title * "  —  " * label)
    Makie.record(fig, path, 1:viewer_gif_frames; framerate = viewer_gif_fps) do i
        θ = θ0 + 2π * (i - 1) / viewer_gif_frames
        eye = Vec3f(look0[1] + radius_xy * cos(θ), look0[2] + radius_xy * sin(θ), look0[3] + z_off)
        Makie.update_cam!(ls.scene, eye, look0, up0)
    end
    title_label.text[] = old_title
    return path
end

############ Bottom controls: toggle + field selector + ppm readout ############
# Placed under the 3D-view column only (not spanning into the colorbar column),
# so the controls never affect the main view's width.
if show_field
    controls = GridLayout(fig[2, 1])

    field_btn = Button(controls[1, 1],
        label = show_field_initially ? "Hide field" : "Show field",
        tellwidth = false, halign = :left)

    Label(controls[1, 2], "Show:", halign = :right, tellwidth = false)
    field_menu = Menu(controls[1, 3], options = field_names, default = default_field,
        width = 150, tellwidth = false)

    ppm_text = @lift(string($name_obs, ": ", round($ppm_obs, digits = 1), " ppm   [(max−min)/mean]"))
    Label(controls[1, 4], ppm_text; halign = :right, tellwidth = false, fontsize = 18)

    on(field_btn.clicks) do _
        new_state = !field_plot.visible[]
        field_plot.visible[] = new_state
        field_btn.label[]    = new_state ? "Hide field" : "Show field"
    end

    on(field_menu.selection) do sel
        name_obs[]   = sel
        color_obs[]  = field_samples[sel]
        crange_obs[] = field_lims[sel]
        ppm_obs[]    = field_ppm[sel]
    end
end

############ GIF export controls (share the view without a live session) ############
gif_row  = GridLayout(fig[show_field ? 3 : 2, 1])
gif_status = Observable("")

gif_btn = Button(gif_row[1, 1], label = "Save rotating GIF (current view)", tellwidth = false)
on(gif_btn.clicks) do _
    gif_status[] = "Recording…"
    stamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    label = show_field ? name_obs[] : (has_shim ? "Shimmed" : "Measured")
    path  = joinpath(viewer_gif_dir, "3Dview_$(label)_$(stamp).gif")
    save_rotation_gif(path; label = label)
    gif_status[] = "Saved → " * path
end

if show_field && has_shim
    compare_btn = Button(gif_row[1, 2], label = "Save GIF pair: Measured vs Shimmed", tellwidth = false)
    on(compare_btn.clicks) do _
        gif_status[] = "Recording Measured…"
        stamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
        # freeze the current field selection, run both, then restore it
        sel0 = name_obs[]
        paths = String[]
        for nm in ("Measured", "Shimmed")
            name_obs[]   = nm
            color_obs[]  = field_samples[nm]
            crange_obs[] = field_lims[nm]
            ppm_obs[]    = field_ppm[nm]
            gif_status[] = "Recording $(nm)…"
            p = joinpath(viewer_gif_dir, "3Dview_$(nm)_$(stamp).gif")
            save_rotation_gif(p; label = nm)
            push!(paths, p)
        end
        name_obs[]   = sel0
        color_obs[]  = field_samples[sel0]
        crange_obs[] = field_lims[sel0]
        ppm_obs[]    = field_ppm[sel0]
        field_menu.i_selected[] = findfirst(==(sel0), field_names)
        gif_status[] = "Saved 2 GIFs → " * dirname(paths[1])
    end
end

Label(gif_row[1, show_field && has_shim ? 3 : 2], gif_status; halign = :left, tellwidth = false, fontsize = 13)

############ Show ############
screen = display(fig)
println("3D window open. Use the menu to switch fields and the button to show/hide. Close to exit.")
wait(screen)

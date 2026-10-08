####### Field Slice & Line Viewer - GLMakie ########
# Created by: Low Field Project - Pontificia Universidad Católica de Chile
#
# Inspects three By fields on the jld2 grid:
#   Measured  -> By_grid loaded from pipeline_config.jl's fieldmap_path (mT)
#   Shim      -> the magnets-only contribution (CSV placements in Data/, dipole
#                field computed on the same grid via Boost_codes/imanes.jl, GPU)
#   Shimmed   -> Measured + Shim
#
# LEFT panel: a 2D heatmap of a chosen field on a chosen plane (XY/XZ/YZ) at a
#             chosen slice position along the perpendicular axis.
# RIGHT panel: a 1D profile along a line in that slice. The "Profile axis" menu
#              picks the axis the profile runs ALONG (the free one); the line
#              slider fixes the other in-plane axis. All three fields overlaid.
#
# Usage: julia field_slice_viewer.jl   (requires a CUDA GPU for the shim field)

using Pkg
Pkg.activate(@__DIR__)                 # same pinned project as the rest of the pipeline
@isdefined(ITERATION) || include(joinpath(@__DIR__, "pipeline_config.jl"))

using GLMakie
using CSV
using DataFrames
using Statistics
using JLD2
using CUDA
using Dates                                          # timestamped GIF filenames

include(joinpath(@__DIR__, "utils/pos_trays.jl"))   # ringpos_from_tray_mm
include(joinpath(@__DIR__, "utils/imanes.jl"))      # dipole field (GPU)
include(joinpath(@__DIR__, "utils/viewer_frame.jl")) # optimizer <-> scan frame (display)

############ Config (from pipeline_config.jl) ############
# Shim solution and field map come from the pipeline config. Each ring's axial z is
# derived straight from its own RingNumber (real InsertPos) below — not from
# positions_in_tray_new_wished/positions_in_tray_occupied, which only describe what
# THIS run of the optimizer was asked to place, not what the CSV on disk actually holds.
field_colormap      = :plasma
# Shim magnet strength: pipeline_config.jl's magnet_moment_Am2, derived from
# config.toml's magnet_Br_T (remanence) + magnet_side_mm (cube side length) —
# passed straight to build_dipoles_device as m_mag (matches setup.jl exactly).
shim_moment_Am2     = magnet_moment_Am2
magnet_axis         = :x           # x–y plane convention, matches the optimizer

############ Load shim-magnet placements (only if a shim solution exists) ############
# FIELD-ONLY mode: with no shim CSV yet (e.g. right after Stage 0, before optimizing),
# the viewer shows just the MEASURED field map — no shim, no GPU — so you can inspect
# the field before shimming. Once a shim CSV exists it adds Shim / Shimmed as before.
has_shim = isfile(shim_csv_path)
if has_shim
    csv_path = shim_csv_path        # Stage 1.5 output (the seam)
    df = CSV.read(csv_path, DataFrame)
    x         = Float32.(df[:, 1])
    y         = Float32.(df[:, 2])
    ring      = Int.(df[:, 3])
    angle_deg = Float32.(df[:, 4])

    unique_rings = sort(unique(ring))
    # RingNumber is the REAL, physical InsertPos (signed tray slot) — so each ring's
    # z comes from calling ringpos_from_tray_mm on THAT ring value directly, not from
    # zipping unique_rings against wished_trays positionally (wished_trays' array
    # order need not match unique_rings' ascending sort, and a run can also drop/add
    # rings vs. config, e.g. an OSII import) — same config-driven tray geometry the
    # optimizer used (was hardcoded to half_shift_mm = 0.0, so a non-zero shift put
    # these rings at the wrong z).
    ring_z_mm    = ringpos_from_tray_mm(unique_rings;
                                        tray_slot_spacing_mm = tray_slot_spacing_mm,
                                        front_tray_shift_mm  = front_tray_shift_mm,
                                        back_tray_shift_mm   = back_tray_shift_mm)
    ring_to_z = Dict(r => Float32(z) for (r, z) in zip(unique_rings, ring_z_mm))
    z_all     = Float32[ring_to_z[r] for r in ring]
    println("Loaded $(nrow(df)) magnets across $(length(unique_rings)) ring(s); Ring -> Z (mm): " *
            join(["$(r)=>$(ring_to_z[r])" for r in unique_rings], ", "))
else
    println("No shim CSV at $(shim_csv_path)  →  FIELD-ONLY mode (measured field map, no shimming).")
end

############ Load measured field ############
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
end
xg32, yg32, zg32 = Float32.(xg), Float32.(yg), Float32.(zg)
nx, ny, nz = size(By_grid)
@assert (nx, ny, nz) == (length(xg), length(yg), length(zg)) "By_grid dims $(size(By_grid)) don't match axes lengths"
measured_full = field_sign .* Float32.(By_grid)
println("Loaded field grid $(nx)×$(ny)×$(nz);  display frame: $(frame_note)")

# How far the field map can be trusted (written by Stage 0). :reshape saves Inf
# (the whole grid is literally measured); :spherical_harmonics saves the
# measured shell's radius (mm) — beyond it the model is extrapolating with NO
# supporting data. Missing key (older jld2) -> Inf, i.e. nothing gets masked.
field_valid_radius_mm = Float32(let r = Inf
    jldopen(field_path, "r") do f
        haskey(f, "field_valid_radius_mm") && (r = f["field_valid_radius_mm"])
    end
    r
end)

############ Shim field on the same grid (GPU dipoles) ############
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

if has_shim
    posiciones   = [(Float64(x[i]), Float64(y[i]), Float64(z_all[i])) for i in eachindex(x)]
    θdeg         = Float64.(angle_deg)
    shim_full    = field_sign .* magnet_By_on_axes(posiciones, θdeg, xg, yg, zg;
                                     m_mag = shim_moment_Am2, axis = magnet_axis, to_mT = true)
    shimmed_full = measured_full .+ shim_full
end

# Display frame. Everything above (dipole field included) is computed in the OPTIMIZER frame
# (B0 -> +y); for "scan" re-express the field arrays + axes in the scan's own frame. This is
# display-only: field values are unchanged (apart from the sign set above).
if use_scan && field_dir != "+y"
    global measured_full, xg_s, yg_s = grid_to_scan(measured_full, xg, yg, field_dir)
    if has_shim
        global shim_full    = grid_to_scan(shim_full,    xg, yg, field_dir)[1]
        global shimmed_full = grid_to_scan(shimmed_full, xg, yg, field_dir)[1]
    end
    global xg32, yg32 = Float32.(xg_s), Float32.(yg_s)
    global nx, ny = size(measured_full, 1), size(measured_full, 2)
end

# Blank out (NaN) anything beyond the trusted radius — viewer-local copies
# only, this never touches the jld2 or anything setup.jl/the optimizer reads.
# GLMakie's heatmap!/lines! both skip NaN by default, so this just leaves
# those cells/line segments empty instead of showing the SH fit's unsupported
# extrapolation past the measured shell.
if isfinite(field_valid_radius_mm)
    rgrid32 = sqrt.(reshape(xg32, :, 1, 1) .^ 2 .+ reshape(yg32, 1, :, 1) .^ 2 .+ reshape(zg32, 1, 1, :) .^ 2)
    outside = rgrid32 .> field_valid_radius_mm
    measured_full[outside] .= NaN32
    if has_shim
        shim_full[outside]    .= NaN32
        shimmed_full[outside] .= NaN32
    end
    println("  field map trusted only to r ≤ $(round(field_valid_radius_mm, digits=1)) mm — " *
            "$(round(100*count(outside)/length(outside), digits=1))% of the grid hidden as extrapolation")
end

# NaN-safe min/max (some cells may now be NaN — see masking above).
nan_extrema(A) = (v = Iterators.filter(!isnan, A); isempty(v) ? (0f0, 0f0) : extrema(v))
nan_minimum(A) = nan_extrema(A)[1]
nan_maximum(A) = nan_extrema(A)[2]

# ppm = (max-min)/mean * 1e6 over whatever's actually visible (NaN-safe), so it
# reflects the sphere mask below when it's on.
function ppm_nan(A)
    v = collect(Iterators.filter(!isnan, A))
    isempty(v) && return NaN32
    lo, hi = extrema(v)
    m = mean(v)
    (m == 0 || !isfinite(m)) && return NaN32
    return Float32((hi - lo) / abs(m) * 1e6)
end

nan_mean(A) = (v = Iterators.filter(!isnan, A); isempty(v) ? NaN32 : Float32(mean(collect(v))))

if has_shim
    field_names = ["Measured", "Shim", "Shimmed"]
    fields = Dict("Measured" => measured_full, "Shim" => shim_full, "Shimmed" => shimmed_full)
else
    field_names = ["Measured"]
    fields = Dict("Measured" => measured_full)
end
default_field = has_shim ? "Shimmed" : "Measured"   # selected on open + colour-scale seed
for k in field_names
    lo, hi = nan_extrema(fields[k])
    println("$(rpad(k,9)) By range [$(round(lo,digits=3)), $(round(hi,digits=3))] mT")
end

############ Plane / slice helpers ############
# Per plane: in-plane horizontal (H) and vertical (V) axes + labels + letters.
plane_axes(p)    = p == "XY" ? (xg32, yg32, "X (mm)", "Y (mm)") :
                   p == "XZ" ? (xg32, zg32, "X (mm)", "Z (mm)") :
                               (yg32, zg32, "Y (mm)", "Z (mm)")
plane_letters(p) = p == "XY" ? ("X", "Y") : p == "XZ" ? ("X", "Z") : ("Y", "Z")
perp_axis(p)     = p == "XY" ? (zg32, "Z") : p == "XZ" ? (yg32, "Y") : (xg32, "X")

# 2D image (H, V) of field F on plane p at perpendicular index si.
plane_image(F, p, si) = p == "XY" ? F[:, :, si] :
                        p == "XZ" ? F[:, si, :] :
                                    F[si, :, :]

nearest_index(vec, val) = argmin(abs.(vec .- Float32(val)))
safe_range(lo, hi) = lo == hi ? (lo - 1f0, hi + 1f0) : (lo, hi)
mid(v) = v[cld(length(v), 2)]

############ Sphere mask (optional, radius-adjustable) ############
# Independent of field_valid_radius_mm above (which permanently hides
# unsupported SH extrapolation): this is a live, togglable crop to a sphere
# r ≤ mask radius, centered at the origin, so the cube's corners (largest r,
# usually the most inhomogeneous / least trustworthy region for a shimming
# check) can be excluded from BOTH the picture and the ppm readouts on demand.
max_r = sqrt(maximum(abs.(xg32))^2 + maximum(abs.(yg32))^2 + maximum(abs.(zg32))^2)
default_mask_r = isfinite(field_valid_radius_mm) ? min(field_valid_radius_mm, max_r) : min(Float32(Rmax), max_r)
mask_on = Observable(false)

############ Observables ############
crange_obs     = Observable(safe_range(nan_extrema(fields[default_field])...))
line_meas_obs  = Observable(Point2f[])
line_shimd_obs = Observable(Point2f[])
mean_meas_obs  = Observable(Point2f[])   # horizontal reference line @ visible mean
mean_shimd_obs = Observable(Point2f[])
slice_title    = Observable("Slice")
line_title     = Observable("Line")

############ Figure ############
fig = Figure(size = (1700, 1000))
Label(fig[0, 1:3], (has_shim ? "Field Slice & Line Viewer" :
      "Field Map Viewer — measured field, pre-shim") * "   —   " * frame_note, fontsize = 22)

# --- top selectors (above each panel) ---
plane_box = GridLayout(fig[1, 1]; tellwidth = false, halign = :center)
Label(plane_box[1, 1], "Plane:", halign = :right)
plane_menu = Menu(plane_box[1, 2], options = ["XY", "XZ", "YZ"], default = "XY", width = 90)

vary_box = GridLayout(fig[1, 3]; tellwidth = false, halign = :center)
Label(vary_box[1, 1], "Profile axis:", halign = :right)
vary_menu = Menu(vary_box[1, 2], options = collect(plane_letters("XY")), default = "X", width = 90)

# --- panels ---
ax_img  = Axis(fig[2, 1]; aspect = DataAspect())
Colorbar(fig[2, 2]; limits = crange_obs, colormap = field_colormap, label = field_label, width = 22)
ax_line = Axis(fig[2, 3]; ylabel = field_label)

lines!(ax_line, line_meas_obs; color = :dodgerblue, linewidth = 2, label = "Measured")
has_shim && lines!(ax_line, line_shimd_obs; color = :seagreen, linewidth = 2, label = "Shimmed")
has_shim && axislegend(ax_line; position = :rt)

# Mean reference lines (same colour as their series, dashed) — flat lines make
# it easy to see how much the shim pulls the profile in TOWARD its own mean.
# Not added to the legend (unlabeled, added after axislegend()).
lines!(ax_line, mean_meas_obs; color = :dodgerblue, linewidth = 1.5, linestyle = :dash)
has_shim && lines!(ax_line, mean_shimd_obs; color = :seagreen, linewidth = 1.5, linestyle = :dash)

# --- bottom controls: sliders (title on top, real units) + centered field menu ---
bottom = GridLayout(fig[3, 1:3])

Label(bottom[1, 1], slice_title; fontsize = 16)
slice_slider = Slider(bottom[2, 1], range = zg32, startvalue = mid(zg32))
Label(bottom[1, 3], line_title; fontsize = 16)
line_slider  = Slider(bottom[2, 3], range = yg32, startvalue = mid(yg32))

field_box = GridLayout(bottom[3, 1:3]; tellwidth = false, halign = :center)
Label(field_box[1, 1], "Field:", halign = :right)
field_menu = Menu(field_box[1, 2], options = field_names, default = default_field, width = 130)

mask_box = GridLayout(bottom[4, 1:3]; tellwidth = false, halign = :center)
mask_btn    = Button(mask_box[1, 1], label = "Sphere mask: OFF", tellwidth = false)
Label(mask_box[1, 2], "Radius:", halign = :right, tellwidth = false)
mask_slider = Slider(mask_box[1, 3], range = range(0f0, max_r; length = 201), startvalue = default_mask_r, width = 220)
Label(mask_box[1, 4], @lift(string(round($(mask_slider.value), digits = 1), " mm")); tellwidth = false)

############ Reactive update ############
updating   = Ref(false)
refreshing = Ref(false)   # true while refresh! runs, so the zoom listener stays quiet

# Current slice image + its axes + base title, stashed by refresh! so the slice ppm
# can be recomputed over just the heatmap's VISIBLE region (updates on zoom, too).
cur_img       = Ref{Matrix{Float32}}()
cur_Hvec      = Ref{Vector{Float32}}()
cur_Vvec      = Ref{Vector{Float32}}()
cur_titlebase = Ref{String}("")
# Profile stash (full, sphere-masked) so it can be clipped to the heatmap's visible region.
cur_xcoord    = Ref{Vector{Float32}}()
cur_meas      = Ref{Vector{Float32}}()
cur_shimd     = Ref{Vector{Float32}}()
cur_free_is_H = Ref{Bool}(true)
cur_fixedval  = Ref{Float32}(0f0)
cur_profbase  = Ref{String}("")

# ppm over the pixels inside the heatmap's current visible limits (NaN-safe, and
# honouring the sphere mask already baked into the stashed image). Sets the slice title.
function update_slice_ppm!()
    isassigned(cur_img) || return
    img, H, V = cur_img[], cur_Hvec[], cur_Vvec[]
    lims = ax_img.finallimits[]
    xlo, ylo = lims.origin[1], lims.origin[2]
    xhi, yhi = xlo + lims.widths[1], ylo + lims.widths[2]
    hsel = (H .>= xlo) .& (H .<= xhi)
    vsel = (V .>= ylo) .& (V .<= yhi)
    zoomed = count(hsel) < length(H) || count(vsel) < length(V)
    sub = (any(hsel) && any(vsel)) ? img[hsel, vsel] : Float32[]
    ax_img.title[] = cur_titlebase[] *
        "\nppm = $(round(ppm_nan(sub), digits = 1))" * (zoomed ? "   (visible region)" : "")
    return
end

# Right-panel profile clipped to the heatmap's VISIBLE region: keep only the part of
# the line inside the current zoom (its span along the free axis, and only if the
# line's fixed coordinate is inside the zoom on the other axis), and report ppm over
# just that. Called from refresh! and whenever the heatmap is zoomed/panned.
function update_profile_region!()
    isassigned(cur_xcoord) || return
    X = cur_xcoord[]
    m = copy(cur_meas[]); s = copy(cur_shimd[])
    lims = ax_img.finallimits[]
    xlo, ylo = lims.origin[1], lims.origin[2]
    xhi, yhi = xlo + lims.widths[1], ylo + lims.widths[2]
    freelo, freehi = cur_free_is_H[] ? (xlo, xhi) : (ylo, yhi)   # heatmap span on the profile axis
    fixlo, fixhi   = cur_free_is_H[] ? (ylo, yhi) : (xlo, xhi)   # …and on the line's fixed axis
    fixed_in = fixlo <= cur_fixedval[] <= fixhi
    insel  = fixed_in ? ((X .>= freelo) .& (X .<= freehi)) : falses(length(X))
    zoomed = count(insel) < length(X)

    m[.!insel] .= NaN32; s[.!insel] .= NaN32          # keep only the in-region segment
    line_meas_obs[]  = Point2f.(X, m)
    line_shimd_obs[] = Point2f.(X, s)

    mm = nan_mean(m); sm = nan_mean(s)
    xin = any(insel) ? X[insel] : X
    x0, x1 = first(xin), last(xin)
    mean_meas_obs[]  = isnan(mm) ? Point2f[] : [Point2f(x0, mm), Point2f(x1, mm)]
    mean_shimd_obs[] = isnan(sm) ? Point2f[] : [Point2f(x0, sm), Point2f(x1, sm)]

    ppmtxt = has_shim ?
        "ppm — Measured: $(round(ppm_nan(m), digits=1))  Shimmed: $(round(ppm_nan(s), digits=1))" :
        "ppm = $(round(ppm_nan(m), digits=1))"
    ax_line.title[] = cur_profbase[] * "\n" * ppmtxt * (zoomed ? "   (visible region)" : "")

    # x-axis follows the heatmap's visible span when zoomed, else the whole line.
    (zoomed && any(insel)) ? xlims!(ax_line, freelo, freehi) : xlims!(ax_line, first(X), last(X))
    # y-axis sized to the visible profile so shim detail stays readable.
    ylo2, yhi2 = safe_range(nan_extrema(vcat(m, s))...)
    ypad = 0.03f0 * (yhi2 - ylo2)
    ylims!(ax_line, ylo2 - ypad, yhi2 + ypad)
    return
end

function refresh!(; reset_view::Bool = false)
    f = field_menu.selection[]; p = plane_menu.selection[]; freelet = vary_menu.selection[]
    (f === nothing || p === nothing || freelet === nothing) && return
    refreshing[] = true

    Hvec, Vvec, Hlab, Vlab = plane_axes(p)
    pvec, plab = perp_axis(p)
    Hlet, Vlet = plane_letters(p)
    si = nearest_index(pvec, slice_slider.value[])

    # ----- left heatmap (recreated so plane changes can't size-race) -----
    img = collect(plane_image(fields[f], p, si))

    # Sphere mask: r here is the TRUE 3D radius at each pixel — H and V are two
    # of (x,y,z) and the perpendicular coordinate pvec[si] is the third, so
    # r = sqrt(H^2 + V^2 + pperp^2) regardless of which plane is selected.
    Rm = Float32(mask_slider.value[])
    if mask_on[]
        rimg = sqrt.(reshape(Hvec, :, 1) .^ 2 .+ reshape(Vvec, 1, :) .^ 2 .+ pvec[si]^2)
        img[rimg .> Rm] .= NaN32
    end
    mask_note = mask_on[] ? "  [sphere r ≤ $(round(Rm, digits=1)) mm]" : ""
    # Stash this slice + its base title so the ppm readout can follow the visible region.
    cur_img[] = img; cur_Hvec[] = Hvec; cur_Vvec[] = Vvec
    cur_titlebase[] = "$f — $p plane @ $plab = $(round(pvec[si], digits = 1)) mm$(mask_note)"

    # Colour scale: while masked, rescale to what's actually visible in THIS
    # slice (so the corner/outlier range that's now hidden stops washing out
    # the contrast of the region you asked to see); unmasked keeps the
    # original whole-volume scale so slices stay comparable to each other.
    crange = mask_on[] ? safe_range(nan_extrema(img)...) : safe_range(nan_extrema(fields[f])...)
    crange_obs[] = crange
    # Remember the current heatmap view (the user's zoom/region) before we recreate
    # the plot; empty!+heatmap! would otherwise auto-refit to the whole slice.
    prev_lims = reset_view ? nothing : ax_img.finallimits[]
    empty!(ax_img)
    heatmap!(ax_img, Hvec, Vvec, img; colormap = field_colormap, colorrange = crange)

    # ----- line selection + guide -----
    if freelet == Hlet                       # profile runs along H; fix V
        vi = nearest_index(Vvec, line_slider.value[]); v = Vvec[vi]
        guide  = [Point2f(first(Hvec), v), Point2f(last(Hvec), v)]
        xcoord, xlab, freelabel = Hvec, Hlab, Hlet
        fixedlet, fixedval = Vlet, v
        getprof = F -> collect(plane_image(F, p, si))[:, vi]
    else                                     # profile runs along V; fix H
        hi = nearest_index(Hvec, line_slider.value[]); h = Hvec[hi]
        guide  = [Point2f(h, first(Vvec)), Point2f(h, last(Vvec))]
        xcoord, xlab, freelabel = Vvec, Vlab, Vlet
        fixedlet, fixedval = Hlet, h
        getprof = F -> collect(plane_image(F, p, si))[hi, :]
    end
    lines!(ax_img, guide; color = :white, linewidth = 2, linestyle = :dash)

    # Keep the user's zoomed region across line/slice/field/mask changes; only a
    # PLANE change (reset_view) refits to the full slice (its coordinates changed).
    if reset_view
        autolimits!(ax_img)
    elseif prev_lims !== nothing
        o, w = prev_lims.origin, prev_lims.widths
        limits!(ax_img, o[1], o[1] + w[1], o[2], o[2] + w[2])
    end

    ax_img.xlabel = Hlab
    ax_img.ylabel = Vlab
    update_slice_ppm!()   # ppm over the currently VISIBLE region (recomputed on zoom, too)

    # ----- right profiles (all three fields) -----
    meas_line  = getprof(measured_full)
    shimd_line = has_shim ? getprof(shimmed_full) : fill(NaN32, length(xcoord))
    if mask_on[]
        rprof = sqrt.(xcoord .^ 2 .+ Float32(fixedval)^2 .+ pvec[si]^2)
        outp  = rprof .> Rm
        meas_line[outp]  .= NaN32
        shimd_line[outp] .= NaN32
    end
    # Stash the full (sphere-masked) profile so it can be clipped to the heatmap's
    # visible region — both here and whenever the heatmap is zoomed/panned.
    ax_line.xlabel = xlab
    cur_xcoord[] = xcoord; cur_meas[] = meas_line; cur_shimd[] = shimd_line
    cur_free_is_H[] = (freelet == Hlet); cur_fixedval[] = Float32(fixedval)
    cur_profbase[] = "Profile @ $plab = $(round(pvec[si], digits=1)), $fixedlet = $(round(fixedval, digits=1)) (along $freelabel)$(mask_note)"
    update_profile_region!()

    # ----- live slider titles -----
    slice_title[] = "Slice:  $plab = $(round(pvec[si], digits=1)) mm"
    line_title[]  = "Line:  $fixedlet = $(round(fixedval, digits=1)) mm"
    refreshing[] = false
    return
end

# Reset slider ranges/menus when the plane (or profile axis) changes structure.
function on_structure_change!(; reset_view::Bool = false)
    updating[] = true
    p = plane_menu.selection[]
    Hvec, Vvec, _, _ = plane_axes(p)
    pvec, _ = perp_axis(p)
    Hlet, Vlet = plane_letters(p)
    freelet = vary_menu.selection[]
    fixedvec = (freelet == Hlet) ? Vvec : Hvec   # the line slider rides the FIXED axis

    slice_slider.range[] = pvec
    line_slider.range[]  = fixedvec
    set_close_to!(slice_slider, mid(pvec))
    set_close_to!(line_slider,  mid(fixedvec))
    updating[] = false
    refresh!(reset_view = reset_view)
    return
end

on(plane_menu.selection) do p
    updating[] && return
    updating[] = true
    vary_menu.options[]    = collect(plane_letters(p))   # show the two in-plane axes
    vary_menu.i_selected[] = 1                            # default: profile along horizontal axis
    updating[] = false
    on_structure_change!(reset_view = true)               # plane changed → coords changed → refit
end

on(vary_menu.selection) do _
    updating[] && return
    on_structure_change!()
end

on(mask_btn.clicks) do _
    new_state = !mask_on[]
    mask_on[] = new_state
    mask_btn.label[] = new_state ? "Sphere mask: ON" : "Sphere mask: OFF"
    refresh!()
end

for obs in (field_menu.selection, slice_slider.value, line_slider.value, mask_slider.value)
    on(_ -> (updating[] || refresh!()), obs)
end

# Zooming/panning the heatmap doesn't fire refresh!, so recompute the visible-region
# slice ppm AND re-clip the profile directly whenever the view limits change.
on(ax_img.finallimits) do _
    refreshing[] && return          # refresh! already updates these explicitly
    update_slice_ppm!()
    update_profile_region!()
end

############ GIF export: sweep the slice slider across its range, record each frame ###
# Drives slice_slider programmatically through its current plane's full perpendicular
# range (e.g. Z for the XY plane), calling refresh! at each step so the heatmap +
# profile + ppm all update exactly as they would if you dragged the slider yourself.
# viewer_gif_frames/viewer_gif_fps come from config.toml (pipeline_config.jl).
function save_slice_sweep_gif(path::AbstractString; field_name::Union{Nothing,String} = nothing)
    sel0 = field_menu.selection[]
    if field_name !== nothing
        field_menu.i_selected[] = findfirst(==(field_name), field_names)
    end
    p = plane_menu.selection[]
    pvec, _ = perp_axis(p)
    lo, hi = Float32(first(pvec)), Float32(last(pvec))
    v0 = slice_slider.value[]
    Makie.record(fig, path, 1:viewer_gif_frames; framerate = viewer_gif_fps) do i
        t = viewer_gif_frames <= 1 ? 0f0 : Float32(i - 1) / (viewer_gif_frames - 1)
        set_close_to!(slice_slider, lo + t * (hi - lo))   # triggers refresh! via its own listener
    end
    set_close_to!(slice_slider, v0)
    field_name !== nothing && (field_menu.i_selected[] = findfirst(==(sel0), field_names))
    refresh!()
    return path
end

gif_row    = GridLayout(fig[4, 1:3])
gif_status = Observable("")

gif_btn = Button(gif_row[1, 1], label = "Save GIF: sweep slice (current field)", tellwidth = false)
on(gif_btn.clicks) do _
    gif_status[] = "Recording…"
    stamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    fname = field_menu.selection[]
    path  = joinpath(viewer_gif_dir, "slice_sweep_$(fname)_$(stamp).gif")
    save_slice_sweep_gif(path)
    gif_status[] = "Saved → " * path
end

if has_shim
    compare_btn = Button(gif_row[1, 2], label = "Save GIF pair: Measured vs Shimmed sweep", tellwidth = false)
    on(compare_btn.clicks) do _
        stamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
        paths = String[]
        for nm in ("Measured", "Shimmed")
            gif_status[] = "Recording $(nm)…"
            p = joinpath(viewer_gif_dir, "slice_sweep_$(nm)_$(stamp).gif")
            save_slice_sweep_gif(p; field_name = nm)
            push!(paths, p)
        end
        gif_status[] = "Saved 2 GIFs → " * dirname(paths[1])
    end
end

Label(gif_row[1, has_shim ? 3 : 2], gif_status; halign = :left, tellwidth = false, fontsize = 13)

############ Layout sizing ############
colsize!(fig.layout, 1, Relative(0.45))   # heatmap
colsize!(fig.layout, 2, Relative(0.04))   # colorbar
colsize!(fig.layout, 3, Relative(0.45))   # line plot
rowsize!(fig.layout, 2, Relative(0.72))   # panels dominate the height

on_structure_change!(reset_view = true)   # initialise ranges + first draw

############ Show ############
screen = display(fig)
println("Viewer open. Plane (top-left) / Profile axis (top-right); sliders set slice & line; Field menu bottom-center. Close to exit.")
wait(screen)

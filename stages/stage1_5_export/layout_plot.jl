# layout_plot.jl  —  STAGE 1.5  (figure: the shim layout, one panel per ring)
#
# Run on its own:        julia stages/stage1_5_export/layout_plot.jl
# Or via the GUI/pipeline: stage "layout" — runs automatically after the shim CSV is exported
#                          (GUI "Run optimization" = optimize → export → this figure).
#
# Reads the current iteration's shim CSV (the seam: X (mm), Y (mm), RingNumber, Angle (deg))
# and draws, for every ring (real InsertPos), a polar quiver panel: one arrow per magnet at
# its (X, Y), pointing along the magnet's in-plane moment angle (0° = +X, CCW) and coloured
# by that angle on a cyclic colormap (0° and 360° share a colour). The B0 axis is drawn as
# a black arrow, in whichever frame the CSV is written in (shim_csv_frame): the scan's own
# B0 direction for "scan", +y for "optimizer".
#
# Works for any layout that ends up in the seam CSV — optimizer export, ring search, insert
# search (sparse: only the placed magnets are drawn) or an OSII/hand-made import. The ring's
# axial z comes from its InsertPos via ringpos_from_tray_mm and the current tray geometry.
#
# Environment: stages/stage1_5_export/layout_env/ (instantiated automatically on first use).
#
#   in :  shim_csv_path
#   out:  Optimizer_Output_per_Iteration/<ITER>/Layout/<ITER>_optimizer_layout.{png,pdf}

# Own environment (CairoMakie, CSV, DataFrames, JLD2): the main project's pinned CUDA/GLMakie stack is left alone.
# The environment must be selected when julia STARTS (--project): activating it from inside a running session
# (after Pkg and other stdlibs are loaded) makes julia recompile Makie on every run (~3 min). So when launched
# normally, this script just re-launches itself in a fresh julia with --project set, and exits with its status.
const LAYOUT_ENV = joinpath(@__DIR__, "layout_env")
if get(ENV, "BOOST_LAYOUT_CHILD", "") != "1"
    cmd = addenv(`$(Base.julia_cmd()) --project=$(LAYOUT_ENV) $(@__FILE__)`, "BOOST_LAYOUT_CHILD" => "1")
    isfile(joinpath(LAYOUT_ENV, "Manifest.toml")) ||
        error("layout_plot.jl: its environment is not set up yet — run  julia install_deps.jl  once (it instantiates $(LAYOUT_ENV)).")
    run(cmd)
    exit(0)                       # run() throws (→ non-zero exit) if the child failed
end

include(joinpath(@__DIR__, "..", "..", "pipeline_config.jl"))
include(joinpath(@__DIR__, "..", "..", "core", "utils", "pos_trays.jl"))      # ringpos_from_tray_mm
using CSV, DataFrames, JLD2, Printf
include(joinpath(@__DIR__, "..", "..", "core", "utils", "placed_shims.jl"))   # read_placed_magnets
include(joinpath(@__DIR__, "..", "..", "core", "utils", "viewer_frame.jl"))   # read_field_direction (needs JLD2)
using CairoMakie

const ARROW_LEN_MM = 25.0          # drawn length of one magnet arrow (neighbours are ~14.5 mm apart)
const CMAP         = :twilight     # cyclic: 0° and 360° have the same colour

isfile(shim_csv_path) || error("layout_plot.jl: shim CSV not found:\n    $(shim_csv_path)\n" *
                               "Run \"Run optimization\" (export) or the OSII import first.")
df = CSV.read(shim_csv_path, DataFrame)
for c in ("X (mm)", "Y (mm)", "RingNumber", "Angle (deg)")
    c in names(df) || error("layout_plot.jl: column '$(c)' missing in $(basename(shim_csv_path)); found $(names(df))")
end
x    = Float64.(df[!, "X (mm)"])
y    = Float64.(df[!, "Y (mm)"])
ring = round.(Int, Float64.(df[!, "RingNumber"]))
ang  = Float64.(df[!, "Angle (deg)"])
# Shims ALREADY placed in the magnet (config placed_shim_csvs): drawn in grey, in the same frame as the shim CSV
# (they are declared in shim_csv_frame too). Display only — they are in the scan, never added to any field.
PL = read_placed_magnets(placed_shim_paths; tray_slot_spacing_mm = tray_slot_spacing_mm,
                         front_tray_shift_mm = front_tray_shift_mm, back_tray_shift_mm = back_tray_shift_mm)
has_placed = !isempty(PL.x)
rings = sort(unique(vcat(ring, PL.ring)))          # panels: every ring that holds new OR placed magnets
n     = length(rings)
R     = sum(hypot.(vcat(x, PL.x), vcat(y, PL.y))) / (length(x) + length(PL.x))
zring = ringpos_from_tray_mm(rings; tray_slot_spacing_mm = tray_slot_spacing_mm,
                             front_tray_shift_mm = front_tray_shift_mm,
                             back_tray_shift_mm  = back_tray_shift_mm)

# --- B0 direction, in the frame the CSV is written in --------------------------
b0 = if shim_csv_frame == "optimizer"
    (0.0, 1.0)                                              # the optimizer frame has B0 → +y by construction
else
    sdir = read_field_direction(shell_fieldmap_path, fieldmap_path)
    sdir === nothing && main_field_direction != "auto" && (sdir = main_field_direction)
    sdir === nothing ? nothing :
        get(Dict("+x" => (1.0, 0.0), "-x" => (-1.0, 0.0), "+y" => (0.0, 1.0), "-y" => (0.0, -1.0)), sdir, nothing)
end
b0 === nothing && @warn "layout_plot.jl: B0 direction unknown (no field_direction in the field map and main_field_direction = \"auto\") — B0 arrow omitted."

# --- grid of panels ----------------------------------------------------------
ncol = n <= 2 ? n : n == 4 ? 2 : n <= 6 ? 3 : n <= 12 ? 4 : 5
nrow = cld(n, ncol)
fig  = Figure(size = (ncol * 310 + 110, nrow * 310 + 30 + (has_placed ? 40 : 0)), fontsize = 13)

# arrow glyph (points along +x at rotation 0), scaled by markersize in data units
arrowshape = CairoMakie.Makie.GeometryBasics.Polygon(Point2f[(-0.5, -0.11), (0.14, -0.11), (0.14, -0.30), (0.5, 0.0),
                             (0.14, 0.30), (0.14, 0.11), (-0.5, 0.11)])
th_circle = range(0, 2π; length = 181)

for (i, r) in enumerate(rings)
    row, col = fld1(i, ncol), mod1(i, ncol)
    ax = Axis(fig[row, col]; aspect = DataAspect(),
              title = @sprintf("Ring %+d   (z = %+.1f mm)", r, zring[i]), titlesize = 14)
    hidedecorations!(ax); hidespines!(ax)
    limits!(ax, -1.12R, 1.12R, -1.12R, 1.12R)
    lines!(ax, R .* cos.(th_circle), R .* sin.(th_circle); color = (:gray, 0.7), linewidth = 1)
    if has_placed                                   # already-placed inserts: grey, drawn first (under the new ones)
        psel = findall(==(r), PL.ring)
        isempty(psel) || scatter!(ax, PL.x[psel], PL.y[psel]; marker = arrowshape, markersize = ARROW_LEN_MM,
                                  markerspace = :data, rotation = deg2rad.(PL.ang[psel]), color = :gray80,
                                  strokewidth = 0.5, strokecolor = :gray35)
    end
    sel = findall(==(r), ring)
    isempty(sel) || scatter!(ax, x[sel], y[sel]; marker = arrowshape, markersize = ARROW_LEN_MM, markerspace = :data,
             rotation = deg2rad.(ang[sel]), color = mod.(ang[sel], 360.0), colormap = CMAP,
             colorrange = (0, 360), strokewidth = 0.4, strokecolor = :black)
    if b0 !== nothing
        L = 55.0
        lines!(ax, [-b0[1] * L, b0[1] * (L - 6)], [-b0[2] * L, b0[2] * (L - 6)]; color = :black, linewidth = 2)
        scatter!(ax, [b0[1] * L], [b0[2] * L]; marker = :utriangle, markersize = 12, color = :black,
                 rotation = atan(b0[2], b0[1]) - π / 2)
        text!(ax, b0[1] * 0.25 * L - b0[2] * 16, b0[2] * 0.25 * L + b0[1] * 16;
              text = rich("B", subscript("0")), fontsize = 14, align = (:center, :center))
    end
end

Colorbar(fig[1:nrow, ncol + 1]; colormap = CMAP, limits = (0, 360), ticks = 0:90:360,
         label = "magnet angle (deg)", height = Relative(0.8))
has_placed && Legend(fig[nrow + 1, 1:ncol],
    [PolyElement(color = :gray80, strokecolor = :gray35, strokewidth = 1),
     PolyElement(color = cgrad(CMAP)[0.3], strokecolor = :black, strokewidth = 0.5)],
    ["already placed (already in the scan)", "new insert (colour = angle)"];
    orientation = :horizontal, framevisible = false, tellheight = true, tellwidth = false)

mkpath(layout_dir)
save(layout_png_path, fig; px_per_unit = 3)
save(layout_pdf_path, fig)
println("Layout figure: ", n, " ring(s) ", rings, ", ", length(x), " new magnets",
        has_placed ? " + $(length(PL.x)) already-placed (grey, from $(join(unique(PL.file), ", ")))" : "",
        ", mean radius ", round(R, digits = 1),
        " mm, B0 ", b0 === nothing ? "unknown" : (b0 == (0.0, -1.0) ? "−y" : b0 == (0.0, 1.0) ? "+y" :
                                                    b0 == (1.0, 0.0) ? "+x" : "−x"),
        " (csv frame = ", shim_csv_frame, ")")
println("  → ", layout_png_path)
println("  → ", layout_pdf_path)

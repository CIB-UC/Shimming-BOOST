# Field_data_file_adapter.jl  —  STAGE 0  (measured CSV → interpolated jld2)
#
# Run on its own:        julia Field_data_file_adapter.jl
# Or via the pipeline:   run_pipeline.jl launches this first.
#
# Reads the measured field CSV (a regular parallelepiped scan, in Gauss) and
# writes the gridded field map the optimizer expects:
#     By_grid  (mT)  on axes  xg, yg, zg  (mm),  with By_grid[i,j,k] = (xg[i],yg[j],zg[k])
#
# Paths come from pipeline_config.jl:
#   in :  measured_field_path        (Measured_Field_Data_csv/<name>.csv)
#   out:  fieldmap_path              (Interpolated_Field_Data_jld2/<same name>.jld2)
#
# Measured CSV layout (this scanner export):
#   line 1            : "Fecha y Hora,<timestamp>"   (metadata, skipped)
#   line 2            : header  ->  X, Y, Z, Gauss, T°
#   line 3 onward     : data rows in mm + Gauss; each z-slice block is preceded
#                       by a spurious (0,0,0) marker row that must be dropped.

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))
@isdefined(ITERATION) || include(joinpath(@__DIR__, "..", "..", "pipeline_config.jl"))

using CSV, DataFrames, JLD2

const GAUSS_TO_MT = 0.1          # 1 Gauss = 0.1 mT
const MACHINE_UNIT_MM = 0.4      # CSV X/Y/Z are machine steps; 1 step = 0.4 mm

# Sign convention: the main field B0 points along `main_field_direction` (the LAB
# frame, set in pipeline_config.jl). It is stored POSITIVE along that axis,
# regardless of the gaussmeter's probe polarity (|Gauss|). Valid because B0 never
# crosses zero inside the bore.
const VALID_DIRS = ("+x", "-x", "+y", "-y")
# "auto" is only possible for scans that record the field VECTOR (the Controlled
# spherical format read by the SH/shell adapters). This reshape adapter takes a
# parallelepiped grid with a single scalar column, so it needs an explicit axis.
main_field_direction == "auto" &&
    error("main_field_direction = \"auto\" is not supported by the reshape adapter (no field-vector data). Set an explicit $(VALID_DIRS), or use the SH/shell adapter with a Controlled scan.")
@assert main_field_direction in VALID_DIRS "main_field_direction = \"$(main_field_direction)\" is invalid; Halbach field lies in the x–y plane, use one of $(VALID_DIRS)."

# Lab frame  →  optimizer frame.
# BOOST always optimizes a +y field. A magnet whose field points elsewhere in the
# x–y plane is handled by rotating the WHOLE measured field about the bore (z) so
# the field becomes +y. The field's inhomogeneity pattern rotates rigidly with
# it, so the shim trays end up on the correct side. Because the trays sit every
# 30° and the x–y grid is square, a 90°/180° rotation maps grid points exactly
# onto grid points (no interpolation). The result lives entirely in the optimizer
# frame: tray 12 is at +y = the field direction; align it to the real field axis
# at assembly. (The lab direction is still saved into the jld2 for the record.)
lab_to_optimizer_xy(x, y, dir) =
    dir == "+y" ? (x, y)  :   #   0°  : already +y
    dir == "+x" ? (-y, x) :   # +90° about z
    dir == "-x" ? (y, -x) :   # -90° about z
                  (-x, -y)    # "-y": 180° about z

# Header is on line 2 of the file; columns are X, Y, Z, Gauss, T°.
df = CSV.read(measured_field_path, DataFrame; header = 2)
rename!(df, :X => :x, :Y => :y, :Z => :z, :Gauss => :By)

# Convert machine-step coordinates → millimetres (everything downstream is mm).
df.x .*= MACHINE_UNIT_MM
df.y .*= MACHINE_UNIT_MM
df.z .*= MACHINE_UNIT_MM

# Drop the spurious (0,0,0) marker rows (z = 0 is not a real measured slice).
filter!(r -> !(r.x == 0.0 && r.y == 0.0 && r.z == 0.0), df)

# Rotate every (x, y) from the lab frame into the optimizer frame (field → +y).
rot  = [lab_to_optimizer_xy(r.x, r.y, main_field_direction) for r in eachrow(df)]
df.x = first.(rot)
df.y = last.(rot)

# Regular-grid axes (mm), ascending.
xg = sort(unique(df.x))
yg = sort(unique(df.y))
zg = sort(unique(df.z))
nx, ny, nz = length(xg), length(yg), length(zg)

# The scan must be a complete regular grid for a direct reshape.
@assert nrow(df) == nx * ny * nz "Measured grid is incomplete: $(nrow(df)) rows ≠ $(nx)×$(ny)×$(nz) = $(nx*ny*nz). Scattered data would need true interpolation here."

# Order so x varies fastest, then y, then z  → matches Julia column-major reshape,
# giving By_grid[i,j,k] at (xg[i], yg[j], zg[k]).
sort!(df, [:z, :y, :x])
By_grid = reshape(abs.(Float64.(df.By)) .* GAUSS_TO_MT, (nx, ny, nz))   # mT, positive along main_field_direction

# The whole grid is a direct reshape of real measured points (no interpolation
# happened), so every voxel is trustworthy -- Inf means "don't clip" to the
# viewers (Shimming_magnets_visualizer.jl / field_slice_viewer.jl), which look
# for this key to avoid showing unsupported extrapolation from the other
# Stage 0 adapter (:spherical_harmonics).
const field_valid_radius_mm = Inf

mkpath(dirname(fieldmap_path))
@save fieldmap_path By_grid xg yg zg main_field_direction field_valid_radius_mm

println("Stage 0  wrote ", fieldmap_path)
println("  main field B0 (lab) → ", main_field_direction,
        main_field_direction == "+y" ? "  (no rotation)" : "  → rotated to +y optimizer frame")
println("  grid ", nx, "×", ny, "×", nz,
        "   |B0| range [", round(minimum(By_grid), digits=3), ", ",
        round(maximum(By_grid), digits=3), "] mT")
println("  x ∈ [", minimum(xg), ", ", maximum(xg), "]  y ∈ [", minimum(yg), ", ",
        maximum(yg), "]  z ∈ [", minimum(zg), ", ", maximum(zg), "] mm")

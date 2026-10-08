### BOOST is a script that receives a fieldmap, in mT, and gives back a shim configuration.

# Guarded so setup.jl stays runnable on its own AND when an entry point
# (run_optim.jl, export_csv.jl) already loaded the config.
@isdefined(ITERATION) || include("pipeline_config.jl")

using DataFrames, StaticArrays, JLD2, Statistics, LinearAlgebra
using Evolutionary, Random, CUDA
using DelimitedFiles, MAT
using GLMakie

include("utils/grid_utils.jl")
include("utils/pos_trays.jl")



# Field map (mT) + axes (mm) from the interpolated jld2. Path comes from config.
@load fieldmap_path By_grid xg yg zg
fieldmap = By_grid

# Shell radius (Rmin/Rmax), occupied/wished tray rings now come from pipeline_config.jl.
title_1 = "posiciones_imanes_shimming"               # nombre de la figura

#Definimos el step de la grilla del fieldmap. Viene del jld2
dx = length(xg) > 1 ? minimum(abs.(diff(xg))) : 0.0
dy = length(yg) > 1 ? minimum(abs.(diff(yg))) : 0.0
dz = length(zg) > 1 ? minimum(abs.(diff(zg))) : 0.0
resmm = (dx, dy, dz)  

#cx, cy, cz = modelBy.center
cx, cy, cz = 0.0, 0.0, 0.0

# radios en cada voxel (CPU)
Rx = reshape(xg .- cx, :, 1, 1)
Ry = reshape(yg .- cy, 1, :, 1)
Rz = reshape(zg .- cz, 1, 1, :)
rgrid = sqrt.(Rx.^2 .+ Ry.^2 .+ Rz.^2)

# Definimos dmask
Δ  = max(dx, dy, dz) 
tol  = 1e-3 * Δ 
mask_shell_bool = (rgrid .>= (Rmin - tol)) .& (rgrid .<= (Rmax + tol))
dmask    = Float32.(mask_shell_bool)                

# resoluciones en metros
dx_m = Float32(dx * 1e-3)
dy_m = Float32(dy * 1e-3)
dz_m = Float32(dz * 1e-3)
dims = size(fieldmap)

posiciones = positions_from_rings_mm(positions_in_tray_new_wished;
    occupied_trays         = positions_in_tray_occupied,
    shim_radius_mm         = shim_radius_mm,
    mags_per_segment       = mags_per_segment,
    num_segments           = num_trays,
    angle_per_segment_deg  = angle_per_segment_deg,
    angular_offset_deg     = angular_offset_deg,
    tray_slot_spacing_mm   = tray_slot_spacing_mm,
    front_tray_shift_mm    = front_tray_shift_mm,
    back_tray_shift_mm     = back_tray_shift_mm)
Nmagshim = length(posiciones)

lower = fill(0.0,   Nmagshim)                  # grados
upper = fill(360.0, Nmagshim)
θ0    = initial_angle_deg .* ones(Nmagshim)     # config: initial_angle_deg (deg)
# Per-magnet dipole moment (A·m²), derived in pipeline_config.jl from the magnet's
# remanence and cube side length (config: magnet_Br_T, magnet_side_mm). One global
# value for now — μ is a per-magnet VECTOR, so mixed magnet grades can be supported
# without touching the math.
μ_base = magnet_moment_Am2 .* ones(Nmagshim)

P_cpu = hcat(posiciones...)             # Convierte a matrix 3x336


fld_field = Float32.(CuArray(fieldmap))
msk = CuArray(dmask)

masked_count(dmask::CuArray{TM,3}) where {TM<:AbstractFloat} = Float64(CUDA.sum(dmask))
Nmsk   = Float32(masked_count(msk))

xg_mm, yg_mm, zg_mm = build_axes_mm(dims, resmm)
grid = make_grid_gpu_from_axes_mm(xg_mm, yg_mm, zg_mm)

P = Float32.(CuArray(P_cpu .* 0.001))

M = similar(P)

B = similar(grid.X)

if mode == "RMS"
    Gx = similar(grid.X)
    Gy = similar(grid.X)
    Gz = similar(grid.X)
end

by_min = CuArray([typemax(Float32)])
by_max = CuArray([typemin(Float32)])

by_mean = CuArray([0.0f0])
stdiv = CuArray([0.0f0])

grad_rms = CuArray([0.0f0])
coef = CuArray([0.0f0])
f_prev = CuArray([0.0f0])

threads = 512 
N = Int32(length(grid.X))
blocks = cld(N, threads) 



shmem_RMS = 3 * threads * sizeof(Float32) 
shmem_sum = threads * sizeof(Float32) 

θ_init = Float32.(CuArray(θ0))
Θmew = similar(θ_init)

mu = Float32.(CuArray(μ_base))
m = Int32(size(P_cpu, 2))
on_off = CuArray(zeros(Nmagshim))
state_new = copy(on_off)

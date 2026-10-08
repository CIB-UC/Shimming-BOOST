### setup_shell.jl — shell-mode setup (eval_domain = :shell)
#
# Drop-in replacement for setup.jl used by the GRADIENT optimizer + ring search
# when scoring on a spherical SHELL at Rmax instead of a dense grid. It provides
# the same globals those scripts read (grid = evaluation coords, fld_field = base field,
# msk, P, mu, M, B, N, m, θ0, Nmagshim, threads, blocks) but the evaluation set is
# the Ns shell points from Field_data_shell.jl — no Cartesian mesh, no ∇B kernels.
#
# `_Btot!` evaluates a dipole's field at ANY points, so it works unchanged on the
# shell coordinates fed in as grid.X/Y/Z. Everything downstream (operator build,
# variance/ppm) then acts on the shell field vector directly.

@isdefined(ITERATION) || include("pipeline_config.jl")

using CUDA, JLD2, LinearAlgebra, Statistics

include("utils/grid_utils.jl")     # GridGPU struct
include("utils/pos_trays.jl")      # positions_from_rings_mm, ringpos_from_tray_mm

# --- load the shell evaluation set (coords in mm, base field in mT) -----------
isfile(shell_fieldmap_path) ||
    error("No shell field map at\n    $(shell_fieldmap_path)\nRun Field_data_shell.jl (Stage 0, shell mode) first.")
@load shell_fieldmap_path shell_xyz By_shell

const Ns_shell = size(shell_xyz, 2)

# evaluation "grid": X/Y/Z are the shell coordinates (metres), shaped (Ns,1,1) so
# they satisfy GridGPU's CuArray{Float32,3} fields and index linearly in the kernels.
_col(v) = CuArray(reshape(Float32.(v), Ns_shell, 1, 1))
gX = _col(shell_xyz[1, :] .* 1f-3)
gY = _col(shell_xyz[2, :] .* 1f-3)
gZ = _col(shell_xyz[3, :] .* 1f-3)
grid = GridGPU(gX, gY, gZ, Int32(Ns_shell), Int32(1), Int32(1))

fld_field = _col(By_shell)                            # base field on the shell (mT)
msk = _col(ones(Float32, Ns_shell))                   # every point is in the shell ⇒ mask = 1
Nmsk = Float32(Ns_shell)

# spacing globals are meaningless on a shell (no ∇B term here) — dummies for API parity
dx_m = 1.0f0; dy_m = 1.0f0; dz_m = 1.0f0
resmm = (1.0, 1.0, 1.0)

threads = 512
N       = Int32(length(grid.X))
blocks  = cld(N, threads)

# --- magnet positions (geometry only — identical to setup.jl) -----------------
posiciones = positions_from_rings_mm(positions_in_tray_new_wished;
    occupied_trays        = positions_in_tray_occupied,
    shim_radius_mm        = shim_radius_mm,
    mags_per_segment      = mags_per_segment,
    num_segments          = num_trays,
    angle_per_segment_deg = angle_per_segment_deg,
    angular_offset_deg    = angular_offset_deg,
    tray_slot_spacing_mm  = tray_slot_spacing_mm,
    front_tray_shift_mm   = front_tray_shift_mm,
    back_tray_shift_mm    = back_tray_shift_mm)
Nmagshim = length(posiciones)

P_cpu  = hcat(posiciones...)                          # 3 × Nmagshim (mm)
P      = Float32.(CuArray(P_cpu .* 0.001))            # → metres
M      = similar(P)
B      = similar(grid.X)

μ_base = magnet_moment_Am2 .* ones(Nmagshim)          # config: magnet_Br_T, magnet_side_mm (matches setup.jl)
mu     = Float32.(CuArray(μ_base))
m      = Int32(size(P_cpu, 2))
θ0     = initial_angle_deg .* ones(Nmagshim)          # config: initial_angle_deg (deg)

lower = fill(0.0,   Nmagshim)
upper = fill(360.0, Nmagshim)

println("setup_shell: ", Ns_shell, " shell points at Rmax=", Rmax, " mm   ·   ", Nmagshim, " shim magnets")

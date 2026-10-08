# optim_grad.jl  —  PHASE 1 (BOOST redesign): the linear operator G
#
#   julia optim_grad.jl
#
# The shim field is LINEAR in the magnet moments. With w_i = (μ cosθ_i, μ sinθ_i)
# (times the on/off state):
#
#     By(r) = By_base(r) + Σ_i [ Gx_i(r)·(μ cosθ_i) + Gy_i(r)·(μ sinθ_i) ]
#
# Gx_i, Gy_i are the By response of magnet i to a UNIT x / y moment — fixed,
# geometry-only (independent of the angles). This file assembles G on the shell
# voxels and VALIDATES that  By_base + G·w  reproduces the existing _Btot! kernel
# to Float32 tolerance, then caches the operator for Phase 2 (variance + adjoint
# gradient + L-BFGS).
#

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))          # project lives in the repo root

# This optimizer only needs the SHARED setup globals + the field kernels — NOT the
# simulated-annealing code. So include setup.jl + kernels/f_kernel.jl directly
# instead of sim_annealing_optim/operation.jl (keeps the two optimizers decoupled).
include(joinpath(@__DIR__, "..", "pipeline_config.jl"))
if eval_domain === :shell                                  # score on a spherical shell at Rmax …
    include(joinpath(@__DIR__, "..", "setup_shell.jl"))
else                                                        # … or the dense grid (default)
    include(joinpath(@__DIR__, "..", "setup.jl"))
end
include(joinpath(@__DIR__, "..", "kernels", "f_kernel.jl")) # _Btot!, _M! used to build/validate G
using LinearAlgebra, JLD2, Printf

const Nmag = Nmagshim
const Ntot = Int(length(grid.X))

# Linear (column-major) indices of the shell voxels — consistent across grid.X,
# fld_field, B and msk (the kernel indexes all of them with the same idx).
const shell_idx = findall(>(0f0), vec(Array(msk)))
const Ns = length(shell_idx)

# ---------------------------------------------------------------------------
# Assemble Gx, Gy  (Ns × Nmag)  by driving the EXACT _Btot! math with a unit
# x- / y-moment on one magnet at a time (base field set to zero). Building G
# from the same kernel guarantees it matches _Btot! by construction.
# ---------------------------------------------------------------------------
function build_G()
    # Shell-restricted operator (used by the variance term & validation) …
    Gx = zeros(Float32, Ns, Nmag)
    Gy = zeros(Float32, Ns, Nmag)
    # … and the FULL-grid operator. The field spatial-gradient penalty needs
    # neighbouring voxels (incl. ones just outside the shell), so the shell-only
    # operator is not enough: cache the whole grid too.  (Ntot × Nmag, Float32.)
    Gx_full = zeros(Float32, Ntot, Nmag)
    Gy_full = zeros(Float32, Ntot, Nmag)
    Munit   = CUDA.zeros(Float32, 3, Nmag)
    zerofld = CUDA.zeros(Float32, Ntot)
    Btmp    = similar(grid.X)
    for i in 1:Nmag
        Munit .= 0f0; CUDA.@allowscalar Munit[1, i] = 1f0            # unit x-moment
        @cuda threads=threads blocks=blocks _Btot!(zerofld, Btmp, grid.X, grid.Y, grid.Z, P, Munit, m, N)
        col = vec(Array(Btmp)); Gx_full[:, i] .= col; Gx[:, i] .= col[shell_idx]

        Munit .= 0f0; CUDA.@allowscalar Munit[2, i] = 1f0            # unit y-moment
        @cuda threads=threads blocks=blocks _Btot!(zerofld, Btmp, grid.X, grid.Y, grid.Z, P, Munit, m, N)
        col = vec(Array(Btmp)); Gy_full[:, i] .= col; Gy[:, i] .= col[shell_idx]

        (i % 64 == 0 || i == Nmag) && @printf("  built G columns %d/%d\r", i, Nmag)
    end
    println()
    return Gx, Gy, Gx_full, Gy_full
end

# ---------------------------------------------------------------------------
# Validation:  By_base + G·w   vs   the _Btot! kernel, on the shell voxels.
# ---------------------------------------------------------------------------
function validate_G(Gx, Gy; θdeg = θ0, state = ones(Nmag))
    # (1) kernel field for these angles/state
    θd = CuArray(Float32.(θdeg)); st = CuArray(Float32.(state))
    @cuda threads=threads blocks=blocks _M!(θd, mu, st, m, M)
    @cuda threads=threads blocks=blocks _Btot!(fld_field, B, grid.X, grid.Y, grid.Z, P, M, m, N)
    B_kernel = Array(B)[shell_idx]

    # (2) linear model in Float64 (so any mismatch reflects G, not Float32 noise)
    μc = Float64.(Array(mu))
    θr = Float64.(θdeg) .* (π / 180)
    ux = μc .* cos.(θr) .* Float64.(state)          # μ cosθ · state
    vy = μc .* sin.(θr) .* Float64.(state)          # μ sinθ · state
    By_base = Float64.(vec(Array(fld_field))[shell_idx])
    By_lin  = By_base .+ Float64.(Gx) * ux .+ Float64.(Gy) * vy

    Bk   = Float64.(B_kernel)
    aerr = maximum(abs.(By_lin .- Bk))
    rerr = aerr / max(1e-9, maximum(abs.(Bk)))
    return aerr, rerr
end

# ---------------------------------------------------------------------------
println("Phase 1 — linear operator G")
println("  field map : ", interpolated_fieldmap_name)
println("  shell     : ", Ns, " voxels   magnets : ", Nmag, "   (G is ", Ns, "×", 2Nmag, ")")

Gx, Gy, Gx_full, Gy_full = build_G()

aerr, rerr = validate_G(Gx, Gy; θdeg = θ0, state = ones(Nmag))
@printf("validation @ θ0, all magnets ON :  max|Δ| = %.3g mT   rel = %.3g\n", aerr, rerr)
println(rerr < 1f-3 ? "  ✓ G reproduces _Btot! within tolerance." :
                       "  ✘ mismatch — check moment convention / scale / voxel indexing.")

# cache the geometry-only operator + base field for Phase 2 (all CPU arrays).
# Shell arrays (Gx/Gy/By_base_shell) drive the variance term; the FULL-grid arrays
# + grid metadata (dims, spacing, shell mask) drive the ∇B (Tikhonov) penalty and
# the metrics, matching the SA kernels (kernels/_grad!: central diff, spacing dy_m).
Gdir = joinpath(optimizer_iter_dir, "GradOpt"); mkpath(Gdir)
By_base_shell = Float32.(vec(Array(fld_field))[shell_idx])
By_base_full  = Float32.(vec(Array(fld_field)))
mask_full     = Float32.(vec(Array(msk)))
Nmask         = Int(round(sum(mask_full)))
nx, ny, nz    = Int(grid.nx), Int(grid.ny), Int(grid.nz)
d_grad_m      = Float64(dy_m)          # SA's _grad! uses dy_m for ALL axes — match it
mu_cpu = Array(mu)
is_shell = (eval_domain === :shell)     # tag so grad_core skips the ∇B (grid-only) machinery
mem_mb = round((length(Gx_full) + length(Gy_full)) * sizeof(Float32) / 1e6, digits = 1)
@printf("%s operator: %d × %d  (%.1f MB), shell voxels %d / %d\n",
        is_shell ? "shell" : "full-grid", Ntot, Nmag, mem_mb, Ns, Ntot)
@save joinpath(Gdir, "operator_G.jld2") shell_idx Ns Nmag Gx Gy By_base_shell mu_cpu Gx_full Gy_full By_base_full mask_full Nmask nx ny nz d_grad_m is_shell
println("Saved operator → ", joinpath(Gdir, "operator_G.jld2"))

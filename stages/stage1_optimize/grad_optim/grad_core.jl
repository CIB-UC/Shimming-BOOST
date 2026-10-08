# grad_core.jl  —  shared math for the gradient-based optimizer + metrics.
#
# Included by run_grad.jl, run_grad_lcurve.jl and utils/eval_metrics.jl (each of
# which activates the root project and includes pipeline_config.jl FIRST). Loads
# the cached linear operator G (built by optim_grad.jl) and provides everything
# needed to build a smooth objective and to score any solution.
#
# The field is LINEAR in the magnet moments:
#     By = By_base + Gx·(μ cosθ · state) + Gy·(μ sinθ · state)
# so every term below is a smooth function of θ with an analytic gradient:
#   • variance      Var  = mean over shell of (By − mean)²                 [mT²]
#   • soft-range    ≈ (max − min) over shell via log-sum-exp               [mT]
#   • ∇B penalty    GradPen = mean over shell of |∇B|²                     [(mT/m)²]
#
# The ∇B stencil MATCHES kernels/_grad! exactly (central differences interior,
# one-sided at the faces, single spacing d = dy_m for all three axes, masked to
# the shell) so grad_rms here == the SA/BOOST √mean|∇B|² — an apples-to-apples
# comparison between the two optimizers.

using JLD2, LinearAlgebra, Statistics, SparseArrays

include(joinpath(@__DIR__, "grad_math.jl"))     # pure shared math (_data_variance, make_data_cost, …)

@isdefined(optimizer_iter_dir) ||
    error("grad_core.jl: include pipeline_config.jl before this file.")

const _GOP = joinpath(optimizer_iter_dir, "GradOpt", "operator_G.jld2")
isfile(_GOP) ||
    error("grad_core: no operator_G.jld2 at\n    $_GOP\nRun  `julia stages/stage1_optimize/grad_optim/optim_grad.jl`  first.")

# The full-grid operator is required. An operator file from the older optim_grad.jl
# (shell-only) is missing it — fail with a clear instruction rather than a KeyError.
let ks = jldopen(f -> keys(f), _GOP, "r")
    ("Gx_full" in ks) ||
        error("operator_G.jld2 has no full-grid operator (older optim_grad.jl).\n" *
              "Re-run  `julia stages/stage1_optimize/grad_optim/optim_grad.jl`  to rebuild it.")
end

@load _GOP shell_idx Ns Nmag Gx Gy By_base_shell mu_cpu Gx_full Gy_full By_base_full mask_full Nmask nx ny nz d_grad_m
# operator built in shell mode? (older grid operators have no is_shell key)
const IS_SHELL = jldopen(f -> (haskey(f, "is_shell") ? f["is_shell"] : false), _GOP, "r")

# --- CPU (Float64) operator + constants ------------------------------------
const GXF   = Float64.(Gx_full)        # Ntot × Nmag   (unit x-moment response, full grid)
const GYF   = Float64.(Gy_full)        # Ntot × Nmag   (unit y-moment response, full grid)
const GXS   = Float64.(Gx)             # Ns   × Nmag   (shell subset — variance term)
const GYS   = Float64.(Gy)
const BF    = Float64.(By_base_full)   # base field, full grid            [mT]
const MU    = Float64.(mu_cpu)         # per-magnet moment magnitude
const SHELL = Int.(shell_idx)          # shell voxel linear indices into the full grid
const NS    = Int(Ns)
const NM    = Int(Nmag)
const MASKV = Float64.(mask_full)      # 0/1 shell mask over the full grid
const NMASK = Float64(Nmask)
const NXf, NYf, NZf = Int(nx), Int(ny), Int(nz)
const NT    = NXf * NYf * NZf

# --- staleness guard: magnet strength is BAKED INTO the cached operator -------
# operator_G.jld2 stores the μ it was built with. If config.toml's magnet_B1cm_mT
# has changed since, every ppm computed below would silently describe the OLD
# magnets. Warn loudly rather than reporting a wrong number as if it were right.
if @isdefined(magnet_moment_Am2) && !isempty(MU)
    _mu_cfg = Float64(magnet_moment_Am2)
    if abs(MU[1] - _mu_cfg) > 1e-9 * max(1.0, abs(_mu_cfg))
        @warn """operator_G.jld2 was built with a DIFFERENT magnet strength — results below are stale.
                 cached μ = $(MU[1]) A·m²   ·   config μ = $(_mu_cfg) A·m² (magnet_B1cm_mT = $(magnet_B1cm_mT) mT)
                 Rebuild it:  julia stages/stage1_optimize/grad_optim/optim_grad.jl   (GUI: Stage 1 → "Grad build+solve")"""
    end
end

# ---------------------------------------------------------------------------
# Finite-difference operators DX, DY, DZ  (Ntot × Ntot, sparse), replicating
# kernels/_grad!: central /(2d) in the interior, one-sided /d at each face, with
# the SAME single spacing d = d_grad_m (= dy_m) on every axis. Grid is column-
# major (x fastest): lin(i,j,k) = i + (j-1)nx + (k-1)nx*ny.
# ---------------------------------------------------------------------------
function _fd_axis(nx, ny, nz, d, axis::Symbol)
    N = nx * ny * nz
    I = Int[]; J = Int[]; V = Float64[]
    lin(i, j, k) = i + (j - 1) * nx + (k - 1) * nx * ny
    n = axis === :x ? nx : axis === :y ? ny : nz
    for k in 1:nz, j in 1:ny, i in 1:nx
        idx = lin(i, j, k)
        a   = axis === :x ? i : axis === :y ? j : k
        nbr(t) = axis === :x ? lin(t, j, k) : axis === :y ? lin(i, t, k) : lin(i, j, t)
        if n == 1
            continue                                   # no gradient along a singleton axis
        elseif 1 < a < n
            push!(I, idx); push!(J, nbr(a + 1)); push!(V,  1 / (2d))
            push!(I, idx); push!(J, nbr(a - 1)); push!(V, -1 / (2d))
        elseif a == 1
            push!(I, idx); push!(J, nbr(2)); push!(V,  1 / d)
            push!(I, idx); push!(J, nbr(1)); push!(V, -1 / d)
        else # a == n
            push!(I, idx); push!(J, nbr(n));     push!(V,  1 / d)
            push!(I, idx); push!(J, nbr(n - 1)); push!(V, -1 / d)
        end
    end
    return sparse(I, J, V, N, N)
end

# On a shell there is no Cartesian mesh, so the ∇B finite-difference operators are
# undefined (and the Tikhonov term is disabled). Only build them on a real grid.
const DX, DY, DZ = IS_SHELL ? (nothing, nothing, nothing) :
    (_fd_axis(NXf, NYf, NZf, d_grad_m, :x),
     _fd_axis(NXf, NYf, NZf, d_grad_m, :y),
     _fd_axis(NXf, NYf, NZf, d_grad_m, :z))

# ---------------------------------------------------------------------------
# Field + metrics
# ---------------------------------------------------------------------------
_moments(θ, state) = (MU .* state .* cos.(θ), MU .* state .* sin.(θ))   # (u, v)

byfield(θ, state = ones(NM)) = (u = MU .* state .* cos.(θ);
                                v = MU .* state .* sin.(θ);
                                BF .+ GXF * u .+ GYF * v)

# range / mean / ppm over the shell + grad_rms = √mean|∇B|² over the shell,
# IDENTICAL definition to the SA kernels (so the two optimizers are comparable).
function field_metrics(θ, state = ones(NM))
    By  = byfield(θ, state)
    Bs  = By[SHELL]
    rng = maximum(Bs) - minimum(Bs)
    mn  = mean(Bs)
    ppm = mn == 0 ? NaN : 1e6 * rng / mn
    if IS_SHELL                                   # no grid ⇒ no spatial-gradient metric
        return (range_mT = rng, mean_mT = mn, ppm = ppm, grad_rms = NaN)
    end
    gx = DX * By; gy = DY * By; gz = DZ * By
    gradsq = sum(MASKV .* (gx.^2 .+ gy.^2 .+ gz.^2)) / NMASK
    return (range_mT = rng, mean_mT = mn, ppm = ppm, grad_rms = sqrt(max(0.0, gradsq)))
end

# ---------------------------------------------------------------------------
# Objective terms for the CACHED full-grid operator — each returns (value,
# ∂value/∂By) with ∂/∂By a full-grid vector (data terms act on the shell slice).
# The pure data-term math (_data_variance / _data_softrange) + make_data_cost +
# shell_metrics + fd_gradcheck come from grad_math.jl (included at the top).
# ---------------------------------------------------------------------------
function _var_term(By)
    val, adjs = _data_variance(By[SHELL])
    adj = zeros(length(By)); @inbounds adj[SHELL] .= adjs
    return val, adj
end
function _softrange_term(By, β)
    val, adjs = _data_softrange(By[SHELL], β)
    adj = zeros(length(By)); @inbounds adj[SHELL] .= adjs
    return val, adj
end

# ∇B penalty: mean over shell of |∇B|²  (matches SA grad_rms², see field_metrics)
function _gradpen_term(By)
    gx = DX * By; gy = DY * By; gz = DZ * By
    val = sum(MASKV .* (gx.^2 .+ gy.^2 .+ gz.^2)) / NMASK
    adj = (2 / NMASK) .* (DX' * (MASKV .* gx) .+ DY' * (MASKV .* gy) .+ DZ' * (MASKV .* gz))
    return val, adj
end

# ---------------------------------------------------------------------------
# build_cost — returns an Optim `only_fg!`-style closure fg!(F, G, θ):
#     J(θ) = data_term(By) + grad_lambda · mean|∇B|²
# assuming all magnets ON (the θ-parametrization fixes |m_i| = μ).
# ---------------------------------------------------------------------------
function build_cost(; data_term::Symbol = :variance,
                      grad_lambda::Real = 0.0,
                      beta::Real = 200.0)
    IS_SHELL && grad_lambda != 0 && @warn "grad_lambda ignored in shell mode (no ∇B term)."
    λg = IS_SHELL ? 0.0 : Float64(grad_lambda); β = Float64(beta)
    data_term in (:variance, :softrange) ||
        error("grad_data_term = $(data_term) invalid; use :variance or :softrange.")
    function fg!(F, G, θ)
        u = MU .* cos.(θ); v = MU .* sin.(θ)
        By = BF .+ GXF * u .+ GYF * v
        dval, dadj = data_term === :variance ? _var_term(By) : _softrange_term(By, β)
        gval = 0.0; adj = dadj
        if λg != 0.0
            gval, gadj = _gradpen_term(By)
            adj = dadj .+ λg .* gadj
        end
        if G !== nothing
            gu = GXF' * adj; gv = GYF' * adj          # ∂J/∂u, ∂J/∂v
            @. G = -MU * sin(θ) * gu + MU * cos(θ) * gv   # chain u=μcosθ, v=μsinθ
        end
        F === nothing ? nothing : dval + λg * gval
    end
    return fg!
end

# grad_math.jl  —  pure, side-effect-free gradient-optimizer math.
#
# NO globals, NO file loads — just functions. Included by BOTH grad_core.jl (the
# cached full-grid objective) and ring_search.jl (operator subsets), so the
# variance / soft-range math lives in exactly one place. Safe to include without
# any operator_G.jld2 present.

using LinearAlgebra, Statistics

# --- data terms as PURE functions of a field vector By → (value, ∂value/∂By) ---
function _data_variance(By)
    n = length(By); mn = mean(By); r = By .- mn
    return dot(r, r) / n, (2 / n) .* r
end
# soft-(max−min) via log-sum-exp; β in 1/mT. max/min shifts are for numerical
# stability and drop out of the derivative (softmax weight p, softmin weight q).
function _data_softrange(By, β)
    mx = maximum(By); ex = exp.(β .* (By .- mx)); sx = sum(ex)
    mn = minimum(By); en = exp.(-β .* (By .- mn)); sn = sum(en)
    return (mx + log(sx) / β) - (mn - log(sn) / β), (ex ./ sx) .- (en ./ sn)
end

# --- operator-AGNOSTIC data-term cost (used by ring_search on subset operators) -
#   GX, GY : Ns×k  shell response of unit x / y moments for k magnets
#   b      : Ns    base field on the shell (mT)
#   mu     : k     per-magnet moment magnitude
# By = b + GX·(μcosθ) + GY·(μsinθ) is the shell field directly (no full grid, no
# ∇B term — the search ranks by variance/soft-range → ppm). Returns Optim fg!.
function make_data_cost(GX, GY, b, mu; data_term::Symbol = :variance, beta::Real = 200.0)
    β = Float64(beta)
    data_term in (:variance, :softrange) ||
        error("data_term = $(data_term) invalid; use :variance or :softrange.")
    function fg!(F, G, θ)
        u = mu .* cos.(θ); v = mu .* sin.(θ)
        By = b .+ GX * u .+ GY * v
        val, adj = data_term === :variance ? _data_variance(By) : _data_softrange(By, β)
        if G !== nothing
            gu = GX' * adj; gv = GY' * adj
            @. G = -mu * sin(θ) * gu + mu * cos(θ) * gv
        end
        F === nothing ? nothing : val
    end
    return fg!
end

# range / mean / ppm of a shell field for an operator + angles (ring-search score)
function shell_metrics(GX, GY, b, mu, θ)
    By = b .+ GX * (mu .* cos.(θ)) .+ GY * (mu .* sin.(θ))
    rng = maximum(By) - minimum(By); mn = mean(By)
    return (range_mT = rng, mean_mT = mn, ppm = mn == 0 ? NaN : 1e6 * rng / mn)
end

# finite-difference sanity check of an analytic gradient (self-test)
function fd_gradcheck(fg!, θ; k = 6, ε = 1e-6)
    g = similar(θ); fg!(true, g, θ)
    maxrel = 0.0
    for i in rand(1:length(θ), k)
        θp = copy(θ); θp[i] += ε; θm = copy(θ); θm[i] -= ε
        fd = (fg!(true, nothing, θp) - fg!(true, nothing, θm)) / (2ε)
        maxrel = max(maxrel, abs(fd - g[i]) / max(1e-9, abs(g[i])))
    end
    return maxrel
end

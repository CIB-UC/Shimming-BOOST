# run_grad.jl  —  PHASE 2 (BOOST redesign): gradient-based joint optimization
#
#   julia grad_optim/run_grad.jl        (run grad_optim/optim_grad.jl first to build/cache G)
#
# The field is exactly linear in the moments (Phase 1 validated G to ~1e-8), so a
# weighted sum of the shell field's variance / soft-range and its spatial-gradient
# penalty is a SMOOTH function of the angles with an analytic gradient:
#
#     J(θ) = data_term(By)  +  grad_lambda · mean|∇B|²           (Tikhonov term)
#     By(θ) = By_base + Gx·(μcosθ) + Gy·(μsinθ)
#
# data_term / grad_lambda / grad_softrange_beta come from pipeline_config.jl. We
# minimize J over ALL magnets jointly with L-BFGS (multi-start); |m_i| = μ is built
# in by the θ-parametrization. The math + FD operators live in grad_core.jl.
#
# CPU-only (uses the cached operator). Additive — does NOT touch the SA.

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))          # project lives in the repo root
include(joinpath(@__DIR__, "..", "pipeline_config.jl"))
include(joinpath(@__DIR__, "grad_core.jl"))
using Optim, Random, Printf, JLD2, Statistics

fg! = build_cost(data_term  = grad_data_term,
                 grad_lambda = grad_lambda,
                 beta        = grad_softrange_beta)

println("Phase 2 — smooth objective (G-based, L-BFGS)")
println("  operator    : ", NS, " shell voxels × ", NM, " magnets  (full grid ", NT, ")")
println("  data term   : ", grad_data_term,
        grad_data_term === :softrange ? "   (β = $(grad_softrange_beta) /mT)" : "")
println("  grad_lambda : ", grad_lambda, "   (Tikhonov weight on mean|∇B|²)")
@printf("  analytic-gradient check (finite diff) : max rel err = %.2e\n",
        (Random.seed!(0); fd_gradcheck(fg!, 2π .* rand(NM))))

# Split the combined fg! into Optim's separate value / gradient callbacks
# (portable core API; avoids only_fg!, which lives in NLSolversBase).
f_cost(θ)     = fg!(true, nothing, θ)
g_cost!(G, θ) = (fg!(nothing, G, θ); G)
run_lbfgs(θ0) = Optim.minimizer(optimize(f_cost, g_cost!, θ0, LBFGS(),
                                         Optim.Options(g_tol = 1e-12, iterations = 5000)))

function run_multistart(seeds)
    ppms = Float64[]; bestθ = nothing; bestppm = Inf
    for s in seeds
        Random.seed!(s)
        θ = run_lbfgs(2π .* rand(NM))
        mt = field_metrics(θ); push!(ppms, mt.ppm)
        @printf("  start %2d : ppm = %8.1f   range = %.4f mT   grad = %.3f mT/m\n",
                s, mt.ppm, mt.range_mT, mt.grad_rms)
        if mt.ppm < bestppm; bestppm = mt.ppm; bestθ = θ; end
    end
    return ppms, bestθ, bestppm
end

ppms, bestθ_rad, bestppm = run_multistart(grad_seeds)
mt = field_metrics(bestθ_rad)

@printf("\nGRAD over %d starts : ppm = %.1f ± %.1f   best = %.1f\n",
        length(grad_seeds), mean(ppms), length(ppms) > 1 ? std(ppms) : 0.0, bestppm)
@printf("  best solution : range = %.4f mT   mean = %.4f mT   grad_RMS = %.3f mT/m   ppm = %.1f\n",
        mt.range_mT, mt.mean_mT, mt.grad_rms, mt.ppm)
println("  reference — SA baseline ≈ 9981 ± 112 ppm;  variance-only optimum ≈ 7878 ppm")

# --- save the best solution in the standard result format (deg, verifiable) ----
bestθ       = Float32.(mod.(rad2deg.(bestθ_rad), 360.0))
final_state = ones(Float32, NM)
λ           = Float64(grad_lambda)      # store the Tikhonov weight actually used
ppm         = bestppm
mkpath(dirname(grad_result_path))
@save grad_result_path λ bestθ final_state ppm       # tagged copy for SA-vs-grad comparison
println("\nSaved best solution → ", grad_result_path)

# Also write the canonical Stage-1 location/name so export_csv.jl → CSV_to_STL.jl
# pick up the gradient result exactly like an SA result (same variables/format).
mkpath(dirname(optimizer_result_path))
@save optimizer_result_path λ bestθ final_state ppm
println("Also saved standard Stage-1 result → ", optimizer_result_path)
println("Compare against other results with:  julia utils/eval_metrics.jl")

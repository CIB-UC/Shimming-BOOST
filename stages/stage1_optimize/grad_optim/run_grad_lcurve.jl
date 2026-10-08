# run_grad_lcurve.jl  —  λ sweep for the GRADIENT optimizer (Tikhonov L-curve)
#
#   julia stages/stage1_optimize/grad_optim/run_grad_lcurve.jl   (run stages/stage1_optimize/grad_optim/optim_grad.jl first)
#
# Solves the smooth objective  J(θ) = data_term(By) + grad_lambda·mean|∇B|²  to the
# global optimum for each grad_lambda in `grad_lambda_sweep` (pipeline_config.jl),
# recording range / ppm / grad_RMS. Produces the gradient method's own range-vs-
# gradient L-curve and OVERLAYS the SA sweep (Lcurve/sweep_lambda.csv) if present,
# so the two optimizers can be compared on the same axes.
#
# Outputs (Optimizer_Output_per_Iteration/<ITERATION>/GradOpt/Lcurve/):
#   grad_sweep_lambda.csv   — grad_lambda, ppm, range_mT, mean_mT, grad_rms
#   grad_sweep_lambda.png   — L-curve (range vs grad) with SA overlay + ppm vs λ
#   gradlam_<λ>.jld2        — best solution per λ (re-exportable via Stage 2)

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))          # project lives in the repo root
include(joinpath(@__DIR__, "..", "..", "..", "pipeline_config.jl"))
include(joinpath(@__DIR__, "grad_core.jl"))
using Optim, Random, Printf, JLD2, Statistics, DataFrames, CSV

outdir = joinpath(optimizer_iter_dir, "GradOpt", "Lcurve"); mkpath(outdir)

function run_lbfgs(fg!, θ0)
    f_cost(θ)     = fg!(true, nothing, θ)
    g_cost!(G, θ) = (fg!(nothing, G, θ); G)
    return Optim.minimizer(optimize(f_cost, g_cost!, θ0, LBFGS(),
                                    Optim.Options(g_tol = 1e-12, iterations = 5000)))
end

# Best-of-multistart solution at one grad_lambda.
function solve_at(λg)
    fg! = build_cost(data_term = grad_data_term, grad_lambda = λg, beta = grad_softrange_beta)
    bestθ = nothing; bestppm = Inf
    for s in grad_seeds
        Random.seed!(s)
        θ = run_lbfgs(fg!, 2π .* rand(NM))
        p = field_metrics(θ).ppm
        if p < bestppm; bestppm = p; bestθ = θ; end
    end
    return bestθ, field_metrics(bestθ)
end

println("Gradient-method λ sweep  (data term: ", grad_data_term, ")   × ", length(grad_seeds), " seed(s)")
rows = NamedTuple[]
for λg in grad_lambda_sweep
    θ, mt = solve_at(λg)
    @printf("  grad_lambda = %-8.3g : ppm = %8.1f   range = %.4f mT   grad = %.3f mT/m\n",
            λg, mt.ppm, mt.range_mT, mt.grad_rms)
    push!(rows, (grad_lambda = Float64(λg), ppm = mt.ppm, range_mT = mt.range_mT,
                 mean_mT = mt.mean_mT, grad_rms = mt.grad_rms))
    tag = replace(string(λg), "." => "p")
    bestθ = Float32.(mod.(rad2deg.(θ), 360.0)); final_state = ones(Float32, NM)
    λ = Float64(λg); ppm = mt.ppm
    @save joinpath(outdir, "gradlam_$(tag).jld2") λ bestθ final_state ppm
end

df  = DataFrame(rows)
csv = joinpath(outdir, "grad_sweep_lambda.csv"); CSV.write(csv, df)
println("\nSweep results → ", csv)

# --- L-curve plot (range vs gradient) with SA overlay + ppm-vs-λ ------------
sa_csv = joinpath(lcurve_dir, "sweep_lambda.csv")
try
    using GLMakie
    fig = Figure(size = (1040, 470))
    ax1 = Axis(fig[1, 1]; title = "L-curve: field range vs gradient RMS",
               xlabel = "field range  max−min  [mT]",
               ylabel = "gradient RMS  √mean|∇B|²  [mT/m]")
    scatterlines!(ax1, df.range_mT, df.grad_rms; color = :crimson, markersize = 10,
                  label = "grad (L-BFGS)")
    for (x, y, l) in zip(df.range_mT, df.grad_rms, df.grad_lambda)
        text!(ax1, x, y; text = "  $(l)", fontsize = 9, align = (:left, :center))
    end
    if isfile(sa_csv)
        sa = CSV.read(sa_csv, DataFrame)
        if all(in(names(sa)), ("range_mean", "grad_mean"))
            scatterlines!(ax1, sa.range_mean, sa.grad_mean; color = :dodgerblue,
                          markersize = 10, label = "SA (annealing)")
        end
    end
    axislegend(ax1; position = :rt)
    ax2 = Axis(fig[1, 2]; title = "ppm vs grad_lambda", xlabel = "grad_lambda", ylabel = "ppm")
    scatterlines!(ax2, Float64.(df.grad_lambda), df.ppm; color = :crimson, markersize = 10)
    png = joinpath(outdir, "grad_sweep_lambda.png"); save(png, fig)
    println("Sweep plot → ", png)
catch e
    @warn "sweep plot failed (CSV still written)." exception = e
end

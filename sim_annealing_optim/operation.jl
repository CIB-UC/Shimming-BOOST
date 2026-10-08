# This file lives in sim_annealing_optim/; shared infra is one level up at the repo root.
include(joinpath(@__DIR__, "..", "utils", "wrap.jl"))
include(joinpath(@__DIR__, "..", "utils", "ppm_report.jl"))
include(joinpath(@__DIR__, "..", "setup.jl"))
using CSV, DataFrames
using Random

# Phase 0: seed both RNG streams the SA uses — CPU rand()/randn() (accept/reject)
# and the device RNG inside the kernels — so a run is reproducible.
seed_rng!(s) = (Random.seed!(s); CUDA.seed!(s); nothing)


# Evaluate the best solution and read back the two cost terms (run on GPU):
#   range  = max(By) - min(By)          [mT]   — the homogeneity / data term (×w)
#   gradr  = sqrt(mean(|∇By|²))         [mT/m] — the smoothness / reg. term (×λ)
#   ppm    = 1e6 * range / mean(By)
function _eval_terms(bestθ, λ, final_state)
    RMS_operation(bestθ, λ, final_state)                 # fills by_min/by_max/grad_rms/coef
    fill!(by_mean, 0f0)
    @cuda threads=threads blocks=blocks shmem=shmem_sum _mean!(B, by_mean, N, Nmsk)
    rng   = Float64(Array(by_max)[1] - Array(by_min)[1])
    gradr = sqrt(max(0.0, Float64(Array(grad_rms)[1])))
    mn    = Float64(Array(by_mean)[1])
    cost  = Float64(Array(coef)[1])
    ppm   = mn == 0 ? NaN : 1e6 * rng / mn
    return (range_mT = rng, grad_rms = gradr, mean_mT = mn, cost = cost, ppm = ppm)
end

# Run the SA once, overriding exactly ONE parameter with value v.
#   param ∈  :lambda :T0 :alpha :iters :restarts :step0 :step_min
# All other parameters stay at their pipeline_config.jl settings.
const _SWEEPABLE = (:lambda, :T0, :alpha, :iters, :restarts, :step0, :step_min)

# Run the SA once with a NamedTuple of parameter OVERRIDES (one or many params).
# Does NOT seed — the caller seeds, so a point can be averaged over seeds.
function _run_sa_over(overrides)
    kw = Dict{Symbol,Any}(:iters => SA_iters, :restarts => SA_restarts,
                          :T0 => SA_T0, :alpha => SA_alpha,
                          :step0 => SA_step0, :step_min => SA_step_min,
                          :report_every => SA_report_every)
    λ = lambda_weight
    for (k, v) in pairs(overrides)
        if k == :lambda
            λ = v
        elseif haskey(kw, k)
            kw[k] = (k in (:iters, :restarts)) ? Int(round(v)) : v   # these must be Int
        else
            error("sweep override :$(k) is not sweepable; use one of $(_SWEEPABLE)")
        end
    end
    bestθ, final_state = naive_SA_RMS!(RMS_operation, λ, test_ring_seq; kw...)
    return bestθ, final_state, λ
end

# Left: ppm mean ± std vs the x-parameter (tells a real effect from RNG noise).
# Right: mean-range vs mean-gradient (the L-curve; corner ≈ best λ for :lambda).
function _plot_sweep(xs, pm, ps, rm, gm, xkey, outpng)
    try
        fig = Figure(size = (1040, 470))
        ax1 = Axis(fig[1, 1]; title = "ppm vs $(xkey)  (mean ± std over seeds)",
                   xlabel = String(xkey), ylabel = "ppm  =  1e6 · range / mean")
        errorbars!(ax1, xs, pm, ps; whiskerwidth = 10, color = (:gray, 0.8))
        scatterlines!(ax1, xs, pm; color = :dodgerblue, markersize = 10)
        ax2 = Axis(fig[1, 2]; title = "mean range vs mean gradient  (L-curve)",
                   xlabel = "field range  max−min  [mT]",
                   ylabel = "gradient RMS  √mean|∇B|²  [mT/m]")
        scatterlines!(ax2, rm, gm; color = :seagreen, markersize = 10)
        for (x, y, l) in zip(rm, gm, xs)
            text!(ax2, x, y; text = "  $(l)", fontsize = 10, align = (:left, :center))
        end
        save(outpng, fig)
        println("  sweep plot → ", outpng)
    catch e
        @warn "sweep plot failed (CSV/jld2 still written)." exception = e
    end
end

# Sweep over `sweep_points` (each a NamedTuple of SA overrides), averaging each
# point over `sweep_seeds` and reporting ppm MEAN ± STD. `sweep_xkey` selects the
# x-axis field. Only needs Stage 1 machinery. Writes
#   <lcurve_dir>/sweep_<xkey>.csv , sweep_<xkey>.png , <xkey>_<x>.jld2 (1st seed)
function generate_sweep(; points = sweep_points, xkey = sweep_xkey, seeds = sweep_seeds)
    mkpath(lcurve_dir)
    n = length(seeds)
    xs = Float64[]; pm = Float64[]; ps = Float64[]; pbest = Float64[]
    rm = Float64[]; rs = Float64[]; gm = Float64[]; tmean = Float64[]
    for (i, pt) in enumerate(points)
        x = getproperty(pt, xkey)
        println("\n", "─"^60)
        println("[point $i/$(length(points))]  $xkey = $x   overrides = $(pt)   × $n seed(s)")
        ppmi = Float64[]; rngi = Float64[]; gri = Float64[]; ti = Float64[]
        for s in seeds
            seed_rng!(s)
            t0 = time()
            bestθ, final_state, λ = _run_sa_over(pt)
            dt = time() - t0
            t = _eval_terms(bestθ, λ, final_state)
            push!(ppmi, t.ppm); push!(rngi, t.range_mT); push!(gri, t.grad_rms); push!(ti, dt)
            if s == first(seeds)                              # keep 1st-seed solution
                ppm = t.ppm; tag = replace(string(x), "." => "p")
                @save joinpath(lcurve_dir, "$(xkey)_$(tag).jld2") xkey x pt λ bestθ final_state ppm
            end
        end
        push!(xs, Float64(x))
        push!(pm, mean(ppmi)); push!(ps, n > 1 ? std(ppmi) : 0.0); push!(pbest, minimum(ppmi))
        push!(rm, mean(rngi)); push!(rs, n > 1 ? std(rngi) : 0.0); push!(gm, mean(gri))
        push!(tmean, mean(ti))
        println("   ppm = $(round(mean(ppmi), digits=1)) ± $(round(n > 1 ? std(ppmi) : 0.0, digits=1))" *
                "   (best $(round(minimum(ppmi), digits=1)))   range = $(round(mean(rngi), digits=4)) mT")
    end
    df = DataFrame(String(xkey) => xs, "ppm_mean" => pm, "ppm_std" => ps, "ppm_best" => pbest,
                   "range_mean" => rm, "range_std" => rs, "grad_mean" => gm, "runtime_mean_s" => tmean)
    csv = joinpath(lcurve_dir, "sweep_$(xkey).csv")
    CSV.write(csv, df)
    println("\nSweep results → ", csv)
    _plot_sweep(xs, pm, ps, rm, gm, xkey, joinpath(lcurve_dir, "sweep_$(xkey).png"))
    return df
end

# Phase-0 benchmark harness: run the SA once per seed (fixed field map + params),
# score ppm via the same GPU recompute verify_solution uses (_eval_terms), and
# report ppm mean ± std + runtime. A new method only "wins" if it beats this
# beyond the run-to-run noise band. Writes  <benchmark_dir>/SA_benchmark.csv
function benchmark_SA(seeds = benchmark_seeds; λ = lambda_weight, save = true)
    mkpath(benchmark_dir)
    sd = Int[]; pp = Float64[]; rng = Float64[]; gr = Float64[]; tm = Float64[]
    for s in seeds
        seed_rng!(s)
        t0 = time()
        bestθ, final_state = naive_SA_RMS!(RMS_operation, λ, test_ring_seq,
                                           T0=SA_T0, alpha=SA_alpha,
                                           iters=SA_iters, restarts=SA_restarts,
                                           step0=SA_step0, step_min=SA_step_min,
                                           report_every=SA_report_every)
        dt = time() - t0
        t = _eval_terms(bestθ, λ, final_state)
        push!(sd, s); push!(pp, t.ppm); push!(rng, t.range_mT); push!(gr, t.grad_rms); push!(tm, dt)
        println("  seed $s : ppm = $(round(t.ppm, digits=1))   range = $(round(t.range_mT, digits=4)) mT   ($(round(dt, digits=1))s)")
    end
    n = length(pp)
    μp, σp = mean(pp), (n > 1 ? std(pp) : 0.0)
    println("\n", "="^64)
    println("SA BENCHMARK   λ=$λ   over $(n) seed(s)   (field: $(interpolated_fieldmap_name))")
    println("  ppm        : $(round(μp, digits=1)) ± $(round(σp, digits=1))   (best $(round(minimum(pp), digits=1)))")
    println("  range mT   : $(round(mean(rng), digits=4)) ± $(round(n>1 ? std(rng) : 0.0, digits=4))")
    println("  runtime s  : $(round(mean(tm), digits=1)) ± $(round(n>1 ? std(tm) : 0.0, digits=1))")
    println("="^64)
    if save
        csv = joinpath(benchmark_dir, "SA_benchmark.csv")
        CSV.write(csv, DataFrame(seed=sd, ppm=pp, range_mT=rng, grad_rms=gr, runtime_s=tm))
        println("  → ", csv)
    end
    return (ppm_mean = μp, ppm_std = σp, ppm_best = minimum(pp), ppms = pp)
end

function optim_run(Iteration_folder_name, Iteration_file_name)

    seed_rng!(rng_seed)              # reproducible production run
    λ = lambda_weight
    bestθ, final_state =  naive_SA_RMS!(RMS_operation, λ, test_ring_seq,
                                        T0=SA_T0, alpha=SA_alpha,
                                        iters=SA_iters, restarts=SA_restarts,  step0=SA_step0, step_min=SA_step_min,
                                        report_every=SA_report_every)
    ppm = get_ppm_RMS(bestθ, λ, final_state, Iteration_folder_name)

    outdir = joinpath(OPTIMIZER_OUTPUT_DIR, Iteration_folder_name)
    mkpath(outdir)
    @save joinpath(outdir, Iteration_file_name) λ bestθ final_state ppm
    @save sa_result_path λ bestθ final_state ppm      # tagged copy for SA-vs-grad comparison
    println("Saved optimizer result → ", joinpath(outdir, Iteration_file_name))
    println("Saved SA tagged copy   → ", sa_result_path)
end

function optim_run_SD(Iteration_folder_name, Iteration_file_name)

    λ = lambda_weight
    bestθ, final_state = naive_SA_STDIV!(STDIV_operation, test_ring_seq, T0=SA_T0, alpha=SA_alpha)
    ppm = get_ppm_RMS(bestθ, λ, final_state, Iteration_folder_name)

    outdir = joinpath(OPTIMIZER_OUTPUT_DIR, Iteration_folder_name)
    mkpath(outdir)
    @save joinpath(outdir, Iteration_file_name) λ bestθ final_state ppm
    @save sa_result_path λ bestθ final_state ppm      # tagged copy for SA-vs-grad comparison
    println("Saved optimizer result → ", joinpath(outdir, Iteration_file_name))
    println("Saved SA tagged copy   → ", sa_result_path)
end


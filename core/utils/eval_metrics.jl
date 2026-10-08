# eval_metrics.jl  —  unified metrics for ANY saved result (SA or gradient)
#
#   julia core/utils/eval_metrics.jl [result1.jld2 result2.jld2 ...]
#
# Scores each result with ONE definition (the cached linear operator + grad_core),
# so SA and gradient solutions are directly comparable: field range, ppm, mean By,
# and gradient RMS √mean|∇B|² (the SA/BOOST smoothness metric). With no arguments
# it compares the current iteration's standard Stage-1 result against the gradient
# result. A no-shims baseline row is always shown for reference.
#
# Requires the cached full-grid operator (run `julia stages/stage1_optimize/grad_optim/optim_grad.jl`
# once for the current iteration first).

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))          # project lives in the repo root
include(joinpath(@__DIR__, "..", "..", "pipeline_config.jl"))
include(joinpath(@__DIR__, "..", "..", "stages", "stage1_optimize", "grad_optim", "grad_core.jl"))
using JLD2, Printf, Statistics

function metrics_of(path)
    @load path λ bestθ final_state ppm
    θ  = deg2rad.(Float64.(Array(bestθ)))          # results store degrees
    st = Float64.(Array(final_state))
    # The cached operator is built for ONE specific magnet set. Scoring a result from
    # a different ring set against it would fail deep inside the matmul (or silently
    # mis-score), so check up front and say what to do about it.
    length(θ) == NM || error("""
        $(basename(path)) holds $(length(θ)) magnets, but the cached operator
        (GradOpt/operator_G.jld2) was built for $(NM).
        They belong to different ring sets. Rebuild the operator for the current
        positions_in_tray_new_wished, then re-run this:
            julia stages/stage1_optimize/grad_optim/optim_grad.jl
        """)
    m  = field_metrics(θ, st)
    saved = ppm isa AbstractArray ? Array(ppm)[1] : ppm
    return (m..., nact = Int(round(sum(st))), saved_ppm = saved, lambda = λ)
end

# results to compare: CLI args, else the per-optimizer TAGGED copies (which never
# clobber each other) so an SA-with-Tikhonov result and a grad result show side by
# side. Run stages/stage1_optimize/sim_annealing_optim/run_optim.jl (λ>0) and stages/stage1_optimize/grad_optim/run_grad.jl to
# populate them.
# insert_result_path is included so a sequential per-insert layout appears here too:
# its `final_state` mask is honoured by field_metrics, so `#on` reports how many
# magnets it actually placed, alongside dense SA/grad runs that fill every slot.
default_files = [sa_result_path, grad_result_path, insert_result_path]
files = length(ARGS) >= 1 ? ARGS : default_files

base = field_metrics(zeros(NM), zeros(NM))         # all shims OFF

println("\n===================================== METRIC COMPARISON =====================================")
println("iteration : ", ITERATION, "    field map : ", interpolated_fieldmap_name)
@printf("%-32s %9s %11s %10s %12s %6s\n", "result", "ppm", "range[mT]", "mean[mT]", "grad[mT/m]", "#on")
println("-"^92)
@printf("%-32s %9.1f %11.4f %10.4f %12.3f %6s\n",
        "baseline (no shims)", base.ppm, base.range_mT, base.mean_mT, base.grad_rms, "0")
for f in files
    if !isfile(f); @warn "missing — skipped" file=f; continue; end
    m = metrics_of(f)
    name = basename(f); length(name) > 32 && (name = "…" * name[end-30:end])
    @printf("%-32s %9.1f %11.4f %10.4f %12.3f %6d\n",
            name, m.ppm, m.range_mT, m.mean_mT, m.grad_rms, m.nact)
end
println("="^92)
@printf("grad metric = √mean|∇B|² over the shell (central diff, spacing dy = %.3f mm), identical for every row.\n",
        d_grad_m * 1e3)

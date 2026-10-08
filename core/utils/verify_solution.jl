# verify_solution.jl
# Independently re-checks a saved optimization result.
#   julia verify_solution.jl  [path/to/best_*.jld2]
#
# It reloads the field map + geometry from setup.jl, then evaluates the field
# homogeneity (1) with NO shims (baseline) and (2) with the saved best angles,
# and reports range (max-min, mT) and homogeneity (ppm) before vs after.

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))              # project lives in the repo root

include(joinpath(@__DIR__, "..", "..", "stages", "stage1_optimize", "setup.jl"))       # setup globals + loads pipeline_config.jl (so optimizer_result_path is defined)
include(joinpath(@__DIR__, "wrap.jl"))              # RMS_operation + field kernels (optimizer-agnostic; no SA driver needed)

using CUDA
using Printf

# Default to the current iteration's Stage 1 result (pipeline_config.jl); ARGS[1] overrides.
resfile = length(ARGS) >= 1 ? ARGS[1] : optimizer_result_path
@info "Loading result" resfile
@load resfile λ bestθ final_state ppm

# make sure they live on the GPU regardless of how they were stored
bestθ_d       = cu(Float32.(Array(bestθ)))
final_state_d = cu(Array(final_state))
zero_state_d  = CUDA.zeros(eltype(final_state_d), Nmagshim)   # all shims OFF

# evaluate B over the masked shell for a given angle/state set, return (min,max,mean) in mT
function eval_state(θ, state, λ)
    RMS_operation(θ, λ, state)                 # fills B (masked), by_min, by_max
    fill!(by_mean, 0f0)
    @cuda threads=threads blocks=blocks shmem=shmem_sum _mean!(B, by_mean, N, Nmsk)
    mn = CUDA.@allowscalar by_min[1]
    mx = CUDA.@allowscalar by_max[1]
    me = CUDA.@allowscalar by_mean[1]
    return Float64(mn), Float64(mx), Float64(me)
end

mn0, mx0, me0 = eval_state(bestθ_d, zero_state_d,  0.0)   # baseline, no shims
mn1, mx1, me1 = eval_state(bestθ_d, final_state_d, λ)     # optimized

rng0, rng1 = mx0 - mn0, mx1 - mn1
ppm0 = 1e6 * rng0 / me0
ppm1 = 1e6 * rng1 / me1
nact = Int(round(CUDA.@allowscalar sum(final_state_d)))

println("\n========== SOLUTION RE-CHECK ==========")
@printf("active shim magnets : %d / %d\n", nact, Nmagshim)
@printf("angles (deg)        : min %.2f  max %.2f  mean %.2f  std %.2f\n",
        minimum(Array(bestθ_d)), maximum(Array(bestθ_d)),
        Statistics.mean(Array(bestθ_d)), Statistics.std(Array(bestθ_d)))
@printf("lambda              : %.3g\n", λ)
println("------ shell field over Rmin..Rmax mm ------")
@printf("                     %12s   %12s\n", "BASELINE", "OPTIMIZED")
@printf("mean By  (mT)      : %12.5f   %12.5f\n", me0, me1)
@printf("range max-min (mT) : %12.5f   %12.5f\n", rng0, rng1)
@printf("homogeneity (ppm)  : %12.1f   %12.1f\n", ppm0, ppm1)
println("-------------------------------------------")
@printf("range reduction    : %.1f%%   (%.5f -> %.5f mT)\n",
        100*(rng0-rng1)/rng0, rng0, rng1)
@printf("ppm  reduction     : %.1f%%   (%.1f -> %.1f ppm)\n",
        100*(ppm0-ppm1)/ppm0, ppm0, ppm1)
@printf("saved 'ppm' field  : %s\n", string(ppm isa AbstractArray ? Array(ppm) : ppm))
println("=======================================")

# run_lcurve.jl  —  L-CURVE sweep (Stage 1 only, no geometry)
#
# Run on its own:        julia run_lcurve.jl
# Or via the pipeline:   run_pipeline.jl launches this when pipeline_mode == "lcurve"
#                        (after Stage 0 has produced the field map).
#
# Sweeps λ (the gradient-term weight) over `lambda_sweep` from pipeline_config.jl,
# optimizing at each value, and records the L-curve: the field-range term vs the
# gradient (smoothness) term. Its corner is the best λ. Writes CSV + plot + per-λ
# jld2 into  Optimizer_Output_per_Iteration/<ITERATION>/Lcurve/.
#
# To change the sweep, w, rings, or SA params, edit pipeline_config.jl — not here.

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))          # project lives in the repo root

include(joinpath(@__DIR__, "..", "..", "..", "pipeline_config.jl"))   # sweep_points, sweep_xkey, sweep_seeds, SA params, ...
include(joinpath(@__DIR__, "operation.jl"))               # defines generate_sweep()

generate_sweep()                # runs sweep_points × sweep_seeds → ppm mean ± std (config)

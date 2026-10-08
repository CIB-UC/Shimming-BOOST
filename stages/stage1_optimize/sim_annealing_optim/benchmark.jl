# benchmark.jl  —  PHASE 0 benchmark harness (SA baseline)
#
#   julia benchmark.jl
#
# Runs the current SA (pipeline_config.jl params) once per seed in `benchmark_seeds`,
# scores each with the same GPU field recompute verify_solution.jl uses, and reports
# ppm mean ± std + runtime. This is the baseline any new optimizer must beat beyond
# the run-to-run noise band (see OPTIMIZER_REDESIGN_BRIEF.md, Phase 0).
#
# Reads the field map + geometry + SA params + seeds from pipeline_config.jl.
# Writes  Optimizer_Output_per_Iteration/<ITERATION>/Benchmark/SA_benchmark.csv

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))          # project lives in the repo root

include(joinpath(@__DIR__, "..", "..", "..", "pipeline_config.jl"))   # SA params, benchmark_seeds, benchmark_dir, ...
include(joinpath(@__DIR__, "operation.jl"))               # defines benchmark_SA()

benchmark_SA()                  # uses benchmark_seeds + lambda_weight from config

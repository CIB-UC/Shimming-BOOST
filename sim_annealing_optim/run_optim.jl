# run_optim.jl  —  STAGE 1 entry point (BOOST optimizer)
#
# Run on its own:        julia run_optim.jl
# Or via the pipeline:   run_pipeline.jl launches this as a subprocess.
#
# What it does:
#   - activates the local project created by install_deps.jl
#   - loads pipeline_config.jl  (iteration name, paths, geometry, SA params)
#   - includes operation.jl     (which loads setup.jl + kernels + utils)
#   - runs simulated annealing and saves the result to
#       Optimizer_Output_per_Iteration/<ITERATION>/<ITERATION>_BOOST_result.jld2
#
# To change the iteration, field map, geometry or search params, edit
# pipeline_config.jl — NOT this file.

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))          # project lives in the repo root

include(joinpath(@__DIR__, "..", "pipeline_config.jl"))   # ITERATION, paths, geometry, SA params
include(joinpath(@__DIR__, "operation.jl"))               # defines optim_run(), optim_run_SD()

# RMS objective:  w*(max-min) + λ*sqrt(grad_RMS).  Needs mode="RMS" in setup.jl.
optim_run(Iteration_folder_name, Iteration_file_name)

# To run the std-deviation objective instead, comment the line above and use:
#   optim_run_SD(Iteration_folder_name, Iteration_file_name)

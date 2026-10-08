# install_deps.jl
# Run ONCE to set up the Julia environment for this project.
#   Windows PowerShell, from inside the BOOST_FKERNEL folder:
#       julia install_deps.jl
#
# This creates a local Project.toml + Manifest.toml in this folder so the exact
# package set is pinned and reproducible. Requires an NVIDIA GPU for CUDA.jl.

import Pkg
Pkg.activate(@__DIR__)          # use a project local to this folder

Pkg.add([
    "CUDA",            # GPU kernels (needs NVIDIA GPU + driver)
    "JLD2",            # .jld2 field maps / results
    "DataFrames",
    "StaticArrays",
    "Evolutionary",
    "MAT",             # .mat sim files
    "GLMakie",         # heatmap PNG reports
    "FileIO",
    "DelimitedFiles",
    "CSV",             # used by test_mags.jl save_to_csv / export_csv
    "GPUArrays",
    "Gmsh",            # Stage 2: insert/STL geometry (CSV_to_STL.jl)
    "Dates",           # Stage 2: timestamps in JIG helpers
    "Optim",           # Phase 2: L-BFGS gradient optimizer (run_grad.jl)
    "HTTP",            # GUI: local web server (gui/server.jl)
    "JSON"             # GUI: config <-> browser JSON
])
# Statistics, LinearAlgebra, Random, TOML, SparseArrays are stdlibs (ship with Julia).

Pkg.instantiate()
Pkg.precompile()

println("\n=== Done. Environment ready in: ", @__DIR__, " ===")
println("CUDA functional check:")
using CUDA
@show CUDA.functional()
if CUDA.functional()
    CUDA.versioninfo()
end

using FileIO
using DataFrames, StaticArrays, JLD2, Statistics, LinearAlgebra
include("wrap.jl")
include("../kernels/f_kernel.jl")


function get_ppm_RMS(bestθ, λ, final_state, save_name)
    by_mean = CuArray([0.0f0])
    RMS_operation(bestθ, λ, final_state)
    @cuda threads=threads blocks=blocks shmem=shmem_sum   _mean!(B, by_mean, N, Nmsk)

    B_res = Array(B)

    dir = joinpath(OPTIMIZER_OUTPUT_DIR, save_name, "imgs")
    mkpath(dir)

    for i = 1:size(B_res, 3)
        slice = B_res[:,:,i]
        mask_slice = dmask[:,:,i]

        # Mask out-of-shell voxels with NaN (shown black).
        plot_data = copy(slice)
        plot_data[mask_slice .== 0] .= NaN

        # Skip slices with NO voxels inside the homogeneity shell. Otherwise the
        # colorrange would be (Inf, -Inf) on all-NaN data and GLMakie HANGS while
        # rendering. (The measured field spans z = ±525 mm but the shell is only
        # r = 100 mm, so most z-slices are entirely outside it.)
        valid = filter(!isnan, vec(plot_data))
        isempty(valid) && continue

        val_min, val_max = minimum(valid), maximum(valid)
        val_mean = mean(valid)
        val_max ≈ val_min && (val_max = val_min + 1f-6)   # avoid a zero-width colorrange

        ppm_slice = 1_000_000 * (val_max - val_min) / val_mean

        fig = Figure(size = (800, 600))
        ax = Axis(fig[1, 1];
            title  = "Fieldmap slice z = $(i)  (local ppm = $(round(ppm_slice, digits=2)))",
            xlabel = "x mm", ylabel = "y mm")
        hm = heatmap!(ax, xg, yg, plot_data;
            colormap = :viridis, colorrange = (val_min, val_max), nan_color = :black)
        Colorbar(fig[1, 2], hm, label = "mT")
        save(joinpath(dir, "$(i)_slice.png"), fig)
    end

    return 1000000 * (by_max - by_min) / by_mean
end


function regen_from_file(file)

    include(joinpath(@__DIR__, "..", "..", "stages", "stage1_optimize", "setup.jl"))

    @load file λ bestθ final_state ppm
    bestθ = CuArray(bestθ)
    final_state = CuArray(final_state)


    by_mean = CuArray([0.0f0])
    RMS_operation(bestθ, λ, final_state)
    @cuda threads=threads blocks=blocks shmem=shmem_sum   _mean!(B, by_mean, N, Nmsk)

    B_res = Array(B)

    for i = 1:201
       slice = B_res[:,:,i]
        mask_slice = dmask[:,:,i]

        # Create a copy for plotting and apply the mask
        # We replace 0s (or whatever your 'masked' value is) with NaN
        plot_data = copy(slice)
        plot_data[mask_slice .== 0] .= NaN 

        val_min = minimum(x -> isnan(x) ? Inf : x, plot_data)
        val_max = maximum(x -> isnan(x) ? -Inf : x, plot_data)
        val_mean = mean(filter(!isnan, plot_data))

        ppm_slice = 1000000*(val_max - val_min)/val_mean

        fig = Figure(size = (800, 600))

        ax = Axis(fig[1, 1];
            title = "Fieldmap slice z = $(i) with local ppm = $(round(ppm_slice, digits=2))",
            xlabel = "x mm",
            ylabel = "y mm"
        )

        # Use nan_color to make the masked areas black
        hm = heatmap!(ax, xg, yg, plot_data; 
            colormap = :viridis,
            colorrange = (val_min,  val_max),
            nan_color = :black  # This turns the NaNs black
        )
        
        Colorbar(fig[1, 2], hm, label = "mT")

        dir = string("runs/runs_lab/r100_lab_imgs/")
        mkpath(dir) 

        save(string("runs/runs_lab/r100_lab_imgs/","$(i)_slice.png"), fig)
    end

    return B_res
end

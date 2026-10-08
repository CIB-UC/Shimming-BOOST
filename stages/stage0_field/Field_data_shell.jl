# Field_data_shell.jl  —  STAGE 0 (shell mode)  measured CSV → spherical-shell jld2
#
#   julia Field_data_shell.jl        (or via run_pipeline.jl when eval_domain = :shell)
#
# Produces the evaluation set the optimizer scores on when `eval_domain = "shell"`:
# a single spherical SHELL at radius Rmax, rather than a dense grid. Motivation: in
# the current-free bore the field is HARMONIC, so by the maximum principle its
# min/max over the ball occur on the bounding sphere — scoring range/ppm on the
# Rmax shell is exact for the whole volume, with far fewer points. For a spherical
# measurement (Josh scan) this also skips the lossy interpolate-to-grid step.
#
# Shell source (config `shell_source`):
#   :measured      → use the RAW measured points + measured |B| directly (no fit, no
#                    resampling). Best when the scan already lies on the Rmax sphere —
#                    the optimizer then scores real data. `shell_n_points` is ignored.
#   :sh_fibonacci  → fit REGULAR solid spherical harmonics to the measured points (same
#                    fit as Field_data_SH_interpolator.jl — valid for a shell OR grid
#                    scan, since the field is harmonic), then evaluate that fit on a
#                    Fibonacci sphere of `shell_n_points` at Rmax (uniform, but smoothed).
# Either way it writes:
#     shell_xyz (3×Ns mm, optimizer frame),  By_shell (Ns mT),  main_field_direction
# to `shell_fieldmap_path`.  (The SH fit is always computed — it also feeds the viewer grid.)
#
# Reads the measured CSV exactly like the SH adapter (header row 2; X,Y,Z,Gauss;
# X/Y/Z scaled by sh_measured_unit_mm; drop (0,0,0); Gauss→mT; lab→optimizer rotate).
# Needs only CSV/DataFrames/JLD2/LinearAlgebra/Statistics/Printf — no GPU, no new deps.

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))
@isdefined(ITERATION) || include(joinpath(@__DIR__, "..", "..", "pipeline_config.jl"))
using CSV, DataFrames, JLD2, LinearAlgebra, Statistics, Printf
include(joinpath(@__DIR__, "..", "..", "core", "utils", "read_measured.jl"))   # read_measured_points, detect_field_direction
include(joinpath(@__DIR__, "..", "..", "core", "utils", "sh_select.jl"))       # sh_select_columns, sh_refit, sh_summary, sh_write_csv

const GAUSS_TO_MT = 0.1
const VALID_DIRS = ("+x", "-x", "+y", "-y")
@assert main_field_direction in (VALID_DIRS..., "auto") "main_field_direction = \"$(main_field_direction)\" invalid; use one of $(VALID_DIRS) or \"auto\"."

lab_to_optimizer_xy(x, y, dir) =
    dir == "+y" ? (x, y) : dir == "+x" ? (-y, x) : dir == "-x" ? (y, -x) : (-x, -y)

# --- solid-harmonic basis (identical to Field_data_SH_interpolator.jl) --------
function assoc_legendre_table(L::Int, ct::Float64, st::Float64)
    P = zeros(Float64, L + 1, L + 1); P[1, 1] = 1.0
    for m in 0:L-1; P[m+2, m+2] = -(2m + 1) * st * P[m+1, m+1]; end
    for m in 0:L-1; P[m+2, m+1] = (2m + 1) * ct * P[m+1, m+1]; end
    for m in 0:L, l in (m+2):L
        P[l+1, m+1] = ((2l - 1) * ct * P[l, m+1] - (l + m - 1) * P[l-1, m+1]) / (l - m)
    end
    return P
end
fact_ratio(l::Int, m::Int) = m == 0 ? 1.0 : 1.0 / prod(Float64(k) for k in (l-m+1):(l+m))
function real_Ylm(l::Int, m::Int, φ::Float64, P::Matrix{Float64})
    Nlm = sqrt((2l + 1) / (4π) * fact_ratio(l, abs(m)))
    m == 0 ? Nlm * P[l+1, 1] :
    m > 0  ? sqrt(2) * Nlm * P[l+1, m+1] * cos(m * φ) :
             sqrt(2) * Nlm * P[l+1, -m+1] * sin(-m * φ)
end
function solid_harmonic_row!(row, x, y, z, L, Rn)
    r = sqrt(x^2 + y^2 + z^2); ct = r > 0 ? clamp(z / r, -1.0, 1.0) : 1.0
    st = sqrt(max(0.0, 1 - ct^2)); φ = atan(y, x); P = assoc_legendre_table(L, ct, st)
    idx = 1
    for l in 0:L
        rl = (r / Rn)^l
        for m in -l:l; row[idx] = rl * real_Ylm(l, m, φ, P); idx += 1; end
    end
    return row
end

# --- 1. read + clean + fit ---------------------------------------------------
mx, my, mz, bgauss, meanvec = read_measured_points(measured_field_path; unit_mm = sh_measured_unit_mm)

# main field direction: auto-detect from the measured vector (rich/Controlled scan),
# otherwise use the config value.
detected = meanvec === nothing ? nothing : detect_field_direction(meanvec)
dir = main_field_direction == "auto" ?
      (detected === nothing ?
          error("main_field_direction = \"auto\" needs a scan with field components (Controlled format).") :
          detected) :
      main_field_direction
@assert dir in VALID_DIRS "main_field_direction resolved to \"$(dir)\" — must be one of $(VALID_DIRS)."

rotxy = [lab_to_optimizer_xy(mx[i], my[i], dir) for i in eachindex(mx)]
X = first.(rotxy); Y = last.(rotxy); Z = mz
b = bgauss .* GAUSS_TO_MT                              # |B| in mT (positive)
n = length(b)
radii = sqrt.(X .^ 2 .+ Y .^ 2 .+ Z .^ 2)
Rn = mean(radii)
ncoef = (sh_degree + 1)^2
n < ncoef && error("Not enough data points ($n) for sh_degree=$(sh_degree) ($(ncoef) terms).")

A = Matrix{Float64}(undef, n, ncoef); rowv = Vector{Float64}(undef, ncoef)
for i in 1:n; solid_harmonic_row!(rowv, X[i], Y[i], Z[i], sh_degree, Rn); A[i, :] = rowv; end
colnorms = [norm(@view A[:, j]) for j in 1:ncoef]; colnorms[colnorms .== 0] .= 1.0
c = (A ./ colnorms') \ b ./ colnorms

# --- which coefficients build the field map? (config: sh_select / sh_use_degree / sh_top_k)
# The decomposition of the FULL fit is always reported; for :first_n / :top_k the kept
# columns are re-fitted to the data and everything below (shell values, grid, saved `c`) uses that.
c_full   = copy(c)
sel_cols = sh_select_columns(c_full, sh_degree, sh_select; use_degree = sh_use_degree, k = sh_top_k)
sh_select === :all || (c = sh_refit(A, b, sel_cols))
resid = A * c .- b

# --- 2. build the SHELL the optimizer scores on ------------------------------
# `shell_source` (config) chooses where the optimizer's evaluation points come from:
#   :measured      → the RAW measured points + measured |B| (NO interpolation). Honest
#                    scoring on real data; valid because a shell scan already lies on
#                    the Rmax sphere (max-principle ⇒ its extrema bound the whole ball).
#                    `shell_n_points` is ignored (Ns = number of measured points).
#   :sh_fibonacci  → evaluate the SH fit on a uniform Fibonacci sphere at Rmax
#                    (`shell_n_points` pts). Use for scattered/partial scans or when a
#                    uniform point set is wanted; this incurs the SH-fit residual above.
function fibonacci_sphere(N, R)
    pts = Matrix{Float64}(undef, 3, N); ga = π * (3 - sqrt(5.0))
    for i in 1:N
        z = 1 - 2 * (i - 0.5) / N; rr = sqrt(max(0.0, 1 - z^2)); θ = ga * (i - 1)
        pts[1, i] = R * rr * cos(θ); pts[2, i] = R * rr * sin(θ); pts[3, i] = R * z
    end
    return pts
end

grow = Vector{Float64}(undef, ncoef)                   # SH row scratch (shell + viewer grid)
if shell_source === :measured
    shell_xyz = permutedims(hcat(X, Y, Z))             # 3×n measured coords (mm, optimizer frame)
    # :all → the real data, unmodified. With an SH selection active the SAME points carry the
    # SH-filtered field instead (row i of A is exactly the basis row at shell point i).
    By_shell  = sh_select === :all ? copy(b) : abs.(A * c)
else
    shell_xyz = fibonacci_sphere(shell_n_points, Rmax)  # 3×Ns, mm, optimizer frame
    By_shell  = Vector{Float64}(undef, shell_n_points)
    for i in 1:shell_n_points
        solid_harmonic_row!(grow, shell_xyz[1, i], shell_xyz[2, i], shell_xyz[3, i], sh_degree, Rn)
        By_shell[i] = abs(dot(grow, c))
    end
end
Ns = size(shell_xyz, 2)

mkpath(dirname(shell_fieldmap_path))
field_direction = dir                                 # resolved (possibly auto-detected)
sh_select_name = String(sh_select)               # JLD2-friendly copy of the mode
@save shell_fieldmap_path shell_xyz By_shell field_direction Rmax Rn sh_degree c c_full sel_cols sh_select_name

println("Stage 0 (shell)  wrote ", shell_fieldmap_path)
println("  main field B0 : ", dir,
        main_field_direction == "auto" ? "  (auto-detected from field vector $(round.(meanvec, digits=1)) G)" : "  (from config)")
@printf("  SH fit: degree %d (%d terms), %d measured points, ref radius Rn=%.2f mm%s\n",
        sh_degree, ncoef, n, Rn, shell_source === :measured ? "   (viewer grid only)" : "")
@printf("  fit residual: RMS=%.4f mT, max=%.4f mT\n", sqrt(mean(resid .^ 2)), maximum(abs.(resid)))
if shell_source === :measured
    rlo, rhi = extrema(radii)
    @printf("  shell: %d MEASURED points (%s), r ∈ [%.2f, %.2f] mm\n", Ns,
            sh_select === :all ? "no interpolation" : "positions as measured, values SH-filtered: sh_select = \"$(sh_select)\"", rlo, rhi)
    (rhi - rlo) > 2.0 && @warn "measured points span radii $(round(rlo,digits=1))–$(round(rhi,digits=1)) mm — looks like a VOLUME scan, not a single shell. Still scored on the real points, but the Rmax max-principle shortcut no longer applies exactly."
    abs(mean(radii) - Rmax) > 2.0 && @warn "measured mean radius $(round(mean(radii),digits=1)) mm ≠ Rmax $(Rmax) mm — check Rmax/units so the scored shell matches the DSV boundary."
else
    @printf("  shell: %d SH-fit points on sphere r=%.1f mm (interpolated)\n", Ns, Rmax)
end
@printf("  |B0| on shell ∈ [%.3f, %.3f] mT\n", minimum(By_shell), maximum(By_shell))

# --- 3. ALSO evaluate the same fit on a cube grid, ONLY for the viewers -------
# The optimizer uses the shell above; Shimming_magnets_visualizer.jl /
# field_slice_viewer.jl are grid-based, so write the standard grid jld2 too
# (identical format to Field_data_SH_interpolator.jl). Cheap: sh_grid_n³ points.
Rg = sh_grid_radius_mm; ng = sh_grid_n
xg = collect(range(-Rg, Rg; length = ng)); yg = collect(xg); zg = collect(xg)
By_grid = Array{Float64}(undef, ng, ng, ng)
for (i, x) in enumerate(xg), (j, y) in enumerate(yg), (k, z) in enumerate(zg)
    solid_harmonic_row!(grow, x, y, z, sh_degree, Rn)
    By_grid[i, j, k] = abs(dot(grow, c))
end
field_valid_radius_mm = Rn
mkpath(dirname(fieldmap_path))
@save fieldmap_path By_grid xg yg zg field_direction sh_degree c Rn field_valid_radius_mm c_full sel_cols sh_select_name
println("  viewer grid   → ", fieldmap_path, "  (", ng, "³, for the 3D/slice viewers)")
sh_summary(stdout, c_full, c, sel_cols, sh_degree; mode = sh_select, use_degree = sh_use_degree, k = sh_top_k, A = A, b = b)
println("  decomposition → ", sh_write_csv(joinpath(sh_report_dir, "sh_decomposition.csv"), c_full, c, sel_cols, sh_degree))
